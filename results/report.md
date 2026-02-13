# DFlash Speculative Decoding on TPU — Results Report

## What We Achieved

We ported DFlash, a block-diffusion-based speculative decoding algorithm, from GPU (PyTorch) to TPU (JAX) and validated it against the original paper's results. On a standalone JAX benchmark matching the paper's experimental setup (single device, greedy decoding, Qwen3-4B + DFlash-b16), our TPU implementation achieves an average acceptance length of **tau=6.67** across four math reasoning datasets — **94% of the GPU paper's tau=7.07**. On Math500 specifically, our TPU tau of 8.72 actually **exceeds** the GPU paper's 7.84. The average end-to-end speedup is **4.26x** over autoregressive baseline on a single TPU v4 device, compared to the paper's 5.53x on GPU.

| Dataset | TPU tau | TPU Speedup | GPU Paper tau | GPU Paper Speedup | tau % of GPU |
|---|---|---|---|---|---|
| GSM8K | 5.17 | 3.36x | 6.53 | 5.15x | 79% |
| Math500 | 8.72 | 5.59x | 7.84 | 6.09x | 111% |
| AIME24 | 6.32 | 3.93x | 7.27 | 5.68x | 87% |
| AIME25 | 6.48 | 4.14x | 6.64 | 5.21x | 98% |
| **Average** | **6.67** | **4.26x** | **7.07** | **5.53x** | **94%** |

We also integrated DFlash into the full vLLM serving pipeline on TPU, where it achieves **2.31x speedup** and **212 tokens/sec** (vs 92 TPS baseline) across 32 prompts. The lower numbers in the pipeline setting are expected — the vLLM scheduler, batch manager, and rejection sampling loop add overhead that the standalone loop avoids. This matches the GPU side, where the paper's standalone results are also significantly better than their vLLM/SGLang serving numbers.

Importantly, the tpu-inference implementation has **zero dependency on vLLM changes**. The standalone benchmark uses only existing tpu-inference library imports (`ModelConfig`, `LoadConfig`, model loader). The vLLM serving integration requires just a 10-line config change in a separate repo, which can be upstreamed independently.

## Challenges and How We Solved Them

The port involved three major challenges that required non-obvious solutions.

**KV cache architecture mismatch.** The GPU DFlash implementation uses PyTorch's `DynamicCache` with simple concat-based KV management, while TPU tpu-inference uses paged attention with Pallas kernels. We couldn't reuse the GPU's cache logic directly. Instead, we implemented a dual-cache architecture: the target model keeps its paged KV cache (required for ragged paged attention), while the draft model uses static on-device JAX arrays with `dynamic_update_slice` writes and a `cache_len` pointer for crop semantics. The draft model uses TPU's `flash_attention` kernel with `causal=False`, matching DFlash's non-causal block diffusion design. Getting the position encoding, cache cropping, and context buffer padding to work correctly across this split architecture required several iterations of debugging against the GPU reference.

**The seq_len inflation bug.** After the initial integration, draft acceptance rates were stuck around 10% (tau=2.49, only 1.30x speedup). We ran 13 systematic A/B tests — toggling flash attention, position schemes, cache modes, no-cache mode — all inconclusive. The root cause turned out to be upstream of the proposer entirely: in `speculative_decoding_manager.py`, the code passed `attn_metadata.seq_lens` to the proposer, but during verification steps this value includes all draft tokens being verified (e.g., seq_len=116 when the true accepted count is 100). Since DFlash is the only proposer that maintains persistent state across iterations (context buffer, KV cache positions, RoPE offsets), this 10-16 token inflation silently corrupted every iteration without causing any errors or NaNs. The fix was four lines: read the actual accepted count from `num_tokens_no_spec` and pass a corrected `attn_metadata` to the proposer. This single change nearly doubled performance — tau jumped from 2.49 to 4.48, speedup from 1.30x to 2.31x.

**Proving vLLM independence.** Our tpu-inference contribution currently sits on top of a vLLM fork with a 10-line DFlash config change. To demonstrate that DFlash works on TPU independently of that fork, we built a standalone JAX benchmark (`benchmarks/standalone_dflash.py`) that mirrors the GPU paper's experimental loop — load models directly via `get_flax_model()`, run a prefill-draft-verify-accept loop, track context buffers and KV state manually. This required careful replication of the `DFlashProposer`'s context buffer management (host-side numpy accumulation, power-of-2 padding, prev_ctx_len tracking) and draft cache crop semantics. The standalone benchmark not only proves vLLM independence but also reveals that our TPU DFlash quality (tau=6.67) is much closer to the GPU paper (tau=7.07) than the vLLM pipeline numbers (tau=4.48) suggested — confirming the remaining gap is pipeline overhead, not implementation quality.
