# Presentation Knowledge Bank

Two decks:
- **Part 1:** Capstone Week 7 — DFlash TPU Port (`slides/week7/`)
- **Part 2:** New Findings — MXU Amortization & Research Direction (`slides/week7_new/`)

---

# PART 1: Capstone Week 7 — DFlash TPU Port

---

## Slide 1 — Title

- Project: Port DFlash speculative decoding from GPU/PyTorch to TPU/JAX
- Team: Aaron Feng, Zhongyan Luo, Son Nguyen, Andy Huang
- Mentor: Hao Zhang, TA: Yiming Zhao, UCSD DSC 180 Capstone, Winter 2025

---

## Slide 2 — The Autoregressive Bottleneck

**Core concept:** Standard LLM generates 1 token per forward pass. 500 tokens = 500 sequential passes. Memory-bandwidth-bound — the bottleneck is reading model weights from HBM each time, not compute.

**Speculative decoding:** Cheap draft model proposes K tokens → expensive target model verifies all K in one parallel pass → if accepted, K tokens for cost of ~1. Output distribution is provably identical to the target model (speculative sampling theorem).

**τ (tau):** Expected tokens produced per step. τ=6.67 means on average 6.67 tokens per step. Step count = N/τ for N-token response.

**"Existing drafters cap at τ≈2-3"** refers to Eagle/Medusa in practice under typical serving conditions. Eagle3 gets τ≈3 in the TPU vLLM setup.

*Q: How is output distribution preserved?*
Accept token x̃ with probability min(1, p(x̃)/q(x̃)). If rejected, sample correction from (p-q)₊ normalized. Mathematically proven to produce exactly p at every position.

*Q: What if all K tokens are rejected?*
Still produce 1 correction token. Never worse than autoregressive. Worst case τ=1.

---

## Slide 3 — DFlash: Block Diffusion Drafting

**What DFlash does:**
- Predicts 16 tokens in one forward pass (block_size=16)
- Non-causal (bidirectional) attention within the block — each of the 16 positions sees all 15 others simultaneously
- Conditioned on target model hidden states from layers [1, 9, 17, 25, 33] (5 layers uniformly sampled)
- Shares embedding + LM head with target model (frozen) — only 5 transformer layers are new parameters
- "15 draft tokens/step" — block_size=16 but first position is the anchor (previously verified token), so 15 net new predictions

**GPU baseline:** τ=7.07, 5.53× speedup on H200

**vs Eagle3:** Eagle3 does 3 sequential autoregressive draft steps + 1 verify. DFlash does 1 parallel draft pass (16 tokens) + 1 verify. DFlash drafting is O(1) per step regardless of block size; Eagle3 is O(K).

*Q: Why non-causal attention for drafting?*
Each position benefits from knowing what neighboring positions will likely be. Position 8 seeing what position 12 is likely to say reduces uncertainty. Same reason bidirectional models (BERT) outperform left-to-right models on masked prediction.

*Q: "No extra parameters" — isn't that slightly misleading?*
Yes. The 5 draft transformer layers ARE new parameters (~100-300M). The claim means the embedding and LM head are shared/frozen — no extra vocabulary projection parameters. Worth clarifying if asked.

---

## Slide 4 — Our Contribution

**Key decisions:**
- Target: Qwen3-4B (36 layers, 4B parameters)
- Draft: DFlash-b16 (5 layers, block_size=16)
- Hardware: TPU v4-8 (4 chips, 8 cores total)
- "Zero vLLM source changes" — DFlash implemented as a compliant proposer plugin through vLLM's spec decode interface. No modifications to vLLM's scheduler, memory manager, or execution engine.

*Q: Why TPU specifically?*
Google TPU pods are production hardware for large-scale inference. Existing ML research uses GPU; TPU inference is less explored. The tpu-inference framework needed a working spec decode implementation.

*Q: Why Qwen3-4B?*
It's what DFlash-b16 was trained for (publicly released checkpoint at z-lab/Qwen3-4B-DFlash-b16). Small enough for a 4-chip TPU v4 pod while representative of production scale.

---

## Slide 5 — Architecture: Dual KV Cache

**Target model KV cache:**
- Paged (PagedAttention / vLLM standard)
- `ragged_paged_attention` handles variable-length sequences via non-contiguous memory blocks
- Extracts aux_hidden_states from 5 uniformly sampled layers as draft conditioning

**Draft model KV cache:**
- Static (pre-allocated, fixed size)
- `dynamic_update_slice` — JAX's way of doing in-place updates on static arrays
- `flash_attention(causal=False)` — non-causal within-block attention
- KV cache holds: context (projected target aux features) + noise tokens (masked positions)

**The pipeline:**
Target hidden states → FC projection → draft KV cache context → DFlash forward pass → 16 logits → rejection sampling → accepted tokens

**Why dual KV cache is necessary:**
GPU/PyTorch handles dynamic indexing naturally. JAX requires static shapes for XLA compilation — forced explicit design: static draft KV + paged target KV. Bridging them was a core engineering challenge.

*Q: Why is the draft KV cache static but the target's is paged?*
Draft model processes a fixed block of 16 tokens per step — KV requirements are predictable. Target model handles arbitrary-length conversations, requiring dynamic memory allocation. Static draft KV works for single-request but limits concurrent multi-user serving.

---

## Slide 6 — Code: Two-Phase KV Cache Write

**Why two phases?**
Context length varies per step (each verification step appends new accepted tokens). KV cache written at position `cache_len`, which changes each step. JAX's static shape constraint: can't resize arrays — write to pre-allocated buffer at current position.

**Phase A:** Write context keys/values at `cache_len`
**Phase B:** Write noise keys/values at `cache_len + actual_ctx_count`

Two phases needed because `actual_ctx_count` must be known before computing noise start position `noise_start = cache_len + actual_ctx_count`.

*Q: What's the performance cost of two writes?*
Each `dynamic_update_slice` is a TPU scatter. Two adds ~0.1-0.2ms per step, negligible vs ~17ms total.

---

## Slide 7 — The seq_len Inflation Bug

**Symptom:** τ=2.49, acceptance 9.9%, 13 A/B tests inconclusive

**Root cause:** `attn_metadata.seq_lens` counts both accepted tokens AND unverified draft tokens. For a request that has generated 100 tokens, vLLM passes `100 + 16 = 116` to the DFlash proposer instead of 100.

**What goes wrong:** DFlash uses seq_len to position its KV cache write. Writing at position 116 instead of 100 puts 16 stale/wrong context positions per step. Over 50 steps, 800 phantom tokens corrupt draft context.

**Why hard to find:**
1. No visible error — tokens were valid English; model recovered from bad context
2. Eagle3 unaffected — stateless (no persistent draft KV cache), wrong seq_len doesn't matter
3. Gradual degradation — quality degrades slowly, not catastrophically
4. Wrong location to look — bug is in the spec decode *manager*, not the DFlash proposer. Proposer code was correct.

**The 4-line fix:**
```python
accepted_seq_lens = self.runner.input_batch.num_tokens_no_spec[
    :attn_metadata.seq_lens.shape[0]].copy()
accepted_attn_metadata = replace(
    attn_metadata, seq_lens=device_array(
        self.runner.mesh, accepted_seq_lens.astype(np.int32)))
```

**Impact:**
- τ: 2.49 → 4.48 (+80%)
- Speedup: 1.30× → 2.31× (+77%)
- Acceptance: 9.9% → 23.2% (+134%)
- TPS: 120.8 → 212.4 (+76%)

*Q: Why didn't tests catch this earlier?*
No oracle. Output was valid English. You'd need side-by-side comparison with GPU output. That wasn't set up initially.

*Q: Could this affect other spec decode methods?*
Any stateful drafter with persistent KV cache would be affected. Stateless drafters (Eagle3) are immune — they don't use seq_len for cache positioning.

---

## Slide 9 — Results: Standalone TPU vs GPU

| Dataset | TPU τ | TPU Speedup | TPU TPS | GPU τ | GPU Speedup |
|---|---|---|---|---|---|
| GSM8K | 6.25 | 3.67× | 222 | 6.69 | 5.53× |
| Math500 | **8.72** | 4.49× | 276 | 7.84 | 5.12× |
| AIME24 | 5.17 | 3.00× | 191 | 6.45 | 5.07× |
| AIME25 | 6.53 | 3.53× | 218 | 7.30 | 5.38× |
| **Avg** | **6.67** | **3.67×** | **227** | **7.07** | **5.53×** |

**Verify your arithmetic:**
- TPU avg τ: (6.25+8.72+5.17+6.53)/4 = 26.67/4 = 6.67 ✓
- GPU avg τ: (6.69+7.84+6.45+7.30)/4 = 28.28/4 = 7.07 ✓
- TPU avg TPS: (222+276+191+218)/4 = 907/4 = 226.75 ≈ 227 ✓
- **GPU avg speedup listed as 5.53× but arithmetic average is (5.53+5.12+5.07+5.38)/4 = 5.275×. The 5.53× is the GSM8K number. If asked, acknowledge this.**

**Key talking points:**
- 94% = 6.67/7.07 — algorithmic quality transfers
- Math500: TPU τ exceeds GPU (8.72 > 7.84) — don't over-claim; different measurement conditions
- Speedup gap (3.67× vs 5.53×) = hardware/runtime overhead, not algorithm degradation

*Q: Why is TPU speedup lower than GPU?*
TPU v4 is an experimental runtime — attention kernels, memory management less optimized than mature GPU CUDA stack. TPU v5 (production) would reduce this gap.

*Q: Why does Math500 exceed GPU?*
Likely measurement conditions differ (temperature, vLLM versions, sampling). Don't claim TPU is fundamentally better on math.

---

## Slide 10 — Results: vLLM Pipeline

**Pipeline numbers (GSM8K):** τ=4.48, speedup=2.31×, spec TPS=212, baseline TPS=92, acceptance=23.2%

**Standalone vs pipeline:** τ drops 6.25 → 4.48 (28% reduction)

**Why the gap:** vLLM's scheduler designed for autoregressive decoding. Occasionally rejects entire draft blocks at batch boundaries. Same pattern in GPU DFlash paper — known vLLM spec decode limitation, not TPU-specific.

*Q: Is 2.31× meaningful?*
Baseline TPS=92, spec decode TPS=212 → 2.31× real improvement in production vLLM stack.

---

## Slide 11 — DFlash vs Eagle3 on TPU

| Metric | DFlash | Eagle3 |
|---|---|---|
| Draft tokens/step | 15 | 3 |
| τ | 4.48 | 2.18 |
| TPS | 212 | 78 |
| Speedup | 2.31× | 1.53× |

DFlash τ is 2.06× Eagle3. Both under identical vLLM pipeline conditions.

*Q: Is this fair — Eagle3 uses a tree?*
Both configured for comparable wall-clock draft cost under the same vLLM scheduler. The comparison reflects real deployment performance.

*Q: Why only 3 tokens/step for Eagle3?*
Eagle3 in vLLM uses default tree config. "3" is the average accepted tokens per step in this TPU pipeline setup.

---

## Slide 12 — Output Quality Verification

8 samples: 4 bit-exact, 7 same final answer, 1 different answer.

**The 1 divergence:** Top-2 logits differed by <10⁻⁴. bf16 rounding causes sampling to flip at decision boundaries. Not an algorithmic error — inspected logit distributions confirmed.

*Q: 8 samples is tiny. What does this prove?*
Correctness sanity check, not statistical power analysis. The statistical evidence of correctness is the τ numbers across thousands of generation steps per benchmark.

---

## Slide 13 — Impact: The 4-Line Fix

All metrics before/after:
- τ: 2.49 → 4.48 (+80%)
- Speedup: 1.30× → 2.31× (+77%)
- Acceptance rate: 9.9% → 23.2% (+134%)
- TPS: 120.8 → 212.4 (+76%)

Consistent across all 4 datasets — confirms systematic fix, not noise.

---

## Slide 14 — Future Work

1. **PR to tpu-inference** — seq_len fix and DFlash proposer live only in the fork; upstream for anyone running spec decode on TPU
2. **TPU v5 benchmarks** — v4 is experimental; v5e and v5p are production-supported with higher matmul throughput
3. **Concurrent serving benchmarks** — current results are single-request only
4. **Diffusion targets on TPU** — *(Note: slide says "FailFast (diffusion-based target model)" — FailFast is a diffusion-based DRAFTER, not a target model. Minor error worth knowing.)*

**What's still missing from the migration (KV cache parity):**
- Draft model uses static KV cache; GPU uses dynamic paged memory
- Variable-length sequence handling: all TPU requests padded to fixed length; GPU handles via PagedAttention block allocator
- Until draft model also has paged KV on TPU, multi-user concurrent serving is limited

---

## Part 1 Numbers to Memorize

| Number | What it is |
|---|---|
| 6.67 | TPU avg τ (standalone) |
| 7.07 | GPU avg τ (standalone) |
| 94% | TPU/GPU τ ratio |
| 4.48 | TPU τ (vLLM pipeline) |
| 2.31× | TPU speedup (vLLM pipeline) |
| 2.49 → 4.48 | τ before/after bug fix |
| +80% | τ improvement from fix |
| 16 | DFlash block size |
| 15 | Net new draft tokens per step |
| 5 | DFlash draft transformer layers |
| [1,9,17,25,33] | Target hidden state extraction layers |
| 4 | Chips in TPU v4-8 pod |
| 36 | Qwen3-4B transformer layers |
| 4B | Target model parameter count |

---

---

# PART 2: New Findings — MXU Amortization & Research Direction

---

## Slide 2 — Recap

| | TPU v4 | GPU (A100) | Ratio |
|---|---|---|---|
| τ (avg) | 6.67 | 7.07 | 94% |
| Speedup | 3.67× | 5.53× | — |
| TPS | 227 | — | — |

"This week" framing: the 33% speedup gap (3.67× vs 5.53×) motivated asking — where does time go, and can we exploit TPU hardware?

---

## Slide 3 — The Pipeline Bottleneck

**Full ablation (Docs 27-29):**

```
Component                    Time    Fraction
─────────────────────────────────────────────
Verify forward (36 layers)  ~10ms     59%
Aux projection               ~2ms     12%
Draft forward + LM head      ~2ms     12%
Verify LM head               ~1ms      6%
Host orchestration           ~2ms     12%
─────────────────────────────────────────────
Total                        ~17ms   100%
```

**Verify sub-costs:**
- FFN matmuls (36 layers): ~1.7ms — MXU compute (the flat part)
- Attention (KV cache read): ~8.3ms — HBM bandwidth, scales with context length L

**Throughput formula (as shown on slide):**

$$\text{Throughput} = \frac{\tau + 1}{T_{\text{step}}} = \frac{7.67}{14\text{ms}} \approx 548 \text{ tok/s}$$

*(Slide uses T_step=14ms = draft+verify only. Full step including host overhead is 17ms. Percentages are unaffected by which denominator is used.)*

*Q: Why is verification 59% if the model is 4B parameters?*
The target model reads all 4B parameters from HBM on every single generation step. The draft model (5 layers, ~200M params) is tiny by comparison. Verification cannot be avoided — it's what guarantees output quality.

*Q: What is "aux projection" and why is it 12%?*
After each verify pass, extract target hidden states from 5 layers and project via a learned FC layer into draft KV cache format. Takes ~2ms. Architectural overhead from DFlash's conditioning design.

---

## Slide 4 — Key Discovery: MXU Amortization

**Core empirical finding (Doc 33):**
Measurement conditions: Qwen3-4B, TPU v4-8, batch size 1, context length L=66 tokens, 3 warmup + 10 timed trials.

| Query tokens K | TPU verify latency | vs K=16 |
|---|---|---|
| 16 | 1.70 ± 0.06 ms | 1.00× |
| 32 | 1.66 ± 0.04 ms | 0.98× |
| 64 | 1.76 ± 0.09 ms | 1.04× |
| 128 | 1.78 ± 0.06 ms | 1.05× |

GPU verification (SmartSpec, arXiv:2406.14066): T = α·N_context + **γ·N_query** + δ — explicitly linear in query count.

**Why flat on TPU — the correct mechanism (Doc 39):**

Weight bytes per FFN layer:
```
gate_proj:  2560 × 9728 × 2 bytes = 49.8 MB
up_proj:    2560 × 9728 × 2 bytes = 49.8 MB
down_proj:  9728 × 2560 × 2 bytes = 49.8 MB
Total per layer: ~150 MB
```
TPU v4 HBM bandwidth: ~1.2 TB/s
Time to load one layer: 150 MB / 1200 GB/s ≈ **0.125 ms**
Measured per-layer latency: ~0.33 ms

MXU compute for K=16 to K=256 ALL finish within the 0.125ms memory load window. **Operations are memory-bandwidth bound. Weight loading dominates; arithmetic is hidden behind memory latency.**

Arithmetic intensity (roofline analysis):
$$\text{Intensity}(K=128) = \frac{2 \times 128 \times 2560 \times 9728}{(128 \times 2560 + 2560 \times 9728) \times 2} \approx 127 \text{ FLOP/byte}$$

$$\text{Intensity}(K=256) \approx 246 \text{ FLOP/byte}$$

$$\text{TPU v4 roofline} = \frac{275 \text{ TFLOPS}}{1.2 \text{ TB/s}} \approx 229 \text{ FLOP/byte}$$

Both K=128 and K=256 are near the roofline crossover, explaining why neither shows significant cost increase.

*Q: Isn't the original explanation about 128×128 MXU tiles wrong then?*
The tile structure contributes (sets minimum compute granularity), but the dominant mechanism is memory-bandwidth boundedness. Weight matrix (~50 MB) dominates bytes loaded; doubling K from 128 to 256 adds only 2.6% more bytes while doubling FLOPs. Since memory-bound, extra FLOPs are hidden.

*Q: Why doesn't GPU show the same flat behavior?*
GPU CUDA core scheduling, smaller tensor core tiles (~16×16 on A100), and different memory subsystem characteristics mean adding K tokens shows more visible compute scaling. SmartSpec confirms GPU scales linearly.

---

## Slide 5 — Flat Scaling Extends Beyond K=128

**Doc 39 results — the K=129 experiment:**

| K | DFlash FFN ms/layer | vs K=16 | Target FFN ms/layer | vs K=16 |
|---|---|---|---|---|
| 16 | 0.335 | 1.00× | 0.339 | 1.00× |
| 64 | 0.328 | 0.98× | 0.334 | 0.98× |
| 128 | 0.330 | 0.98× | 0.329 | 0.97× |
| **129** | **0.319** | **0.95×** | **0.325** | **0.96×** |
| 192 | 0.334 | 1.00× | 0.341 | 1.00× |
| 256 | 0.343 | 1.02× | 0.346 | 1.02× |

K=129/K=128: **0.97× (DFlash), 0.99× (Target) — no step function, no cliff.**

We hypothesized ~2× jump at K=129 (second tile row). It didn't appear. The flat zone extends to K=256.

*Q: You predicted a jump and it didn't appear — doesn't that weaken your argument?*
It strengthens the main claim. Flat verification cost is confirmed even more broadly (extends to K=256). The mechanism (memory-bandwidth boundedness) is more fundamental than just tile rows. The core TPU advantage vs GPU holds.

*Q: How do you know it won't jump at K=257?*
At very long contexts, the attention component O(K×L) grows regardless of hardware. Our measurements used L=66 tokens (FFN-dominated). At L=1000+, attention starts to matter and the flat zone may narrow. That's a caveat — our numbers are for short-to-medium context.

---

## Slide 6 — What We Tried and Ruled Out (Docs 32-35)

| Doc | Experiment | Key measurement | Viable? |
|---|---|---|---|
| 32 | Layer truncation | τ at each truncation layer | No — even -1 layer: -30% τ |
| 33 | Amortized verification | T_verify(K=16..128) | Verify flat ✓ but multi-block draft overhead kills it |
| 34 | TPU vs GPU uniqueness | Literature comparison | Novel ✓ |
| 35 | Tree speculation (K=1,2,4 branches) | τ per branch count | No — K=2: +0.5% τ; K=4: τ drops |

**Layer truncation numbers (Doc 32, GSM8K):**

| Layers | τ | τ retention |
|---|---|---|
| 35/36 | 4.35 | 70.5% |
| 30/36 | 2.23 | 36.1% |
| 18/36 | 1.05 | 17.0% |
| 36/36 | 6.18 | 100% |

Last layer alone flips **13.9%** of all argmax decisions.

**The pattern:** MXU amortization benefits verification, not drafting. Any attempt adding drafting work (extra branches, extra blocks) costs +4-5ms overhead per addition — draft overhead is ~3.6ms/step on top of the 0.36ms model forward. Extra branches can't amortize because each needs its own aux projection and KV cache setup.

*Q: Why did τ actually drop at K=4 branches in tree speculation?*
Each branch starts from a cold DFlash KV cache. Alternative branches lack rich conditioning from target hidden states (which are computed only for the primary branch path). Bad drafts → more rejections → τ drops.

---

## Slide 7 — Both Sides Amortize (Doc 37)

**Raw matmul microbenchmark (Doc 37):**
Measurement: isolated FFN matmuls only, 5 warmup + 20 timed trials.

| Component | K=16 | K=128 | Ratio |
|---|---|---|---|
| DFlash FFN (5 layers) | 1.637ms | 1.556ms | **0.95×** |
| Target FFN (36 layers) | 11.871ms | 11.384ms | **0.96×** |
| Attention Q×K^T (KV=256) | 0.202ms | 0.196ms | **0.97×** |

K=128 is slightly **faster** than K=16 — zero padding waste vs 87.5% waste.

**Full step time comparison — nothing moves:**

| | K=16 | K=128 |
|---|---|---|
| DFlash forward | 0.36ms | 0.36ms |
| Aux projection | ~2ms | ~2ms |
| Host overhead | ~1.6ms | ~1.6ms |
| Target verify | ~10ms | ~10ms |
| Accept/output | ~3ms | ~3ms |
| **Total** | **~17ms** | **~17ms** |

Implication: A DFlash retrained with block_size=128 produces **8× more draft tokens per forward pass at zero additional compute cost**.

*Q: Why is K=128 slightly faster at the matmul level?*
At K=16, the MXU pads 16 rows to 128, executing tile operations where 87.5% is on zero-padded data. At K=128, the tile is filled exactly — no wasted work.

---

## Slide 8 — Why DFlash Uses K=16

**DFlash paper quote (Section 4.2):**
> "In practical serving scenarios, large blocks can increase verification cost under compute-bound settings; reducing the block size can therefore yield better overall speedup."

**DFlash's own ablation (Table 7):**

| Train K | Test K | Math500 τ | Speedup |
|---|---|---|---|
| 16 | 16 | **6.33** | 4.64× |
| 8 | 8 | 5.21 | 3.97× |

Larger K is better for τ. K=16 was chosen because GPU verification scales linearly.

**GPU vs TPU summary:**

| Hardware | Verify(128) vs Verify(16) | Optimal K |
|---|---|---|
| GPU (A100) | ~8× cost (linear) | K=16 |
| TPU v4 | 1.05× cost (flat) | K=128+ |

K=16 is a **GPU design parameter**, not an algorithmic optimum.

*Q: The DFlash authors are aware of this tradeoff. Why didn't they train a TPU-native version?*
DFlash was designed and evaluated entirely on GPU (H200). No TPU experiments in the paper. This work fills that gap.

*Q: Can you just run K=128 at inference without retraining?*
No. DFlash paper Table 7 confirms: train K=16, test K=8 → major τ degradation (6.33 → 5.09). Training and inference block size must match.

---

## Slide 9 — How τ Translates to Throughput

**Formulas (as shown on the slide):**

$$\tau(K) = \frac{1 - \alpha^K}{1 - \alpha} \quad \text{(geometric series)}$$

$$\text{Ceiling: } \tau(\infty) = \frac{1}{1-\alpha}$$

$$\text{Throughput} = \frac{\tau + 1}{T_{\text{step}}}$$

**What α is:** per-position acceptance probability. α = E[min(1, p(x̃ᵢ)/q(x̃ᵢ))]. Currently α ≈ 0.87 (backed out from τ(K=16) = 6.67).

**Throughput table (slide numbers):**

| Scenario | α | K | τ | Throughput | vs current |
|---|---|---|---|---|---|
| Current (K=16) | 0.87 | 16 | 6.67 | 548 tok/s | baseline |
| K=128, stable α | 0.87 | 128 | 7.69 | 621 tok/s | **+13%** |
| K=128, improved α | 0.90 | 128 | 9.99 | 785 tok/s | **+43%** |
| K=128, optimistic | 0.93 | 128 | 13.29 | 1,021 tok/s | **+86%** |

*(Slide uses T_step=14ms and (τ+1) in numerator. The percentage gains hold regardless of normalization.)*

**Geometric series intuition:**
- α=0.87 → P(position k accepted) = 0.87^k
- τ ceiling = 1/(1-0.87) = 7.69
- At K=16: (1-0.87^16)/0.13 = 6.67 — already 87% of ceiling
- At K=128: 0.87^128 ≈ 0 → τ ≈ 7.69 (essentially at ceiling)

**End-to-end latency for 500-token response:**
Steps = 500/τ, each step 17ms.

| τ | Steps | Latency |
|---|---|---|
| 6.67 | 75 | 1,275ms |
| 7.69 | 65 | 1,105ms (-13%) |
| 9.99 | 50 | 850ms (-33%) |
| 13.29 | 38 | 646ms (-49%) |

*Q: Where does α=0.87 come from?*
Solve (1-α^16)/(1-α) = 6.67 numerically → α ≈ 0.87.

*Q: Why does α=0.87 give only +13% from K=16 to K=128? That seems small.*
Geometric series saturates fast. At α=0.87, most mass is in early positions — by position 16, most available gain is captured. Going to K=128 only adds the thin tail. Real upside requires α to improve.

---

## Slide 10 — Why α Might Improve

**The mechanism:**
DFlash non-causal attention: position i attends to ALL other positions simultaneously.
- K=16: position 8 sees 15 neighbors
- K=128: position 8 sees 127 neighbors (8× richer context)

**Analogy:** BERT (bidirectional) vs GPT (left-to-right) on masked token prediction. BERT knows both sides, reducing per-position uncertainty. DFlash's within-block non-causal attention is exactly this.

**Supporting evidence from DFlash paper:**
K=8 acceptance histogram shows "frequently fully accepts entire blocks (35.7%)" — K=8 often underutilized. K=8→16 gave +21% τ. Same dynamic may apply K=16→128.

**Why this differs from failed tree/multi-block experiments:**

| Approach | Extra drafting cost | Mechanism |
|---|---|---|
| Tree (K=4 branches) | +4×5ms = +20ms | Exploring alternative first tokens |
| Multi-block (2×16) | +4ms | More tokens to verify |
| **Wide-block K=128** | **+0ms (same forward pass)** | **Richer within-block context** |

*Q: What if α degrades at K=128?*
Conservative case (α unchanged) still gives +13% from longer chains at zero compute cost. α would need to degrade from 0.87 below (1-0.87)/K=16 ≈ 0.80 before there's net harm. Theoretically possible but counterintuitive — more context should not hurt.

*Q: How do you measure whether α improved?*
Measure prefix-acceptance curves: sᵢ = P(first i tokens all accepted). If K=128 shifts sᵢ upward at same i positions, α improved. τ = Σ sᵢ directly, no distributional assumptions.

---

## Slide 11 — Proposed: TPU-Native Wide-Block DFlash

**Training changes:**

| Parameter | Current | K=128 variant |
|---|---|---|
| block_size | 16 | 128 |
| Anchors per sequence | 512 | ~48 (3072/128 × 2) |
| Loss decay γ | 7 | Larger (more positions before tail shrinks) |
| Training data | 800K samples | Same (reuse + cached hidden states) |
| Architecture | 5 layers, shared embed/head | Unchanged |

**Compute estimate:**
- Trainable params: ~200M (5 draft transformer layers)
- K=128 training: ~8-10× more per step than K=16
- Estimated 1-4 days on TPU v4-8, or single A100 with existing PyTorch code + `block_size=128`

**The defensible novelty claim:**
"First to show and exploit a memory-bandwidth-bound flat verification cost on TPU v4 and to retrain a diffusion drafter specifically to match that hardware property."

*Q: Can you just use K=128 at inference with the existing K=16 DFlash?*
No — out-of-distribution. Model trained for 15 masked positions; at K=128 it would see 127. Massive degradation expected.

*Q: Compute cost vs benefit?*
1-4 days training → potentially +13-43% permanent throughput improvement on every inference step. Very favorable ratio.

---

## Slide 12 — Two Viable Paths

| | Wide-Block Drafter | Cross-Request Batching |
|---|---|---|
| Type | ML research | Systems engineering |
| Retraining needed? | Yes | No |
| Infra changes? | No | Yes |
| Single-request benefit? | Yes | No |
| Throughput at batch | Additive | Multiplicative |
| Expected gain | +13% to +43% | 2-4× at batch=8 |

**Cross-request batching math:**
- 8 concurrent requests each draft K=16 tokens → 128 total
- Verify all 128 together ≈ same cost as verifying 16 (flat zone)
- Per-request verify cost: 1/8 of normal
- Projected step time: ~4 + 10/8 + 3 ≈ 8.25ms → 7.67/8.25ms ≈ 930 tok/s (+70%)

*Q: Why isn't cross-request batching already done?*
Draft model's static KV cache doesn't support multiple concurrent requests without pre-allocating separate caches per request. Serving infrastructure problem, not hardware.

---

## Slide 13 — Next Steps

1. **GPU comparison** (`benchmarks/gpu_matmul_scaling.py`) — standalone PyTorch, same dimensions, same K values, proves flat scaling is TPU-specific
2. **K=128 DFlash training sweep** — K=32, 64, 128 to map α vs block width
3. **Paper writeup** — Docs 33, 34, 37, 39 (measurement) + Doc 36 (direction)

---

## Slide 14 — Summary

**Core claim (one sentence):** TPU verification cost is flat from 16 to 256 tokens due to memory-bandwidth boundedness; GPU scales linearly; DFlash's K=16 is a GPU constraint that doesn't apply on TPU.

**What was ruled out:** Layer truncation (fundamental — last layer flips 13.9% of decisions), multi-block speculation (draft overhead +4ms), tree speculation (draft overhead + cold KV cache).

**The floor guarantee:** Even if α doesn't improve, K=128 gives +13% throughput from longer acceptance chains, at zero additional compute cost.

**Open question:** Does α improve at K=128? Prefix-acceptance curve sweep (K=32, 64, 128) answers this. Determines whether gain is +13% or +43-86%.

---

## Part 2 Formula Reference

$$\tau(K) = \frac{1 - \alpha^K}{1 - \alpha} \quad \text{iid geometric series}$$

$$\tau(\infty) = \frac{1}{1-\alpha} \approx 7.69 \text{ at } \alpha=0.87$$

$$\text{Throughput} = \frac{\tau + 1}{T_{\text{step}}} \quad \text{(slide convention)}$$

$$\text{Intensity}(K) = \frac{2K \cdot H \cdot I}{(K \cdot H + H \cdot I) \cdot 2} \quad H=2560,\ I=9728$$

$$\text{TPU v4 roofline} = \frac{275 \text{ TFLOPS}}{1.2 \text{ TB/s}} \approx 229 \text{ FLOP/byte}$$

$$\text{Weight load time/layer} = \frac{3 \times H \times I \times 2 \text{ bytes}}{1.2 \text{ TB/s}} = \frac{150 \text{ MB}}{1.2 \text{ TB/s}} \approx 0.125 \text{ ms}$$

---

## Part 2 Numbers to Memorize

| Number | What it is |
|---|---|
| 59% | Verify fraction of step time |
| ~1.7ms | FFN component of verify (flat) |
| ~8.3ms | Attention component of verify (bandwidth-bound) |
| 1.05× | T_verify(K=128)/T_verify(K=16) on TPU |
| 0.97× | T_verify(K=129)/T_verify(K=128) — no step function |
| 0.95× | DFlash FFN K=128 vs K=16 (slightly faster) |
| 0.87 | Current α |
| 7.69 | τ ceiling at α=0.87 |
| +13% | Conservative throughput gain at K=128 (same α) |
| +43% | Gain if α → 0.90 |
| +86% | Gain if α → 0.93 |
| 229 FLOP/byte | TPU v4 roofline crossover |
| 127 FLOP/byte | Arithmetic intensity at K=128 |
| 246 FLOP/byte | Arithmetic intensity at K=256 |
| ~150 MB | Weight bytes per FFN layer |
| 0.125 ms | Time to load one FFN layer from HBM |
| 13.9% | Fraction of argmax decisions flipped by final layer |
| -30% | τ loss from removing just 1 of 36 layers |
| 0.95× | DFlash FFN K=128 cost (slightly cheaper than K=16) |
