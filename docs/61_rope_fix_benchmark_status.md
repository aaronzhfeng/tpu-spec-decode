# Doc 61: RoPE Position Fix — Benchmark Status

Date: 2026-03-15

---

## What we found

Context and noise keys in `qwen3_dflash.py` receive identical RoPE positions
(`md.input_positions`), but should receive different positions:
- Context keys: positions starting at `cache_len` (earlier in sequence)
- Noise keys: positions at current block offset (later in sequence)

Both the GPU reference (`z-lab/dflash/model/dflash.py:77-82`) and the standalone
TPU proposer (`models/jax/dflash.py:529-531`) use separate positions.

See doc 60 for full analysis.

## What we confirmed

Synthetic test (`pr-ready/test_rope_positions.py`) ran in Docker on v5p-8:
- Max absolute difference between same vs different positions: **0.86**
- Mean absolute difference: **0.15**
- Outputs are **materially different**

The fix is not a no-op. Different positions change the attention output.

## Fix location

Code in `pr-ready/pr_dflash_1a/tpu_inference/models/jax/qwen3_dflash.py`
(uncommitted). Changes:
- `Qwen3DFlashModel.__call__`: unpacks 3-tuple, constructs `ctx_positions`
- `Qwen3DFlashDecoderLayer.__call__`: accepts and forwards `ctx_positions`
- `Qwen3DFlashAttention.__call__`: uses `ctx_positions` for k_ctx RoPE

Backward-compatible: `ctx_positions=None` default falls back to old behavior.

## E2e benchmark blocked

Attempted to run the full vLLM serving pipeline to measure tau impact.

**Blockers:**
1. `dflash` not registered as a valid speculative method in vLLM
   - Patched `vllm/config/speculative.py` (SpeculativeMethod Literal + __post_init__)
2. `DFlashDraftModel` architecture not in vLLM model registry
   - Available in tpu-inference's model_loader but vLLM validates first
   - `--trust-remote-code` gets past this
3. vLLM LKG version mismatch: tpu_platform expects `vllm.v1.attention.backends.pallas`
   which doesn't exist in current vllm-lkg
   - This is a version compatibility issue between pr-ready/vllm-lkg and the
     dflash-integration tpu-inference code

The PR split (1a/1b/1c) and the vLLM LKG version make it impossible to run
the full pipeline from the current pr-ready setup without also updating vllm-lkg.

## What's needed to benchmark

1. A vLLM version that has `vllm.v1.attention.backends.pallas`
2. Or: run the benchmark using the original Docker setup that was used for
   the tau=4.48 vLLM pipeline results (which vLLM commit was that?)
3. The fix only affects the `qwen3_dflash.py` path (vLLM-integrated), not
   the standalone proposer (which already has correct positions and tau=6.67)

## Progress on e2e benchmark (2026-03-15)

### What worked
- Docker image's built-in vLLM is too new (v0.13.0). Must use vllm-lkg.
- Mount tpu-inference over `/workspace/tpu_inference` (underscore, not hyphen)
  to replace Docker's built-in copy.
- PYTHONPATH: `/workspace/vllm-lkg:/workspace/tpu_inference`
- Combined pr_dflash_1c + cherry-pick 1b + cherry-pick 1a into `pr_dflash_test`
- Fixed `get_model` 8-value unpack in `spec_decode/jax/dflash.py:85`
- **Server started successfully**: target model loaded, DFlash draft model
  loaded, KV caches allocated, compilation completed, Application startup
  complete.

### What's still broken
- First inference request crashes the server (container exits).
- Need to debug the runtime error in the speculative decoding path.
- This is a separate issue from the RoPE fix.

### Working Docker command
```bash
sudo docker run -d --rm --privileged --net=host \
  --name dflash_test \
  -v /path/to/pr_dflash_test:/workspace/tpu_inference \
  -v /path/to/vllm-lkg:/workspace/vllm-lkg \
  -v /dev/shm/hf-cache:/dev/shm/hf-cache \
  -e PYTHONPATH=/workspace/vllm-lkg:/workspace/tpu_inference \
  -e HF_HOME=/dev/shm/hf-cache \
  vllm/vllm-tpu:latest \
  bash -c 'pip install -q flax==0.12.2 && python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B --trust-remote-code \
    --speculative-config "{...dflash config...}" --port 8000'
```

## E2e benchmark results (2026-03-15)

Server ran successfully with `--max-model-len 4096` (full 40960 exceeds VMEM).
5 requests of 128 tokens each, same prompt, temperature=0.

| Metric | Baseline (same positions) | RoPE Fix (separate positions) |
|--------|--------------------------|-------------------------------|
| Mean acceptance length | **3.53** | 2.89 |
| Avg draft acceptance rate | **16.9%** | 12.6% |
| Accepted throughput | **7.34** tok/s | 2.77 tok/s |

**The baseline outperforms the RoPE fix.** Separate context positions made
acceptance rate worse, not better.

Possible explanations:
1. The vLLM-integrated path may work differently from the standalone proposer.
   The standalone proposer writes both context and noise into the same on-device
   KV cache at different offsets (naturally different positions). The vLLM path
   uses paged KV cache with separate context handling.
2. The 3-tuple unpacking may have introduced a subtle issue (e.g., ctx_positions
   computed from cache_len may not align with what the paged cache expects).
3. Both acceptance rates are low compared to standalone (tau=6.67), suggesting
   the pipeline integration has other bottlenecks.

## Recommendation

**Do NOT apply the RoPE position fix.** The benchmark shows it degrades
acceptance rate. The current code (same positions for context and noise keys)
performs better in the vLLM pipeline path.

Artifacts:
- `pr-ready/pr_dflash_1a`: RoPE fix (uncommitted)
- `pr-ready/pr_dflash_test`: combined 1a+1b+1c with fix + get_model unpack fix
- `pr-ready/test_rope_positions.py`: synthetic comparison script
