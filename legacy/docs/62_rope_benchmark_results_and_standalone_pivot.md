# Doc 62: RoPE Benchmark Results and Standalone Pivot

Date: 2026-03-16

---

## RoPE position fix benchmark results

We benchmarked the RoPE position fix (doc 60) through the vLLM pipeline on
TPU v5p-8 with `--max-model-len 4096`. Five requests of 128 tokens each,
same prompt, temperature=0.

| Metric | Baseline (same positions) | RoPE Fix (separate positions) |
|--------|--------------------------|-------------------------------|
| Mean acceptance length | **3.53** | 2.89 |
| Avg draft acceptance rate | **16.9%** | 12.6% |
| Accepted throughput | **7.34** tok/s | 2.77 tok/s |

**The baseline outperforms the RoPE fix.** Separate context positions made
acceptance rate worse. The fix should NOT be applied.

---

## Why vLLM was involved

To benchmark the RoPE fix, we needed the full vLLM pipeline because the fix
targets `qwen3_dflash.py`, which is the vLLM-integrated model path. This
required:

1. Combining pr_dflash_1a (model+proposer), 1b (pipeline), 1c (tests) into
   `pr_dflash_test`
2. Mounting over `/workspace/tpu_inference` (underscore) to replace Docker's
   built-in copy
3. Using vllm-lkg on PYTHONPATH (Docker's built-in vLLM is too new)
4. Patching vLLM's `SpeculativeConfig` to accept `"dflash"` method
5. Fixing `get_model` 8-value unpack in `spec_decode/jax/dflash.py:85`
6. Using `--max-model-len 4096` (full 40960 exceeds VMEM)

---

## Two code paths in the PR

The PR (#1868) contributes both paths as part of the upstream tpu-inference PR
chain. Both are part of the stated scope (see `docs/51_pr_dflash_1_description.md`).

1. **Standalone benchmark harness** (`models/jax/dflash.py`, measured via
   `standalone_dflash.py`): Uses `flash_attention` directly with its own
   on-device KV cache. This produced tau=6.67, 4.26x speedup. Does NOT use
   `dflash_concat_attention`.

2. **vLLM-integrated model** (`models/jax/qwen3_dflash.py` +
   `layers/common/dflash_attention_interface.py`): Uses paged KV cache
   through `attention()`. Part of the PR scope, but full pipeline support
   also requires an upstream PR to vLLM to register the `dflash` method
   and `DFlashDraftModel` architecture.

The reviewer's comments target `dflash_attention_interface.py`, which is used
by path #2. The RoPE investigation only affects path #2. Path #1 already has
correct RoPE positions and is not affected.

Note: for the reviewer reply, we frame `dflash_attention_interface.py` as
forward-looking for vLLM pipeline integration. This is an internal framing
choice for the reply, not a scope reduction of the PR.

---

## Impact on PR reply

Updated `response/08_reply_pr_kyuyeunk_review.md` to:
1. Clarify standalone is the primary deliverable
2. Note `dflash_attention_interface.py` is forward-looking for future vLLM
   integration
3. Explain the original GPU repo also wraps standard attention in a custom
   module (same pattern, not missing a reuse opportunity)

---

## Artifacts

- `pr-ready/pr_dflash_1a`: RoPE fix code (uncommitted, DO NOT apply)
- `pr-ready/pr_dflash_test`: combined 1a+1b+1c with fixes for benchmarking
- `pr-ready/test_rope_positions.py`: synthetic RoPE comparison script
- `pr-ready/patch_vllm.py`: vLLM runtime patch for dflash method
- `docs/62_benchmark_raw_logs.md`: raw vLLM SpecDecoding metrics log lines
