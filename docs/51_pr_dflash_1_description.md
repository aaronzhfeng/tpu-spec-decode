# PR #1868 (updated)
# Title: [Spec Decoding] Add DFlash model and proposer

---

# Description

Add DFlash draft model and proposer for block-diffusion speculative decoding on JAX/TPU. DFlash predicts multiple tokens in parallel using discrete diffusion, unlike Eagle3's autoregressive drafting. This follows the same proposer pattern as Eagle3.

This is PR 1 of 3 for DFlash support:
1. **This PR:** Model, proposer, and unit tests (all new files)
2. Pipeline integration (modifications to existing files)
3. E2E tests and Buildkite CI

**New files:**
- `tpu_inference/models/jax/dflash.py` -- DFlash draft model (DFlashForCausalLM)
- `tpu_inference/models/jax/qwen3_dflash.py` -- Qwen3-specific DFlash variant with attention
- `tpu_inference/layers/common/dflash_attention_interface.py` -- dflash_concat_attention kernel
- `tpu_inference/spec_decode/jax/dflash.py` -- DFlashProposer (prepare_inputs, propose, sampling)
- `tests/models/jax/test_qwen3_dflash_attention.py` -- DFlash attention unit tests
- `tests/models/jax/test_qwen3_dflash.py` -- target layer ID selection tests
- `tests/spec_decode/test_dflash.py` -- proposer sampling tests

# Tests

- Unit tests for DFlash attention (concat, additive bias, GQA): `tests/models/jax/test_qwen3_dflash_attention.py`
- Unit tests for target layer ID selection: `tests/models/jax/test_qwen3_dflash.py`
- Unit tests for proposer sampling: `tests/spec_decode/test_dflash.py`

# Checklist

- [x] I have performed a self-review of my code.
- [x] I have necessary comments in my code, particularly in hard-to-understand areas.
- [x] I have made or will make corresponding changes to any relevant documentation.

---

# PR 2 (pr_dflash_1b)
# Title: [Spec Decoding] Integrate DFlash into speculative decoding pipeline

---

# Description

Wire DFlash block-diffusion speculative decoding into the existing TPU inference pipeline. The DFlash model and proposer were added in #1868; this PR connects them to the runner, KV cache manager, and speculative decoding manager so DFlash can be used end-to-end.

No changes to existing Eagle3 or ngram code paths -- DFlash gets its own `propose_dflash_draft_token_ids` method and a separate `elif "dflash"` dispatch branch.

**Modified files:**
- `tpu_inference/models/common/model_loader.py` -- register DFlashDraftModel in model registry
- `tpu_inference/models/jax/qwen3.py` -- collect aux_hidden_states from target layers during forward pass (needed by DFlash proposer to inject target context)
- `tpu_inference/runner/tpu_runner.py` -- add DFlashProposer initialization for `method="dflash"`
- `tpu_inference/runner/speculative_decoding_manager.py` -- add dflash method dispatch and `propose_dflash_draft_token_ids` (uses `accepted_attn_metadata` with correct seq_lens for drafter)
- `tpu_inference/runner/kv_cache_manager.py` -- extend draft KV cache allocation to cover dflash, read `num_hidden_layers` from config instead of hardcoding 1

Usage (after both #1868 and this PR):
```python
args['speculative_config'] = {
    'model': 'z-lab/Qwen3-4B-DFlash-b16',
    'num_speculative_tokens': 5,
    'method': 'dflash',
    'draft_tensor_parallel_size': 1,
}
```

# Tests

E2e tests are in a follow-up PR.

# Checklist

- [x] I have performed a self-review of my code.
- [x] I have necessary comments in my code, particularly in hard-to-understand areas.
- [x] I have made or will make corresponding changes to any relevant documentation.

---

# PR 3 (pr_dflash_1c)
# Title: [Spec Decoding] Add DFlash e2e tests and Buildkite CI

---

# Description

Add e2e tests and Buildkite CI for DFlash block-diffusion speculative decoding. The DFlash model/proposer were added in #1868, and pipeline integration in #1869. This PR adds the test coverage and CI.

Verified on both TPU v4 and v5p across 9 datasets (math, code, chat) with Qwen3-4B target + z-lab/Qwen3-4B-DFlash-b16 draft, achieving 3x average speedup.

**Files:**
- `tests/e2e/test_speculative_decoding.py` -- add `test_dflash_correctness` (Qwen3-4B + DFlash draft, output correctness) and `test_dflash_performance` (1.5x speedup threshold)
- `.buildkite/features/Speculative_Decoding-_DFlash.yml` -- Buildkite CI pipeline for DFlash correctness and performance, modeled after Eagle3's `Speculative_Decoding-_Eagle3.yml`

# Tests

```bash
pytest tests/e2e/test_speculative_decoding.py::test_dflash_correctness
pytest tests/e2e/test_speculative_decoding.py::test_dflash_performance
```

# Checklist

- [x] I have performed a self-review of my code.
- [x] I have necessary comments in my code, particularly in hard-to-understand areas.
- [x] I have made or will make corresponding changes to any relevant documentation.
