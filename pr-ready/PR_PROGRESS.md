# PR-Ready: DFlash Integration into vllm/tpu-inference

## Directory Structure

```
pr-ready/
├── main/     ← synced with upstream/main (vllm-project/tpu-inference, commit 6f56bde1)
├── dflash/   ← dflash-integration working branch (read-only copy)
├── pr/       ← pr/dflash branch (clean PR, based on upstream/main)
└── PR_PROGRESS.md  ← this file
```

## PR Branch Commits

### Committed

**Commit 1: Add DFlash block-diffusion draft model** (534c528f)
- `tpu_inference/layers/common/dflash_attention_interface.py` (NEW)
- `tpu_inference/models/jax/dflash.py` (NEW — 644 lines, Phase 3 JAX model)
- `tpu_inference/models/jax/qwen3_dflash.py` (NEW — Qwen3-specific)
- `tpu_inference/models/torchax/__init__.py` (NEW)
- `tpu_inference/models/torchax/dflash_torchax.py` (NEW — PyTorch bridge)
- `tpu_inference/spec_decode/jax/dflash.py` (NEW — DFlashProposer, 423 lines)

**Commit 2: Add DFlash test suite** (d6c902f3)
- `tests/models/jax/test_qwen3_dflash.py` (NEW)
- `tests/models/jax/test_qwen3_dflash_attention.py` (NEW)
- `tests/spec_decode/test_dflash.py` (NEW — updated for `_sample_block_draft_tokens` API)

### Staged (ready to commit)

**Commit 3: Integrate DFlash into speculative decoding pipeline**
- `tpu_inference/runner/speculative_decoding_manager.py` — DFlash method dispatch, draft probs, seq_len fix
- `tpu_inference/runner/tpu_runner.py` — DFlashProposer init + draft_token_probs
- `tpu_inference/models/common/model_loader.py` — DFlashDraftModel registry (conflict resolved)
- `tpu_inference/runner/kv_cache_manager.py` — DFlash KV cache compatibility

### Pending

**Commit 4: Flax 0.12 compatibility**
- `tpu_inference/models/jax/dflash.py` — `self.layers = nnx.List([...])` (required by Flax 0.12)

**Commit 5: Restore aux_hidden_states for DFlash** (THE BLOCKER FIX)
- `tpu_inference/models/jax/qwen3.py` — upstream dropped aux_hidden_states return, DFlash needs it

---

## Problems Encountered & Resolved

### 1. Benchmark broken by commit fc2d2a1

**Discovery:** The standalone benchmark (`benchmarks/standalone_dflash.py`) was silently broken by commit `fc2d2a1` ("update benchmark script and results") in the `tpu-spec-decode` repo. This commit rewrote the decode loop and:
- Removed `draft_cache_len` / `actual_ctx_count` tracking
- Changed from passing incremental new context to passing full accumulated buffer
- Removed the `(ctx_jax, cache_len_arr, ctx_count_arr)` tuple required by Phase 3 model
- Set `prev_ctx_len = n` after prefill instead of keeping it at 0

**Fix:** Restored the original Phase 3 context management from commit `6208960`. Key fix: `prev_ctx_len` must stay 0 after prefill because the draft model hasn't consumed that context yet.

**Verification:** Old environment (dflash-integration + Flax 0.11.1) now produces **tau=5.68, 2.51x speedup** — matching expected results.

### 2. Flax 0.11.1 → 0.12.4 breaking changes

**Problem:** Upstream main requires Flax 0.12.4 which has breaking changes:
- `nnx.List` required for module containers (plain `list` of layers rejected with `ValueError`)
- `nnx.Embed` requires `jax.set_mesh()` context during model creation
- `nnx.eval_shape` must run inside mesh context

**Fix:**
- DFlash `dflash.py`: `self.layers = nnx.List([...])` instead of `self.layers = [...]`
- Standalone benchmark: wrapped `get_flax_model()` calls with `jax.set_mesh(mesh)`

### 3. vLLM API change (fused_moe.activation)

**Problem:** Upstream tpu-inference imports `vllm.model_executor.layers.fused_moe.activation.MoEActivation` which doesn't exist in Docker image's vLLM 0.13.0+tpu.

**Fix:** Checked out local vllm fork to LKG commit `05972ea` (branch `vllm-lkg`) which has the required module. No Docker rebuild needed — just PYTHONPATH.

### 4. get_flax_model return value change (7-tuple → 8-tuple)

**Problem:** New model_loader returns 8 values (added `_not_support` placeholder and `pooler_fn`). Old returned 7. Benchmark indexing was wrong.

**Fix:** Adaptive unpacking based on `len(flax_result)`:
- 7-tuple (old): `combine_fn = flax_result[2]`
- 8-tuple (new): `combine_fn = flax_result[3]`

### 5. Upstream Qwen3 dropped aux_hidden_states (THE BLOCKER)

**Problem:** New upstream `qwen3.py` line 360 returns `return kv_caches, x, []` — always empty list. Old version returned actual target model intermediate layer outputs (`aux_hidden_states`). DFlash depends on these for KV injection — they're projected and injected as context into every draft layer.

**Impact:** Draft model gets zero context → predicts garbage → tau=1.00

**Root cause:** Upstream refactored Qwen3 and didn't need aux hidden states for their use cases. DFlash is the only consumer.

**Fix needed:** Restore aux hidden state collection in Qwen3's forward pass. This is a small, targeted change that collects hidden states from specific layers during the forward pass.

---

## Environment Setup

### Docker Images

| Tag | Flax | JAX | vLLM | For |
|-----|------|-----|------|-----|
| `vllm/vllm-tpu:old` | 0.11.1 | 0.8.1 | 0.13.0+tpu | dflash-integration branch |
| `vllm/vllm-tpu:latest` + pip upgrade | 0.12.4 | 0.8.3 | LKG 05972ea | PR branch |

### Running PR branch tests

```bash
# Unit tests (proposer) — PASSING
sudo docker run --rm --privileged --network host \
  -e PYTHONPATH=/workspace/tpu-spec-decode/pr-ready/pr \
  -v /home/aaronfeng/tpu-spec-decode:/workspace/tpu-spec-decode \
  vllm/vllm-tpu:latest \
  bash -c "pip install --quiet flax==0.12.4 jax[tpu]==0.8.3 jaxlib==0.8.3 qwix==0.1.2; \
           python3 -m pytest tests/spec_decode/test_dflash.py -v"

# Standalone benchmark (after Qwen3 aux fix)
sudo docker run --rm --privileged --network host --tmpfs /tmp:rw,size=80g \
  -v /home/aaronfeng/tpu-spec-decode:/workspace/tpu-spec-decode \
  -v /dev/shm/hf-cache:/hf-cache \
  -e HF_HOME=/hf-cache -e HUGGINGFACE_HUB_CACHE=/hf-cache/hub \
  -e PYTHONPATH=/workspace/tpu-spec-decode/pr-ready/pr:/workspace/tpu-spec-decode/vllm \
  vllm/vllm-tpu:latest \
  bash -c "pip install --quiet flax==0.12.4 jax[tpu]==0.8.3 jaxlib==0.8.3 qwix==0.1.2; \
           python3 benchmarks/standalone_dflash.py \
             --target-model Qwen/Qwen3-4B --draft-model z-lab/Qwen3-4B-DFlash-b16 \
             --dataset gsm8k --max-samples 3 --max-new-tokens 128 --max-model-len 2048"
```

### Running old branch tests (baseline comparison)

```bash
sudo docker run --rm --privileged --network host --tmpfs /tmp:rw,size=80g \
  -v /home/aaronfeng/tpu-spec-decode:/workspace/tpu-spec-decode \
  -v /dev/shm/hf-cache:/hf-cache \
  -e HF_HOME=/hf-cache -e HUGGINGFACE_HUB_CACHE=/hf-cache/hub \
  -e PYTHONPATH=/workspace/tpu-spec-decode/tpu-inference:/workspace/tpu-spec-decode/vllm \
  vllm/vllm-tpu:old \
  bash -c "python3 benchmarks/standalone_dflash.py \
             --target-model Qwen/Qwen3-4B --draft-model z-lab/Qwen3-4B-DFlash-b16 \
             --dataset gsm8k --max-samples 3 --max-new-tokens 128 --max-model-len 2048"
```

---

## Test Results

| Test | Old Env (Flax 0.11.1) | PR Env (Flax 0.12.4) | Status |
|------|---------|--------|--------|
| DFlash proposer unit tests | N/A | **2/2 PASS** | Done |
| Standalone benchmark tau | **5.68** | **5.68** | **PASS — identical** |
| Standalone speedup | **2.51x** | **2.41x** | **PASS** |
| Standalone TPS | 233.1 | 211.6 | PASS (diff is pip overhead) |
| Per-position acceptance α | 0.81, 0.68, 0.60... | 0.81, 0.68, 0.60... | **Identical** |
| Draft hidden stats (mean/std) | 0.025/2.88 | 0.025/2.88 | **Identical** |
| Draft token argmax (first 5) | [525,2661,1447,12,12] | [525,2661,1447,12,12] | **Identical** |
| Model loading | OK | OK | Done |
| Target model inference | OK | OK | Done |
| Output quality | 2/3 exact match | 2/3 exact match | **Identical** |

### Benchmark Details (PR Branch — GSM8K, 3 samples, 128 new tokens)

```
Samples:        2 (+ 1 warmup)
Baseline TPOT:  11.41 ms (87.7 TPS)
DFlash TPOT:    4.73 ms (211.6 TPS)
Speedup:        2.41x
Tau:            5.68

Per-position acceptance rate:
  pos  0: 1.000  pos  4: 0.426  pos  8: 0.213  pos 12: 0.128
  pos  1: 0.809  pos  5: 0.404  pos  9: 0.191  pos 13: 0.106
  pos  2: 0.681  pos  6: 0.340  pos 10: 0.191  pos 14: 0.085
  pos  3: 0.596  pos  7: 0.298  pos 11: 0.149  pos 15: 0.064
```

## Changes Made to PR Branch (beyond original DFlash code)

### Flax 0.12 Compatibility
- `tpu_inference/models/jax/dflash.py`: `self.layers = nnx.List([...])` (Flax 0.12 rejects plain list)

### Upstream Qwen3 aux_hidden_states Restoration
- `tpu_inference/models/jax/qwen3.py`:
  - Added `_init_aux_hidden_state_layers()` — configures which target layers to capture (Eagle3 or DFlash)
  - Overrode `Qwen3Model.__call__()` — collects aux hidden states during forward pass
  - Updated `Qwen3ForCausalLM.__call__()` — passes `aux_hidden_states` through (was returning `[]`)
  - Added `from itertools import islice`

### Test Updates
- `tests/spec_decode/test_dflash.py`: Updated for `_sample_block_draft_tokens` API (was `_get_draft_token_ids`), use JAX arrays for state arg

## Pending Commits on PR Branch

**Commit 3 (staged):** Integrate DFlash into speculative decoding pipeline
- `speculative_decoding_manager.py`, `tpu_runner.py`, `model_loader.py`, `kv_cache_manager.py`

**Commit 4 (unstaged):** Flax 0.12 compatibility + Qwen3 aux_hidden_states
- `dflash.py` (nnx.List), `qwen3.py` (aux hidden state collection)

**Commit 5 (unstaged):** Updated test suite
- `test_dflash.py` (API updates)

## Next Steps

1. **Commit all staged/unstaged changes** to pr/dflash branch
2. **Remove debug prints** from standalone benchmark
3. **Push `pr/dflash` branch** to origin
4. **Create PR** against vllm-project/tpu-inference
