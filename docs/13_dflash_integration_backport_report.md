# DFlash Integration Backport Report (zhongyan_dev -> dflash-integration)

Status: **Runnable smoke verified after backport (not committed yet)**  
Date: 2026-02-10

## Goal

Document what `dflash-integration` originally had, why it did not run, what `zhongyan_dev` changed, how we backported those changes, and what works now.

## Scope

This work touches two local repos under this workspace:

- `vllm` (`main` branch, local path: `tpu-spec-decode/vllm`)
- `tpu-inference` (`dflash-integration` branch, local path: `tpu-spec-decode/tpu-inference`)

Reference source branches/commits used for backport:

- `vllm` `origin/zhongyan_dev` @ `5400a8b90` (`dflash v1`)
- `tpu-inference` `origin/zhongyan_dev` @ `06c823e` (`dflash v1`)

Note: `tpu-inference origin/zhongyan_dev` later advanced to `ab7ef0e`, but the runnable baseline backported here is `06c823e`.

## 1) What `dflash-integration` originally had

Originally, `dflash-integration` implemented DFlash mainly as a JAX-native path:

- `qwen3_dflash.py` model path
- custom DFlash attention interface
- DFlash proposer wiring in TPU runner/spec decode manager

This branch already had significant DFlash logic and tests, but runtime depended on assumptions that did not match the actual vLLM + TPU runtime combination we tested.

## 2) Why original `dflash-integration` was problematic at runtime

Two key runtime blockers were observed:

1. **vLLM method validation blocker (hard fail before runtime):**
   - Current `vllm/main` did not accept speculative method `"dflash"` in `SpeculativeConfig`.
   - Engine failed during config parsing before DFlash model code could run.

2. **Even with `zhongyan_dev` vLLM, original `dflash-integration` still failed:**
   - Draft model load path still followed incompatible assumptions from Eagle3/JAX init path.
   - Error observed in mixed matrix tests: JAX initialization/type errors during draft model setup (for example, `ShapeDtypeStruct ... is not a valid JAX type` in draft model loading flow).

In short, original branch was not runnable end-to-end under the tested Docker TPU environment.

## 3) How `zhongyan_dev` solved it

`zhongyan_dev` solved runability with a coordinated pair of changes:

### vLLM side

- Added explicit `dflash` support in speculative config handling:
  - method typing/acceptance
  - method auto-detection for DFlash draft model
  - graph-hash and method flow handling compatible with DFlash

### TPU-inference side

- Switched to a runtime approach centered on:
  - `DFlashForCausalLM` draft model registration
  - `DFlashProposer` using `torchax` wrapper (`dflash_torchax.py`) to run HF DFlash model path
- Updated runner/manager/cache/compilation behavior to align with DFlash runtime contract rather than Eagle3-only assumptions.

## 4) How we modified `dflash-integration` (backport process)

Backport was done locally via git operations (not manual editor typing, no commit yet):

1. In `vllm/main`:
   - applied `5400a8b90` changes (`vllm/config/speculative.py`)

2. In `tpu-inference/dflash-integration`:
   - applied `06c823e` DFlash runtime set
   - resolved cherry-pick conflicts
   - then normalized the runtime-critical files to exactly match `06c823e` snapshots to avoid mixed partial-merge artifacts

Backported `tpu-inference` runtime set:

- `tpu_inference/models/common/model_loader.py`
- `tpu_inference/models/jax/dflash.py` (new)
- `tpu_inference/models/jax/qwen3.py`
- `tpu_inference/models/torchax/__init__.py` (new)
- `tpu_inference/models/torchax/dflash_torchax.py` (new)
- `tpu_inference/runner/compilation_manager.py`
- `tpu_inference/runner/kv_cache_manager.py`
- `tpu_inference/runner/speculative_decoding_manager.py`
- `tpu_inference/runner/tpu_runner.py`
- `tpu_inference/spec_decode/jax/dflash.py`

## 5) How it works now

After the backport:

- `dflash` is accepted by `vllm` config path.
- Draft architecture resolution/mapping aligns with DFlash runtime path.
- One-prompt DFlash smoke generation now succeeds on the backported current branches.

Validated output artifact:

- `/tmp/dflash_output/current_backport_smoke_generate.json`

Observed result:

- `status: ok`
- `num_output_tokens: 16`
- generation completed successfully in smoke run.

## 6) Current repo state

At time of writing:

- changes are **staged** in `vllm` and `tpu-inference`
- no new commits were created yet
- this allows review before commit/push.

## 7) Practical conclusion

To make `dflash-integration` runnable now, using the `zhongyan_dev` support model is the correct baseline:

- `vllm` must include DFlash speculative support
- `tpu-inference` must include the compatible DFlash runtime loading/proposal path

Trying to keep original `dflash-integration` runtime behavior with only partial `vllm` changes is not sufficient for runability in the tested TPU Docker environment.

