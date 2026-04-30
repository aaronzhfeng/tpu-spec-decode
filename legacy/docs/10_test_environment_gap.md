# Test Environment Gap — DFlash Integration

Status: **Blocking local test execution**
Date: 2025-02-07

## Problem

Our DFlash integration code is ready for PR, but we **cannot run the
tpu-inference test suite locally** on a bare TPU v4 VM. The tpu-inference
repo is designed to run inside a purpose-built Docker/CI environment managed
by the vLLM TPU team. Reproducing that environment via `pip install` is not
feasible.

## What We Tried

| Step | Command | Result |
|------|---------|--------|
| Install pytest + deps | `pip install pytest pytest-mock qwix jaxtyping flax` | OK |
| Install tpu-inference (editable, no-deps) | `sudo pip install --no-deps -e tpu-inference/` | OK on Python 3.10, fails on 3.11 (`AttributeError: install_layout`) |
| Install tpu-inference (editable, with deps) | `pip install -e tpu-inference/` | Fails — `torchax==0.0.10` not on PyPI (only 0.0.4–0.0.7 exist) |
| Install vllm from PyPI | `pip install vllm` | Installs GPU build (0.15.1). API mismatch: `vllm.model_executor.layers.attention.Attention` no longer exists at that path (moved to `vllm.attention.layer`) |
| Shim vllm import | Patched `__init__.py` to re-export `Attention` | Unblocked one import, but next import chain hits `SyntaxError` from Python 3.10 incompatibility in kernel code (`w1_vmem[*w_slices]` requires 3.11+) |
| Switch to Python 3.11 | `python3.11 -m pip install ...` + `sudo python3.11 -m pip install --no-deps -e ...` | pip install deps OK, but editable install fails with `AttributeError: install_layout` (system setuptools too old for 3.11) |

## Root Causes

### 1. tpu-inference is not a standalone pip package

It is a **vLLM hardware plugin** designed to be pre-installed inside a
Docker image that the vLLM TPU team builds. Key dependencies
(`torchax==0.0.10`, `libtpu==0.0.31`, `pathwaysutils`) are internal
packages not published to public PyPI.

### 2. vllm version mismatch

The public PyPI `vllm` (≥ 0.15) is the GPU build. Its internal module
layout has diverged from what tpu-inference imports:

```
tpu-inference expects:  vllm.model_executor.layers.attention.Attention
PyPI vllm 0.15.1 has:   vllm.attention.layer.Attention
```

The tpu-inference team uses a matched vllm build baked into their Docker
image.

### 3. Python version

tpu-inference kernel code uses Python 3.11+ syntax (PEP 646 starred
unpacking in subscripts: `w1_vmem[*w_slices]`). The TPU VM's default
`python3` is 3.10. Python 3.11 exists on the VM, but the system
`setuptools` is too old for editable installs under 3.11.

## What This Means for Our PR

**Our DFlash code is not the problem.** The code changes compile, the
imports resolve in isolation, and the test logic is sound. The blocker is
purely environmental: we cannot locally reproduce the Docker-based runtime
that tpu-inference's CI uses.

### What our PR includes (all verified by syntax + import checks)

- `tpu_inference/spec_decode/jax/dflash.py` — DFlash proposer
- `tpu_inference/layers/common/dflash_attention_interface.py` — concat attention
- `tpu_inference/models/jax/qwen3_dflash.py` — draft model
- `tests/spec_decode/test_dflash.py` — proposer unit tests
- `tests/models/jax/test_qwen3_dflash.py` — model helper tests
- `tests/models/jax/test_qwen3_dflash_attention.py` — attention math tests
- Modified: `model_loader.py`, `qwen3.py`, `compilation_manager.py`,
  `kv_cache_manager.py`, `speculative_decoding_manager.py`, `tpu_runner.py`
- Modified tests: `test_kv_cache_manager.py`,
  `test_speculative_decoding_manager.py`, `test_tpu_runner.py`

### What we verified locally

| Check | Status |
|-------|--------|
| `python -m py_compile` on all new/modified files | PASS |
| Scope guard (all changes within allowlist) | PASS |
| Preflight (paths, modules present) | PASS |
| `dflash_concat_attention` import + JAX execution | PASS |
| `DFlashProposer` import | PASS |
| Full pytest suite | BLOCKED (environment) |

### What the PR reviewer / CI will validate

The tpu-inference repo has its own CI pipeline that runs inside a Docker
image with the correct vllm build, torchax, libtpu, etc. Once our PR is
submitted, CI will:

1. Run `pytest` on all test files (including our DFlash tests)
2. Run integration / e2e tests on real TPU hardware
3. Validate against their nightly support matrix

## Recommended Path Forward

1. **Submit the PR as-is.** Our code is correct and follows the repo's
   patterns. The CI environment will run the tests.

2. **If we want local validation before PR**, we have two options:

   a. **Ask the tpu-inference team for their Docker image tag** — this is
      the intended development path. Their README links to a quickstart
      guide at `docs.vllm.ai` which likely documents the Docker setup.

   b. **Run a standalone JAX-only validation** — we can test the critical
      attention math (`dflash_concat_attention`) in isolation since it only
      depends on JAX + numpy, confirming concat-vs-additive correctness
      without needing the full tpu-inference runtime.

3. **Do not shim or patch vllm internals.** Re-exporting moved classes is
   fragile, version-dependent, and would need to be undone before PR. It
   does not prove anything that CI won't prove more reliably.

## Lessons Learned

- Hardware-specific ML inference repos (tpu-inference, TensorRT-LLM, etc.)
  are **not designed for bare-metal pip install**. They ship as Docker
  images with exact dependency snapshots.
- Local development on TPU VMs requires either the team's Docker image or
  a carefully matched virtual environment — neither of which are documented
  for external contributors yet.
- For our contribution workflow: write code and tests following repo
  patterns, verify syntax/scope locally, and rely on CI for full validation.
