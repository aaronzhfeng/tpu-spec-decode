# TPU-Inference Repository Deep Dive

## Scope and intent

This page is now the entry point for a split deep-dive set covering the local `tpu-inference/` repository vendored in this workspace.

This is a snapshot of the repository state as reviewed on **February 6, 2026**.

---

## Reading map

- `docs/04_tpu_inference_runtime_architecture.md`
  - boot/init sequence
  - platform integration
  - end-to-end execution flow
  - runner manager responsibilities

- `docs/05_tpu_inference_models_and_kernels.md`
  - model backend selection (`flax_nnx` vs `vllm`/TorchAX)
  - model loading pipeline
  - layers and kernel families
  - quantization, LoRA, speculative/structured decoding, multimodal

- `docs/06_tpu_inference_distributed_operations_and_validation.md`
  - DP/PP/disaggregated serving internals
  - executors and KV transfer connector
  - support matrices and capability tracking
  - test surface and operational scripts

---

## What `tpu-inference` is

At a high level, `tpu-inference` is the TPU backend/plugin layer used by vLLM to run inference on Google TPUs. It supports two model implementation paths behind a common runtime:

1. `flax_nnx` (JAX-native model definitions and kernels)
2. `vllm` (PyTorch/vLLM model definitions executed on TPU through TorchAX + JAX lowering)

Key design themes visible in the codebase:

- Preserve vLLM user/API surface while swapping execution internals for TPU.
- Use JAX compilation/sharding as the core runtime substrate.
- Keep model path pluggable (`MODEL_IMPL_TYPE=auto|flax_nnx|vllm`).
- Implement TPU-specialized kernels for core performance hotspots.
- Support both single-host and multi-host/disaggregated serving.

---

## Top-level repository layout

Under `tpu-inference/`:

- `README.md`: top-level project overview and links.
- `docs/`: user/developer docs (MkDocs site source).
- `tpu_inference/`: main Python package.
- `tests/`: unit, core, distributed, kernel, model, runner, e2e tests.
- `examples/`: runnable inference/profiling/disagg examples.
- `scripts/`: benchmarking/integration/multihost operational scripts.
- `support_matrices/`: recommended model/feature/kernel/parallelism/quantization matrices.
- `docker/`: Dockerfiles for containerized usage/release.
- `setup.py`, `requirements*.txt`: packaging and dependency wiring.
- `verified_commit_hashes.csv`: version pairing history of vLLM/tpu-inference commits.

Notable packaging detail: `pyproject.toml` currently contains only license header, while effective packaging logic is in `setup.py`.

---

## Documentation surface in the repo

Primary docs entry points:

- `docs/README.md`
- `docs/getting_started/tpu_setup.md`
- `docs/getting_started/installation.md`
- `docs/getting_started/quickstart.md`
- `docs/getting_started/out-of-tree.md`
- `docs/developer_guides/jax_model_development.md`
- `docs/developer_guides/torchax_model_development.md`
- `docs/recommended_models_features.md`
- `docs/profiling.md`

MkDocs navigation is declared in `mkdocs.yml`.

---

## Suggested reading order for contributors

1. `docs/getting_started/quickstart.md`
2. `docs/recommended_models_features.md`
3. `docs/04_tpu_inference_runtime_architecture.md`
4. `docs/05_tpu_inference_models_and_kernels.md`
5. `docs/06_tpu_inference_distributed_operations_and_validation.md`

---

## Summary

`tpu-inference` is a substantial TPU runtime/plugin layer for vLLM with:

- a flexible dual model path (`flax_nnx` and `vllm`/TorchAX),
- strong runtime decomposition around `TPUModelRunner`,
- dedicated TPU kernel stack for attention/MoE/quantized ops,
- support for async scheduling, speculative decoding, structured decoding, multimodal, LoRA, DP/PP, and disaggregated serving,
- and a broad test+matrix ecosystem that makes maturity/coverage visible by feature.

Use `03` as the map and the `04`-`06` documents for detail.
