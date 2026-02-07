# TPU-Inference Runtime Architecture

## Scope

This document focuses on runtime behavior: initialization, platform integration, request execution flow, and runner manager responsibilities.

This is a snapshot of the repository state as reviewed on **February 6, 2026**.

---

## Boot sequence and runtime initialization

### Package import side effects

`tpu_inference/__init__.py` intentionally performs early runtime setup:

- imports `tpu_inference.env_override` first to set env overrides
- inspects `envs.JAX_PLATFORMS`
- if `proxy` is in platforms: initializes Pathways flow and eagerly resolves vLLM platform
- otherwise logs TPU info via `tpu_info` helper

`tpu_inference/env_override.py` currently sets:

- `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` (avoid CUDA stream assumptions on TPU)

### Environment variable model

`tpu_inference/envs.py` centralizes lazy-parsed environment variables and validation.  
Important controls include:

- `JAX_PLATFORMS`
- `TPU_ACCELERATOR_TYPE`, `TPU_NAME`, `TPU_WORKER_ID`
- `TPU_MULTIHOST_BACKEND` (notably `ray`)
- `PREFILL_SLICES`, `DECODE_SLICES` (disaggregated serving control)
- `SKIP_JAX_PRECOMPILE`
- `VLLM_XLA_CHECK_RECOMPILATION`
- `MODEL_IMPL_TYPE` (`auto|vllm|flax_nnx|jetpack`)
- `NEW_MODEL_DESIGN`, `USE_2D_TP`, `NUM_SLICES`
- `PHASED_PROFILING_DIR`, `PYTHON_TRACER_LEVEL`
- `USE_MOE_EP_KERNEL`, `USE_MEGABLOCKS`
- `ENABLE_QUANTIZED_MATMUL_KERNEL`, `REQUANTIZE_BLOCK_SIZE`, `REQUANTIZE_WEIGHT_DTYPE`

`enable_envs_cache()` can freeze/cached resolved env values after initialization.

---

## Platform integration with vLLM

`tpu_inference/platforms/tpu_platform.py` is the core platform adapter.

Responsibilities:

- identifies platform as TPU (`PlatformEnum.TPU`)
- registers TPU attention backend selection (`PallasAttentionBackend`)
- sets supported quantization methods list
- initializes sharding configuration (`ShardingConfigManager`)
- adjusts vLLM compilation/backend defaults for TPU
- chooses executor backend by topology/mode:
  - single-host, PP=1: uniprocess
  - single-host with PP: custom multiprocess executor
  - multi-host with `TPU_MULTIHOST_BACKEND=ray`: custom Ray executor
- enforces multimodal chunked-input constraints on TPU
- enables DP scheduler replacement when DP size > 1

---

## Execution architecture (end-to-end request flow)

High-level path in serving mode:

1. vLLM creates TPU platform/executor/worker.
2. `TPUWorker` initializes device context, distributed state, and `TPUModelRunner`.
3. Scheduler produces `SchedulerOutput`.
4. `TPUModelRunner` prepares padded/sharded inputs, executes model forward, computes logits, samples tokens.
5. Runner returns `ModelRunnerOutput` (or async wrapper).
6. Scheduler updates request states and repeats.

### Worker layer

`tpu_inference/worker/tpu_worker.py`:

- handles device binding and process bounds setup
- initializes vLLM distributed stubs + JAX PP communication setup
- initializes KV transfer infrastructure
- delegates model execution to `TPUModelRunner`
- offers profiling hooks (`jax.profiler`)
- precompile-before-kv-allocation option to avoid OOM from late compilation

Pipeline parallel integration is handled through JAX transfer group utilities in `distributed/jax_parallel_state.py`.

### Runner layer (core of inference)

`tpu_inference/runner/tpu_runner.py` is the primary runtime engine.

Major responsibilities:

- model/runtime initialization (mesh, random seeds, multimodal/spec decode managers)
- input-state management and request batch persistence
- input tensor construction (non-DP and DP codepaths)
- model forward execution
- logits selection and token sampling
- optional speculative decoding and structured decoding
- async scheduling placeholder token substitution
- KV-cache access helpers for disaggregation

The runner is decomposed into specialized manager components.

---

## Runner manager modules and their roles

### `runner/persistent_batch_manager.py`

- tracks request lifecycle state across steps
- updates per-request cached state from scheduler output
- manages add/remove/resume/preemption handling
- maintains persistent batch tables
- reorders requests to favor kernel-friendly decode-first layout

### `runner/input_batch.py`

- stores per-request token buffers, lengths, block table pointers, sampling params
- tracks LoRA mappings, bad words, allowed tokens, logit bias, generators
- supports in-place swapping/condensing for efficient batch mutation

### `runner/kv_cache_manager.py`

- builds `KVCacheSpec` per layer from model configuration/attention modules
- supports full/sliding/MLA attention specs
- initializes sharded KV cache arrays
- supports cross-layer shared KV cache mapping
- provides gather/transfer/insert utilities for disaggregated flows

### `runner/compilation_manager.py`

- orchestrates JAX precompilation for expected shape buckets
- precompiles:
  - backbone (text and embedding paths)
  - sampling and gather-logprobs
  - select-from-array helpers
  - structured decoding
  - disagg KV utility paths
  - speculative decoding helpers (incl. Eagle3 helper graph fragments)
- respects `SKIP_JAX_PRECOMPILE` and compile-cache settings

### `runner/speculative_decoding_manager.py`

- supports ngram and Eagle3 draft proposal paths
- computes index metadata for target/draft/bonus logits in speculative decode
- returns draft token IDs via `DraftTokenIds`

### `runner/structured_decoding_manager.py`

- applies grammar bitmask to logits using packed-bitmask expansion on JAX
- prepares per-batch structured decoding control tensors

### `runner/multimodal_manager.py`

- batches and executes multimodal encoder inputs by modality
- caches multimodal embeddings by hash
- gathers embeddings for current scheduled token windows
- computes M-RoPE positions for multimodal models

### `runner/lora_utils.py`

- activates LoRA adapters for current padded schedule
- updates model params/buffers after LoRA switch
- extracts and swaps LoRA metadata for TorchAX-backed model path

### `runner/utils.py`

- request/token padding utilities
- compile-forbid context (`ForbidCompile`) for recompilation checks
- phased profiler logic (`PhasedBasedProfiler`) for prefill/decode/balanced phase capture
