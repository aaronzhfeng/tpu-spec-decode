# TPU-Inference Models and Kernels

## Scope

This document focuses on model backend selection, model loading paths, layer/kernel organization, and model-level features (quantization, LoRA, speculative decoding, structured decoding, multimodal).

This is a snapshot of the repository state as reviewed on **February 6, 2026**.

---

## Model loading and backend selection

`tpu_inference/models/common/model_loader.py` is the central router.

### Implementation selection

`get_model(...)` picks backend based on `MODEL_IMPL_TYPE`:

- `auto`: resolves via architecture and RunAI streaming constraints
- `flax_nnx`: JAX-native model class path
- `vllm`: PyTorch/vLLM model wrapped through TorchAX

### JAX (`flax_nnx`) model path

- architecture registry populated in `_get_model_architecture(...)`
- currently includes model families such as Llama3/Llama4, Qwen2/Qwen3/Qwen2.5-VL, DeepSeek-V3, Llama Guard 4, Eagle3 Llama, GPT-OSS
- abstract model creation + weight loading strategy to reduce memory duplication
- Qwix quantization integration hooks for abstract or concrete model paths
- returns compiled callable trio:
  - `model_fn`
  - `compute_logits_fn`
  - `combine_hidden_states_fn` (used for Eagle3 flows)

### vLLM/TorchAX model path

`tpu_inference/models/vllm/vllm_model_wrapper.py`:

- loads vLLM PyTorch model (CPU load path first)
- shards/places params to TPU through cleanup sharding utilities
- wraps forward/logit calls using `torch.func.functional_call` and TorchAX interop (`jax_view`/`torch_view`)
- supports LoRA manager integration and metadata replacement
- patches PP group calls to JAX PP group where needed

`models/vllm/vllm_model_loader.py` adds an incremental weight loader (`tpu_streaming_loader`) to process/shard weights as they load.

---

## Layers and kernel stack

The codebase is organized in three layer namespaces:

- `tpu_inference/layers/common`: shared core logic (attention metadata, sharding, linear/moe helpers, quant helpers)
- `tpu_inference/layers/jax`: JAX-native module building blocks
- `tpu_inference/layers/vllm`: TPU backend implementations for vLLM PyTorch modules

### Kernel families in `tpu_inference/kernels/`

- `ragged_paged_attention/v2` and `v3` (+ `kernel_hd64`): primary attention kernels
- `flash_attention/`: flash attention path
- `quantized_matmul/`: quantized matmul kernels + tuning tables + utils
- `fused_moe/v1/`: fused MoE kernel
- `megablox/gmm.py`: grouped GEMM path
- `mla/v1/`: MLA attention kernel
- `collectives/all_gather_matmul.py`: collective communication matmul kernel

These kernels are wired into layer implementations and selected by model/runtime configuration.

---

## Quantization and LoRA support model

Quantization is spread across:

- common quant helpers/configs in `layers/common/quantization`
- JAX quantization in `layers/jax/quantization`
- vLLM-compat quantization in `layers/vllm/quantization`
- kernel support in quantized matmul and attention paths

LoRA support includes:

- Torch path adapter wrappers in `lora/torch_punica_tpu.py` and `lora/torch_lora_ops.py`
- per-step LoRA activation and metadata exchange in `runner/lora_utils.py`

---

## Speculative decoding support

Spec decode methods currently implemented:

- `ngram`
- `eagle3` (JAX proposer in `spec_decode/jax/eagle3.py`)

Core wiring:

- runner detects speculative config
- target run produces logits and sampled tokens
- rejection sampler and draft proposer generate speculative tokens for next steps
- metadata indices are built in `SpeculativeDecodingManager`

---

## Structured decoding support

Structured decoding path is integrated at runner sampling time:

- scheduler provides grammar output bitmask
- `StructuredDecodingManager` maps masks to active batch requests
- logits are masked before sampling via JAX kernelized bit unpacking

DP-specific caveat: structured decoding precompilation is skipped for DP path in current manager logic.

---

## Multimodal support

Multimodal flow is implemented in runner and model interfaces:

- multimodal encoder execution and cache (`MultiModalManager`)
- embedding merge via model-provided `embed_input_ids_fn`
- M-RoPE position handling for multimodal token windows

Current docs and support matrices indicate multimodal support is present but model/backend coverage remains uneven across nightly snapshots.
