# TPU-Inference Distributed Operations and Validation

## Scope

This document focuses on distributed/parallel execution, disaggregated serving, executors, support matrices, testing surface, and operational scripts.

This is a snapshot of the repository state as reviewed on **February 6, 2026**.

---

## Parallelism, distributed, and disaggregated serving

### Data parallel scheduler

`core/sched/dp_scheduler.py` provides `DPScheduler`:

- spawns one scheduler worker process per DP rank
- load balances new requests by:
  - prefix cache hit first
  - then least token load
- combines per-rank scheduler outputs into a unified output
- handles rank-aware grammar bitmask/output splitting

### Pipeline parallel communication

`distributed/jax_parallel_state.py` defines `GroupCoordinator` for send/recv tensor dict operations between PP stages using JAX transfer server connections.

### Disaggregated prefill/decode mode

- `core/disagg_utils.py`: parses `PREFILL_SLICES` / `DECODE_SLICES`
- `core/core_tpu.py`: disaggregated orchestrator threads:
  - prefill
  - KV transfer
  - decode
- `core/disagg_executor.py`: single-worker disagg executor wrapper

### KV connector for remote KV transfer

`distributed/tpu_connector.py` implements TPU connector (scheduler + worker sides):

- producer side marks finished prefill requests and exposes KV payload
- consumer side loads/pulls remote KV and writes it locally
- uses JAX transfer servers for payload transport
- uses ZeroMQ side-channel notifications for completion/freeing semantics

---

## Executors

- `executors/multiproc_executor.py`:
  - TPU-specific multiprocess executor
  - MPMD behavior for pipeline parallel
- `executors/ray_distributed_executor.py`:
  - Ray cluster worker orchestration for TPU
  - TPU-aware placement/resource assignment
  - worker initialization for multihost topologies

---

## Support matrices (capability tracking)

`support_matrices/` gives the “recommended/stress-tested” view.  
`support_matrices/nightly/` gives moving nightly status, plus split views by backend path:

- `nightly/flax_nnx/*`
- `nightly/vllm/*`

Tracked categories:

- text-only model support
- multimodal model support
- feature support
- kernel support (+ microbenchmark matrix)
- parallelism support
- quantization support
- hardware acceleration/dtype notes

This matrix set is the best place to check what is validated vs “unverified” at a given snapshot.

---

## Test surface

`tests/` is broad and layered:

- `core/`: core runtime/disagg/DP scheduler coverage
- `distributed/`: distributed utils + TPU connector tests
- `runner/`: runner/input batch/kv/spec decode/structured decode tests
- `kernels/`: kernel correctness tests for major kernels
- `layers/`: common + jax + vllm layer/quant tests
- `models/`: model-specific loading and behavior tests
- `platforms/`, `worker/`, `executors/`, `spec_decode/`, `e2e/`

Also present:

- `scripts/vllm/integration/`: LM-eval and safety accuracy integration tests
- `tests/e2e/benchmarking/*.sh`: benchmark recipe scripts

---

## Examples and operational scripts

Examples under `examples/`:

- `offline_inference.py`
- `offline_lora_inference.py`
- `offline_safety_model_inference.py`
- `multi_modal_inference.py`
- `tpu_profiling.py`
- `disagg/*` scripts for local and multihost disagg experimentation

Operational scripts:

- `scripts/multihost/run_cluster.sh`
- `scripts/multihost/deploy_cluster.sh`
- `scripts/vllm/benchmarking/*`

These scripts reflect practical deployment modes that the codebase is targeting.

---

## Current implementation notes and visible constraints

From code paths + support docs in this snapshot:

- The repo supports both JAX-native and TorchAX/vLLM model paths, but per-feature parity differs by model/backend.
- Many advanced kernel/quantization/parallel combinations are still marked `unverified` in matrices.
- PP on Ray has explicit guardrails in TPU platform logic.
- Structured decoding with DP has explicit precompilation limitation in current compilation manager.
- Several TODOs in kernels/managers indicate active optimization and feature completion work.
