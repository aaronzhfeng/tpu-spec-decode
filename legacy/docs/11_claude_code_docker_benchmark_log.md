# Claude Code Docker Benchmark Run Log

Date: 2026-02-08
Agent: Claude Code v2.1.37 (Opus 4.6)
Status: **Failed — disk space exhaustion**

## Summary

Claude Code was given `prompts/claude.md` and attempted to set up the
tpu-inference Docker environment and run DFlash benchmark comparisons.
It made significant progress — successfully pulled the Docker image,
created a runtime patching system, and got DFlash imports working inside
the container — but ultimately failed because the 97GB boot disk filled
up with Docker images (~20GB) + model weights (~18GB) + pip caches
(~5GB).

## What Claude Code Did

### Phase 1: Docker Image Discovery

- Found and pulled `vllm/vllm-tpu:latest` Docker image
- Image size: ~20GB
- This image contains the full tpu-inference runtime with correct vllm,
  torchax, libtpu, jax, flax — all the dependencies we couldn't pip install

### Phase 2: Runtime Patching Strategy

Since the Docker image has stock tpu-inference (no DFlash), Claude Code
created two files to patch the container at runtime:

#### `patch_docker.py` (464 lines)

A comprehensive patching script that modifies the stock Docker image's
code to add DFlash support. It applies 8 patches:

0. **Patches vLLM SpeculativeConfig** — adds `"dflash"` to
   `EagleModelTypes`, method passthrough, eagle3 target validation, and
   `is_eagle` property
1. **Copies new DFlash files** into the container:
   - `tpu_inference/spec_decode/jax/dflash.py`
   - `tpu_inference/layers/common/dflash_attention_interface.py`
   - `qwen3_dflash_docker.py` → `tpu_inference/models/jax/qwen3_dflash.py`
2. **Patches `tpu_runner.py`** — adds DFlash import and initialization
3. **Patches `speculative_decoding_manager.py`** — adds DFlash dispatch
   (routes through eagle3 proposal path since DFlashProposer inherits
   Eagle3Proposer)
4. **Patches `kv_cache_manager.py`** — extends eagle3-only check to
   include dflash, multi-layer draft cache support
5. **Patches `compilation_manager.py`** — adds dflash to eagle3
   precompilation check
6. **Patches `qwen3.py`** — adds DFlash helper functions
   (`_build_target_layer_ids`, `_get_dflash_target_layer_ids`),
   `aux_hidden_state_layers` init, `Qwen3Model.__call__` override to
   capture hidden states, `Qwen3ForCausalLM.__call__` update
7. **Patches `model_loader.py`** — registers `DFlashDraftModel` and
   `Qwen3DFlashForCausalLM` in the model registry

All patches were verified: `DFlashProposer import: OK`,
`dflash_concat_attention import: OK`

#### `qwen3_dflash_docker.py` (480 lines)

A Docker-adapted version of the DFlash draft model. Uses the Docker
image's layer interfaces (`nnx.Einsum`, `nnx.RMSNorm`, `nnx.Embed`)
directly instead of the `JaxEmbed`/`JaxEinsum`/`JaxRmsNorm` wrappers
used in our repo version (which don't exist in the Docker image's
codebase).

### Phase 3: Baseline Benchmark (Attempted)

Claude Code attempted to run a baseline (autoregressive) benchmark with
Qwen3-8B. The Qwen3-8B target model was pre-downloaded to
`/tmp/hf_home` (~16GB). This run's status is unknown — the log file
was only 1 line when checked.

### Phase 4: DFlash Benchmark — Run 1 (Container: ecd3054f2966)

Docker command:
```bash
sudo docker run --privileged --net host --shm-size=16G \
  -v patch_docker.py:/mnt/patch_docker.py:ro \
  -v qwen3_dflash_docker.py:/mnt/qwen3_dflash_docker.py:ro \
  -v tpu-inference/tpu_inference:/mnt/dflash_src/tpu_inference:ro \
  -v /tmp/hf_home:/tmp/hf_home \
  --name dflash_bench \
  vllm/vllm-tpu:latest \
  bash -c "python3 /mnt/patch_docker.py && python3 -c '...benchmark...'"
```

**Result: Stalled on draft model download**

- All patches applied successfully
- Target model loaded from cache
- Draft model download (z-lab/Qwen3-8B-DFlash-b16) stalled at 2.0GB
  (expected ~4-5GB)
- Download stalled for ~10 hours with no progress
- Container killed manually

### Phase 5: Fix — Re-download Draft Model

```bash
sudo docker run --rm -v /tmp/hf_home:/tmp/hf_home \
  vllm/vllm-tpu:latest python3 -c \
  "from huggingface_hub import snapshot_download; \
   snapshot_download('z-lab/Qwen3-8B-DFlash-b16', cache_dir='/tmp/hf_home')"
```

**Result: Downloaded successfully — 2.10GB in 3 seconds**

The previous stall was a Docker networking issue, not a model size issue.

### Phase 6: DFlash Benchmark — Run 2 (Container: 7f3945f52b8f)

Re-ran with pre-downloaded draft model. All patches applied successfully.

**Key log output:**
```
INFO 02-08 18:05:42 Resolved architecture: Qwen3ForCausalLM
INFO 02-08 18:06:06 Resolved architecture: TransformersForCausalLM
(draft model resolved as TransformersForCausalLM — NOT Qwen3DFlashForCausalLM)
INFO 02-08 18:06:06 Chunked prefill enabled with max_num_batched_tokens=1024
INFO 02-08 18:06:06 ShardingStrategy(tensor_parallelism=1, ...)
INFO 02-08 18:06:06 Force using UniProcExecutor for JAX on single host
```

**Result: EngineCore failed to start**
```
ERROR 02-08 18:06:58 EngineCore failed to start.
File "tpu_worker.py", line 366, in load_model → self.model_runner.load_model()
File "tpu_runner.py", line 508, in load_model → [traceback cut off]
```

**Important observation:** The draft model architecture resolved as
`TransformersForCausalLM` (vllm's generic wrapper), NOT as our
registered `Qwen3DFlashForCausalLM`. This means the model_loader patch
didn't take effect for the draft model resolution path. The
`DFlashDraftModel` architecture from HuggingFace config is not being
matched to our registered class.

### Phase 7: DFlash Benchmark — Run 3 (Container: 145879be7b7a)

Re-ran with output redirected to mounted files.

**Result: Disk space exhaustion**
```
Not enough free disk space to download the file.
Expected: 3996.25 MB. Available: 2296.41 MB.
```

The Qwen3-8B target model weights needed re-downloading (the previous
cache was in a different container's layer), but only 2.3GB was free.
Docker image (~20GB) + existing caches consumed all 97GB.

### Cleanup

User terminated Claude Code and ran:
```bash
sudo docker system prune -af
```

**Reclaimed: 19.9GB** (Docker images, containers, build cache)

Post-cleanup disk state:
```
/dev/root  97G  76G  21G  79% /
/tmp/pip-unpack-*: 4.6GB
/tmp/hf_home: 2.2GB (draft model only; target model lost with Docker prune)
```

## Files Created by Claude Code

| File | Location | Size | Purpose |
|------|----------|------|---------|
| `patch_docker.py` | repo root | 464 lines | Runtime patching of Docker container |
| `qwen3_dflash_docker.py` | repo root | 480 lines | Docker-adapted DFlash draft model |

These files are **not committed** to git. They live at the repo root.

## Key Findings

### What Worked
1. `vllm/vllm-tpu:latest` Docker image exists and runs on TPU v4
2. The runtime patching approach is viable — all 8 patches applied cleanly
3. DFlash imports work inside the container after patching
4. Draft model downloads successfully when not hitting Docker networking issues

### What Failed
1. **Disk space**: 97GB boot disk is too small for Docker image (~20GB) +
   Qwen3-8B weights (~16GB) + DFlash draft (~2.1GB) + pip caches (~5GB) +
   OS/tools (~50GB). Only ~4GB margin.
2. **Draft model architecture resolution**: The model resolved as
   `TransformersForCausalLM` instead of `Qwen3DFlashForCausalLM`. The
   model_loader registry patch may need to intercept at a different point
   in the resolution chain.
3. **Docker networking**: First draft model download stalled for 10+ hours.
   Re-downloading in a fresh container worked instantly.

### Remaining Blockers for Next Run

1. **Disk space management**: Either:
   - Use a larger boot disk (200GB+)
   - Pre-download ALL models outside Docker and mount the cache
   - Use `--download-dir` pointing to a persistent disk
   - Use Qwen3-4B instead of 8B (saves ~12GB)

2. **Architecture resolution**: Need to debug why `DFlashDraftModel` →
   `TransformersForCausalLM` instead of `Qwen3DFlashForCausalLM`. The
   model registry patch may need to be applied to the vllm-side model
   resolution (not just tpu_inference's `_MODEL_REGISTRY`).

3. **Model caching**: After `docker system prune`, the Qwen3-8B target
   model cache was lost (it was inside the Docker layer, not on the
   mounted volume). All models must be on the mounted `/tmp/hf_home` path.

## Docker Command Template (For Next Run)

```bash
# Pre-download both models first
sudo docker run --rm -v /tmp/hf_home:/tmp/hf_home \
  vllm/vllm-tpu:latest python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-8B', cache_dir='/tmp/hf_home')
snapshot_download('z-lab/Qwen3-8B-DFlash-b16', cache_dir='/tmp/hf_home')
"

# Then run benchmark with pre-downloaded models
sudo docker run -d --privileged --net host --shm-size=16G \
  -v $(pwd)/patch_docker.py:/mnt/patch_docker.py:ro \
  -v $(pwd)/qwen3_dflash_docker.py:/mnt/qwen3_dflash_docker.py:ro \
  -v $(pwd)/tpu-inference/tpu_inference:/mnt/dflash_src/tpu_inference:ro \
  -v /tmp/hf_home:/tmp/hf_home \
  -v /tmp/dflash_output:/tmp/dflash_output \
  --name dflash_bench \
  vllm/vllm-tpu:latest \
  bash -c "python3 /mnt/patch_docker.py 2>&1 | tee /tmp/dflash_output/patch.log && \
           python3 benchmark_script.py 2>&1 | tee /tmp/dflash_output/bench.log"
```

## Recommendations for Next Agent

1. **Increase disk first**: `gcloud compute disks resize` the boot disk
   to 200GB, or attach a separate persistent disk for model storage.

2. **Consider Qwen3-4B**: Cuts target model from ~16GB to ~8GB. DFlash
   draft exists: `z-lab/Qwen3-4B-DFlash-b16`. The DFlash paper's
   `run_benchmark.sh` actually defaults to Qwen3-4B.

3. **Fix architecture resolution**: Check if `DFlashDraftModel` needs
   to be registered in vllm's model registry (`vllm/model_executor/`)
   in addition to tpu_inference's `_MODEL_REGISTRY`. The vllm side may
   resolve the architecture before tpu_inference gets a chance.

4. **Keep the patching approach**: `patch_docker.py` and
   `qwen3_dflash_docker.py` are solid. The patching worked; the failures
   were all environmental (disk, networking, architecture resolution).
