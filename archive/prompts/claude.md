# DFlash TPU Benchmark — Environment Setup & Comparison Runs

## Context

We are integrating DFlash (a block-diffusion speculative decoding method)
into `vllm-project/tpu-inference` for Google Cloud TPUs. The code
integration is complete and lives on the `dflash-integration` branch of our
fork: `https://github.com/aaronzhfeng/tpu-inference`.

The code passes syntax checks, scope guards, and standalone JAX attention
math validation. However, we **cannot run the full test suite or benchmark
comparisons** because the tpu-inference repo requires a Docker-based runtime
environment with internal dependencies (torchax, libtpu, matched vllm
build) that cannot be pip-installed on a bare TPU VM.

See `docs/10_test_environment_gap.md` for the full problem description and
everything we already tried.

## Your Task

Explore and execute the following options to get a working runtime
environment on our TPU v4 VM so we can run benchmark comparisons. Work
through them in priority order. Stop and report findings after each option.

---

### Option 1: Follow the official quickstart

The tpu-inference README links to a quickstart guide:
https://docs.vllm.ai/projects/tpu/en/latest/getting_started/quickstart/

Read this page. It likely documents a Docker image or install process.
If it provides a Docker image tag, try pulling and running it on our VM.

Our TPU VM info:
- Node: `t1v-n-b8ef674a-w-0`
- TPU type: v4-8 (4 chips, 8 cores)
- OS: Ubuntu 22.04, Python 3.10 (default), Python 3.11 available
- The tpu-inference repo is at `/home/aaronfeng/tpu-spec-decode/tpu-inference`
- Our DFlash branch: `dflash-integration`

If a Docker container works, verify by running:
```bash
python -c "from vllm import LLM; print('vLLM OK')"
python -c "from tpu_inference.spec_decode.jax.dflash import DFlashProposer; print('DFlash OK')"
```

---

### Option 2: Build from source with correct dependencies

If no pre-built Docker image is available for v4, try building from source.

The key dependencies that blocked us:
- `torchax==0.0.10` — not on public PyPI (only 0.0.4–0.0.7 exist)
- `libtpu==0.0.31` — internal Google package
- `pathwaysutils` — internal Google package
- `vllm` — needs the TPU-specific build, not the GPU PyPI build

Possible approaches:
- Check if there's a `requirements_tpu.txt` or build script we missed
- Check if `torchax` is available via Google's internal PyPI index
  (e.g., `https://storage.googleapis.com/jax-releases/...`)
- Check if `pip install vllm` with special index URLs works for TPU
- Try `pip install --no-deps -e tpu-inference/` under Python 3.11 with
  upgraded setuptools (`pip install --upgrade setuptools`) to fix the
  `install_layout` error we hit

---

### Option 3: Run benchmark comparisons inside Docker

Once a working environment exists (from Option 1 or 2), run the
following benchmark comparisons. Each should be a separate run.

#### Running 3A–3C with Docker

From repo root `~/tpu-spec-decode`, mount the patch assets and `scripts/`, then run **one benchmark per container** (release TPU between runs). Results are written to `results/` (one JSON per run plus `benchmarks.jsonl`); mount `-v $(pwd)/results:/mnt/results` to save them on the host.

**3A — Baseline (no patch):**
```bash
sudo docker run --rm --privileged --net host --shm-size=16G \
  -v $(pwd)/scripts:/mnt/scripts:ro \
  -v $(pwd)/results:/mnt/results \
  vllm/vllm-tpu:latest \
  python3 /mnt/scripts/benchmark_baseline.py
```

**3B — DFlash (patch first, then benchmark):**
```bash
sudo docker run --rm --privileged --net host --shm-size=16G \
  -v $(pwd)/patch_docker.py:/mnt/patch_docker.py:ro \
  -v $(pwd)/qwen3_dflash_docker.py:/mnt/qwen3_dflash_docker.py:ro \
  -v $(pwd)/dflash_src:/mnt/dflash_src:ro \
  -v $(pwd)/scripts:/mnt/scripts:ro \
  -v $(pwd)/results:/mnt/results \
  vllm/vllm-tpu:latest \
  bash -c "python3 /mnt/patch_docker.py && python3 /mnt/scripts/benchmark_dflash.py"
```

**3C — Eagle-3 (no patch):**
```bash
sudo docker run --rm --privileged --net host --shm-size=16G \
  -v $(pwd)/scripts:/mnt/scripts:ro \
  -v $(pwd)/results:/mnt/results \
  vllm/vllm-tpu:latest \
  python3 /mnt/scripts/benchmark_eagle3.py
```

Optional: add `-v /tmp/hf_home:/tmp/hf_home` to cache Hugging Face downloads across runs.

---

### Option 4: If Docker doesn't work — standalone verification

If Options 1-2 both fail, fall back to verifying our code without vLLM:

**On host (with local tpu-inference):**
```bash
cd /home/aaronfeng/tpu-spec-decode
JAX_PLATFORMS=cpu PYTHONPATH=tpu-inference python3 -c "
import jax
import jax.numpy as jnp
import numpy as np
from tpu_inference.layers.common.dflash_attention_interface import dflash_concat_attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata

# [... run the 4-test validation suite ...]
# This was already verified to pass. See conversation history.
"
```

**In Docker (patch first so dflash_attention_interface is present):**
```bash
sudo docker run --rm --privileged --net host --shm-size=16G \
  -v $(pwd)/patch_docker.py:/mnt/patch_docker.py:ro \
  -v $(pwd)/qwen3_dflash_docker.py:/mnt/qwen3_dflash_docker.py:ro \
  -v $(pwd)/dflash_src:/mnt/dflash_src:ro \
  vllm/vllm-tpu:latest \
  bash -c "python3 /mnt/patch_docker.py && JAX_PLATFORMS=cpu python3 -c \"
from tpu_inference.layers.common.dflash_attention_interface import dflash_concat_attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
print('dflash_concat_attention import: OK')
\""
```

And confirm the pytest suite runs inside the Docker environment by
examining their CI configuration if accessible.

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `tpu-inference/tpu_inference/spec_decode/jax/dflash.py` | Our DFlash proposer |
| `tpu-inference/tpu_inference/layers/common/dflash_attention_interface.py` | Concat attention implementation |
| `tpu-inference/tpu_inference/models/jax/qwen3_dflash.py` | Draft model (JAX) |
| `tpu-inference/tests/models/jax/test_qwen3_dflash_attention.py` | Attention math unit tests |
| `tpu-inference/tests/spec_decode/test_dflash.py` | Proposer unit tests |
| `verification/sh/run_tpu_inference_dflash_eval.sh` | Our eval shell script |
| `verification/py/run_tpu_dflash_eval.py` | Our eval Python script |
| `docs/10_test_environment_gap.md` | Full description of what we tried |

## Expected Deliverables

1. A working runtime environment (Docker or native) where `from vllm import LLM` works on TPU
2. Benchmark numbers: tok/s for Baseline, DFlash, and (if possible) Eagle-3 on Qwen3-8B
3. A summary of what worked, what didn't, and any issues encountered
4. If nothing works: a clear description of the remaining blockers and what
   the tpu-inference team would need to provide

## Models on HuggingFace

- Target: `Qwen/Qwen3-8B` (or `Qwen/Qwen3-4B` if 8B is too large)
- DFlash draft: `z-lab/Qwen3-8B-DFlash-b16` (or `z-lab/Qwen3-4B-DFlash-b16`)
- Eagle-3 draft (community): `Tengyunw/qwen3_8b_eagle3`
- Eagle-3 draft (tpu-inference tested): `unkmaster/EAGLE3-LLaMA3.1-Instruct-8B` (Llama target, not Qwen)
