# TPU v5p Setup Manual

Goal: Get DFlash inference running on v5p-8 to validate the tpu-inference PR.

## What We Need

1. JAX with libtpu that sees 4 TPU v5p chips
2. tpu-inference repo (dflash-integration branch) with DFlash model
3. vLLM (dflash-speculative-config branch) for pipeline integration
4. HuggingFace access for Qwen3-4B + DFlash-b16 weights
5. Standalone DFlash benchmark passing (tau > 5.0)

## Key Constraints

- Python 3.10.6 is the system Python (cannot easily change on the base image)
- JAX >= 0.7.0 requires Python 3.11+ (dropped 3.10 support)
- Therefore we MUST pin JAX < 0.7 OR upgrade Python first
- v5p uses VFIO (/dev/vfio/) not /dev/accel* - libtpu handles this transparently
- No ML packages pre-installed - everything must be installed from scratch
- SSH must remain functional throughout - verify after each major step
- Boot disk is only 97 GB - Docker image (~20GB) + model weights (~18GB) can fill it (see Doc 11)
- tpu-inference has internal deps (torchax, libtpu, pathwaysutils) not on public PyPI (see Doc 10)

## Two Setup Paths

### Path A: Bare-metal pip install in venv (primary, used by setup_v5p_safe.sh)

Install JAX, tpu-inference, and vLLM via pip inside a venv. This is simpler
and avoids Docker disk overhead. The standalone DFlash benchmark (Doc 22-23)
proved this works on v4 - it only needs tpu-inference deps, not the full
Docker environment.

Risk: Some tpu-inference dependencies (torchax, pathwaysutils) may not be
on public PyPI. If pip install fails, benchmarks can still run via Docker
(tests/standalone_benchmark.sh, tests/benchmark.sh mount the repo).

### Path B: Docker-based fallback

If pip install fails (e.g. torchax, pathwaysutils not on PyPI), use the
`vllm/vllm-tpu:latest` Docker image with tpu-inference mounted. The primary
benchmarks (`tests/standalone_benchmark.sh`, `tests/benchmark.sh`) run in
Docker and mount the repo; no separate patching scripts are needed.

## Strategy (Path A)

Use a Python virtual environment (not global pip) to avoid corrupting the system.
Install incrementally with SSH checks between steps.
Do NOT run apt-get update or apt-get upgrade.
Store model weights on /dev/shm to avoid filling boot disk.

## Setup Script

Run: `bash preparation/setup_v5p_safe.sh`

The script does these steps (each can also be run manually):

### Step 1: Docker group (one-time, requires re-login)
```bash
sudo usermod -aG docker $USER
# Must log out and back in for this to take effect
```

### Step 2: Create venv
```bash
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install --upgrade pip
```

### Step 3: Install JAX for v5p
```bash
# Option A (recommended): Pin to latest JAX 0.6.x (last Python 3.10 series)
pip install "jax[tpu]<0.7" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Option B: If Option A fails or we need newer JAX features,
# upgrade Python to 3.11 first:
#   sudo add-apt-repository ppa:deadsnakes/ppa -y
#   sudo apt-get install python3.11 python3.11-venv python3.11-dev -y
#   python3.11 -m venv ~/venv311
#   source ~/venv311/bin/activate
#   pip install --upgrade pip
#   pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Option C: If libtpu doesn't detect v5p with stable release, try nightly:
#   pip install -U --pre jax jaxlib libtpu \
#     -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### Step 4: Verify JAX sees TPU
```bash
python3 -c "import jax; print('JAX:', jax.__version__); print('Devices:', jax.devices()); print('Count:', jax.device_count())"
# Expected: 4 TpuDevice entries
```

### Step 5: Clone repos
```bash
cd ~/tpu-spec-decode
bash preparation/clone_repos.sh
```

### Step 6: Install tpu-inference dependencies
```bash
pip install -r tpu-inference/requirements.txt
pip install -r tpu-inference/requirements_benchmarking.txt
pip install -e tpu-inference --no-build-isolation
```

### Step 7: Install vLLM
```bash
# This is the heavyweight step - vLLM compiles C extensions
# Verify SSH works before and after
pip install -e vllm --no-build-isolation
```

### Step 8: Verify DFlash import
```bash
python3 -c "from tpu_inference.spec_decode.jax.dflash import DFlashProposer; print('DFlash import OK')"
```

### Step 9: Run smoke test
```bash
python3 preparation/tpu_sanity_check.py       # auto-detects JAX (Path A) or torch_xla (Path B)
bash preparation/check_dflash_support.sh host  # verifies DFlash import + vLLM support
```

### Step 10: Run DFlash acceptance benchmark (host mode)
```bash
bash preparation/run_dflash_acceptance_smoke.sh host
# Expected: tau > 5.0, speedup > 2x
# Results saved to /dev/shm/contrib-out/<run_id>/results.json
```

## Alternatives When Things Go Wrong

### JAX install fails
- Check Python version: `python3 --version` (must be 3.10 for jax<0.7)
- Try without version pin first: `pip install "jax[tpu]"` - if it installs but JAX is 0.7+, it won't run on Python 3.10
- Try nightly builds (Option C above)
- Check pip cache isn't stale: `pip cache purge`

### JAX installs but doesn't see TPU
- Check VFIO devices exist: `ls -la /dev/vfio/`
- Check no other process holds them: `sudo lsof /dev/vfio/*`
- Try explicit backend: `PJRT_DEVICE=TPU python3 -c "import jax; print(jax.devices())"`
- Enable verbose logging:
  ```bash
  TPU_MIN_LOG_LEVEL=0 TF_CPP_MIN_LOG_LEVEL=0 python3 -c "import jax; print(jax.devices())"
  ```
- If falls back to CPU silently, likely libtpu version mismatch with v5p hardware

### vLLM install fails
- Usually a compilation issue with C extensions
- Try: `pip install -e vllm --no-build-isolation 2>&1 | tail -50` to see actual error
- If disk space issue: `df -h /` (97 GB total, compilation can use temp space)
- Set TMPDIR to /dev/shm if /tmp runs out: `TMPDIR=/dev/shm pip install -e vllm --no-build-isolation`
- If memory issue during compilation: unlikely with 440 GiB RAM
- Alternative: install vLLM without editable mode: `pip install vllm` (uses pre-built wheel)

### tpu-inference internal deps not on PyPI (torchax, pathwaysutils)
- These are internal Google packages baked into the vllm/vllm-tpu Docker image
- If pip install fails on these: switch to Path B (Docker-based setup)
- Or try installing from the Docker image's site-packages:
  ```bash
  docker run --rm -v /tmp/extracted:/out vllm/vllm-tpu:latest \
    bash -c "cp -r /usr/local/lib/python3.*/site-packages/torchax /out/ 2>/dev/null; \
             cp -r /usr/local/lib/python3.*/site-packages/pathwaysutils /out/ 2>/dev/null"
  # Then: pip install /tmp/extracted/torchax /tmp/extracted/pathwaysutils
  ```

### Disk space runs out
- Boot disk is only 97 GB. Docker image + model weights + pip cache can fill it
- Use /dev/shm (221 GiB) for model weights: `export HF_HOME=/dev/shm/hf_home`
- Use /dev/shm for pip cache: `pip install --cache-dir=/dev/shm/pip-cache ...`
- Monitor disk: `df -h /` and `du -sh /tmp/* ~/.cache/*`
- Clean pip cache: `pip cache purge`
- Clean Docker: `sudo docker system prune -af` (reclaims ~20GB)

### SSH breaks during setup
- This is the critical failure mode that bricked the previous node
- NEVER run apt-get update or apt-get upgrade
- NEVER modify system Python or system pip globally
- If SSH breaks: from local machine, try `gcloud compute tpus tpu-vm ssh --tunnel-through-iap`
- If that fails: delete and re-create node (see docs/V5_ACCESS_LOG.md)

### tpu-inference requirements conflict
- The requirements.txt may pin versions that conflict with JAX < 0.7
- If so, install requirements first, then force-reinstall JAX:
  ```bash
  pip install "jax[tpu]<0.7" --force-reinstall -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
  ```
- Then verify JAX still sees TPU

### DFlash benchmark runs but tau is low
- This is expected behavior, not a setup issue
- v5p may have different performance characteristics than v4
- Compare with v4 numbers: standalone tau should be ~6.67, speedup ~4.26x
- If tau < 3.0, likely a code issue not hardware

## Known Package/Sync Issues (from PR_PROGRESS.md)

When syncing with upstream tpu-inference, we hit these issues. They WILL
recur on v5p if versions don't match. The setup script targets the PR
branch environment (Flax 0.12.4, JAX < 0.7).

### 1. Flax 0.11.1 -> 0.12.4 breaking changes
- `nnx.List` required for module containers (plain list of layers rejected)
- `nnx.Embed` requires `jax.set_mesh()` context during model creation
- Fix already applied in PR branch `dflash.py`: `self.layers = nnx.List([...])`

### 2. vLLM API change (fused_moe.activation)
- Upstream tpu-inference imports `vllm.model_executor.layers.fused_moe.activation.MoEActivation`
- This doesn't exist in all vLLM versions
- Fix: use vLLM fork at LKG commit `05972ea` (branch `vllm-lkg`)

### 3. get_flax_model return value (7-tuple -> 8-tuple)
- New model_loader returns 8 values (added `_not_support` and `pooler_fn`)
- Standalone benchmark must handle both: check `len(flax_result)`

### 4. Upstream Qwen3 dropped aux_hidden_states (THE BLOCKER)
- New upstream `qwen3.py` returns `return kv_caches, x, []` - always empty
- DFlash depends on aux_hidden_states for KV injection into draft layers
- Without this: draft model gets zero context, predicts garbage, tau=1.00
- Fix in PR branch: restore aux hidden state collection in Qwen3 forward pass

### 5. Benchmark script silently broken by later commit
- Commit `fc2d2a1` rewrote the decode loop and broke context management
- Fix: use the PR branch benchmark which has the original Phase 3 logic

### Docker image versions that work

| Tag | Flax | JAX | vLLM | Use Case |
|-----|------|-----|------|----------|
| `vllm/vllm-tpu:old` | 0.11.1 | 0.8.1 | 0.13.0+tpu | dflash-integration branch (original) |
| `vllm/vllm-tpu:latest` + pip upgrade | 0.12.4 | 0.8.3 | LKG 05972ea | PR branch (current) |

Note: both Docker images use JAX 0.8.x which requires Python 3.11+. This is
why the Docker images work but bare-metal pip on v5p (Python 3.10) needs
JAX < 0.7. If there are Flax/JAX API incompatibilities with JAX < 0.7,
the Docker path (Path B) may be the only option.

## What NOT to Do

1. Do NOT run `apt-get update` or `apt-get upgrade` - can break SSH daemon
2. Do NOT `pip install --upgrade pip` globally (only inside venv)
3. Do NOT install JAX 0.7+ on Python 3.10 - it will fail
4. Do NOT run outdated setup scripts; use setup_v5p_safe.sh
5. Do NOT install everything in one command - install incrementally
6. Do NOT forget to activate the venv before installing packages
7. Do NOT download model weights to root disk - use /dev/shm (221 GiB)
8. Do NOT pull Docker images without checking disk space first (`df -h /`)
