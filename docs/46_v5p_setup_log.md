# V5P Setup Log

**Date:** 2026-03-01
**VM:** TPU v5p-8, tpu-ubuntu2204-base
**Operator:** aaronfeng + Claude Code
**Script:** `preparation/setup_v5p_safe.sh` (step-by-step)

## Pre-Setup Baseline

| Property | Value |
|---|---|
| OS | Ubuntu 22.04.2 LTS |
| Python | 3.10.6 (`/usr/bin/python3`) |
| TPU chips | 4 v5p at `/dev/vfio/[0-3]` |
| Boot disk | 97 GB (86 GB free) |
| `/dev/shm` | 221 GB |
| Docker | 24.0.4 (user not in docker group) |
| JAX | Not installed |
| Venv | Not created |

---

## Step 1: Add User to Docker Group

**Status:** DONE
**Started:** 2026-03-01
**Output:**
```
[INFO]  Step 1: Adding user to docker group
[WARN]  Added to docker group - log out and back in for it to take effect
[OK]    SSH daemon is running
```
**Notes:** User added to docker group. Group membership confirmed: `docker` now in `groups aaronfeng`. Re-login needed for `docker` commands without `sudo` in new shells.

---

## Step 2: Create Python Venv

**Status:** DONE
**Started:** 2026-03-01
**Output:**
```
[INFO]  Step 2: Creating Python venv at /home/aaronfeng/venv
[OK]    Venv created
[OK]    pip upgraded inside venv: pip 26.0.1
[OK]    SSH daemon is running
```
**Notes:** First attempt failed — `python3.10-venv` package was missing on the bare VM. Fixed with `sudo apt install -y python3.10-venv` (safe targeted install, not `apt-get upgrade`). Python also upgraded from 3.10.6 to 3.10.12 as a side effect. First failed attempt left a partial `~/venv` directory that had to be `rm -rf`'d before retry. SSH verified safe after apt install.

---

## Step 3: Install JAX (pinned < 0.7)

**Status:** DONE
**Started:** 2026-03-01
**Output:**
```
[INFO]  Step 3: Installing JAX (pinned < 0.7 for Python 3.10)
Successfully installed jax-0.6.2 jaxlib-0.6.2 libtpu-0.0.17 numpy-2.2.6 scipy-1.15.3 ml_dtypes-0.5.4
[OK]    JAX installed: 0.6.2
[OK]    SSH daemon is running
```
**Notes:** Clean install. JAX 0.6.2 + jaxlib 0.6.2 + libtpu 0.0.17. No conflicts.

---

## Step 4: Verify JAX Sees TPU Devices

**Status:** DONE
**Started:** 2026-03-01
**Output:**
```
[INFO]  Step 4: Verifying JAX TPU device visibility
  JAX version:  0.6.2
  Device count: 4
    TPU_0(process=0,(0,0,0,0))
    TPU_1(process=0,(1,0,0,0))
    TPU_2(process=0,(0,1,0,0))
    TPU_3(process=0,(1,1,0,0))
  TPU chips: 4
[OK]    JAX sees TPU devices
[OK]    SSH daemon is running
```
**Notes:** All 4 v5p chips detected via VFIO. Topology is 2x2x1x1.

---

## Step 5: Clone Repos

**Status:** DONE
**Started:** 2026-03-01
**Output:**
```
==> [1/7] Root working repos
  tpu-inference  → dflash-integration
  vllm           → dflash-speculative-config (+ vllm-lkg at 05972ea7)

==> [2/7] zhongyan_dev repos
  dflash         → main (Zhongyan0721/dflash)
  tpu-inference  → zhongyan_dev
  vllm           → zhongyan_dev

==> [3/7] PR-ready setup
  main/          → main (upstream synced)
  dflash/        → dflash-integration
  pr/            → pr/dflash (clean PR branch)

==> [4/7] External references (7 repos)
==> [5/7] dflash-wide → main
==> [6/7] Brainstorm repos (2 repos)
==> [7/7] Slides theme (mtheme → master)
```
**Notes:** Initial attempt failed — the VM had empty placeholder directories (`tpu-inference/`, `vllm/`, `zhongyan_dev/`, `dflash-wide/`, `brainstorm-*`, `pr-ready/{main,dflash,pr}`, `slides/mtheme/`) that `clone_or_checkout_branch` and `clone_if_missing` refuse to overwrite. Cleaned all empty dirs with `rmdir`, preserved real files (`pr-ready/PR_PROGRESS.md`, `slides/` content, `external/.gitkeep`). Ran `clone_repos.sh` directly (not via `setup_v5p_safe.sh step5`) since it's the same script underneath. All 7 groups cloned cleanly.

---

## Step 6: Install tpu-inference Dependencies

**Status:** DONE
**Started:** 2026-03-01
**Output:**
```
tpu-inference-0.0.0+v6 installed (IS_FOR_V7X=false to avoid jax==0.8.1 override)
```
**Notes:** Multiple Python 3.10 incompatibilities resolved:
- `requirements.txt` pins `jax==0.8.0`, `jaxlib==0.8.0`, `flax==0.11.1`, `numba==0.62.1` — all require Python 3.11+.
- Installed deps manually, filtering out 3.11+ packages and using best-effort 3.10-compatible versions:
  - `flax==0.10.7` (latest for 3.10, vs pinned 0.11.1)
  - `numba==0.61.2` (latest for 3.10, vs pinned 0.62.1)
  - `torchax==0.0.7` (latest available for 3.10, vs pinned 0.0.10)
  - `qwix==0.1.2` (installed OK)
- Build also needed `wheel` package and `IS_FOR_V7X=false` (setup.py defaults to `true`, which loads `requirements_v7x.txt` bumping JAX to 0.8.1 and libtpu to 0.0.31).
- tpu-inference imports fail until vLLM is installed (step 7) — `tpu_inference.logger` depends on `vllm.logger`.

---

## Step 7: Install vLLM

**Status:** DONE
**Started:** 2026-03-02
**Output:**
```
vllm: /home/aaronfeng/tpu-spec-decode/vllm/vllm/__init__.py
tpu_inference: /home/aaronfeng/tpu-spec-decode/tpu-inference/tpu_inference/__init__.py
INFO TPU info: node_name=aaron-v5p-node6 | tpu_type=v5p-8 | worker_id=0 | num_chips=4 | num_cores_per_chip=2
```
**Notes:** `pip install -e vllm` does not work on TPU (no PEP 660 support, setup.py fails without CUDA_HOME). With `VLLM_TARGET_DEVICE=tpu`, it builds a 6KB empty shell. Switched both vLLM and tpu-inference to **PYTHONPATH-based import** instead of pip install:
```bash
export PYTHONPATH="/home/aaronfeng/tpu-spec-decode/vllm:/home/aaronfeng/tpu-spec-decode/tpu-inference:${PYTHONPATH:-}"
```
This matches the approach already used by `check_dflash_support.sh`. Both imports verified. Uninstalled the pip packages to avoid import path conflicts.

---

## Step 8: Re-pin JAX if Overwritten

**Status:** DONE
**Started:** 2026-03-02
**Output:**
```
JAX: 0.6.2
Devices: 4
```
**Notes:** JAX 0.6.2 survived all dependency installs — no re-pin needed. This is because we used `--no-deps` for the package installs and manually filtered the requirements.txt.

---

## Step 9: Smoke Tests and DFlash Import

**Status:** DONE
**Started:** 2026-03-02
**Output:**
```
# DFlash import
DFlash import: OK

# check_dflash_support.sh host
supported_speculative_methods=deepseek_mtp,dflash,draft_model,eagle,eagle3,...,ngram,...
dflash_supported=True
tpu_inference_dflash_import=ok
[OK] DFlash support check passed.

# tpu_sanity_check.py
TPU chips: 4
Matmul OK: shape=(1024, 1024), dtype=bfloat16
Attention OK: shape=(1, 2, 16, 128), dtype=bfloat16
TPU sanity check passed (JAX).
```
**Notes:** All three checks pass. Warnings about Triton (no GPU) and `vllm._C` (no compiled C extension on TPU) are expected and harmless. Required installing additional vLLM runtime deps (`cbor2`, `transformers`, `pyzmq`, etc.) from `requirements/common.txt` since we couldn't use `pip install vllm`.

---

## Summary

| Step | Status | Issues |
|---|---|---|
| 1. Docker group | DONE | None |
| 2. Venv | DONE | Needed `apt install python3.10-venv`; cleaned partial dir |
| 3. JAX install | DONE | Clean — JAX 0.6.2 + jaxlib 0.6.2 |
| 4. JAX TPU verify | DONE | 4 v5p chips detected (2x2x1x1) |
| 5. Clone repos | DONE | Empty placeholder dirs blocked cloning; cleaned with `rmdir` |
| 6. tpu-inference | DONE | Filtered 3.11+-only deps; `IS_FOR_V7X=false`; used `--no-deps` |
| 7. vLLM | DONE | Editable install impossible on TPU; switched to PYTHONPATH |
| 8. JAX re-pin | DONE | No re-pin needed — survived all installs |
| 9. Smoke tests | DONE | Installed vLLM runtime deps from `requirements/common.txt` |

## Final Environment

```bash
# Activate
source ~/venv/bin/activate
export PYTHONPATH="/home/aaronfeng/tpu-spec-decode/vllm:/home/aaronfeng/tpu-spec-decode/tpu-inference:${PYTHONPATH:-}"

# Verify
python3 -c "import jax; print(jax.__version__, jax.device_count())"  # 0.6.2, 4
python3 -c "from tpu_inference.spec_decode.jax.dflash import DFlashProposer; print('OK')"
```

### Key Version Pins

| Package | Installed | Required (3.11+) | Note |
|---|---|---|---|
| Python | 3.10.12 | 3.11+ for upstream | VM constraint |
| JAX | 0.6.2 | 0.8.0 | Pinned < 0.7 for Python 3.10 |
| jaxlib | 0.6.2 | 0.8.0 | Matches JAX |
| libtpu | 0.0.17 | 0.0.31 (v7x) | Matches JAX 0.6.x |
| flax | 0.10.7 | 0.11.1 | Latest for 3.10 |
| numba | 0.61.2 | 0.62.1 | Latest for 3.10 |
| torchax | 0.0.7 | 0.0.10 | Latest for 3.10 |
| torch | 2.9.0 | — | Via torchvision dep |

### Known Warnings (Expected)

- `The vLLM package was not found` — PYTHONPATH-based install, no version metadata
- `Triton is installed but 0 active driver(s) found` — No GPU, TPU only
- `Failed to import from vllm._C` — No compiled C extension on TPU
