# TPU Setup Guide — Speculative Decoding Project

**Team:** Aaron Feng, Andy Huang, Son Nguyen, Zhongyan Luo
**GCP Project:** `hao-ai-lab-trc`
**Coordinator:** Yiming Zhao
**Last updated:** February 8, 2026

---

## 1. Current Access & Quota

We've been added to the GCP project `hao-ai-lab-trc` under the **Tensor Research Cloud (TRC)** program. As of the email Yiming forwarded (Feb 7 2026):

### TRC free v5 quota (from TRC-Support)

- **Quota:** 4 on-demand Cloud TPU v5 chips in zone **`us-east1-d`**.
- **Duration:** 86-day free trial for TPUs created in the zone above.
- **Critical:** Create new Cloud TPUs **only in the zone listed above** (`us-east1-d`). Creating TPUs in other zones can incur charges.
- **Architecture:** TRC quotas are intended for the **TPU VM architecture** and the **Queued Resource API**. Use the correct flags for your quota type ([on-demand](https://cloud.google.com/tpu/docs/queued-resources#on-demand) vs preemptible).
- **Prefer on-demand** over preemptible when both are available; preemptible (spot) can be reclaimed at any time.
- **Do not exceed the free quota.** Delete unused TPUs and Queued Resources so quota is not consumed unnecessarily.

### General rules

- **v4 TPUs** may also be available with free quota; use only the **free** resources specified for the project.
- Official v5 docs: [Cloud TPU v5p](https://docs.cloud.google.com/tpu/docs/v5p). Get started with the [QuickStart guide](https://cloud.google.com/tpu/docs/quickstart) and, for v4, the [v4 user's guide](https://cloud.google.com/tpu/docs/users-guide-tpu-vm).
- **Do not delete resources created by other users.**

> ⚠️ The quota information at the end of the [JAX & Google CLI Notion guide](https://married-spell-e7e.notion.site/JAX-GOOGLE-CLI-Guide-24df509095f180abbcf7ddc7ff0e9252) is **outdated and expired**. Ignore it. Use the quotas from Yiming's forwarded TRC email instead.

---

## 2. TPU Hardware Overview

**TPU v5 (v5p)** is now available; see [Cloud TPU v5p](https://docs.cloud.google.com/tpu/docs/v5p) for specs and regions. The sections below also describe **v4** for teams still using it.

### TPU v4

Each TPU v4 chip has:

- **32 GiB unified HBM** across the entire chip (shared by both on-chip TensorCores)
- **Up to 275 peak TFLOPS** per chip
- **128×128 systolic array** MXUs (pick tensor dimensions as multiples of 128 for best performance)
- 3D torus interconnect topology for multi-chip communication

Common v4 configurations (small topologies, < 64 chips):

| Name    | Topology | Chips | TensorCores |
|---------|----------|-------|-------------|
| v4-8    | 2×2×1    | 4     | 8           |
| v4-16   | 2×2×2    | 8     | 16          |
| v4-32   | 2×2×4    | 16    | 32          |
| v4-64   | 2×4×4    | 32    | 64          |

For our initial work, **v4-8** (single host, 4 chips) is the starting point.

---

## 3. Essential Documentation & Links

| Resource | URL |
|----------|-----|
| JAX & Google CLI Guide (Notion) | https://married-spell-e7e.notion.site/JAX-GOOGLE-CLI-Guide-24df509095f180abbcf7ddc7ff0e9252 |
| gcloud CLI Cheatsheet | https://cloud.google.com/sdk/docs/cheatsheet |
| Cloud TPU Documentation | https://docs.cloud.google.com/tpu/docs |
| **TPU v5p Docs** | https://docs.cloud.google.com/tpu/docs/v5p |
| TPU v4 Docs | https://docs.cloud.google.com/tpu/docs/v4 |
| TPU Software Versions | https://docs.cloud.google.com/tpu/docs/runtimes |
| Attach Durable Block Storage | https://docs.cloud.google.com/tpu/docs/attach-durable-block-storage |
| vLLM TPU Inference Repo | https://github.com/vllm-project/tpu-inference |
| DFlash (reference implementation) | https://github.com/z-lab/dflash |

---

## 4. Setting Up gcloud CLI

If you haven't already installed the Google Cloud SDK:

```bash
# Install gcloud CLI (follow interactive prompts)
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Initialize and authenticate
gcloud init
gcloud auth login

# Set the project
gcloud config set project hao-ai-lab-trc

# Verify
gcloud config list
```

Useful gcloud commands:

```bash
# Check your current config
gcloud info

# List available TPU accelerator types in a zone
# For our TRC v5 quota, use zone us-east1-d (see Section 1).
gcloud compute tpus accelerator-types list --zone=us-east1-d

# List your current TPU resources
gcloud compute tpus tpu-vm list --zone=us-east1-d

# List queued resource requests
gcloud compute tpus queued-resources list --zone=us-east1-d
```

---

## 5. Creating a TPU VM

Our TRC quota (Section 1) is for **v5** chips in `us-east1-d`. If you also have v4 quota, you can create v4 VMs in the same zone. Use `gcloud compute tpus accelerator-types list --zone=us-east1-d` to see available types (e.g. `v5p`, `v4-8`).

### Option A: Queued Resources (Recommended)

Queued resources queue your request until TPU capacity is available.

**For TRC v5 quota (4 v5 chips):**
```bash
gcloud compute tpus queued-resources create YOUR-NAME-queued \
    --node-id=YOUR-NAME-node \
    --project=hao-ai-lab-trc \
    --zone=us-east1-d \
    --accelerator-type=v5p \
    --runtime-version=tpu-ubuntu2204-base
```

**If you have v4 quota instead:**
```bash
gcloud compute tpus queued-resources create YOUR-NAME-queued \
    --node-id=YOUR-NAME-node \
    --project=hao-ai-lab-trc \
    --zone=us-east1-d \
    --accelerator-type=v4-8 \
    --runtime-version=tpu-ubuntu2204-base
```

Check status:

```bash
gcloud compute tpus queued-resources describe YOUR-NAME-queued \
    --zone=us-east1-d
```

Provisioning typically takes up to 30 minutes.

### Option B: Direct Creation

Use the same `--accelerator-type` as your quota (e.g. `v5p` for TRC v5, or `v4-8` if you have v4 quota):

```bash
gcloud compute tpus tpu-vm create YOUR-NAME-node \
    --project=hao-ai-lab-trc \
    --zone=us-east1-d \
    --accelerator-type=v5p \
    --version=tpu-ubuntu2204-base
```

> **Naming convention:** Please prefix your resources with your name (e.g., `aaron-node`, `andy-queued`) to avoid conflicts. Use zone **us-east1-d** for the TRC v5 quota (Section 1).

---

## 6. Connecting to Your TPU VM

```bash
# SSH into the TPU VM
gcloud compute tpus tpu-vm ssh YOUR-NAME-node --zone=us-east1-d

# For multi-host TPUs, specify the worker
gcloud compute tpus tpu-vm ssh YOUR-NAME-node --zone=us-east1-d --worker=0
```

Once connected, you'll have a standard Ubuntu 22.04 environment with a 100 GiB boot disk.

---

## 7. Installing JAX on the TPU VM

After SSH-ing in:

```bash
# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Install JAX for TPU (check https://github.com/google/jax#pip-installation for latest)
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Verify JAX sees the TPU
python3 -c "import jax; print(jax.devices())"
# Should show TpuDevice entries
```

For PyTorch/XLA:

```bash
pip install torch torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
```

---

## 8. Storage

The TPU VM comes with a **100 GiB boot disk**. If you need more space (e.g., for model weights), you can attach additional block storage:

### Create a Disk

```bash
gcloud compute disks create YOUR-NAME-disk \
    --size=200GB \
    --zone=us-east1-d \
    --type=pd-balanced
```

### Attach to Your TPU VM

```bash
gcloud alpha compute tpus tpu-vm attach-disk YOUR-NAME-node \
    --zone=us-east1-d \
    --disk=YOUR-NAME-disk \
    --mode=read-write
```

### Format and Mount (from inside the TPU VM)

```bash
# List disks to find the new one (likely /dev/sdb)
sudo lsblk

# Format (only if new/blank — this erases all data!)
sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb

# Create mount point and mount
sudo mkdir -p /mnt/disks/data
sudo mount -o discard,defaults /dev/sdb /mnt/disks/data
sudo chmod a+w /mnt/disks/data
```

> ⚠️ **Additional storage may NOT be free.** Delete extra disks when you're done with them:
> ```bash
> gcloud compute disks delete YOUR-NAME-disk --zone=us-east1-d
> ```

---

## 9. Running a Quick Sanity Check

Verify your TPU works with a minimal JAX computation:

```python
import jax
import jax.numpy as jnp

# Check devices
print(f"TPU devices: {jax.devices()}")
print(f"Number of devices: {jax.device_count()}")

# Simple matrix multiply to exercise the MXU
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (1024, 1024))
y = x @ x.T
print(f"Matrix multiply result shape: {y.shape}")
print(f"Computation device: {y.devices()}")
print("TPU is working correctly!")
```

### Optional: PyTorch/XLA sanity check (recommended before DFlash)
If you plan to run DFlash on TPU, run the repo script:
```bash
python preparation/tpu_sanity_check.py
```
This checks the XLA device, a small matmul, and SDPA attention.

---

## 10. Starter Task: Integrating DFlash into TPU Inference

Prof. Zhang's recommended approach is to **integrate DFlash into the tpu-inference repo** (add DFlash as a speculative decoding method alongside ngram and EAGLE-3), not to port or modify the DFlash repo itself for TPU.

**What is DFlash?** A block-diffusion-based speculative decoding method that uses a lightweight diffusion model for drafting. It already outperforms EAGLE-3 on GPUs despite being trained on significantly less data (289K vs 1.4M samples).

**Integration approach:**

1. Use the **tpu-inference** repo as the main codebase. Add DFlash as a new speculative method (e.g. `method == "dflash"`) in the runner and speculative decoding manager, following the existing EAGLE-3 pattern.
2. Implement the DFlash draft model and attention in JAX/Flax inside tpu-inference (e.g. `tpu_inference/models/jax/`, `tpu_inference/spec_decode/jax/`), using the [z-lab/dflash](https://github.com/z-lab/dflash) repo as the **reference** for behavior and math.
3. The [speculative_decoding_manager.py](https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/runner/speculative_decoding_manager.py) and EAGLE-3 proposer in tpu-inference are the right reference for how to plug in a new method.

**Repos to use:**

- **tpu-inference** — where the integration code lives (fork and work on a branch, e.g. `dflash-integration`).
- **z-lab/dflash** — reference implementation (GPU/PyTorch); do not modify for TPU; use it to understand DFlash behavior and to compare results.

**Available DFlash draft models (for loading in tpu-inference):**
- `z-lab/Qwen3-8B-DFlash-b16`
- `z-lab/Qwen3-4B-DFlash-b16`
- `z-lab/Qwen3-Coder-30B-A3B-DFlash`

---

## 11. Cleanup — Do This When You're Done

Always tear down resources when not actively using them to stay within quota:

```bash
# Delete the TPU VM
gcloud compute tpus tpu-vm delete YOUR-NAME-node --zone=us-east1-d

# Or if created via queued resources
gcloud compute tpus queued-resources delete YOUR-NAME-queued --zone=us-east1-d

# Delete any extra disks
gcloud compute disks delete YOUR-NAME-disk --zone=us-east1-d

# Verify everything is cleaned up
gcloud compute tpus tpu-vm list --zone=us-east1-d
gcloud compute disks list --filter="zone:(us-east1-d)"
```

---

## 12. Troubleshooting

**TPU VM not provisioning?**
- Resources might take up to 30 minutes. Use queued resources and check status with `describe`.
- Use zone **us-east1-d** for TRC v5 quota to avoid charges (Section 1).

**`jax.devices()` returns empty or CPU only?**
- Ensure you installed `jax[tpu]` with the correct libtpu release URL.
- Try `pip install --upgrade jax jaxlib` and reinstall.

**Out of disk space?**
- The boot disk is 100 GiB. Attach additional block storage (see Section 8).
- Use `df -h` to check disk usage.

**Permission denied / quota errors?**
- Confirm you're in the `hao-ai-lab-trc` project: `gcloud config get project`
- Stick to the free resource types from Yiming's email.
- Use on-demand over spot when available.

**Need help?**
- Post in the Slack thread or reach out to Yiming Zhao.
- vLLM TPU team contact: vllm-tpu@google.com
- Google Cloud TPU docs: https://docs.cloud.google.com/tpu/docs

---

## Appendix: Key Project Repos at a Glance

| Repo | Purpose | Link |
|------|---------|------|
| **DFlash** | Reference implementation (GPU); use for behavior and baselines | https://github.com/z-lab/dflash |
| **vLLM TPU Inference** | **Integration target** — add DFlash as a speculative method here | https://github.com/vllm-project/tpu-inference |
| **Prompt Lookup Decoding** | Baseline SD via string matching | https://github.com/apoorvumang/prompt-lookup-decoding |
| **Lookahead Reasoning** | GPU→TPU porting reference | https://github.com/hao-ai-lab/LookaheadReasoning |
| **Speculators** | Unified SD training/eval library | https://github.com/vllm-project/speculators |