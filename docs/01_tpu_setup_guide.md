# TPU v4 Quickstart Guide — Speculative Decoding Project

**Team:** Aaron Feng, Andy Huang, Son Nguyen, Zhongyan Luo
**GCP Project:** `hao-ai-lab-trc`
**Coordinator:** Yiming Zhao
**Last updated:** February 5, 2026

---

## 1. Current Access & Quota

We've been added to the GCP project `hao-ai-lab-trc`. As of now:

- **v5p is NOT yet available** for free — it may come later.
- **v4 TPUs are available** with free quota. Use only the **free** TPU resources.
- Yiming has forwarded an email with the full list of available free quotas — **refer to that email** for the authoritative quota details.
- **"On-demand"** resources are preferred over **"spot"** (spot is preemptable and can be interrupted at any time).
- **Do not exceed the quotas** specified in the forwarded email.
- **Do not delete resources created by other users.**

> ⚠️ The quota information at the end of the [JAX & Google CLI Notion guide](https://married-spell-e7e.notion.site/JAX-GOOGLE-CLI-Guide-24df509095f180abbcf7ddc7ff0e9252) is **outdated and expired**. Ignore it. Use the quotas from Yiming's email instead.

---

## 2. TPU v4 Hardware Overview

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
| TPU v4 Docs | https://docs.cloud.google.com/tpu/docs/v4 |
| TPU Software Versions | https://docs.cloud.google.com/tpu/docs/runtimes |
| Attach Durable Block Storage | https://docs.cloud.google.com/tpu/docs/attach-durable-block-storage |
| vLLM TPU Inference Repo | https://github.com/vllm-project/tpu-inference |
| DFlash (Starter Task) | https://github.com/z-lab/dflash |

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
gcloud compute tpus accelerator-types list --zone=us-central2-b

# List your current TPU resources
gcloud compute tpus tpu-vm list --zone=us-central2-b

# List queued resource requests
gcloud compute tpus queued-resources list --zone=us-central2-b
```

---

## 5. Creating a TPU v4 VM

### Option A: Queued Resources (Recommended)

Queued resources queue your request until TPU capacity is available. This is the method shown in the Slack thread:

```bash
gcloud compute tpus queued-resources create YOUR-NAME-queued \
    --node-id=YOUR-NAME-node \
    --project=hao-ai-lab-trc \
    --zone=us-central2-b \
    --accelerator-type=v4-8 \
    --runtime-version=tpu-ubuntu2204-base
```

Check status:

```bash
gcloud compute tpus queued-resources describe YOUR-NAME-queued \
    --zone=us-central2-b
```

Provisioning typically takes up to 30 minutes.

### Option B: Direct Creation

```bash
gcloud compute tpus tpu-vm create YOUR-NAME-node \
    --project=hao-ai-lab-trc \
    --zone=us-central2-b \
    --accelerator-type=v4-8 \
    --version=tpu-ubuntu2204-base
```

> **Naming convention:** Please prefix your resources with your name (e.g., `aaron-node`, `andy-queued`) to avoid conflicts. Make sure you select the correct zone as specified in Yiming's email.

---

## 6. Connecting to Your TPU VM

```bash
# SSH into the TPU VM
gcloud compute tpus tpu-vm ssh YOUR-NAME-node --zone=us-central2-b

# For multi-host TPUs, specify the worker
gcloud compute tpus tpu-vm ssh YOUR-NAME-node --zone=us-central2-b --worker=0
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
    --zone=us-central2-b \
    --type=pd-balanced
```

### Attach to Your TPU VM

```bash
gcloud alpha compute tpus tpu-vm attach-disk YOUR-NAME-node \
    --zone=us-central2-b \
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
> gcloud compute disks delete YOUR-NAME-disk --zone=us-central2-b
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

## 10. Starter Task: Porting DFlash to TPU

Prof. Zhang's recommended first milestone is porting **DFlash** to TPU.

**What is DFlash?** A block-diffusion-based speculative decoding method that uses a lightweight diffusion model for drafting. It already outperforms EAGLE-3 on GPUs despite being trained on significantly less data (289K vs 1.4M samples).

**Key porting considerations:**

1. DFlash uses **flash-attention-2**, which is GPU-specific. This needs to be replaced with a TPU-compatible attention implementation (e.g., JAX's built-in attention or Pallas kernels).
2. The vLLM TPU Inference repo ([speculative_decoding_manager.py](https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/runner/speculative_decoding_manager.py)) already has EAGLE-3 support on TPU — use this as reference architecture.
3. The [Lookahead Reasoning GPU→TPU diff](https://github.com/ConstBob/Lookahead-Reasoning/compare/main...tpu) shows concrete examples of modifications needed when porting from GPU to TPU.

**Repos to clone on the TPU VM:**

```bash
git clone https://github.com/z-lab/dflash.git
git clone https://github.com/vllm-project/tpu-inference.git
```

**Available DFlash draft models:**
- `z-lab/Qwen3-8B-DFlash-b16`
- `z-lab/Qwen3-4B-DFlash-b16`
- `z-lab/Qwen3-Coder-30B-A3B-DFlash`

---

## 11. Cleanup — Do This When You're Done

Always tear down resources when not actively using them to stay within quota:

```bash
# Delete the TPU VM
gcloud compute tpus tpu-vm delete YOUR-NAME-node --zone=us-central2-b

# Or if created via queued resources
gcloud compute tpus queued-resources delete YOUR-NAME-queued --zone=us-central2-b

# Delete any extra disks
gcloud compute disks delete YOUR-NAME-disk --zone=us-central2-b

# Verify everything is cleaned up
gcloud compute tpus tpu-vm list --zone=us-central2-b
gcloud compute disks list --filter="zone:(us-central2-b)"
```

---

## 12. Troubleshooting

**TPU VM not provisioning?**
- Resources might take up to 30 minutes. Use queued resources and check status with `describe`.
- Make sure you're using the correct zone from the quota email.

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
| **DFlash** | Starter task — port to TPU | https://github.com/z-lab/dflash |
| **vLLM TPU Inference** | TPU inference backend with SD support | https://github.com/vllm-project/tpu-inference |
| **Prompt Lookup Decoding** | Baseline SD via string matching | https://github.com/apoorvumang/prompt-lookup-decoding |
| **Lookahead Reasoning** | GPU→TPU porting reference | https://github.com/hao-ai-lab/LookaheadReasoning |
| **Speculators** | Unified SD training/eval library | https://github.com/vllm-project/speculators |