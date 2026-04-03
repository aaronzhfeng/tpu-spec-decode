# Environment Setup Guide

## Project Context

This is the working environment for porting DFlash block-diffusion speculative decoding to TPU, led by Aaron Feng as an HDSI capstone at UC San Diego under Professor Hao Zhang's group, mentored by Yiming Zhao. The primary repo is `tpu-spec-decode`, which contains benchmarks, docs, and orchestrates multiple clones of `aaronzhfeng/tpu-inference` (the upstream Google TPU inference repo) under `pr-ready/`. Three PRs are open upstream: #1868 (model + proposer), #1869 (pipeline integration), #1870 (e2e tests). The standalone DFlash proposer achieves tau=6.67 on TPU v5p. Key findings include K-flat verification (no step function through K=1024 on datacenter hardware). Current work streams: addressing PR reviewer feedback from Google (kyuyeunk, Lumosis), scoping a torchax port requested by Yarong Mu (Google), and waiting on wider-block checkpoints (K=32/64) from the DFlash authors (Zhijian Liu). All outbound communications are tracked in `brainstorm/response/`. Technical investigation docs are in `docs/` (numbered 01-63). Memory for cross-session context is in `.claude/projects/`. The TPU VM runs Docker with `vllm/vllm-tpu:latest` for inference; code editing should be done locally, with SSH commands queued to the TPU for benchmarks.

## Clone tpu-spec-decode (primary repo)

```bash
git clone https://github.com/aaronzhfeng/tpu-spec-decode.git
cd tpu-spec-decode
git submodule update --init --recursive
```

This pulls in:
- `brainstorm/` -> `aaronzhfeng/brainstorm-20-spec-decode-diffusion` (response drafts, proposals, literature)

## pr-ready/ subfolders (manual clones, not submodules)

These are clones of `aaronzhfeng/tpu-inference` at different branches. Clone them manually:

```bash
cd pr-ready

# Main tpu-inference (tracks upstream)
git clone https://github.com/aaronzhfeng/tpu-inference.git main
cd main && git checkout main && cd ..

# PR branch (model + proposer + inline comments)
git clone https://github.com/aaronzhfeng/tpu-inference.git pr_dflash_1
cd pr_dflash_1 && git checkout pr_dflash_1 && cd ..
# Also fetch the other PR branches:
cd pr_dflash_1 && git fetch origin pr_dflash_1b:pr_dflash_1b pr_dflash_1c:pr_dflash_1c && cd ..

# vLLM last-known-good (matching version for tpu-inference)
git clone https://github.com/vllm-project/vllm.git vllm-lkg
cd vllm-lkg && git checkout d15c3b9 && cd ..
```

### Branch reference

| Folder | Remote | Branch | Commit | PR | Purpose |
|--------|--------|--------|--------|-----|---------|
| `main` | aaronzhfeng/tpu-inference | main | `a73366c4` | - | Upstream tracking |
| `pr_dflash_1` | aaronzhfeng/tpu-inference | pr_dflash_1 | `82adc7d8` | #1868 | Model + proposer + inline comments |
| (same repo) | aaronzhfeng/tpu-inference | pr_dflash_1b | `20ec612c` | #1869 | Pipeline integration (tpu_runner, kv_cache, model_loader) |
| (same repo) | aaronzhfeng/tpu-inference | pr_dflash_1c | `bdc64e95` | #1870 | E2e tests + Buildkite CI |
| `vllm-lkg` | vllm-project/vllm | main @ `d15c3b9` | `d15c3b9` | - | Matching vLLM for tpu-inference |

All three PR branches share the same base (`a73366c4`) and are independent (not stacked). Each adds one commit on top of main.

### Folders safe to skip on fresh setup

These are investigation artifacts, not needed for development:
- `pr_dflash_1a/` - RoPE fix experiment (local clone, DO NOT apply)
- `pr_dflash_test/` - Combined branches for benchmarking
- `pr_dflash/` - Old pre-split PR branch
- `dflash-integration/` - Old integration branch
- `pr_dflash_1b/`, `pr_dflash_1c/` - Empty stubs

## Sibling repos (workspace root)

Clone these at the same level as tpu-spec-decode:

| Repo | URL | Purpose |
|------|-----|---------|
| `brainstorm-00-core` | `aaronzhfeng/brainstorm-00-core` | Research template and prompts |
| `research-copilot` | `aaronzhfeng/research-copilot` | Research tooling |
| `tpu-dflash-paper` | `aaronzhfeng/tpu-dflash-paper` | Paper and poster artifacts |
| `tpu-dflash-web` | `zhongyan0721/tpu-dflash` | Project website |
| `raw_ideas` | `aaronzhfeng/raw_ideas` | Idea scratchpad |

Optional / not needed for development:
- `marp-style-showcase/` - Presentation template (non-git)
- `son_dev/` - Son's dev workspace (non-git)
- `texmf/` - LaTeX config (non-git)
- `venv/` - Bare-metal Python venv (broken, use Docker instead)

## Remote inference via SSH

All code editing can be done locally. Use SSH to queue inference on the TPU VM:

```bash
# One-off command
ssh <user>@<TPU-VM-IP> "cd tpu-spec-decode && git pull && <command>"

# Docker inference
ssh <user>@<TPU-VM-IP> "sudo docker run --rm --privileged --net=host \
  -v /home/aaronfeng/tpu-spec-decode/pr-ready/pr_dflash_1:/workspace/tpu_inference \
  -v /home/aaronfeng/tpu-spec-decode/pr-ready/vllm-lkg:/workspace/vllm-lkg \
  -v /dev/shm/hf-cache:/dev/shm/hf-cache \
  -e PYTHONPATH=/workspace/vllm-lkg:/workspace/tpu_inference \
  -e HF_HOME=/dev/shm/hf-cache \
  vllm/vllm-tpu:latest \
  bash -c 'pip install flax==0.12.2 && <command>'"
```

### Key Docker details (from doc 63)

- Mount to `/workspace/tpu_inference` (underscore) to override built-in
- PYTHONPATH: vllm-lkg first, then tpu_inference
- `flax==0.12.2` required
- `--max-model-len 4096` if VMEM issues
- Models cached at `/dev/shm/hf-cache/` on the TPU VM
