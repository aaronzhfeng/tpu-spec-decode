# DFlash on TPU: vLLM pipeline and benchmarks

DFlash speculative decoding on TPU: run the vLLM + tpu-inference pipeline and compare with GPU baselines.

## Project overview

**Speculative decoding** speeds up autoregressive decoding by using a fast **draft model** to propose a block of tokens; the main **target model** verifies them in one forward pass. **DFlash** is a diffusion-style draft model: it predicts a block of draft tokens in parallel (one draft forward pass) instead of token-by-token, which fits TPUs well.

This repo wires DFlash into the **tpu-inference** backend (JAX/Flax on TPU) behind **vLLM**. You get the same vLLM API and scheduling, with DFlash as a speculative method alongside ngram and EAGLE-3. The reference implementation and algorithm live in [z-lab/dflash](https://github.com/z-lab/dflash); we use it for datasets and behavioral alignment.

---

## Repository structure

```
tpu-spec-decode/
├── deps/                    # Bootstrap deps (created by clone_repos.sh)
│   ├── tpu-inference/       # TPU backend + DFlash integration
│   ├── vllm/                # vLLM with TPU/speculative support
│   └── dflash/              # Datasets, load_and_process_dataset
├── preparation/             # clone_repos.sh, preflight/smoke helpers
├── tests/                   # preflight, smoke, benchmark, compare, standalone_benchmark
│   ├── configs/             # Manifest JSONs (smoke_*, benchmark_*.json)
│   └── lib/                 # common.sh, docker_run.sh
├── verification/            # Full validation harness + contribution matrix
├── results/                 # generate_csvs.py, output CSVs
├── benchmarks/              # standalone_dflash.py
└── docs/                    # Conceptual overview, design docs
```

## Setup and run pipeline

Follow these steps to set up the environment and run the DFlash pipeline end-to-end. All commands assume you are at the **repo root** (`tpu-spec-decode/`).

### 1. Create TPU v4 VM and SSH in

Request a v4-8 TPU (e.g. on-demand queue), wait until `state` is ACTIVE, then SSH in. (More thorough investigation will be run on **v5p**; this guide uses v4 for current development stage.)

```bash
gcloud compute tpus queued-resources create YOUR_NAME-dflash-queued \
  --node-id=YOUR_NAME-dflash-node \
  --project=hao-ai-lab-trc \
  --zone=us-central2-b \
  --accelerator-type=v4-8 \
  --runtime-version=tpu-ubuntu2204-base

gcloud compute tpus queued-resources describe YOUR_NAME-dflash-queued \
  --project=hao-ai-lab-trc --zone=us-central2-b
# Wait until state == ACTIVE

gcloud compute tpus tpu-vm ssh YOUR_NAME-dflash-node --zone=us-central2-b --project=hao-ai-lab-trc
```

### 2. Clone repo and bootstrap dependencies

Clone this repo, then run the bootstrap script. It creates a **`deps/`** directory with three clones: **tpu-inference**, **vllm**, and **dflash** (for datasets). No symlinks or `forks/` directory.

```bash
mkdir -p ~/dflash_spec && cd ~/dflash_spec
git clone https://github.com/aaronzhfeng/tpu-spec-decode.git
cd tpu-spec-decode

bash preparation/clone_repos.sh
```

### 3. Pull Docker image

```bash
sudo docker pull vllm/vllm-tpu:latest
```

### 4. Preflight and smoke

Preflight checks repo layout, Docker, and vLLM DFlash support. Smoke runs a short DFlash pipeline to confirm everything works.

```bash
bash tests/preflight.sh
bash tests/smoke.sh
```

### 5. vLLM pipeline benchmark

Run the math benchmark (gsm8k, math500, aime24, aime25). Results are written **outside the repo** under `/dev/shm/dflash-test-outputs/<run_id>/` (e.g. `bench_math_20260214_120000`; the script prints the path).

```bash
# DFlash math (~30 min)
bash tests/benchmark.sh math
```

Optional: compare to reported GPU speedups and regenerate CSVs in `results/`:

```bash
bash tests/compare.sh latest
python results/generate_csvs.py
```

Other suites: `bash tests/benchmark.sh code | chat | full`. Eagle3 (Llama-3.1-8B + Eagle3): `bash tests/benchmark.sh tests/configs/eagle3_llama_math.json`.

### 6. Standalone DFlash benchmark

Run the standalone JAX/TPU benchmark (no vLLM engine) on the same four math datasets. To have results picked up by `generate_csvs.py`, pass `--output-json results/standalone_<dataset>.json` and then run `python results/generate_csvs.py`.

```bash
bash tests/standalone_benchmark.sh --dataset gsm8k --max-samples 8 --max-new-tokens 256 --output-json results/standalone_gsm8k.json
bash tests/standalone_benchmark.sh --dataset math500 --max-samples 8 --max-new-tokens 256 --output-json results/standalone_math500.json
bash tests/standalone_benchmark.sh --dataset aime24 --max-samples 8 --max-new-tokens 256 --output-json results/standalone_aime24.json
bash tests/standalone_benchmark.sh --dataset aime25 --max-samples 8 --max-new-tokens 256 --output-json results/standalone_aime25.json

python results/generate_csvs.py
```

---

## Full validation run (verification harness)

For one command that runs preflight, scope guard, pytest subset, TPU smoke, TPU eval, and optional external reference (and compares to reported baselines), use the verification harness:

```bash
STRICT_PRECHECK=1 RUN_PYTEST=1 RUN_SMOKE=1 RUN_EVAL=1 RUN_EXTERNAL=0 \
bash verification/sh/run_dflash_validation_matrix.sh
```

See `verification/README.md` for per-stage commands and toggles.

---

## Cleanup

When finished, release the TPU to avoid charges. If the queued resource is **ACTIVE** (VM is running), you must delete the TPU VM first, then the queued resource:

```bash
# 1. Delete the TPU VM (required when state is ACTIVE)
gcloud compute tpus tpu-vm delete YOUR_NAME-dflash-node --zone=us-central2-b --project=hao-ai-lab-trc

# 2. Delete the queued resource
gcloud compute tpus queued-resources delete YOUR_NAME-dflash-queued --zone=us-central2-b --project=hao-ai-lab-trc
# If that fails with "Invalid choice", use:
# gcloud alpha compute tpus queued-resources delete YOUR_NAME-dflash-queued --zone=us-central2-b --project=hao-ai-lab-trc
```
