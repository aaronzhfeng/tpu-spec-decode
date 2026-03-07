# GPU Verification Benchmarks (V14)

Rerun all GPU verification experiments on datacenter hardware (H100/A100).

## Quick Start

```bash
# On the H100 machine:
git clone <repo-url> && cd tpu-spec-decode

# One-command setup (installs PyTorch, downloads model ~8 GB)
bash benchmarks/gpu/setup.sh

# Run all experiments (~40 min on H100)
source .venv-gpu-bench/bin/activate
bash benchmarks/gpu/run_all.sh
```

Results go to `results/v14_h100/`.

## What Gets Run

| ID | What | Script | Time |
|----|------|--------|------|
| V14a | Full forward K=16,128 at L=64-1024 | `gpu_verify_full.py` | 5 min |
| V14b | Context scaling L=256-2048 | `gpu_verify_context_scaling.py` | 10 min |
| V14c | K-sweep K=16-128 step 16 | `gpu_verify_full.py` | 10 min |
| V14d | Isolated matmul scaling | `gpu_matmul_scaling.py` | 1 min |
| V14e | DFlash draft speed | `gpu_draft_speed.py` | 5 min |
| V14f | Component decomposition | `gpu_forward_decomposition.py` | 10 min |

## What We're Looking For

**Key question:** Is the 1.24x GPU verify penalty (K=128 vs K=16) an RTX-specific artifact or a real GPU characteristic?

- **H100 roofline predicts** FFN stays memory-bound through K=256 (OI=229 < ridge=295.5). So H100 should show a **smaller** penalty than RTX 2000 Ada's 1.24x.
- If H100 is near-flat (<1.05x): memory-boundedness is universal at 4B scale.
- If H100 matches RTX (~1.24x): genuine GPU effect, not hardware-specific.
- If K=80 step function disappears on H100: it was RTX tensor core tiling.

## RTX 2000 Ada Baseline (for comparison)

| Experiment | RTX Result |
|------------|-----------|
| V7 (full forward K=128/K=16) | 1.24x constant across L |
| V12 (context scaling) | 1.24x at L=256-2048 |
| V13 (K-sweep) | Step function at K=80 (1.07x→1.16x) |
| V10 (component split) | 40% FFN / 40% attention / 20% other |

## Hardware Comparison

| Spec | RTX 2000 Ada | H100 SXM | TPU v5p |
|------|-------------|----------|---------|
| BF16 TFLOPS | 12 | 990 | 459 |
| HBM BW (TB/s) | 0.29 | 3.35 | 4.8 |
| Ridge (FLOP/byte) | 41.7 | 295.5 | 95.6 |
| HBM capacity | 16 GB | 80 GB | 95 GB |

## Running Individual Experiments

```bash
source .venv-gpu-bench/bin/activate

# Just V14a
python benchmarks/gpu_verify_full.py \
    --k-values 16,128 --context-lengths 64,256,512,1024 \
    --trials 50 --output-json results/v14_h100/v14a_full_forward.json

# Just V14c (K-sweep)
python benchmarks/gpu_verify_full.py \
    --k-values 16,32,48,64,80,96,112,128 --context-lengths 256,1024 \
    --trials 50 --output-json results/v14_h100/v14c_k_sweep.json
```

## Dependencies

- Python 3.10+
- NVIDIA GPU with CUDA 12.x drivers
- ~8 GB disk for Qwen3-4B weights
- ~16 GB VRAM minimum (H100's 80 GB is plenty)

Installed by `setup.sh`: `torch`, `transformers`, `accelerate`, `huggingface-hub`.
