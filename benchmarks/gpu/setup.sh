#!/usr/bin/env bash
# H100/GPU quick-start setup for V14 verification experiments.
# Usage: bash benchmarks/gpu/setup.sh
# Run from repo root (tpu-spec-decode/).

set -euo pipefail

echo "=== GPU Benchmark Setup ==="

# 1. Check NVIDIA GPU
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Need NVIDIA GPU with CUDA drivers."
    exit 1
fi
echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# 2. Create venv (skip if already exists)
VENV_DIR=".venv-gpu-bench"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
else
    echo "Venv exists: $VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# 3. Install dependencies
echo "Installing PyTorch + dependencies..."
pip install --quiet --upgrade pip
pip install --quiet torch --index-url https://download.pytorch.org/whl/cu124
pip install --quiet transformers accelerate huggingface-hub

# 4. Verify CUDA is accessible from PyTorch
echo ""
echo "=== Verification ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'VRAM: {props.total_memory / 1e9:.1f} GB')
    print(f'Compute capability: {props.major}.{props.minor}')
else:
    print('WARNING: CUDA not available. Check driver/PyTorch compatibility.')
    exit(1)
"

# 5. Download model weights (biggest time sink — do it once)
echo ""
echo "=== Downloading Qwen3-4B (target model) ==="
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('Downloading tokenizer...')
AutoTokenizer.from_pretrained('Qwen/Qwen3-4B')
print('Downloading model weights (~8 GB)...')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-4B', torch_dtype='auto')
print('Done. Model cached for all experiments.')
"

echo ""
echo "=== Setup Complete ==="
echo "Run experiments:  source $VENV_DIR/bin/activate && bash benchmarks/gpu/run_all.sh"
