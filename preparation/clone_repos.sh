#!/bin/bash
#
# Clone all repositories needed for the TPU Speculative Decoding project
# - external/  : Upstream originals (read-only reference)
# - forked/    : Our forks (where we make changes)
#
# Run this script from anywhere in the repo
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXTERNAL_DIR="${REPO_ROOT}/external"
FORKED_DIR="${REPO_ROOT}/forked"

# ============================================================================
# EXTERNAL: Upstream originals (read-only reference)
# ============================================================================

echo "==> Cloning upstream repositories into ${EXTERNAL_DIR}"
mkdir -p "${EXTERNAL_DIR}"
cd "${EXTERNAL_DIR}"

# ============================================================================
# Core Repositories
# ============================================================================

# DFlash - STARTER TASK: Block diffusion-based speculative decoding
if [ ! -d "dflash" ]; then
    echo "Cloning DFlash..."
    git clone https://github.com/z-lab/dflash.git
else
    echo "DFlash already exists, skipping..."
fi

# vLLM TPU Inference - TPU backend with speculative decoding support
if [ ! -d "tpu-inference" ]; then
    echo "Cloning vLLM TPU Inference..."
    git clone https://github.com/vllm-project/tpu-inference.git
else
    echo "vLLM TPU Inference already exists, skipping..."
fi

# Prompt Lookup Decoding - Simple string-matching SD baseline
if [ ! -d "prompt-lookup-decoding" ]; then
    echo "Cloning Prompt Lookup Decoding..."
    git clone https://github.com/apoorvumang/prompt-lookup-decoding.git
else
    echo "Prompt Lookup Decoding already exists, skipping..."
fi

# Lookahead Reasoning - GPU→TPU porting reference
if [ ! -d "LookaheadReasoning" ]; then
    echo "Cloning Lookahead Reasoning..."
    git clone https://github.com/hao-ai-lab/LookaheadReasoning.git
else
    echo "Lookahead Reasoning already exists, skipping..."
fi

# Lookahead Reasoning TPU fork - contains GPU→TPU diff
if [ ! -d "Lookahead-Reasoning-TPU" ]; then
    echo "Cloning Lookahead Reasoning TPU fork..."
    git clone --branch tpu https://github.com/ConstBob/Lookahead-Reasoning.git Lookahead-Reasoning-TPU
else
    echo "Lookahead Reasoning TPU fork already exists, skipping..."
fi

# vLLM Speculators - Unified SD training/eval library
if [ ! -d "speculators" ]; then
    echo "Cloning vLLM Speculators..."
    git clone https://github.com/vllm-project/speculators.git
else
    echo "vLLM Speculators already exists, skipping..."
fi

# ============================================================================
# Diffusion-based SD Research Repos (Optional)
# ============================================================================

# FailFast - Dynamic speculation length with dLLMs
if [ ! -d "failfast" ]; then
    echo "Cloning FailFast..."
    git clone https://github.com/ruipeterpan/failfast.git
else
    echo "FailFast already exists, skipping..."
fi

echo ""
echo "==> External repositories cloned!"

# ============================================================================
# FORKED: Our forks (where we make changes)
# ============================================================================

echo ""
echo "==> Cloning forked repositories into ${FORKED_DIR}"
mkdir -p "${FORKED_DIR}"
cd "${FORKED_DIR}"

# DFlash fork - our TPU port
if [ ! -d "dflash" ]; then
    echo "Cloning forked DFlash..."
    git clone https://github.com/aaronzhfeng/dflash.git
    cd dflash
    git remote add upstream https://github.com/z-lab/dflash.git
    cd ..
else
    echo "Forked DFlash already exists, skipping..."
fi

echo ""
echo "==> All repositories cloned successfully!"
echo ""
echo "External (upstream reference):"
ls -1 "${EXTERNAL_DIR}"
echo ""
echo "Forked (our working copies):"
ls -1 "${FORKED_DIR}"