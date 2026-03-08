# TPU Speculative Decoding Project — Documentation

This folder contains the project documentation: the conceptual foundation (below) and the development log (`DEV_LOG.md`). The content here is synthesized from the original docs 0 (research overview and conceptual overview).

---

## Overview

Speculative decoding is a technique to accelerate inference for large language models. This project explores implementing and optimizing speculative decoding on Google Cloud TPUs, with particular interest in diffusion-model-based approaches that have shown 5–6× speedups on GPUs.

**Key motivation:** TPUs are actually friendlier to diffusion models than GPUs, making this a promising research direction.

## Project Goals

1. Achieve speculative decoding performance on TPUs comparable to or exceeding GPU implementations
2. Explore further research ideas building on this foundation

## Starter Task

**Prof. Zhang's suggestion:** "How about you guys try to first port dflash to TPU"

DFlash is the recommended first milestone before tackling more complex diffusion-based speculative decoding implementations.

---

# Part I: Conceptual Foundation

## 1. The Sequential Bottleneck in Autoregressive Inference

Modern LLM deployments operate at massive scale, where even a single sequential component in the inference pipeline can dominate end-to-end latency. Inference has two stages:

### 1.1 Prefilling

In the prefilling stage, the model processes the input prompt to initialize its internal state (KV cache) for decoding. All tokens in the prompt are known in advance, so there is no sequential runtime dependency. This stage can be fully parallelized across the prompt length.

### 1.2 Decoding

In the decoding stage, the model generates new tokens autoregressively. Each prediction at step *t*+1 depends on the token generated at step *t*. Vanilla decoding requires one forward pass per generated token. Although each forward pass can reuse the KV cache, the process cannot be parallelized across time. **Decoding is the primary bottleneck in LLM inference.**

---

## 2. Speculative Decoding

Speculative decoding introduces parallelism into decoding by using a **fast draft model** to propose a sequence of tokens to the **target model**. The target model verifies these draft tokens and produces the final output.

**Terms:**
- **Draft model:** Fast, cheap model proposing draft tokens
- **Target model:** Main autoregressive model (single source of truth)

**Mechanism:**
1. Draft model generates *n* draft tokens
2. Target model verifies the sequence in a **single forward pass** across positions *t*+1 … *t*+*n*
3. At each position, a rejection-sampling rule compares draft vs target probabilities
4. Verification stops at the first rejected token; remaining drafts are discarded

This relaxes the past–future dependency: the target evaluates a known sequence in one parallel pass instead of waiting for each token to be sampled before the next step.

---

## 3. Draft Model

Draft models range from simple heuristics to learned neural models. The TPU Inference repo supports:

- **ngram:** Non-neural; very low cost but poor approximation → low acceptance
- **EAGLE-3:** One-layer transformer; better approximation but autoregressive drafting → sequential proposal

Diffusion-based draft models (like DFlash) parallelize the proposal phase while achieving high acceptance rates.

---

## 4. DFlash (Conceptual)

DFlash uses a **diffusion-style draft model** to propose a **block of tokens in a single forward pass**, instead of one-by-one autoregressive generation.

**Key distinction:**
- Autoregressive drafter: *O*(*k*) sequential forward passes for *k* draft tokens
- DFlash: *O*(1) forward passes for a block of *k* tokens

**Mechanism:**
- Target model provides contextual hidden states from the last generated token
- Draft model takes context + a block of placeholder embeddings (masks/noise)
- In one forward pass, it predicts logits for all block positions
- Draft tokens are sampled and verified by the target using standard speculative decoding

DFlash uses a **lightweight diffusion draft model conditioned on target context** to achieve both fast drafting and high acceptance. See: https://z-lab.ai/projects/dflash/

---

## 5. DFlash Implementation Overview (Reference Repo)

The reference implementation is [z-lab/dflash](https://github.com/z-lab/dflash). High-level design:

- **No separate embedding or LM head:** Reuses target model's `embed_tokens` and LM head
- **Input:** Block of token IDs (with masks) → target embedding; plus **target hidden states** (concatenated, projected) as context
- **Draft stack:** Small decoder layers with **non-causal attention** over the block (all positions attend to each other)
- **Output:** Block hidden states → target LM head → draft logits → sampled draft tokens

DFlash is a thin, context-conditioned block predictor that inherits vocabulary and output semantics from the target.

---

## 6. vLLM and TPU-Inference

- **vLLM:** High-throughput inference server for LLMs; default GPU backend
- **tpu-inference** ([vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference)): TPU backend for vLLM; plugs in as platform layer; supports ngram, EAGLE-3, and JAX/PyTorch models

**Integration goal:** Add DFlash as a new speculative decoding method **inside tpu-inference**, implementing the draft model and proposer in JAX/Flax, and reusing the reference repo only for behavior and baselines.

---

# Part II: Research Overview & Resources

## Core Repositories

### DFlash ⭐ STARTER TASK

- **GitHub:** https://github.com/z-lab/dflash
- **Authors:** Jian Chen, Zhijian Liu (Z Lab)
- **Status:** Paper 2026; SGLang supported, vLLM in progress
- **Models:** z-lab/Qwen3-8B-DFlash-b16, z-lab/Qwen3-4B-DFlash-b16, z-lab/Qwen3-Coder-30B-A3B-DFlash
- **TPU task:** Adapt DFlash to run on TPU; replace flash-attention-2 with TPU-compatible attention

### vLLM TPU Inference

- **GitHub:** https://github.com/vllm-project/tpu-inference
- **Key file:** [speculative_decoding_manager.py](https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/runner/speculative_decoding_manager.py)
- **Contact:** vllm-tpu@google.com
- **Features:** EAGLE-3, pipeline parallelism, multi-host disaggregated serving, FP8 MoE

### Other References

- **Prompt Lookup Decoding:** [apoorvumang/prompt-lookup-decoding](https://github.com/apoorvumang/prompt-lookup-decoding) — 2–4× on input-grounded tasks
- **Lookahead Reasoning:** [hao-ai-lab/LookaheadReasoning](https://github.com/hao-ai-lab/LookaheadReasoning) — NeurIPS 2025; reasoning-step speculation; GPU→TPU diff reference
- **vLLM Speculators:** [vllm-project/speculators](https://github.com/vllm-project/speculators) — Unified SD training and vLLM integration

## TPU Setup

- **Docs:** [Cloud TPU](https://docs.cloud.google.com/tpu/docs), [JAX & Google CLI](https://married-spell-e7e.notion.site/JAX-GOOGLE-CLI-Guide-24df509095f180abbcf7ddc7ff0e9252)

Example v4-8 creation:

```bash
gcloud compute tpus queued-resources create lmgame-rl-queued \
    --node-id=lmgame-rl-node \
    --project=hao-ai-lab-trc \
    --zone=us-central2-b \
    --accelerator-type=v4-8 \
    --runtime-version=tpu-ubuntu2204-base
```

## Key Papers: Diffusion-Based Speculative Decoding

| Paper | arXiv | Speedup | Key innovation |
|-------|-------|---------|----------------|
| **SpecDiff** | [2408.05636](https://arxiv.org/abs/2408.05636) | up to 8.7× | Original diffusion-based SD |
| **SpecDiff-2** | [2511.00606](https://arxiv.org/abs/2511.00606) | up to 5.5× avg | Drafter-verifier calibration |
| **FailFast** | [2512.20573](https://arxiv.org/abs/2512.20573) | up to 4.9× | Dynamic speculation length |
| **DART** | [2601.19278](https://arxiv.org/abs/2601.19278) | 2.03–3.44× | N-gram tree pruning |
| **DFlash** | [2602.06036](https://arxiv.org/html/2602.06036v1) | 6×+ lossless | Block diffusion drafter |

---

# Documentation Index

- **`DEV_LOG.md`** — Chronological development flow: integration, parity, research, scaling, PR
- **`SCRIPTS_REFERENCE.md`** — Benchmarks, tests, and verification scripts reference
- **`README.md`** (this file) — Conceptual foundation + research overview + resources

---

*Last updated: March 2026*
