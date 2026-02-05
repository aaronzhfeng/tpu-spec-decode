# TPU Speculative Decoding Project

## Overview

Speculative decoding is a technique to accelerate inference for large language models. This project explores implementing and optimizing speculative decoding on Google Cloud TPUs, with particular interest in diffusion-model-based approaches that have shown 5-6x speedups on GPUs.

**Key motivation:** TPUs are actually friendlier to diffusion models than GPUs, making this a promising research direction.

## Project Goals

1. Achieve speculative decoding performance on TPUs comparable to or exceeding GPU implementations
2. Explore further research ideas building on this foundation

## Starter Task

**Prof. Zhang's suggestion:** "How about you guys try to first port dflash to TPU"

This establishes dflash as the recommended first milestone before tackling more complex diffusion-based speculative decoding implementations.

---

## Core Repositories

### 1. DFlash ⭐ STARTER TASK

**Block Diffusion for Ultra-Fast Speculative Decoding**

- **GitHub:** https://github.com/z-lab/dflash
- **Authors:** Jian Chen, Zhijian Liu (Z Lab)
- **Status:** Paper coming soon (2026)
- **Integration:** SGLang supported, vLLM in progress

**Description:**

DFlash is a novel speculative decoding method that utilizes a lightweight block diffusion model for drafting. It enables efficient, high-quality parallel drafting that pushes the limits of inference speed. The drafter component must be used in conjunction with a target model (e.g., Qwen3-8B, Qwen3-Coder-30B-A3B-Instruct). Despite being trained on significantly less data (289K samples vs 1.4M for EAGLE-3), DFlash already outperforms EAGLE-3 in inference acceleration. The repository provides pre-trained draft models for various Qwen3 variants and supports both standalone usage via transformers and production deployment via SGLang. Training recipes will be open-sourced to allow training custom DFlash draft models to accelerate any LLM.

**Available Models:**
- z-lab/Qwen3-8B-DFlash-b16
- z-lab/Qwen3-4B-DFlash-b16
- z-lab/Qwen3-Coder-30B-A3B-DFlash

**TPU Porting Task:** Adapt DFlash to run on TPU infrastructure, replacing flash-attention-2 with TPU-compatible attention implementation.

---

### 2. vLLM TPU Inference

**TPU inference for vLLM, with unified JAX and PyTorch support**

- **GitHub:** https://github.com/vllm-project/tpu-inference
- **Key File:** [speculative_decoding_manager.py](https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/runner/speculative_decoding_manager.py)
- **Contact:** vllm-tpu@google.com

**Description:**

vLLM TPU is now powered by tpu-inference, an expressive and powerful new hardware plugin unifying JAX and PyTorch under a single lowering path within the vLLM project. The new backend provides a framework for developers to: (1) push the limits of TPU hardware performance in open source, (2) provide more flexibility to JAX and PyTorch users by running PyTorch model definitions performantly on TPU without any additional code changes while also extending native support to JAX, and (3) retain vLLM standardization with the same user experience, telemetry, and interface. The repository contains speculative decoding implementations including EAGLE-3 support with verified performance for Llama 3.1-8B. Although vLLM TPU's unified backend makes out-of-the-box high performance serving possible with any supported model, some core components are still being implemented.

**Recent Features:**
- Async scheduler for improved performance on smaller models
- EAGLE-3 speculative decoding support
- Pipeline parallelism on JAX models
- Multi-host disaggregated serving
- FP8 quantized weights support for MoE layers

---

### 3. Prompt Lookup Decoding

**Speculative decoding via simple string matching in the prompt**

- **GitHub:** https://github.com/apoorvumang/prompt-lookup-decoding
- **Author:** Apoorv Saxena
- **Status:** Integrated into HuggingFace transformers library
- **Speedup:** 2x-4x on input-grounded tasks

**Description:**

Prompt Lookup Decoding modifies speculative decoding by replacing the draft model with simple string matching in the prompt to generate candidate token sequences. This results in significant speedups (2x-4x) in input-grounded tasks, with no effect on output quality. The method can be used with any decoder model without model changes or external datastore, and works with both greedy and sampling techniques. The intuition is that in several LLM use cases involving input-grounded generation (summarization, document QA, multi-turn chat, code editing), there is high n-gram overlap between LLM input (prompt) and LLM output—entity names, phrases, or code chunks that the LLM directly copies from the input. Prompt lookup exploits this pattern to speed up autoregressive decoding. The implementation swaps the "draft model" with a lookup function that tries to match the last few tokens to somewhere earlier in the prompt, returning the next-k token continuation as candidate sequence.

**Usage:** Simply add `prompt_lookup_num_tokens=10` to `model.generate(...)` calls in transformers.

**Key Parameters:**
- `max_ngram_size`: Maximum ngram to use when looking for matches in the prompt
- `num_pred_tokens`: Candidate sequence length to return after match is found

---

### 4. Lookahead Reasoning (GPU→TPU Reference)

**Scaling Speculative Decoding with Lookahead Reasoning**

- **GitHub:** https://github.com/hao-ai-lab/LookaheadReasoning
- **GPU→TPU Diff:** https://github.com/ConstBob/Lookahead-Reasoning/compare/main...tpu
- **Authors:** Yichao Fu, Rui Ge, Zelei Shao, Zhijie Deng, Hao Zhang (Hao AI Lab @ UCSD)
- **Venue:** NeurIPS 2025
- **arXiv:** [2506.19830](https://arxiv.org/abs/2506.19830)
- **Speedup:** 1.4x → 2.1x (combined with token-level SD)

**Description:**

Lookahead Reasoning (LR) is a novel method designed to accelerate the inference speed of large reasoning models (LRMs). While traditional token-level speculative decoding offers some speedup, it is bounded on complex, long-form tasks because the probability of correctly guessing a long, exact sequence of tokens is exponentially low. Lookahead Reasoning mitigates this limitation by elevating speculation from the token level to the more abstract 'reasoning step' level. The core insight is that reasoning models generate step-by-step, and each step needs only to be semantically correct, not exact token matching. In Lookahead Reasoning, a lightweight draft model proposes several future steps; the target model expands each proposal in one batched pass, and a verifier keeps semantically correct steps while letting the target regenerate any that fail. Token-level SD still operates within each reasoning step, so the two layers of parallelism multiply. Importantly, LR is orthogonal to token-level approaches and can be combined with them to achieve multiplicative speedups.

**TPU Porting Reference:** The GPU→TPU comparison branch shows modifications needed when adapting code from GPU to TPU implementations—useful as a reference for porting other speculative decoding methods.

---

### 5. vLLM Speculators

**Unified library for speculative decoding algorithms in vLLM**

- **GitHub:** https://github.com/vllm-project/speculators
- **Maintainer:** Red Hat / vLLM Project

**Description:**

Speculators is a unified library for building, evaluating, and storing speculative decoding algorithms for LLM inference in vLLM. The library provides: (1) offline training data generation using vLLM to enable generation of hidden states, with samples saved to disk for draft model training; (2) end-to-end training support of single and multi-layer draft models for both non-MoE and MoE models; (3) a standardized, extensible Hugging Face-compatible format for defining speculative models with tools to convert from external research repositories; and (4) seamless vLLM integration built for direct deployment with minimal overhead. Models trained through Speculators can run seamlessly in vLLM using a simple `vllm serve <speculator_model>` command.

---

## Documentation & Guides

- **JAX & Google CLI Guide:** https://married-spell-e7e.notion.site/JAX-GOOGLE-CLI-Guide-24df509095f180abbcf7ddc7ff0e9252
  - *Note: Free quota info at end of doc is outdated/expired*
- **gcloud CLI Cheatsheet:** https://cloud.google.com/sdk/docs/cheatsheet
- **Cloud TPU Documentation:** https://docs.cloud.google.com/tpu/docs
- **Attaching Durable Block Storage:** https://docs.cloud.google.com/tpu/docs/attach-durable-block-storage

## TPU Setup

### Hardware
- **Target:** TPU v5p (most powerful TPU model)
- **Free TPUs will be provided** for this project

### Example: Creating a 4-chip v4 Single Host

```bash
gcloud compute tpus queued-resources create lmgame-rl-queued \
    --node-id=lmgame-rl-node \
    --project=hao-ai-lab-trc \
    --zone=us-central2-b \
    --accelerator-type=v4-8 \
    --runtime-version=tpu-ubuntu2204-base
```

### Storage Notes
- TPU VM includes a 100 GiB boot disk
- Additional storage may be needed for some scenarios
- **Important:** Additional storage might not be free—delete when not needed

## Technical Context

### Speculative Decoding Methods
The TPU Inference repo contains multiple SD methods:
- **ngram:** Calls ngram implementation from vLLM repo
- **Diffusion-based:** Using diffusion models as draft models (showing 5-6x speedup on GPUs)

### GPU → TPU Porting Considerations
The Lookahead-Reasoning comparison branch shows modifications needed when adapting code from GPU to TPU implementations.

## Key Papers: Diffusion-Based Speculative Decoding

These papers represent the state-of-the-art in using diffusion models for speculative decoding. FailFast and SpecDiff-2 are the most likely candidates for what Prof. Zhang referenced (published shortly before the January 23rd conversation).

---

### 1. FailFast (December 2025) — Most Likely Candidate

**Fail Fast, Win Big: Rethinking the Drafting Strategy in Speculative Decoding via Diffusion LLMs**

- **arXiv:** [2512.20573](https://arxiv.org/abs/2512.20573)
- **Authors:** Rui Pan et al.
- **Published:** December 23, 2025
- **Code:** https://github.com/ruipeterpan/failfast
- **Speedup:** up to 4.9× over vanilla decoding, 1.7× over EAGLE-3

**Abstract:**

Diffusion Large Language Models (dLLMs) offer fast, parallel token generation, but their standalone use is plagued by an inherent efficiency-quality tradeoff. We show that, if carefully applied, the attributes of dLLMs can actually be a strength for drafters in speculative decoding with autoregressive (AR) verifiers. Our core insight is that dLLM's speed from parallel decoding drastically lowers the risk of costly rejections, providing a practical mechanism to effectively realize the (elusive) lengthy drafts that lead to large speedups with speculative decoding. We present FailFast, a dLLM-based speculative decoding framework that realizes this approach by dynamically adapting its speculation length. It "fails fast" by spending minimal compute in hard-to-speculate regions to shrink speculation latency and "wins big" by aggressively extending draft lengths in easier regions to reduce verification latency (in many cases, speculating and accepting 70 tokens at a time!). Without any fine-tuning, FailFast delivers lossless acceleration of AR LLMs and achieves up to 4.9× speedup over vanilla decoding, 1.7× over the best naive dLLM drafter, and 1.7× over EAGLE-3 across diverse models and workloads. We open-source FailFast at https://github.com/ruipeterpan/failfast.

---

### 2. SpecDiff-2 (November 2025)

**SpecDiff-2: Scaling Diffusion Drafter Alignment For Faster Speculative Decoding**

- **arXiv:** [2511.00606](https://arxiv.org/abs/2511.00606)
- **Authors:** Jameson Sandler, Jacob K. Christopher, Thomas Hartvigsen, Ferdinando Fioretto
- **Published:** November 1, 2025
- **Speedup:** up to 5.5× average over standard decoding

**Abstract:**

Speculative decoding has become the standard approach for accelerating Large Language Model (LLM) inference. It exploits a lossless draft-then-verify procedure to circumvent the latency of autoregressive decoding, achieving impressive speed-ups. Yet, current speculative decoding approaches remain limited by two fundamental bottlenecks: (1) the autoregressive dependency during drafting which limits parallelism, and (2) frequent rejections of draft tokens caused by misalignment between the draft and verify models. This paper proposes SpecDiff-2, a novel framework to jointly address these two bottlenecks. It leverages discrete diffusion as a non-autoregressive drafter to address bottleneck (1) and develops novel techniques to calibrate discrete diffusion drafters with autoregressive verifiers, addressing bottleneck (2). Experimental results across a comprehensive benchmark suite show that SpecDiff-2 achieves a new state-of-the-art across reasoning, coding, and mathematical benchmarks, improving tokens-per-second by up to an average of +55% over previous baselines and obtaining up to 5.5× average speed-up over standard decoding, without any loss of accuracy.

---

### 3. SpecDiff (August 2024) — Original Paper

**Speculative Diffusion Decoding: Accelerating Language Generation through Diffusion**

- **arXiv:** [2408.05636](https://arxiv.org/abs/2408.05636)
- **Authors:** Jacob K. Christopher, Brian R. Bartoldson, Bhavya Kailkhura, Ferdinando Fioretto
- **Published:** August 10, 2024
- **Venue:** NAACL 2025
- **Speedup:** up to 8.7× over standard generation, 2.5× over existing speculative decoding

**Abstract:**

Speculative decoding has emerged as a widely adopted method to accelerate large language model inference without sacrificing the quality of the model outputs. While this technique has facilitated notable speed improvements by enabling parallel sequence verification, its efficiency remains inherently limited by the reliance on incremental token generation in existing draft models. To overcome this limitation, this paper proposes an adaptation of speculative decoding which uses discrete diffusion models to generate draft sequences. This allows parallelization of both the drafting and verification steps, providing significant speedups to the inference process. Our proposed approach, Speculative Diffusion Decoding (SpecDiff), is validated on standard language generation benchmarks and empirically demonstrated to provide up to 8.7× speed-up over standard generation processes and up to 2.5× speed-up over existing speculative decoding approaches.

---

### 4. Spiffy (September 2025)

**Spiffy: Multiplying Diffusion LLM Acceleration via Lossless Speculative Decoding**

- **arXiv:** [2509.18085](https://arxiv.org/abs/2509.18085)
- **Authors:** Sudhanshu Agrawal, Risheek Garrepalli, Raghavv Goel, Mingu Lee, Christopher Lott, Fatih Porikli
- **Published:** September 22, 2025
- **Speedup:** 2.8-3.1× standalone, up to 7.9× when combined with other methods

**Abstract:**

Diffusion LLMs (dLLMs) have recently emerged as a powerful alternative to autoregressive LLMs (AR-LLMs) with the potential to operate at significantly higher token generation rates. However, currently available open-source dLLMs often generate at much lower rates, typically decoding only a single token at every denoising timestep in order to maximize output quality. We present Spiffy, a speculative decoding algorithm that accelerates dLLM inference by 2.8-3.1× while provably preserving the model's output distribution. This work addresses the unique challenges involved in applying ideas from speculative decoding of AR-LLMs to the dLLM setting. Spiffy proposes draft states by leveraging the dLLM's distribution itself in an auto-speculative manner. This approach is efficient and effective, and eliminates the overheads of training and running an independent draft model. To structure the candidate draft states, we propose a novel directed draft graph which is uniquely designed to take advantage of the bidirectional, block-wise nature of dLLM generation and can be verified in parallel by the dLLM. To further optimize the structure of these draft graphs, we introduce an efficient, offline calibration algorithm that procedurally determines high-quality graph configurations. These optimized draft graphs, enabling increased acceptance rates, lead to a significant boost in the overall speedup achieved by the system. Crucially, Spiffy is also complementary to other recent innovations in improving dLLM generation speeds such as KV-caching and multi-token unmasking. We demonstrate that when combined with such parallel decoding algorithms, Spiffy is able to effectively multiply the benefits of these methods leading to total speedups of up to 7.9×.

---

### 5. DiffuSpec (September 2025)

**DiffuSpec: Unlocking Diffusion Language Models for Speculative Decoding**

- **arXiv:** [2510.02358](https://arxiv.org/abs/2510.02358)
- **Authors:** Guanghao Li, Zhihui Fu, Min Fang, Qibin Zhao, Ming Tang, Chun Yuan, Jun Wang
- **Published:** September 28, 2025
- **Speedup:** up to 3× wall-clock speedup
- **Key Feature:** Training-free, drop-in framework

**Abstract:**

As large language models (LLMs) scale up, accuracy improves, but the autoregressive (AR) nature of decoding increases latency since each token requires a serial forward pass. Speculative decoding addresses this by employing a fast drafter to propose multi-token drafts, which are then verified in parallel by the target model. However, many deployments still rely on AR drafters, where sequential passes limit wall-clock gains. We revisit the drafting stage and present DiffuSpec, a training-free drop-in framework that uses a pretrained diffusion language model (DLM) to produce multi-token drafts in a single forward pass, while remaining compatible with standard AR verifiers. Because DLM drafts are generated under bidirectional conditioning, parallel per-position candidates form a token lattice in which the locally highest-probability token at each position need not form a causal left-to-right path. Moreover, DLM drafting requires pre-specifying a draft length, inducing a speed-quality trade-off. To address these challenges, we introduce two practical components: (i) a causal-consistency path search (CPS) over this lattice that extracts a left-to-right path aligned with AR verification; and (ii) an adaptive draft-length (ADL) controller that adjusts next proposal size based on recent acceptance feedback and realized generated length. Across benchmarks, DiffuSpec yields up to 3× wall-clock speedup, establishing diffusion-based drafting as a robust alternative to autoregressive drafters for speculative decoding.

---

### 6. DART (January 2026) — Published After Slack Thread

**DART: Diffusion-Inspired Speculative Decoding for Fast LLM Inference**

- **arXiv:** [2601.19278](https://arxiv.org/abs/2601.19278)
- **Authors:** Fuliang Liu, Xue Li, Ketai Zhao, Yinxi Gao, Ziyan Zhou, Zhonghui Zhang, Zhibin Wang, Wanchun Dou, Sheng Zhong, Chen Tian
- **Published:** January 27, 2026
- **Speedup:** 2.03-3.44× wall-clock speedup, surpasses EAGLE-3 by 30%

**Abstract:**

Speculative decoding is an effective and lossless approach for accelerating LLM inference. However, existing widely adopted model-based draft designs, such as EAGLE3, improve accuracy at the cost of multi-step autoregressive inference, resulting in high drafting latency and ultimately rendering the drafting stage itself a performance bottleneck. Inspired by diffusion-based large language models (dLLMs), we propose DART, which leverages parallel generation to reduce drafting latency. DART predicts logits for multiple future masked positions in parallel within a single forward pass based on hidden states of the target model, thereby eliminating autoregressive rollouts in the draft model while preserving a lightweight design. Based on these parallel logit predictions, we further introduce an efficient tree pruning algorithm that constructs high-quality draft token trees with N-gram-enforced semantic continuity. DART substantially reduces draft-stage overhead while preserving high draft accuracy, leading to significantly improved end-to-end decoding speed. Experimental results demonstrate that DART achieves a 2.03×–3.44× wall-clock time speedup across multiple datasets, surpassing EAGLE3 by 30% on average and offering a practical speculative decoding framework.

---

### Summary Table

| Paper | arXiv | Published | Speedup | Key Innovation |
|-------|-------|-----------|---------|----------------|
| **SpecDiff** | [2408.05636](https://arxiv.org/abs/2408.05636) | Aug 2024 | up to 8.7× | Original diffusion-based SD |
| **Spiffy** | [2509.18085](https://arxiv.org/abs/2509.18085) | Sep 2025 | 2.8-3.1× (7.9× combined) | Auto-speculative directed draft graphs |
| **DiffuSpec** | [2510.02358](https://arxiv.org/abs/2510.02358) | Sep 2025 | up to 3× | Training-free CPS + ADL |
| **SpecDiff-2** | [2511.00606](https://arxiv.org/abs/2511.00606) | Nov 2025 | up to 5.5× avg | Drafter-verifier calibration |
| **FailFast** | [2512.20573](https://arxiv.org/abs/2512.20573) | Dec 2025 | up to 4.9× | Dynamic speculation length |
| **DART** | [2601.19278](https://arxiv.org/abs/2601.19278) | Jan 2026 | 2.03-3.44× | N-gram tree pruning |

## Team & Communication

- **Prof. Hao Zhang** - Project lead
- **Yiming Zhao** - TPU resources and tutorials coordinator
- **Aaron's team** - Expressed interest in working on speculative decoding on TPU

---

*Last updated: February 2026*
*Source: Slack thread from Prof. Zhang's lab (January 23, 2025)*