# Speculative Decoding and DFlash

This document is the conceptual foundation for the project: it explains the sequential bottleneck in autoregressive inference, speculative decoding, draft models, and DFlash. It then summarizes how DFlash is implemented in the reference repo and how vLLM and TPU-inference relate to our integration work.

---

## 1. The sequential bottleneck in autoregressive inference

Modern LLM deployments operate at massive scale, where even a single sequential component in the inference pipeline can dominate end-to-end latency and degrade user experience. While LLM inference is often described as inherently sequential, this is only partially true. In practice, inference consists of two distinct stages with very different computational characteristics: prefilling, which can be efficiently parallelized, and decoding, which introduces unavoidable sequential dependencies.

### 1.1 Prefilling

In the prefilling stage, the model processes the input prompt to initialize its internal state for decoding by computing the KV cache for the entire prompt in a single forward pass.

There is no prediction at this stage: all tokens in the prompt are known in advance. As a result, the model does not have a sequential runtime dependency on previously generated tokens, even though positional causality is still enforced via the causal attention mask.

Because the model does not need to wait for the next token to be generated, this stage can be fully parallelized across the prompt length (through batched forward pass), similar to how sequences are processed in parallel during training.

### 1.2 Decoding

In the decoding stage, the model begins generating new tokens autoregressively. At each step, the model predicts the next token conditioned on the most recent tokens retained in the model's context window, appends this token to the sequence, and repeats the process until termination.

Unlike prefilling, decoding has an inherent sequential dependency: the prediction at step *t*+1 depends on the token generated at step *t*. As a result, vanilla decoding requires one forward pass per generated token. Although each forward pass can reuse the KV cache from earlier steps, the overall process cannot be parallelized across time. Consequently, inference latency grows linearly with the number of output tokens.

This strict sequential nature makes decoding the primary bottleneck in LLM inference, especially for long generations or low-latency applications. Since full parallelization is fundamentally incompatible with autoregressive decoding, most inference acceleration techniques focus on relaxing this dependency in controlled ways. One such approach is speculative decoding, which attempts to introduce limited parallelism while preserving the model's autoregressive semantics. We discuss this technique in the next section.

---

## 2. Speculative decoding

Speculative decoding introduces parallelism into the decoding stage by using a fast, computationally cheap model to propose a sequence of tokens to the main model. The main model then verifies these tokens and produces the final generated output. Below, we explain how this mechanism relaxes the strict sequential nature of decoding.

Speculative decoding introduces the following terms:

- **Draft model:** the fast, cheap model proposing tokens (draft tokens) to the main model.
- **Target model:** the main, computationally expensive autoregressive model.

The mechanism works as follows. During inference, after prefilling, suppose the model has generated *t* tokens. The draft model generates a sequence of *n* draft tokens and proposes them to the target model. The target model then verifies this sequence and outputs at least one token (if the first draft token is rejected) and at most *n* tokens (if the entire draft sequence is accepted).

To verify the proposed sequence, the target model computes the probability distribution over the vocabulary at positions *t*+1, *t*+2, …, *t*+*n* in a single forward pass. At each position *t*+*i*, the draft token *t*+*i* is verified by comparing its probability under the target model's distribution with its probability under the draft model's distribution using a rejection-sampling rule. Verification proceeds sequentially from *t*+1 onward and stops at the first rejected draft token. If a rejection occurs at position *t*+*k*, the target model discards all remaining draft tokens and samples a token directly from its own distribution at that position.

It is important to note that although the target model still evaluates all *n* positions in the proposed sequence, the past–future dependency has been relaxed. In contrast to vanilla decoding, where each token must be sampled before the next forward pass can begin, speculative decoding verifies a known sequence of tokens. This allows the target model's computation to be parallelized across the *n* positions in one forward pass, in a pattern similar to prefilling.

In the worst case, where the first draft token is rejected, speculative decoding degenerates to standard decoding with negligible additional cost beyond running the draft model. In typical cases, where the target model accepts several draft tokens before rejection, multiple tokens are generated per target-model forward pass, reducing average decoding latency while preserving the exact output distribution.

---

## 3. Draft model

As introduced above, the draft model is a separate, fast, and computationally cheap model that proposes tokens, while the target model remains the single source of truth. The core mechanism of proposing and verifying tokens to relax strict autoregressive decoding is fundamentally the same across speculative decoding techniques. These techniques are primarily distinguished by the choice of draft model, or equivalently, by how draft tokens are generated.

The TPU Inference repository currently supports ngram and eagle3 draft models. The ngram draft model is non-neural, while eagle3 is a small, one-layer transformer that reuses the target model's context, such as hidden states and the embedding of the last generated token. As we can see, a draft model does not need to belong to any specific family of probabilistic models; it can range from a simple heuristic model to a learned neural model. However, each choice of draft model comes with its own trade-offs.

For ngram, the computation cost is very low and token proposal can be efficiently parallelized. However, it provides a poor approximation of the target model's token distribution, leading to low acceptance rates during verification and, consequently, limited latency speedup. Eagle3, in contrast, produces a much better approximation of the target model's distribution, resulting in higher acceptance rates. Nevertheless, it remains an autoregressive model and relies on sequential sampling to generate draft tokens, though at a lower cost than the target model. This increases the time spent in the proposal phase in exchange for improved acceptance during verification.

In the next section, which is also the main focus of our work, we examine a diffusion-based draft model. Unlike eagle3, this approach parallelizes the draft token proposal process while remaining expressive enough to achieve higher acceptance rates than simple n-gram methods.

---

## 4. DFlash (conceptual)

DFlash is a speculative decoding method that uses a diffusion-style draft model to propose a block of tokens in a single forward pass, rather than generating draft tokens one by one as in autoregressive draft models.

The key distinction lies in how draft tokens are generated. In DFlash, the draft model operates on a block of future positions simultaneously. Instead of predicting the next token conditioned on the previously predicted draft token, the model treats the entire block (matrix of token embeddings corresponding to multiple future positions) as a structured object to be inferred at once.

At a high level, DFlash works as follows. Given the current decoding state, the target model provides contextual information, such as hidden states from the last generated token. The draft model then takes this context together with a block of placeholder token embeddings, typically represented as masks or noise. In a single forward pass, the draft model predicts logits for all positions in the block, from which draft token IDs are sampled. These draft tokens are subsequently verified by the target model using the standard speculative decoding acceptance rule.

This approach is referred to as "diffusion-style" because the block of tokens is modeled as something to be denoised or jointly inferred given context, rather than generated sequentially. Within the block, there is no autoregressive dependency between positions. All positions are predicted in parallel, conditioned on the same contextual signal from the target model.

From a systems perspective, this difference has important implications for time complexity. For an autoregressive draft model, proposing a sequence of *k* draft tokens requires *O*(*k*) sequential forward passes, since each token depends on the previously generated draft token. In contrast, DFlash proposes a block of *k* draft tokens in *O*(1) draft-model forward passes with respect to sequence length, as all positions in the block are processed simultaneously. As a result, DFlash removes the sequential bottleneck in the drafting stage by introducing an additional layer of parallelism on top of the parallelism already enabled by target-model verification in speculative decoding.

This design is particularly well suited to TPU architectures. TPUs are optimized for large, dense, data-parallel matrix operations and benefit from computation patterns that avoid fine-grained sequential control flow. Diffusion-style draft models naturally align with this execution model, as they emphasize parallel prediction over blocks rather than token-by-token generation. This observation motivates our focus on integrating DFlash into the TPU inference stack, which will be discussed in detail in the following docs.

**Note.** DFlash is not the only speculative decoding approach that employs a diffusion-based draft model. Prior work has explored using either large or small diffusion models to propose draft tokens. However, these approaches have been shown to be impractical in practice. Large diffusion models incur a substantial memory footprint that outweighs the latency benefits gained from parallelism, while small diffusion models tend to produce draft tokens that poorly align with the target model's distribution, resulting in low acceptance rates.

DFlash aims to achieve both fast drafting and high acceptance by using a lightweight diffusion draft model that is explicitly conditioned on contextual features extracted from the target model. Rather than generating tokens from scratch, the draft model leverages the target model's intermediate representations to guide block prediction. This design effectively transfers part of the target model's reasoning capability to the draft model while retaining the parallelism advantages of diffusion-based generation. For additional details, we refer readers to the original DFlash blog post: https://z-lab.ai/projects/dflash/

---

## 5. DFlash implementation overview (in the DFlash repo)

The reference implementation of DFlash lives in the [z-lab/dflash](https://github.com/z-lab/dflash) repository. At a high level, the draft model is not a full language model: it has no embedding layer and no language-model head of its own. It consists only of the "middle" transformer stack—a small number of decoder layers with a custom attention pattern—and it reuses the **target model's** embedding and LM head for both input and output. This keeps the draft model lightweight while ensuring that draft token IDs and logits share the same vocabulary and representation space as the target.

**Input side: embeddings.** The block of positions to be predicted is first represented as token IDs (including a mask or placeholder for unknown future positions). These IDs are passed through the **target model's** embedding layer (`embed_tokens`) to produce a block of token embeddings, often referred to as the "noise" or initial block embedding. No separate draft embedding layer is used. In addition, the draft receives **target hidden states**: the target model is run on the current context, and hidden states from a subset of its layers (selected by a fixed or configurable set of layer indices) are extracted, concatenated, and projected down to a single hidden dimension. This projected vector is the contextual signal that conditions the draft on what the target "thinks" at the current position.

**Draft forward pass.** The draft model's own parameters are limited to (i) a linear layer that projects the concatenated target hidden states to the draft's hidden size, (ii) a stack of DFlash decoder layers, and (iii) layer normalization. Each DFlash layer uses a custom attention mechanism: queries come only from the block (the positions being predicted), while keys and values come from both the target context and the block. Crucially, attention is **non-causal** over the block, so all block positions can attend to each other and to the context in one shot. This allows the entire block to be updated in a single forward pass. The output of the draft stack is a block of hidden states in the same shape as the target model's hidden size.

**Output side: logits and sampling.** The block of hidden states produced by the draft is fed into the **target model's** LM head to obtain logits over the vocabulary. Draft token IDs are then sampled from these logits (e.g., via temperature sampling). Thus, the draft model never has its own vocabulary projection; it always uses the target's. This guarantees that draft proposals and target verification speak the same language and avoids any need to align two different heads.

**Summary.** The DFlash draft model in the reference repo is a thin, context-conditioned block predictor: it takes target hidden states plus a block of target-embedded token vectors, runs them through a small non-causal transformer stack, and then uses the target's LM head to produce draft logits and tokens. All embedding and output semantics are inherited from the target; only the middle "reasoning" over the block is implemented by the draft. When we integrate DFlash into the TPU inference stack, we reimplement this same contract (context + block → draft hidden → target LM head → draft tokens) in JAX/Flax within the tpu-inference codebase, using the reference repo only for behavioral and numerical alignment.

---

## 6. vLLM and TPU-inference

**vLLM** is a high-throughput, production-oriented open-source inference server for large language models. It provides a unified API and execution engine that handles batching, memory management (e.g., PagedAttention), and scheduling. By default, vLLM runs on GPU backends (CUDA). To run on Google TPUs, a separate backend layer is required that replaces or adapts GPU-specific components (kernels, memory layout, execution flow) for the TPU.

**TPU-inference** (the [vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference) repository) is that backend. It plugs into vLLM as the TPU platform and execution path. At a high level, it preserves vLLM's user-facing API and request/scheduling model while swapping the underlying implementation: model execution, attention, and other hotspots are implemented or lowered to JAX and run on TPU. The repository supports multiple model implementation types (e.g., JAX/Flax-native vs. PyTorch/vLLM models executed on TPU via TorchAX), single-host and multi-host setups, and speculative decoding. Out of the box, it ships with support for ngram and EAGLE-3 as draft models; the runner and speculative decoding manager are designed so that additional methods can be added alongside these.

**Relevance to our project.** Our goal is to add DFlash as a new speculative decoding method on TPU. The right place to do that is inside the **tpu-inference** repo: we implement the DFlash draft model and proposer in JAX/Flax, wire a new method (e.g., `"dflash"`) into the speculative decoding manager and runner, and keep the DFlash reference repo (z-lab/dflash) as a read-only source for behavior and baselines. In this way, DFlash runs in the same vLLM-driven, TPU-inference execution environment as EAGLE-3 and ngram, reusing the same target model, KV cache, and verification logic. The conceptual and implementation notes in the previous sections (sections 1–5) describe the algorithm and the reference implementation; the rest of the docs in this folder describe the TPU setup, contribution principles, and the tpu-inference codebase where that integration lives.
