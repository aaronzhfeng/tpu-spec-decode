# Doc 30: Insights from the DFlash TPU Port — What We Actually Learned

## Context

Docs 25-29 covered the execution of two near-term research directions:
- Direction 1 (pipeline gap): The standalone loop has ~2ms overhead out of 17ms. Near-optimal.
- Direction 2 (iterative refinement): Negative result. DFlash is OOD when fed predicted tokens instead of mask tokens. Requires retraining.

This document steps back from optimization and records the broader insights
that emerged from the full DFlash TPU porting and experimentation effort
(Docs 00-29). These insights are not about making DFlash faster — they're
about what DFlash reveals about the intersection of diffusion models,
autoregressive generation, and hardware architecture.

---

## Insight 1: Diffusion-Style Parallel Prediction Works Far Better Than Expected

DFlash predicts 15 tokens in a single forward pass and achieves tau=6.67 —
meaning on average ~7 of those 15 predictions are correct (verified by the
target model). This is remarkable because:

- The draft model has **no autoregressive conditioning** between positions.
  Position 8's prediction cannot see what position 7 predicted. It relies
  entirely on the target model's hidden states and non-causal self-attention
  within the block.

- Autoregressive drafters like EAGLE-3 achieve tau~7.07 on the same task,
  using sequential conditioning where each token sees all previous tokens.
  DFlash gets to 94% of that quality **without sequential dependence**.

- This challenges the common assumption that good next-token prediction
  requires left-to-right conditioning. For speculative decoding (where the
  target model provides rich hidden-state context), parallel prediction is
  nearly as good as sequential prediction.

**Open question:** Is 94% a fundamental ceiling for parallel prediction given
target hidden states, or is it an artifact of DFlash's specific architecture
(4 layers, trained with mask tokens)? Could a better-designed parallel
predictor close the remaining 6% gap?

---

## Insight 2: The Target Model Is the Bottleneck, Not the Drafter

Our ablation study (Doc 29) showed the time budget breakdown:

```
Verify forward (36 layers):  ~10ms  (59%)
Aux projection:               ~2ms  (12%)
Draft forward + LM head:      ~2ms  (12%)
Verify LM head:               ~1ms  ( 6%)
Host orchestration:            ~2ms  (12%)
```

The verify forward pass — running the full target model on the drafted
block — dominates wall-clock time. The draft model is already cheap (0.36ms
forward pass). Making drafting cheaper, faster, or more accurate has
diminishing returns because the target verification cost is fixed.

This inverts the usual research focus. Most speculative decoding papers
optimize the drafter (better acceptance, faster drafting). But the real
leverage is in **making verification cheaper**:

- Can we verify with fewer layers? (Early-exit verification)
- Can we verify a subset of positions? (Selective verification)
- Can we amortize verification cost across multiple draft blocks?
- Can we use a cheaper model for verification of "easy" tokens?

**Open question:** Is there a verification scheme that preserves the
statistical guarantee of speculative decoding (output distribution matches
the target model exactly) while using less compute than a full target forward
pass?

---

## Insight 3: TPU Hardware Naturally Favors Parallel-Block Prediction

TPU's MXU operates on 128x128 bf16 tiles. During standard autoregressive
decode, the query matrix is (1, hidden_dim) — a single row. The MXU
processes this as a degenerate case, wasting >99% of its tile capacity.

DFlash's block prediction presents (16, hidden_dim) queries — 16 rows.
This is still small for the MXU but 16x better utilized than single-token
decode. The draft forward pass takes only 0.36ms for 4 transformer layers
processing 16 tokens simultaneously.

More broadly: **any technique that converts single-token operations into
block operations gets a natural speedup on TPU**, because the hardware
is designed for large matrix multiplies that single-token decode cannot
provide.

This applies beyond speculative decoding:
- Batched verification (processing multiple requests' draft blocks together)
- Prefill-heavy workloads (already large matrices)
- Multi-query decoding patterns

**Open question:** What is the optimal block size for TPU v4? Our experiments
used block_size=16 (from DFlash's default). Is there a sweet spot where MXU
utilization, acceptance rate, and verification cost balance optimally? This
likely differs between TPU generations (v4 vs v5e vs v6e) due to different
MXU sizes and memory bandwidth.

---

## Insight 4: The Standalone-to-Production Gap Is the Real Systems Problem

Our most striking measurement: tau=6.67 standalone vs tau=4.48 in the vLLM
pipeline. A **33% quality degradation** from infrastructure overhead alone.

This gap comes from:
- vLLM's scheduler treating spec decode as an afterthought
- Batch management designed for vanilla autoregressive decode
- Rejection sampling loop overhead in `speculative_decoding_manager.py`
- Shape dynamism forcing recompilation

This is not a DFlash-specific problem. **Every speculative decoding method
will lose 30-50% of its standalone performance when deployed in production
serving frameworks.** Yet every paper reports standalone numbers.

This suggests a significant research opportunity: designing serving
frameworks that are spec-decode-native rather than bolting speculative
decoding onto frameworks designed for vanilla autoregressive inference.

**Open question:** What would a serving framework look like if speculative
decoding were the primary execution model rather than an add-on? How would
scheduling, memory management, and batching change?

---

## Insight 5: Stateful Drafters Expose Pipeline Bugs That Stateless Drafters Hide

The seq_len inflation bug (Doc 21) was invisible for months because EAGLE-3
(stateless drafter) silently tolerated it. DFlash (stateful — persistent
context buffer, KV cache across iterations) broke catastrophically. The bug
corrupted context buffer content, KV cache positions, and RoPE embeddings
simultaneously.

This reveals a reliability dimension that the field hasn't addressed:
**stateful draft models are canaries in the pipeline coal mine.** They fail
loudly when the pipeline has subtle bugs, which is arguably a feature.

More practically: as speculative decoding moves toward more sophisticated
drafters (tree-based drafting, multi-model ensembles, persistent state),
pipeline correctness becomes critical. The current approach of testing with
"does the output look reasonable?" is insufficient.

**Open question:** Can we design correctness invariants for speculative
decoding pipelines? E.g., formal checks that the drafter's state is
consistent with the accepted tokens, independent of output quality?

---

## Insight 6: The LM Head Is Surprisingly Cheap on TPU

The LM head matmul — (batch, 2560) @ (2560, 151936) — takes only 0.73ms on
TPU v4. This is a 389M element computation completing in under a millisecond.
Batch size 1 through 32 shows identical latency because the MXU tile size
(128x128) makes the hidden dimension (2560 = 20 tiles) the dominant factor,
not the batch dimension.

This means vocabulary projection is essentially free on TPU relative to the
transformer forward pass. Techniques that trade LM head compute for quality
(top-k vocabulary projection, approximate softmax, vocabulary pruning) are
**not useful on TPU** — they solve a problem that doesn't exist on this
hardware.

This is hardware-specific insight. On GPU, the LM head can be a meaningful
fraction of decode time, making approximate vocabulary projection worthwhile.
On TPU, the MXU makes the full vocabulary projection so fast that
approximations add complexity without saving time.

**Open question:** Does this generalize to larger models? For a 70B parameter
model with hidden_size=8192 and vocab_size=128k, is the LM head still
negligible on TPU?

---

## Insight 7: JAX's Lazy Evaluation Already Provides Most Fusion Benefits

Our JIT fusion experiment (Doc 28) showed that fusing draft+logits+argmax
and verify+logits+argmax into single compiled functions produced **negligible
improvement** (~1ms per step, within noise). The unfused loop already
achieves near-optimal pipelining because:

- JAX dispatches operations asynchronously (lazy evaluation)
- Operations overlap naturally — host dispatches the next op while the
  previous one runs on device
- The only mandatory sync points are when values must cross the device-host
  boundary (e.g., `np.array(jax_tensor)` to read acceptance results)

This means **manual operation fusion is largely unnecessary in JAX/TPU
workloads**. The XLA compiler + JAX's async dispatch already achieve what
manual CUDA kernel fusion achieves on GPU. The optimization effort should
focus on eliminating host-device sync points, not on fusing device-side
operations.

**Open question:** Does this hold for multi-device (TPU pod) workloads where
cross-device communication introduces additional sync points?

---

## Summary of Where Leverage Exists

| Area | Leverage | Why |
|---|---|---|
| Drafter speed | Low | Draft forward is 0.36ms — already negligible |
| Drafter quality | Low-Medium | tau=6.67 is 94% of GPU; marginal gains remain |
| LM head optimization | None on TPU | 0.73ms matmul, hardware makes it free |
| Manual op fusion | None | JAX lazy eval already pipelines |
| Verification cost | **High** | 59% of step time, no published work on reducing it |
| Production pipeline | **High** | 33% quality loss from infrastructure |
| Hardware-aware design | **Medium** | MXU tile alignment, block size optimization |
| Serving framework design | **High** | No spec-decode-native serving framework exists |

The highest-leverage research targets are verification cost reduction and
spec-decode-native serving — both are **underexplored in the literature** and
both emerged directly from our empirical DFlash TPU work.

---

*Created: February 20, 2026*
*Status: Synthesis of Docs 00-29 findings — basis for next-phase research brainstorming*


 What we've proven empirically (Docs 27-29):
  - Direction 1 (pipeline gap): The standalone loop is already near-optimal (~2ms overhead out of 17ms). The real
   gap is in vLLM orchestration (tau 6.67→4.48).
  - Direction 2 (iterative refinement): Negative result. DFlash is trained with mask tokens — feeding predicted
  tokens back is out-of-distribution. Would require retraining with a denoising schedule.

  What remains unexplored — the broader questions you're pointing at:

  - Direction 3 (TPU-native draft architecture): Nobody has designed a draft model for TPU. Current DFlash has
  GPU-shaped hidden dimensions that waste MXU tiles. This requires training, not just inference optimization.
  - Direction 4 (Hybrid adaptive drafting): Switching between cheap (n-gram) and expensive (diffusion) strategies
   based on difficulty. The auxiliary hidden states DFlash already extracts could serve as the difficulty signal.
  - Direction 5 (Disaggregated spec decode on TPU pods): Break the serial draft→verify constraint by pipelining
  across chips via ICI.

  But honestly, those were incremental extensions. I think what you're getting at is bigger than "how to make
  DFlash faster" — it's more like: what does the DFlash experience tell us about the intersection of diffusion
  and autoregressive generation? The broader implications we observed but haven't formalized:

  1. Diffusion-style parallel prediction works surprisingly well as a drafter — tau=6.67 means ~7 tokens
  predicted correctly per step, which challenges the assumption that good token prediction requires
  autoregressive conditioning.
  2. TPU hardware naturally favors parallel-block prediction over sequential decode — MXU utilization is terrible
   for single-token AR decode but great for block prediction. This is a hardware-architecture alignment that
  nobody has explored systematically.
  3. The real bottleneck is the target model, not the drafter — our ablation showed verify forward is 59% of step
   time. This suggests research should focus on making verification cheaper, not drafting better.
