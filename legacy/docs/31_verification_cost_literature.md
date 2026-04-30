# Doc 31: Verification Cost Reduction — Literature Deep-Read & Experiment Assessment

## Context

Our ablation study (Doc 29) showed the target verification forward pass is 59%
of step time (~10ms out of 17ms). The drafter is only 2% (0.36ms). This inverts
the field's optimization focus: most work improves drafters, but the real
bottleneck is verification.

We deep-read the three most relevant papers that explicitly target verification
cost: TriSpec, Sparse Verification, and SpecEE. All are from 2025 — this is
an emerging area.

---

## Paper 1: TriSpec (arXiv 2601.23180)

### Mechanism

Three-model architecture: drafter → proxy verifier → target model.

- **Proxy**: A same-family smaller model (e.g., Qwen3-1.7B for Qwen3-32B target)
- **Routing criterion**: Margin-based — `top1_prob - top2_prob >= lambda` (default 0.5)
- **Case I** (proxy confident at rejection point): Proxy handles verification
  alone. Target model is never called. Bonus token sampled from proxy.
- **Case II** (proxy uncertain): Proxy-verified prefix accepted, remaining
  tokens sent to target for authoritative verification.
- **Adapter**: 1-layer MLP maps proxy features into drafter's expected feature
  space so the drafter can run using proxy features in Case I rounds.

### Results (A100 GPU)

- 25-35% additional speedup over EAGLE-3 alone
- Target invocation reduced by ~50% (e.g., 21% → 10% on MATH500)
- Accuracy degradation: <1% on most benchmarks, up to 1.6% worst case

### Statistical Guarantee

**Breaks exact distribution matching.** In Case I rounds, the bonus token is
sampled from the proxy's distribution, not the target's. Accepted tokens are
verified against the proxy only. This is a lossy method — output matches the
proxy distribution, not the target distribution.

### Gaps Relative to Our Work

| Gap | Detail |
|-----|--------|
| **TPU** | Zero consideration. Three-model orchestration + dynamic routing is hostile to XLA static compilation. |
| **Diffusion drafters** | Entirely autoregressive. Mechanism depends on per-token probability distributions from AR drafting. Would need fundamental redesign for parallel diffusion drafts. |
| **Hardware cost profile** | No hardware-specific cost breakdown. Assumes proxy is cheap relative to target, but this needs validation on TPU where smaller models may have worse MXU utilization. |
| **Same-family requirement** | Requires a smaller model from the same family (82% token-level exact match). Constrains model choices. |

---

## Paper 2: Sparse Verification (arXiv 2512.21911)

### Mechanism

Three-dimensional sparsification of the verification forward pass:

1. **Sparse Attention**: Evict low-importance KV blocks based on query-key dot
   products from the first draft token. Top-N blocks retained, rest pruned.
   Inter-layer reuse: adjacent layers with similar block selections share masks.
   Only activates above sequence length threshold L₀ (default 4K tokens).

2. **Sparse FFN**: Prune channels where gate activation `|h_i| < tau`. At
   tau=0.1, ~64% of channels pruned. FLOPs reduction ~43%.

3. **Sparse MoE**: Skip low-weight experts. For DeepSeek-R1 (8 experts/token),
   skip up to 3 experts per token based on calibration-derived thresholds.

All three applied simultaneously in "hybrid" mode.

### Results (H800 GPU)

- **No wall-clock speedup reported.** Only FLOPs analysis. This is the biggest
  red flag — sparse gather/scatter can be slower than dense matmuls on hardware
  optimized for dense compute.
- Acceptance length: negligible reduction (2.73 → 2.72)
- Quality: task-dependent. CollegeMath drops 58→53 (5 points). ROUGE drops 0-3.5.

### Statistical Guarantee

**Breaks exact distribution matching.** Sparse verification produces approximate
target logits — `p_θ_sparse(x|ctx) ≠ p_θ(x|ctx)`. The acceptance criterion
compares against this approximation, not the true target distribution.

### Gaps Relative to Our Work

| Gap | Detail |
|-----|--------|
| **No wall-clock speedup** | Only theoretical FLOPs. Sparse operations may be slower on hardware designed for dense compute. |
| **TPU** | Zero consideration. Irregular sparse patterns (gather/scatter, dynamic block selection) are the opposite of what MXU wants. TPU requires regular, tiled memory access. |
| **Attention sparsity irrelevant at short context** | SA only activates above L₀=4K tokens. Our experiments use contexts well below this. Only SFFN would apply. |
| **Diffusion drafters** | Untested. The "first token drives block selection" heuristic has no natural analog for parallel diffusion drafts. |
| **Calibration dependency** | Inter-layer reuse and MoE thresholds need calibration data. Distribution sensitivity not analyzed. |

---

## Paper 3: SpecEE (arXiv 2504.08850)

### Mechanism

Per-layer early exit during the target model's verification forward pass:

1. **Features**: At each layer, compute 3 features per speculative token:
   - Dot product of hidden state with LM head columns for speculative tokens only
   - Local softmax probabilities across speculative tokens
   - Probability variation vs previous layer (the "probability shift" signal)

2. **Predictor**: 2-layer MLP (hidden dim 512) outputs exit probability.
   Exit if output ≥ 0.5.

3. **Safety check**: On exit, compute full LM head logits and verify argmax
   is among speculative tokens. If not, continue to next layer.

4. **Scheduling**: Only ~10 of 32 predictors are active per token (offline
   profiling + online circular queue of recent exit layers).

### Results (A100, RTX 4090)

- Standalone early exit: 1.27x speedup over HuggingFace baseline
- **Combined with speculative decoding: only 1.05-1.06x additional speedup**
- Average layers: 32 → ~23 (28% reduction)
- Perplexity: MT-Bench PPL degrades 6.49 → 8.44 (+30%)
- Predictor overhead: ~5.6% of inference time

### Statistical Guarantee

**Breaks exact distribution matching.** Exiting at layer 26 instead of 36
means the output token is determined by an intermediate representation, not the
final one. The paper's safety check only verifies that the intermediate-layer
argmax is a speculative token — it does not verify this matches the final-layer
argmax.

**KV cache problem**: Exiting early leaves layers 27-36 without KV entries.
All future tokens attend to incomplete KV caches, compounding quality loss.

### Gaps Relative to Our Work

| Gap | Detail |
|-----|--------|
| **Marginal with spec decode** | Their own numbers: only 1.05-1.06x over EAGLE alone. Barely worth the complexity. |
| **TPU/XLA** | Dynamic per-token per-layer exit decisions are hostile to XLA static compilation. Would require compiling both branches or padding. |
| **Diffusion drafters** | Untested. "Probability shift" phenomenon measured with EAGLE tokens only. Diffusion-generated tokens may have different statistical properties. |
| **Quality degradation** | 30% PPL increase on generation tasks. KV cache incompleteness affects all future tokens. |
| **Small model tested** | Only Llama2-7B/13B/70B. No analysis of models with different layer dynamics. |

---

## Cross-Paper Comparison

| Dimension | TriSpec | Sparse Verify | SpecEE |
|-----------|---------|---------------|--------|
| Preserves exact distribution | No | No | No |
| Tested on TPU | No | No | No |
| Works with diffusion drafters | No | No | No |
| Wall-clock speedup measured | Yes (GPU) | **No** | Yes (GPU) |
| Speedup over spec decode | +25-35% | Unknown | +5-6% |
| Requires extra model/training | Yes (proxy + adapter) | No (inference-time) | Yes (MLP predictors) |
| Quality loss | <1% accuracy | 0-5 pts task-dependent | 30% PPL on generation |
| Memory overhead | High (3 models) | None | Low (~0.9GB) |

---

## The Open Question

**Can verification cost be reduced while preserving the exact output distribution
matching guarantee?**

All three papers break this guarantee. None attempt to preserve it. The
standard speculative decoding proof (Leviathan et al., Chen et al.) requires
the verification forward pass to produce *exact* target logits for the
acceptance/rejection sampling to be distribution-preserving. Any modification
to the verification pass (sparse, early-exit, proxy) produces approximate
logits, breaking the guarantee.

This creates two possible research directions:
1. **Prove it's impossible**: Show a lower bound that exact-distribution
   verification requires the full target forward pass. This would formalize
   what all three papers implicitly assume.
2. **Find a loophole**: Identify conditions under which partial verification
   is still distribution-preserving. E.g., if layers 27-36 provably don't
   change the argmax for certain token classes, early exit at layer 26 would
   be exact for those tokens.

Either result would be a significant theoretical contribution.

---

## Experiment Assessment: What Can We Run on Our Current Setup?

### What We Have

- TPU v4-8 pod (4 chips, 8 cores)
- Qwen3-4B target model (36 layers, hidden_size=2560, vocab=151936)
- DFlash diffusion drafter (4 layers, block_size=16)
- Working standalone benchmark with per-component timing
- Docker-based experiment infrastructure
- All benchmarks in `benchmarks/`, wrappers in `tests/`

### Feasible Experiments

#### Experiment 1: Layer-Truncated Verification (HIGH FEASIBILITY)

**What**: Run verification through only the first N layers (N = 12, 18, 24,
30, 36) instead of all 36. Measure acceptance rate (tau) and step time at
each truncation point.

**Why**: This is the simplest possible test of "how much of the target model
do you actually need for verification?" No trained predictor needed — just
truncate the forward pass.

**How**: Modify the verify forward call to return hidden states after N layers
instead of 36. Compute logits from the truncated hidden state. Run the standard
acceptance logic. Measure tau and wall-clock time.

**Expected outcome**: A curve showing tau vs number of layers. If tau remains
high (e.g., >5.0) at N=24, that means 12 layers (33% of verification compute)
are unnecessary. If tau collapses immediately, verification truly needs all
layers.

**Infrastructure needed**: Minimal — modify `target_model_fn` call or wrap it.
No new models, no training.

#### Experiment 2: Per-Layer Logit Stability Profiling (HIGH FEASIBILITY)

**What**: During verification, extract the hidden state at every layer and
compute logits. Record where the argmax token stabilizes (stops changing
across layers).

**Why**: This directly tests SpecEE's "probability shift" hypothesis on our
specific model (Qwen3-4B) and hardware (TPU). If the argmax stabilizes at
layer 24, that's evidence early exit at 24 would be lossless for that token.

**How**: Run the full 36-layer verification but hook each layer to extract
hidden states. Compute `jnp.argmax(logits_fn(hidden_state_at_layer_n))` for
n = 1..36. Record the first layer where the argmax matches the final argmax
and never changes afterward.

**Expected outcome**: A distribution of "stabilization layers" across tokens.
If most tokens stabilize by layer 24-28, there's headroom. If they don't
stabilize until layer 34-36, early exit is unviable.

**Infrastructure needed**: Modify the model forward pass to return intermediate
hidden states. This requires reading how the Qwen3 model loops over layers and
adding hooks.

#### Experiment 3: DFlash-as-Proxy Verification (MEDIUM FEASIBILITY)

**What**: Use the DFlash draft model's output or the target model's early
layers as a cheap proxy verifier (TriSpec-inspired, but without a separate
proxy model).

**Why**: We don't have a separate same-family smaller model, but we do have
the DFlash draft model which already processes the block and produces hidden
states. Can its hidden states predict which tokens the target would accept?

**How**: After DFlash produces draft tokens, compute a confidence score
(e.g., entropy of draft logits at each position). Compare positions where
DFlash is confident vs uncertain against the target model's accept/reject
decisions. If high-confidence DFlash positions are always accepted, they
could skip target verification.

**Expected outcome**: Correlation between DFlash confidence and target
acceptance. If strong, this opens the door to selective verification
(skip target for high-confidence positions).

**Infrastructure needed**: Extract draft logits (already computed in our
pipeline). Add entropy computation. Correlate with acceptance.

#### Experiment 4: Exact-Distribution Early Exit via Caching (LOW FEASIBILITY)

**What**: Attempt to make early exit distribution-preserving by caching the
full target logits from the previous step and using them as a reference for
tokens that haven't changed.

**Why**: If a token was accepted in the previous step, and the context hasn't
changed much, the full-layer logits for that token position might be very
close to the previous value. Could we reuse cached logits for accepted
positions and only fully verify new/rejected positions?

**How**: Would need to track per-position logit staleness and validate that
cached logits remain accurate.

**Why low feasibility**: Requires careful theoretical analysis to ensure the
distribution guarantee holds. Probably needs multiple experiments to validate.

### Not Feasible Without External Resources

- **TriSpec-style proxy verification**: Requires a same-family smaller model
  (e.g., Qwen3-0.6B). We don't have one loaded, and loading a 3rd model would
  strain TPU memory.
- **Sparse verification on TPU**: MXU fundamentally opposes irregular sparse
  patterns. Would likely be slower, not faster.
- **Trained early-exit predictors**: Requires generating training data and
  training MLP predictors. Feasible in principle but significant engineering.

### Recommended Order

1. **Experiment 1** (layer-truncated verification) — cheapest to implement,
   highest information value. Answers "is there headroom?" directly.
2. **Experiment 2** (per-layer logit stability) — slightly more engineering,
   explains the mechanism behind any headroom found.
3. **Experiment 3** (DFlash-as-proxy) — novel angle unique to our setup
   (diffusion drafter confidence as verification routing signal).

If Experiment 1 shows tau remains high at N<36, Experiments 2 and 3 become
the foundation for a paper. If tau collapses immediately, we learn that full
verification is necessary and pivot to the theoretical direction (prove a lower
bound).

---

*Created: February 20, 2026*
*Status: Literature analysis complete. Experiments 1-3 ready to implement.*
