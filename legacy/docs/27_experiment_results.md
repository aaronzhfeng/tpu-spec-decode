# Doc 27: Experiment Results — Pipeline Profiling & Iterative Refinement

## Summary

Two experiments were run on TPU v4 (4 chips) with Qwen3-4B target and
DFlash-b16 draft model, gsm8k dataset, 4 samples (1 warmup + 3 measured),
256 max output tokens, greedy decoding.

**Direction 1 (Pipeline Profiling):** Core compute is only 17% of step time.
82.8% is overhead, dominated by logits-to-token conversion (argmax).

**Direction 2 (Iterative Refinement):** Refinement hurts draft quality.
Tau drops from 6.18 to ~2.5 because the model was trained with mask tokens,
not predicted tokens, at draft positions.

---

## Direction 1: Pipeline Profiling

### Per-Phase Timing Breakdown

| Phase | Mean (ms) | Median (ms) | Std (ms) | % of Step |
|---|---|---|---|---|
| acceptance | 8.28 | 8.28 | 0.07 | 46.9% |
| verify_forward | 2.56 | 2.51 | 0.43 | 14.5% |
| draft_sample | 2.41 | 2.34 | 0.40 | 13.6% |
| aux_projection | 1.86 | 0.92 | 7.09 | 10.5% |
| host_device_xfer | 1.79 | 1.67 | 1.40 | 10.2% |
| draft_forward | 0.47 | 0.46 | 0.07 | 2.7% |
| ctx_update | 0.27 | 0.27 | 0.06 | 1.5% |
| cache_mgmt | 0.00 | 0.00 | 0.00 | 0.0% |
| **TOTAL** | **17.64** | **16.49** | **7.22** | **100%** |

### Key Findings

1. **Core compute (draft_forward + verify_forward) = 3.03 ms (17.2%)**
   - Draft forward: 0.47 ms — extremely fast, the 4-layer DFlash model is cheap
   - Verify forward: 2.56 ms — the 36-layer Qwen3-4B target model, processing 16 tokens

2. **Logits-to-token conversion = ~10.7 ms (60.5%)**
   - `acceptance` (8.28 ms): `target_logits_fn` on verify hidden states + `jnp.argmax` + numpy comparison
   - `draft_sample` (2.41 ms): `target_logits_fn` on draft hidden states + `jnp.argmax`
   - Both call `target_logits_fn` which computes `hidden_states @ lm_head.T` (a large matrix multiply against the full vocab)

3. **Host-device overhead = 1.79 ms (10.2%)**
   - Building noise block arrays and uploading context features to device
   - Includes `jnp.array()` calls for small tensors

4. **Auxiliary projection = 1.86 ms (10.5%)**
   - `jnp.concatenate(aux_hidden_states)` + `draft_combine_fn` (FC + RMSNorm)
   - High variance (std=7.09) suggests first-call JIT compilation spikes

### Implications for Direction 1 Research

The biggest optimization target is **not** the model forward passes — it's
the **logits computation**. `target_logits_fn` is called twice per step
(once for draft sampling, once for verification), and each call performs a
`(T, hidden_size) @ (hidden_size, vocab_size)` matmul. For Qwen3-4B:
- hidden_size = 2560, vocab_size = 151,936
- This is a (16, 2560) @ (2560, 151936) matmul = 6.2 billion FLOPs per call

Potential optimizations:
- **Fuse draft_sample into draft_forward**: Run logits + argmax as part of the
  draft model's XLA graph, avoiding a separate `tpu_sync()` + host roundtrip
- **Fuse acceptance into verify_forward**: Compute verify logits and acceptance
  length in one XLA graph, returning only the scalar acceptance count
- **Top-k logits for acceptance**: Instead of full vocab argmax, compute only
  the top-1 token index via a partial sort (Pallas kernel or custom XLA op)
- **XLA while_loop**: Eliminate all host-device sync points by running the
  entire draft→verify→accept cycle in a `jax.lax.while_loop`

### Overhead Budget for an XLA-Fused Loop

If we could fuse the entire step into one XLA graph:
- Keep: draft_forward (0.47) + verify_forward (2.56) = 3.03 ms
- Eliminate or reduce: everything else (14.61 ms)
- Best-case step time: ~3-5 ms (vs current 17.64 ms)
- That would yield ~3.5-5.9x improvement in step latency
- Combined with tau=6.18, this could push TPS well above the current 362.9

---

## Direction 2: Iterative Refinement

### Comparison Table

| k | tau | TPOT (ms) | TPS | Speedup vs AR | Draft (ms) | Refine (ms) | Verify (ms) |
|---|---|---|---|---|---|---|---|
| AR baseline | — | 10.83 | 92.4 | 1.00x | — | — | — |
| **0 (standard)** | **6.18** | **2.76** | **362.9** | **3.93x** | 2.71 | 0 | 10.52 |
| 1 | 2.48 | 7.95 | 125.8 | 1.36x | 2.71 | 2.67 | 10.48 |
| 2 | 2.57 | 8.89 | 112.5 | 1.22x | 2.74 | 5.36 | 10.37 |
| 3 | 2.51 | 10.27 | 97.3 | 1.05x | 2.65 | 7.89 | 10.33 |

### Per-Position Acceptance Rates

| Position | k=0 | k=1 | k=2 | k=3 |
|---|---|---|---|---|
| 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| 1 | 0.886 | 0.542 | 0.584 | 0.550 |
| 2 | 0.754 | 0.352 | 0.401 | 0.371 |
| 3 | 0.640 | 0.236 | 0.274 | 0.268 |
| 4 | 0.509 | 0.148 | 0.164 | 0.161 |
| 5 | 0.456 | 0.099 | 0.080 | 0.082 |

### Key Finding: Refinement Destroys Draft Quality

Refinement does not improve draft quality — it **degrades** it catastrophically.
Tau drops from 6.18 (k=0) to ~2.5 (k=1,2,3), a 60% reduction.

The per-position rates show the damage clearly: position 1 acceptance drops
from 88.6% to 54.2% with just one refinement step. Later positions are
even worse (position 4: 50.9% → 14.8%).

### Root Cause Analysis

The DFlash draft model was **trained** to predict from a specific input pattern:
`[known_token, mask, mask, mask, ..., mask]`. The noise positions (1-15) are
mask tokens (token ID 0), and the model has learned to produce good predictions
conditioned on this specific input distribution.

When we replace masks with predicted tokens for refinement:
`[known_token, predicted_1, predicted_2, ..., predicted_14]`

...we are feeding the model an **out-of-distribution** input. The model has
never seen predicted tokens at positions 1-14 during training. The token
embeddings for real tokens vs mask tokens are very different, and the model's
attention patterns and internal representations are calibrated for the
mask-token input pattern.

This is analogous to the "exposure bias" problem in sequence models: the model
behaves differently on its own outputs than on its training distribution.

### Why k=2 Is Slightly Better Than k=1

The small tau improvement from k=1 (2.48) to k=2 (2.57) is likely noise, but
could also be because the second refinement pass gives the model a slightly
more consistent input (k=1's predictions are better than masks in some
positions, providing a halfway-useful signal).

### Implications

**Naive inference-time refinement does not work** for DFlash. For iterative
refinement to help, the draft model would need to be:

1. **Trained with a denoising schedule**: Expose the model to partially-filled
   blocks during training, not just mask-filled blocks. This would teach the
   model to refine predictions iteratively.

2. **Architecturally modified**: Add a "refinement mode" where the model
   receives previous predictions as conditioning, possibly with a learned
   mixing parameter between mask embeddings and predicted token embeddings.

3. **Multi-step aware**: The training objective could include a multi-step
   denoising loss, where the model is trained to improve predictions from
   step k to step k+1.

These modifications would make iterative refinement a **training-time**
research contribution rather than a pure inference-time technique. This is
a deeper but potentially higher-impact direction.

---

## Timing Observations

### Draft forward is cheap: 0.47 ms (profiling) / 2.71 ms (refinement)

The discrepancy comes from different measurement methodology:
- Profiling: `tpu_sync()` before and after the draft forward call only
- Refinement: includes `target_logits_fn` + `argmax` in the draft timing

The pure draft model forward pass (4 layers of DFlash attention + MLP) is
only ~0.5 ms on TPU v4. This confirms that the draft model is extremely
cheap, and the latency budget for refinement steps (~2.7 ms each including
logits) is well understood.

### Verify forward is stable: ~10.4 ms across all k values

The target model's 16-token verify pass takes ~10.4 ms regardless of
refinement. This is expected — refinement only changes the draft tokens,
not the verification workload.

### Refinement steps are constant-cost: ~2.65 ms each

Each additional refinement step adds a consistent ~2.65 ms (draft forward +
logits + argmax). This matches the initial draft cost, confirming that
refinement passes are not cheaper than the initial pass despite context
K/V being cached.

---

## Updated Research Strategy

### Direction 1: Pipeline Overhead (PROMISING — pursue aggressively)

The profiling data shows a clear and large optimization target. 82.8% of
step time is overhead, and much of it can be eliminated by fusing
operations into XLA graphs. The path forward:

1. **Fuse logits + argmax**: Eliminate the two separate `target_logits_fn`
   calls by incorporating logits computation into the model forward graphs
2. **XLA while_loop prototype**: Build a proof-of-concept fused decode loop
3. **Quantify the improvement**: Compare fused vs current step latency

### Direction 2: Iterative Refinement (NEGATIVE result — pivot needed)

Naive inference-time refinement doesn't work because the model is
out-of-distribution. Options:

- **Pivot to trained refinement**: Train a DFlash variant with denoising
  schedule (multi-step noise → clean) to enable true iterative refinement.
  This is a training contribution, not just inference.
- **Pivot to partial refinement**: Instead of replacing all masks, only
  replace high-confidence positions and keep low-confidence ones as masks.
  This stays closer to the training distribution.
- **Document the negative result**: The finding that single-step denoising
  is already optimal for the current DFlash architecture is valuable and
  publishable on its own.

---

## Data Files

- `results/profiling_gsm8k.json` — Direction 1 per-step timing data
- `results/refinement_gsm8k.json` — Direction 2 comparison data

---

*Created: February 19, 2026*
*Status: Experiments complete, results documented*
