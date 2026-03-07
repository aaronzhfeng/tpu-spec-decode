# Doc 53 — Block Size Scaling Theory: Why K Matters (and When It Doesn't)

**Status:** Reference document
**Date:** 2026-03-06

---

## 1. Core Concepts

### 1.1 What tau (τ) and K are

Both are token counts.

- **K** = number of draft tokens proposed per speculative decoding step (block size)
- **τ (tau)** = average number of draft tokens *accepted* per step

τ ≤ K always. The gap between them is waste — tokens drafted but thrown away.

### 1.2 Throughput

The throughput of speculative decoding is:

```
Throughput = τ(K) / T_step(K)
```

Where T_step(K) is the wall-clock time for one draft-then-verify cycle. If T_step is constant with K (the K-flat property on TPU), then throughput scales linearly with τ.

---

## 2. The Geometric Distribution of Acceptance

### 2.1 Why consecutive prefix matters

Speculative decoding accepts a **strict consecutive prefix**. The target model verifies all K draft tokens, but the moment one position is wrong, everything after it is discarded — even if later positions would have been correct.

Example with K=16, suppose position 5 is rejected:
```
Position:  0  1  2  3  4  5  6  7  8  ... 15
Draft:     A  B  C  D  E  X  G  H  I  ...  P
Target:    A  B  C  D  E  F  G  H  I  ...  Q
                          ^
                     first mismatch
Accepted:  A  B  C  D  E  [F from target]
Discarded:                    G  H  I  ...  P  (all thrown away)
```

Positions 6-15 are discarded regardless of correctness. Only 5 draft tokens + 1 bonus token from the target's prediction at the rejection point are kept.

### 2.2 The math

If each position has independent acceptance probability α (the per-position acceptance rate), then acceptance length follows a geometric distribution:

```
P(accept exactly n tokens) = α^n × (1 - α)    for n = 0, 1, 2, ...
```

The expected acceptance length (tau) with a block of K tokens:

```
τ(K) = Σ(n=0 to K-1) α^n = (1 - α^K) / (1 - α)
```

At infinite K, this converges to:

```
τ(∞) = 1 / (1 - α)
```

### 2.3 Numerical example at α = 0.86

The cumulative probability of still being in an accepted run at position n:

```
pos  0: α^0  = 1.000  (100% — always reaches here)
pos  1: α^1  = 0.860
pos  2: α^2  = 0.740
pos  3: α^3  = 0.636
pos  4: α^4  = 0.547
pos  5: α^5  = 0.470
pos  6: α^6  = 0.405
pos  7: α^7  = 0.348
pos  8: α^8  = 0.299
pos 10: α^10 = 0.221
pos 15: α^15 = 0.105
pos 20: α^20 = 0.049
pos 30: α^30 = 0.011
pos 63: α^63 = 0.00009
```

Expected accepted tokens:

| K | τ(K) | % of τ(∞) | Utilization (τ/K) |
|---|------|-----------|-------------------|
| 8 | 4.90 | 69% | 61% |
| 16 | 6.50 | 91% | 41% |
| 32 | 6.98 | 98% | 22% |
| 64 | 7.12 | 99.7% | 11% |
| 128 | 7.14 | 99.99% | 6% |
| ∞ | 7.14 | 100% | 0% |

**Key observation:** At K=16, you've already captured 91% of the infinite-K ceiling. Going from K=16 to K=128 adds 0.64 tokens per step — less than 1 extra token. Utilization drops from 41% to 6%.

### 2.4 Why α dominates K

The ceiling τ(∞) = 1/(1-α) is entirely determined by α:

| α | τ(∞) | τ(K=16) | Headroom from K=16 to K=∞ |
|---|------|---------|---------------------------|
| 0.80 | 5.00 | 4.68 | +0.32 (7%) |
| 0.86 | 7.14 | 6.50 | +0.64 (10%) |
| 0.90 | 10.00 | 8.15 | +1.85 (23%) |
| 0.95 | 20.00 | 12.58 | +7.42 (59%) |

Going from α=0.86 to α=0.90 at K=16 gains +1.65τ. Going from K=16 to K=∞ at α=0.86 gains +0.64τ. **Improving α is 2.6x more valuable than increasing K** (at these operating points).

At higher α, increasing K becomes more worthwhile because the geometric tail is fatter. But you need high α first for K to matter.

---

## 3. DFlash Draft Model Architecture

### 3.1 How the draft model uses target hidden states

The draft model does NOT see target predictions for the K tokens it's generating. It sees target representations of the **context** (already accepted tokens).

```
                    ┌─────────────────────────────────┐
                    │         Target Model             │
                    │  Forward pass on accepted seq    │
                    │  → hidden states at layer L      │
                    └──────────┬──────────────────────┘
                               │ project via combine_fn
                               ▼
                    ┌──────────────────────┐
                    │   Context K, V       │ ← target's view of accepted tokens
                    └──────────┬───────────┘
                               │
    ┌──────────────────────────┼──────────────────────────┐
    │                    Draft Model                       │
    │                                                      │
    │  Input: [next_tok, MASK, MASK, ..., MASK]  (K pos)  │
    │                                                      │
    │  Attention (non-causal):                             │
    │    Query  = K noise positions                        │
    │    Key    = context hidden states + K positions      │
    │    Value  = context hidden states + K positions      │
    │                                                      │
    │  Each position can attend to:                        │
    │    1. All context positions (target-conditioned)     │
    │    2. All other K positions (bidirectional)          │
    │                                                      │
    │  Output: K logits → argmax → K draft tokens         │
    └─────────────────────────────────────────────────────┘
```

**What the draft model sees:**
- Rich target-model representation of everything *before* the block (via projected hidden states as K/V)
- Other positions *within* the block (via non-causal self-attention) — but these start as MASK tokens

**What the draft model does NOT see:**
- Target model predictions for the K positions being generated
- Any iterative refinement (single pass, not multi-step diffusion at inference)

### 3.2 Why target conditioning changes the dynamics

In standalone block diffusion (BD3LM), the model generates tokens with no external guidance. Each position must independently figure out what token to produce, with only intra-block context from other noise positions.

In DFlash, the target hidden states provide a strong conditioning signal:
- The model knows *exactly* what came before the block (from the target's perspective)
- This anchors predictions, especially at early positions in the block

This is why DFlash Table 7 shows K=16 beating K=8 (+21% τ on Math500) — more positions means more bidirectional context within the block, and the target conditioning prevents the quality collapse seen in BD3LM.

---

## 4. BD3LM vs DFlash: Two Opposing Forces on Block Size

### 4.1 BD3LM: standalone block diffusion

BD3LM generates L' tokens per block with no target model. Quality degrades with block size:

| Block size (L') | Generative PPL |
|-----------------|---------------|
| 1 (autoregressive) | 8.9 |
| 4 | 25.7 |
| 8 | 28.2 |
| 16 | 33.4 |

This is expected: predicting 16 tokens simultaneously from noise is fundamentally harder than predicting 4. There's no external conditioning to anchor the predictions.

### 4.2 DFlash: target-conditioned speculative decoding

DFlash shows the opposite trend (at least from K=8 to K=16):

| Block size (K) | τ on Math500 |
|-----------------|-------------|
| 8 | 5.21 |
| 16 | 6.33 (+21%) |

**But these are measuring different things:**
- BD3LM PPL: how good are the generated tokens (all are kept)
- DFlash τ: how many consecutive tokens match the target (strict prefix)

### 4.3 The opposing forces

When you increase block size in DFlash, two forces compete:

| Force | Direction | Source |
|-------|-----------|--------|
| **Raw diffusion difficulty** | α decreases | Denoising more positions simultaneously is harder (BD3LM effect) |
| **Intra-block context** | α increases | Non-causal attention lets positions inform each other; more positions = richer context |
| **Target conditioning** | α stable | Context K/V from target model anchors predictions regardless of K |

From K=8→K=16, the context benefit won. The open question (B1) is: **does it keep winning at K=32, K=64, K=128?**

### 4.4 What the geometric distribution tells us about BD3LM

BD3LM has no acceptance/rejection — all generated tokens are kept. The geometric distribution does not directly apply.

However, per-position prediction quality does decay within a block. This is why DFlash uses **position-weighted loss** with exponential decay:

```
w_k = exp(-(k-1) / γ)
```

Where γ scales with block size (K=8→γ=4, K=16→γ=7). Later positions are down-weighted during training because they're inherently harder.

The analogy:
- **Speculative decoding:** per-position decay → geometric acceptance → first failure kills the rest (harsh)
- **Standalone diffusion:** per-position decay → later tokens are worse quality → PPL increases (gradual)

Same underlying cause (parallel prediction is harder at later positions), different consequences.

---

## 5. The K-Flat Property and Why It Changes the Calculus

### 5.1 On GPU: bigger K has a cost

GPU verification scales with K:
- K=128 costs 1.24x of K=16 (full forward pass, V7 measurement)
- Draft cost also scales: 1.22x at K=128 vs K=16

So on GPU, throughput at larger K:

```
Throughput(K) = τ(K) / T_step(K)

K=16:  6.50 / T         = 6.50/T
K=64:  7.12 / (1.15×T)  ≈ 6.19/T    ← WORSE than K=16
K=128: 7.14 / (1.24×T)  ≈ 5.76/T    ← even worse
```

At fixed α=0.86, bigger K is a net **loss** on GPU. You need α to improve to break even.

### 5.2 On TPU: bigger K is free

TPU verification is K-flat (0.97–1.02x from K=16 to K=128, confirmed through K=1024):

```
K=16:  6.50 / T         = 6.50/T
K=64:  7.12 / (1.00×T)  = 7.12/T    ← +10% free
K=128: 7.14 / (1.00×T)  = 7.14/T    ← +10% free
```

On TPU, bigger K can never hurt. The worst case is a small free gain from the geometric tail. The best case (if α improves) is a large gain.

### 5.3 The 2x2 intersection

|  | GPU Verify (1.24x at K=128) | TPU Verify (0.97x at K=128) |
|--|---|---|
| **AR Draft (8x at K=128)** | Both scale. Not viable. | Draft scales. Not viable. |
| **Diffusion Draft (1.22x GPU / 1.00x TPU)** | Net loss at fixed α. Needs α↑. | **Free gain. Any α↑ is bonus.** |

Only diffusion + TPU makes wide blocks risk-free.

---

## 6. The Critical Open Question (B1)

**Does per-position acceptance rate (α) improve with wider blocks in a target-conditioned setting?**

Evidence for (α increases):
- DFlash Table 7: K=16 beats K=8 by 21% on Math500
- More intra-block positions → richer bidirectional context → better coherence
- Target conditioning anchors predictions, preventing BD3LM-style collapse

Evidence against (α decreases or flat):
- BD3LM: standalone diffusion quality degrades with block size
- Noise-to-noise attention may not help — positions start as MASK tokens
- Training difficulty increases with block size (optimization challenge)

**Why this is the only question that matters:**
- If α stays at 0.86: τ ceiling is 7.14 regardless of K. K=16 captures 91%. Wider blocks give +10% on TPU (free), net loss on GPU.
- If α improves to 0.90 at K=64: τ(64) = 8.65, a 33% gain over τ(16)=6.50. Significant.
- If α degrades to 0.82 at K=64: τ(64) = 5.43, worse than τ(16)=6.50. Wider blocks hurt even on TPU.

The experiment: train K=32 and K=64 DFlash models, measure per-position α, and determine which force wins.

---

## 7. Summary

| Concept | Key Takeaway |
|---------|-------------|
| τ and K | Same unit (tokens). τ ≤ K. Gap is waste. |
| Geometric distribution | Consecutive prefix acceptance. One failure kills everything after it. |
| τ ceiling | At fixed α, τ saturates around K=32-64. K>64 adds <1 token. |
| α vs K | Improving α is 2-3x more valuable than increasing K. |
| Target conditioning | Draft model sees target hidden states for context, not for generated tokens. |
| BD3LM vs DFlash | Same per-position decay, different consequences (PPL vs prefix rejection). |
| K-flat on TPU | Bigger K is free on TPU, costly on GPU. |
| B1 | Does wider K improve α? Only question that matters for wide-block research. |
