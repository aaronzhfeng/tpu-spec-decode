# Doc 36: TPU-Native Drafter — Synthesis and Research Direction

## Context

Docs 32–35 document four experiments run after the ablation study (Doc 29):

| Doc | Experiment | Finding |
|-----|-----------|---------|
| 32 | Layer truncation | Verification needs all 36 layers. Even -1 layer loses 30% τ. |
| 33 | Amortized verification (microbenchmark + multi-block) | Verify(128) ≈ Verify(16) on TPU. Multi-block speculation fails — draft overhead dominates. |
| 34 | TPU vs GPU uniqueness | Flat scaling is MXU-specific. GPU verification scales linearly (SmartSpec, ICLR 2025). |
| 35 | Tree speculation | Drafting K branches costs K× more draft time for negligible τ gain. Not viable. |

This document synthesizes what those four experiments mean together, provides the pipeline-level view that explains why every single-request exploitation attempt has failed, and identifies the one remaining direction that is both novel and feasible: a TPU-native DFlash variant trained with block_size=128.

---

## 1. The Pipeline View — Caps at Every Stage

One speculative decoding step produces (τ + 1) tokens. The total step time breaks down as:

```
┌──────────────────┬──────────────────────────────┬──────────────────┐
│   DRAFT ~4ms     │   VERIFY ~10ms               │  ACCEPT ~3ms     │
│   (24%)          │   (59%)                      │  (18%)           │
└──────────────────┴──────────────────────────────┴──────────────────┘
     ↑                      ↑                             ↑
  already                 THE WALL                   host overhead
  negligible                                          near floor
```

**Draft stage (~4ms total):**

| Sub-cost | Time | Hard floor |
|---|---|---|
| DFlash forward (5 layers, K=16) | 0.36ms | MXU tile floor — flat to K=128 |
| Aux projection (target hidden → draft KV) | ~2ms | Memory bandwidth, roughly fixed |
| Host overhead, KV cache write | ~1.6ms | Device-host transfer latency |

The 0.36ms forward pass is already negligible. The ~3.6ms of overhead is the real draft cost and does not scale with K for K≤128 — it is fixed per step.

**Verify stage (~10ms total):**

| Sub-cost | Time | Hard floor |
|---|---|---|
| FFN component (36 layers × matmuls) | ~1.72ms | MXU tile floor — flat to K=128 (measured, Doc 33) |
| Attention over KV cache | ~8.3ms | Memory bandwidth reading KV cache, grows with context length |

The ~8.3ms attention component scales as O(K × N_context). This is the wall that does not move: it reads the full KV cache regardless of drafter choices.

**Accept/output stage (~3ms):** Rejection sampling, device→host transfer, correction token sampling. Already near its floor.

**The τ ceiling:**

```
τ(K) = (1 - α^K) / (1 - α)     [expected accepted tokens, iid model]

Measured: τ(K=16) = 6.67  →  backs out α ≈ 0.87 per position
Ceiling (K→∞): τ(∞) = 1/(1-α) ≈ 7.69
```

At K=16, DFlash is already at 87% of the theoretical iid ceiling. The remaining τ headroom is modest even with larger K — unless wider blocks improve α itself.

**The throughput equation:**

```
              τ + 1          7.67
Throughput = ───────── = ──────────── ≈ 548 tok/s
              T_step          14ms
```

Every single-request attempt to improve throughput has hit the same constraint: step time does not move. Verification is dominant and cannot be reduced (Doc 32). Adding more draft tokens requires more drafting work (Docs 33, 35), which increases T_draft proportionally.

---

## 2. Why Every Single-Request Exploitation Failed

All three post-ablation attempts tried to exploit MXU amortization within a single request:

| Attempt | Idea | Why it failed |
|---|---|---|
| Multi-block (Doc 33) | Draft 2×16 tokens, verify 32 | Draft overhead +4ms, verify is flat but draft is not |
| Layer truncation (Doc 32) | Verify fewer layers | τ collapses — not a hardware problem |
| Tree speculation (Doc 35) | Draft K branches, pick best | Each branch costs ~5ms in draft setup, τ barely improves |

The pattern: MXU amortization benefits verification, not drafting. Any attempt that adds drafting work to put more tokens in the verification window negates the gain. The hardware win only materializes if you can fill the 128-token verification window without adding proportional drafting cost.

---

## 3. The One Remaining Single-Request Path

**Key observation:** The MXU tile argument applies equally to the drafter's own forward pass.

DFlash uses K=16, meaning its FFN layers process a 16×hidden query matrix — padding 16 to 128, wasting 87.5% of the tile. For K=128, the query matrix is exactly 128×hidden — one tile, zero waste. The drafter's forward pass at K=128 costs the same ~0.36ms as K=16 because the tile is the atomic unit.

Both sides of the step are flat for K≤128:
- Draft forward (K=128): ~0.36ms (same as K=16, MXU tile)
- Verify forward (K=128): same ~1.72ms FFN floor, same ~8.3ms attention (measured, Doc 33)
- Draft overhead: ~3.6ms (fixed per step regardless of K)

Therefore: step time is approximately constant for any K≤128. Any τ gain from larger K is pure throughput improvement.

**Expected τ improvement under iid model (α=0.87):**

| K | τ(K) | vs K=16 |
|---|------|---------|
| 16 | 6.67 | baseline |
| 32 | 7.40 | +11% |
| 64 | 7.65 | +15% |
| 128 | 7.69 | +15% |

The gain saturates quickly. K=32 captures most of it; K=128 gives marginally more.

**If wider blocks improve α:**

DFlash conditions on target hidden states injected into every draft layer. Wider blocks mean each position attends over 127 neighboring draft positions (vs 15 for K=16), giving richer parallel within-block context. If this improves α from 0.87 to 0.90:

| K | τ (α=0.87) | τ (α=0.90) |
|---|------------|------------|
| 16 | 6.67 | 7.59 |
| 64 | 7.65 | 9.44 |
| 128 | 7.69 | 9.99 |

At K=128 with modest α improvement, τ could exceed EAGLE-3's τ=7.07 GPU ceiling. Whether α actually improves with wider blocks is the core empirical unknown.

---

## 4. Why DFlash Uses K=16 (A GPU Constraint That Does Not Apply on TPU)

DFlash was designed and trained on H200 GPUs. The paper's own ablation (Table 7) shows larger K improves τ significantly:

| Train K | Test K | Math500 τ | Speedup |
|---------|--------|-----------|---------|
| 16 | 16 | 6.33 | 4.64× |
| 8 | 8 | 5.21 | 3.97× |

Larger K is clearly better for τ. But the paper adds this caveat:

> "In practical serving scenarios, large blocks can increase verification cost under compute-bound settings (e.g., large batch sizes); reducing the block size in such cases can therefore yield better overall speedup. We leave adaptive block-size scheduling to future work."

This caveat is GPU-specific. On GPU, verifying K=128 tokens costs ~8× more than verifying K=16 (SmartSpec's linear scaling). The paper is correct for its hardware: large K hurts at GPU serving batch sizes because verification grows proportionally.

On TPU, this constraint does not exist. Verification is flat for K≤128 (measured, Doc 33). The paper's entire motivation for keeping K small evaporates on TPU hardware. The optimal K on TPU is determined purely by τ quality, with no verification cost penalty up to K=128.

**The research novelty:** DFlash's block size is implicitly a GPU design parameter. The paper's own caveat about large block sizes is the exact constraint our MXU amortization finding eliminates on TPU. Retraining for K=128 on TPU is not a scaled-up version of DFlash — it is DFlash redesigned for the hardware's actual cost function.

---

## 5. Training Feasibility

**Model size:** DFlash trains only the 5 draft transformer layers. The embedding and LM head are shared with Qwen3-4B (frozen). Trainable parameters: roughly 100–300M.

**Original training setup (from DFlash paper, Section A.1):**
- 800K samples, 3072 token sequences, 6 epochs
- 512 anchor positions per sequence (each = start of one K-token masked block)
- AdamW lr=6×10⁻⁴, cosine schedule, warmup ratio 0.04
- Offline training: target hidden states precomputed and cached

**Changes needed for K=128:**
1. `block_size: 16 → 128`
2. Anchor count per sequence: reduce from 512 to ~48-60 (3072/128 = 24 non-overlapping max)
3. Loss decay γ: retune from 7 to a larger value (more positions before tail-end loss shrinks)
4. All other hyperparameters: same

**Data reuse:** The 800K training samples and their cached Qwen3-4B hidden states can be fully reused. Only block construction changes.

**Compute estimate on TPU v4-8:**

```
Trainable params: ~200M (estimate)
Total tokens:     800K × 3072 × 6 epochs ≈ 14.7B token-equivalents

K=16 FLOPs  ≈ 6 × 200M × 14.7B ≈ 17.6 ExaFLOPs
             → ~14 hours on TPU v4-8 at ~350 TFLOPS effective

K=128 multiplier (FFN-dominated, ~8–10× per step):
             → 1–4 days on TPU v4-8
```

**Engineering constraint:** The original DFlash training code is PyTorch (trained on H200 GPUs). The current infrastructure is JAX/TPU. Options:
- PyTorch/XLA on TPU (setup required, works)
- Port training loop to JAX (larger effort, cleaner long-term)
- Single A100/H100 GPU via institutional access: run existing PyTorch code as-is with `block_size=128`

The code change is minimal. The infrastructure path is the main decision.

---

## 6. Summary: What We've Ruled Out and What Remains

**Ruled out empirically (Docs 32–35):**
- Layer truncation for cheaper verification — not viable for Qwen3-4B
- Multi-block speculation for higher τ — draft overhead kills it
- Tree speculation for higher τ — draft overhead kills it
- Any single-request MXU exploitation without retraining — all paths blocked

**What remains viable:**

1. **Cross-request verification batching (serving, no retraining):** Verify B requests' K tokens together in one target forward pass. T_verify is shared across B requests. Potential ~2–4× throughput at batch=8. Systems engineering contribution — no model changes needed.

2. **TPU-native wide-block drafter (K=128, requires retraining):** Both draft AND verify forward passes cost the same as K=16 on TPU due to MXU tile structure. Expected τ gain: +15% minimum (iid model), potentially much more if wider blocks improve per-position acceptance. Estimated training time: 1–4 days on TPU v4-8 (or equivalent GPU). The single ML research contribution available given current constraints.

**The open empirical question for #2:**
Does α (per-position acceptance probability) remain stable, improve, or degrade as K grows from 16 to 128? This is what a trained K=32, K=64, K=128 sweep would measure. The iid assumption is conservative; the true answer determines whether this is a 15% win or a potentially larger one.

---

## 7. How τ Translates to Inference Speed

This section makes the throughput math concrete, since τ is the lever that the wide-block drafter directly moves.

### What α is

α is the **per-position acceptance probability** — the probability that any single draft token at a given position passes verification by the target model.

At each position i in the draft block, the acceptance rule from the speculative sampling theorem (Chen et al., 2023) is:

```
Accept token x̃ᵢ with probability  min(1,  p(x̃ᵢ) / q(x̃ᵢ))
```

where p is the target model's probability for that token and q is the drafter's probability. If the drafter was confident and correct, p/q ≈ 1 and acceptance is near-certain. If the drafter predicted something the target model would rarely say, p/q is small and rejection is likely.

α is the expectation of this ratio across all positions and all generation contexts. It is a single number summarizing draft quality.

### How α compounds into τ

Speculative decoding accepts a **prefix** of the draft block — it stops at the first rejection. The acceptance chain works as follows:

```
Draft:  [ t₁   t₂   t₃   t₄   t₅  ...  t₁₆ ]
Check:    ✓    ✓    ✓    ✗   stop
                          ↑
               First rejection — everything after discarded
```

The probability of accepting exactly k tokens is α^k × (1-α) (geometric distribution). The expected total accepted tokens τ is the sum of this series over k=0 to K:

```
τ(K) = (1 - α^K) / (1 - α)
```

For α=0.87 and K=16:  τ = (1 - 0.87^16) / 0.13 ≈ 6.67  (measured, matches experiments)
For α=0.87, K→∞:      τ = 1/(1-α) = 1/0.13 ≈ 7.69       (theoretical ceiling)

At K=16, DFlash is already at 6.67/7.69 = **87% of the iid ceiling** for its current α. The remaining 13% is available by increasing K, but cannot be exceeded without improving α itself.

### How τ translates directly to throughput

Step time is approximately constant at ~14ms for any K≤128 (MXU tile property). Therefore:

```
              τ + 1          tokens generated per step
Throughput = ───────── = ─────────────────────────────
              T_step         fixed cost per step
```

| Scenario | τ | Throughput | vs current |
|---|---|---|---|
| Current (K=16, α=0.87) | 6.67 | 548 tok/s | baseline |
| K=128, α=0.87 (iid ceiling) | 7.69 | 621 tok/s | +13% |
| K=128, α=0.90 (improved) | 9.99 | 785 tok/s | +43% |
| K=128, α=0.93 (optimistic) | 13.29 | 1,021 tok/s | +86% |

### How τ translates to end-to-end latency

The more intuitive way to see the gain: each step costs a fixed ~17ms (at full generation context). Higher τ means fewer steps needed to complete a response.

For a 500-token response:

```
Steps needed = 500 / τ

Current (τ=6.67):    75 steps × 17ms = 1,275ms
K=128, τ=7.69:       65 steps × 17ms = 1,105ms   (-13%)
K=128, τ=9.99:       50 steps × 17ms =   850ms   (-33%)
K=128, τ=13.29:      38 steps × 17ms =   646ms   (-49%)
```

The 10ms verification wall is a **per-step tax**. Higher τ amortizes this tax across more output tokens. With τ=6.67 you pay 10ms to get ~6.67 tokens. With τ=10 you pay the same 10ms to get ~10 tokens — 50% more output for the same verification cost.

### Why α might improve with wider blocks

The current DFlash (K=16) uses non-causal attention: each of the 16 draft positions can attend to all 15 others simultaneously when predicting. With K=128, each position attends to 127 neighbors.

The analogy: bidirectional language models (BERT) consistently outperform left-to-right models on masked token prediction because seeing both sides of context reduces uncertainty. DFlash's within-block non-causal attention is exactly this — each position sees its neighbors to resolve ambiguity. More neighbors (K=128 vs K=16) means more context per prediction, which could increase α above 0.87.

The key difference from multi-block or tree speculation: in those cases, you add drafting work proportional to extra tokens. With K=128 trained end-to-end, you spend the same 0.36ms forward pass to predict 128 tokens that are each informed by 127 neighbors. No extra cost; potentially better per-token quality.

### The threshold that makes this compelling

The iid model gives a conservative lower bound. If K=128 training merely preserves α=0.87 (no improvement), throughput gains +13%. If it improves α to 0.90 (plausible given richer within-block context), throughput gains +43% and total latency for a 500-token response drops by 33%. These are single-request latency improvements with no serving infrastructure changes needed.

---

*Created: February 20, 2026*
*Status: Synthesis and research direction — next step is training K=32/64/128 DFlash variants*
*Builds on: Docs 32–35 (experiments), Doc 34 (TPU uniqueness), proposal_v1.md (Contribution 2)*
