# Doc 35: Tree Speculation — Experiment Results

## Objective

Test whether top-K tree speculation can exploit MXU amortization to
increase DFlash throughput. Draft K candidate blocks with different
first tokens, verify each, take the best.

## Method

For each step:
1. Draft primary block (16 tokens) — standard DFlash
2. Get top-K first-token candidates from previous verification logits
3. For each alternative, draft a separate block via DFlash
4. Verify each block independently (separate target forward passes)
5. Take the block with longest acceptance
6. Report actual and theoretical amortized throughput

**Setup**: Qwen3-4B + DFlash, TPU v4-8, GSM8K, 3 samples, K=1,2,4.

## Results

| K | Tau | Draft (ms) | Verify (ms) | Step (ms) | Tok/s | Amort. Step | Amort. Tok/s |
|---|-----|-----------|------------|----------|-------|------------|-------------|
| 1 | 5.66 | 4.51 | 10.84 | 19.22 | 294.3 | 15.35 | 368.6 |
| 2 | 5.69 | 10.95 | 27.95 | 42.79 | 132.9 | 21.79 | 261.0 |
| 4 | 5.45 | 23.63 | 61.34 | 89.07 | 61.2 | 34.47 | 158.2 |

**Amortized step** = draft_time + 1× baseline_verify (assuming batched
verification). This is the theoretical best case with MXU amortization.

## Key Findings

### 1. Tree speculation barely improves tau

K=2 gives tau=5.69 vs baseline 5.66 — only +0.5% improvement. K=4
actually *decreases* tau to 5.45. The reasons:

- **Primary first token is usually correct**: With ~84% position-0
  acceptance, alternatives help only 16% of the time.
- **Alternative drafts have bad context**: Each alternative block uses
  a freshly allocated DFlash KV cache (no conversation history), so
  DFlash's predictions for alternative branches are low quality.
- **Branches only diverge at position 0**: A flat tree (diverge only
  at the first token) has limited upside. Deeper trees (diverge at
  positions 0, 4, 8, ...) would need even more draft passes.

### 2. Draft overhead dominates, not verification

Even with perfect MXU amortization (batched verification), the amortized
throughput for K=2 (261 tok/s) is **worse** than K=1 (369 tok/s):

```
K=1: draft=4.51ms + verify=10.84ms → throughput 369 tok/s
K=2: draft=10.95ms + verify=10.84ms → throughput 261 tok/s (amortized)
```

Each additional draft branch costs ~5ms (DFlash KV allocation + forward
pass + logits). For tree speculation to break even, each branch would
need to improve tau by at least 5ms/10.84ms ≈ 46%. A +0.5% improvement
is nowhere close.

### 3. The cost structure makes tree speculation unviable for DFlash

DFlash's cost breakdown:
- Draft forward: 0.36ms (cheap)
- Draft KV allocation: ~3-4ms (expensive — JAX buffer allocation)
- Draft logits: ~1ms
- **Total per-branch overhead: ~5ms**

The KV allocation dominates. On GPU, DFlash could reuse KV caches more
efficiently. On TPU/JAX, buffer allocation triggers XLA compilation and
memory management overhead.

---

## What Would Actually Help

### Strategy 1: Cross-Request Batching (Serving)

Instead of multiple branches from ONE request, verify blocks from
MULTIPLE independent requests in a single target forward pass.

- No extra draft cost (each request drafts independently)
- No KV cache allocation (each request has its own cache)
- Pure MXU amortization: verify(8×16) ≈ verify(16)
- Potential 8× throughput per request at batch_size=8

This is the strongest exploitation of MXU amortization because it
eliminates the draft overhead that kills tree approaches.

### Strategy 2: Wider Single-Block Drafters

Train DFlash with block_size=32, 64, or 128.

- One draft pass (no per-branch overhead)
- Verify for free: verify(64) ≈ verify(16) on TPU
- Research question: how does per-position acceptance scale with width?
- Requires retraining, but the verification cost is already proven free

### Strategy 3: Tree with Shared Draft State

The alternative branches in our experiment had empty KV caches. A better
implementation would:
- Share the DFlash KV cache prefix across branches (copy-on-write)
- Only diverge the KV at the branch point
- This reduces per-branch draft cost from ~5ms to ~0.5ms

With ~0.5ms per branch: 8 branches cost 4ms draft + 10ms verify (amortized)
= 14ms total. If tau improves by even 20% (5.66 → 6.79): throughput
= 6.79/14 × 1000 = 485 tok/s vs baseline 369 tok/s = **1.31× speedup**.

This requires engineering the DFlash KV cache management, not just
a naive re-allocation.

---

## Summary of Experimental Series (Docs 32-35)

| Experiment | Finding | Speedup |
|-----------|---------|---------|
| Layer truncation (Doc 32) | Cannot reduce layers; all 36 needed | None |
| MXU amortization (Doc 33) | Verify is flat 16→128 tokens | Confirmed |
| TPU uniqueness (Doc 34) | GPU scales linearly; TPU is flat | Novel finding |
| Tree speculation (Doc 35) | Draft overhead kills tree approaches | Negative |

**The amortization finding is real and confirmed. The exploitation path
is through serving (cross-request batching) or architecture (wider blocks),
not through single-request tree speculation.**

---

*Created: February 20, 2026*
*Status: Negative result for tree approach. Amortization exploitation
requires different strategy.*
