# Doc 29: Ablation Study Results — Where the Time Actually Goes

## Summary

Four targeted microbenchmarks reveal the true optimization targets for DFlash
speculative decoding on TPU v4. The dominant cost is the **verify forward pass
+ LM head** at ~11ms (65% of step time), not the draft LM head or host overhead.

---

## Test 1: LM Head Microbenchmark (Isolated)

| Config | Time (ms) |
|---|---|
| LM head logits+argmax (batch=1) | 1.02 |
| LM head logits+argmax (batch=15) | 1.02 |
| LM head logits+argmax (batch=16) | 1.04 |
| LM head logits+argmax (batch=32) | 1.02 |
| Matmul only (16,2560)@(2560,151936) | 0.73 |
| Argmax only (16,151936) | 0.10 |

**Finding:** The LM head matmul is ~1ms regardless of batch size. TPU's MXU
handles the (batch, 2560) @ (2560, 151936) multiplication in a single
scheduling quantum. Batch 1 to 32 shows identical cost — the MXU tile is
128×128, so the hidden dimension (2560 = 20 tiles) dominates, not the batch.

**Implication:** Optimizing the LM head matmul in isolation (e.g., top-k
vocabulary projection, approximate LM head) would save at most ~0.7ms per
call. With 2 calls per step, ceiling is ~1.4ms — only 8% of step time.

---

## Test 2: Host Loop Overhead

| Component | Time (ms/iter) |
|---|---|
| Empty Python loop | 0.0001 |
| Small jnp ops (dispatch overhead) | 2.36 |
| H→D transfer (16, 2560) bf16 | 0.46 |
| D→H transfer (16,) int32 | 0.006 |
| Numpy ctx buffer ops | 0.02 |
| **Full simulated host iteration** | **4.40** |

**Finding:** The host loop overhead is dominated by JAX operation dispatch
cost (2.36ms for small jnp ops), not data transfer. The D→H transfer for
acceptance tokens (64 bytes) is essentially free at 6 microseconds. The
context buffer H→D transfer (16×2560 bf16) costs 0.46ms.

**Implication:** `jax.lax.while_loop` could save ~4.4ms per step (19% of
step time) by eliminating Python dispatch overhead. However, most of the
2.36ms "jnp ops" cost comes from JAX tracing/dispatch, which would be
absorbed into the on-device loop's XLA compilation anyway. The real savings
from `while_loop` are likely less than 4.4ms.

---

## Test 3: DFlash With vs Without Draft LM Head

| Mode | Avg step (ms) | Tau |
|---|---|---|
| Full (normal spec decode) | 23.11 | 5.68 |
| Skip draft LM head (random tokens) | 14.30 | 1.01 |
| **Draft LM head cost** | **8.82 (38%)** | — |

**Finding:** Removing the draft LM head saves 8.82ms per step, but the
LM head itself is only ~1ms (Test 1). The remaining ~7.8ms is the **forced
synchronization** — `np.array(jnp.argmax(draft_logits, axis=-1))` blocks
until the entire draft forward + LM head chain completes. In "skip" mode,
we use `np.random.randint` which doesn't block on any device computation.

**Implication:** The 8.82ms "LM head cost" is really "the cost of needing
draft tokens on host before verification can begin." An on-device loop
(`jax.lax.while_loop`) that keeps draft tokens on device and feeds them
directly into verify would eliminate this sync point entirely.

---

## Test 4: Component Overlap Analysis (Natural Pipelining)

| Component | Time (ms) |
|---|---|
| A) draft_forward only | 0.36 |
| B) draft + logits + argmax | 1.91 (LM head adds 1.55) |
| C) verify_forward only | 1.35 |
| D) verify + logits + argmax | **10.97** (LM head adds **9.62**) |
| E) aux_projection | 2.28 |
| **Sum of parts** | **15.16** |
| Actual full step | ~16-17 |

**Critical finding:** The verify forward shows 1.35ms alone, but adding the
LM head inflates it to 10.97ms. This is NOT because the LM head is slow —
it's because the sync barrier at the end forces waiting for the **entire
36-layer transformer forward pass** to actually complete. Without the sync
(test C alone), JAX returns immediately because the computation is still
in-flight on the device.

The sum of parts (15.16ms) closely matches the actual step time (~17ms),
confirming these components run **mostly sequentially** with minimal overlap.

---

## Time Budget Breakdown

```
Full Step (~17ms):
  ├─ Draft phase:           ~2 ms  (forward 0.36 + LM head 1.55)
  ├─ Host orchestration:    ~1 ms  (numpy, metadata, token transfers)
  ├─ Verify forward:       ~10 ms  (36 transformer layers, 16 query tokens)
  ├─ Verify LM head:        ~1 ms  (matmul + argmax)
  ├─ Aux projection:        ~2 ms  (FC + RMSNorm on concatenated hidden states)
  └─ Acceptance + misc:     ~1 ms
```

The verify forward pass dominates at ~10ms (59% of step time). The two LM
head calls total ~2ms (12%). Host orchestration is ~1ms (6%).

---

## Optimization Verdict

### Path 1: LM Head Optimization → LOW IMPACT
- Ceiling: ~1.4ms savings (2 × 0.7ms matmul optimization)
- 8% of step time
- Not worth pursuing as primary optimization

### Path 2: `jax.lax.while_loop` → MEDIUM-HIGH IMPACT
- Eliminates host orchestration: ~1ms savings
- **Eliminates draft-to-verify sync**: ~7-8ms savings (the sync forcing
  host to wait for draft forward to complete before building verify block)
- **Total ceiling: ~8-9ms savings (~50% of step time)**
- This is the clear winner because it eliminates the mandatory sync between
  draft sampling and verify block construction

### Path 3: vLLM Pipeline Profiling → SEPARATE CONCERN
- The standalone→vLLM gap (tau 6.67→4.48) is a real issue
- But it's orthogonal to the step-time optimization
- Should be pursued after the standalone loop is optimized

---

## Why `jax.lax.while_loop` Is the Clear Next Step

The Test 3 result is the key: removing the draft LM head saves 8.82ms not
because the matmul is expensive, but because it **eliminates the host sync
point**. In the current loop:

1. Draft forward runs on TPU
2. `np.array(jnp.argmax(draft_logits))` — **HOST BLOCKS** until draft forward
   + LM head complete on TPU
3. Host builds verify block from draft tokens
4. Verify forward runs on TPU
5. `np.array(jnp.argmax(verify_logits))` — **HOST BLOCKS** again

With `jax.lax.while_loop`:
1. Draft forward → LM head → argmax → verify block → verify forward → acceptance
   all happen **on device** without host intervention
2. Host only sees the final result after the full step completes
3. XLA can potentially overlap draft LM head with the start of verify forward

**Expected improvement: step time from ~17ms down to ~10-12ms.**

---

## Data Files

- `results/ablation_gsm8k.json` — Full ablation data
- `results/profiling_gsm8k.json` — Per-phase profiling (Doc 27)
- `results/fused_gsm8k.json` — Fused pipeline comparison (Doc 28)

---

*Created: February 19, 2026*
*Status: Experiment complete — `jax.lax.while_loop` identified as primary optimization target*
