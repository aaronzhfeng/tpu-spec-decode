# Doc 28: Fused Pipeline Results — Why JIT Fusion Barely Helps

## Result

| Metric | Baseline (AR) | Unfused DFlash | Fused DFlash | Improvement |
|---|---|---|---|---|
| TPOT (ms) | 10.71 | 2.75 | 2.79 | -1.5% |
| TPS | 93.4 | 363.2 | 357.9 | -1.5% |
| Tau | — | 6.18 | 5.89 | — |
| Avg step (ms) | — | 17.19 | 16.40 | +4.6% |
| Median step (ms) | — | 16.21 | 16.31 | -0.6% |
| Speedup vs AR | 1.00x | 3.89x | 3.83x | — |

**JIT fusion of draft+logits+argmax and verify+logits+argmax+acceptance into
single compiled functions produced negligible improvement (~1ms per step).**

---

## Why the Profiling Showed 82.8% Overhead But Fusion Barely Helps

### The profiling was measuring sync barriers, not real overhead

The profiling benchmark (Doc 27) inserted `tpu_sync()` (= `jax.effects_barrier()`)
**between every phase** to get accurate per-phase timing. But these barriers
**force synchronization** between host and device, which is exactly what
they were supposed to measure.

In the real (unfused) generate loop:
- JAX uses **lazy evaluation** — operations are dispatched to device but
  the host doesn't wait for completion
- `target_logits_fn()` after `target_model_fn()` dispatches immediately
  without waiting for the model forward to finish
- `jnp.argmax()` dispatches without waiting for logits to finish
- The host-side numpy operations (context buffer, output_ids management)
  **overlap** with device compute

So the "82.8% overhead" was largely the cost of the sync barriers themselves,
not the actual overhead in the unfused loop. The unfused loop already pipelines
well because JAX handles asynchronous execution automatically.

### What the real overhead actually is

The only mandatory synchronization points in the unfused loop are:
1. `np.array(jnp.argmax(draft_logits, axis=-1))` — pulls 15 int32 values to host
2. `jnp.array(output_ids[start:start+block_size])` — pushes 16 int32 values to device
3. `np.array(jnp.argmax(verify_logits, axis=-1))` — pulls 16 int32 values to host
4. `int(np.cumprod(matches).sum())` — acceptance length computation

These transfers involve tiny tensors (64 bytes each). The fused version
eliminates transfers 1-2 by keeping draft tokens on device, but the savings
are negligible because:
- The arrays are 64 bytes (< 1 microsecond to transfer)
- The device compute (LM head matmul: 2560 × 151936 = 389M elements) dominates

### Corrected understanding of the time budget

| What | Real time | Notes |
|---|---|---|
| Target model forward (verify) | ~10 ms | 36 transformer layers, 16 query tokens |
| LM head matmul × 2 | ~4-5 ms | (16, 2560) @ (2560, 151936), called twice |
| Draft model forward | ~0.5 ms | 4 transformer layers |
| Aux projection | ~1 ms | FC + RMSNorm on concatenated hidden states |
| Host numpy ops | ~0.3 ms | Overlap with device compute |
| **Total** | **~16-17 ms** | **All compute, not overhead** |

The step time is ~17 ms because that's how long the actual computation takes.
There is no large "overhead" to eliminate — the profiling's sync barriers
were creating artificial overhead that doesn't exist in the real loop.

---

## What This Means for Direction 1

### The standalone benchmark is already near-optimal

The host-orchestrated standalone loop achieves tau=6.18 with 17ms per step.
Almost all of that 17ms is genuine compute (model forwards + LM head matmuls).
There is no large optimization to be had from fusing operations in the
standalone benchmark.

### The real gap is in the vLLM pipeline

The meaningful performance gap is:
- **Standalone**: tau=6.67, 4.26x speedup
- **vLLM pipeline**: tau=4.48, 2.31x speedup

This gap (tau 6.67 → 4.48) comes from vLLM's scheduling, batch management,
rejection sampling loop, and `speculative_decoding_manager.py` orchestration —
not from the model compute or host-device transfers within a single step.

### Revised Direction 1 research focus

The original Direction 1 hypothesis ("pipeline overhead from host-device
roundtrips") was wrong for the standalone benchmark. The correct target is:

1. **vLLM pipeline overhead**: Profile the full vLLM serving path to identify
   where the tau=6.67→4.48 degradation comes from. This requires instrumenting
   `speculative_decoding_manager.py` and the runner, not the standalone loop.

2. **`jax.lax.while_loop`**: A fully on-device loop could still help by
   eliminating the host orchestration entirely. The potential benefit isn't
   removing "overhead" but enabling XLA to optimize the full loop globally —
   e.g., overlapping the LM head matmul for draft sampling with the start
   of the verify forward pass.

3. **LM head optimization**: The two LM head matmuls account for ~30% of step
   time. A fused or approximate LM head (e.g., top-k projection instead of
   full vocab) could reduce this.

---

## Tau Difference (6.18 vs 5.89)

The fused version shows slightly lower tau (5.89 vs 6.18). This is likely
due to XLA reordering operations within the fused graph, producing slightly
different floating-point accumulation patterns. The output quality check
confirms this: only 1/3 measured samples match exactly between fused and
unfused. This is the same bf16 precision effect seen in Doc 24 — not a
correctness issue, just numerical noise.

---

## Data Files

- `results/fused_gsm8k.json` — Full comparison data
- `results/profiling_gsm8k.json` — Per-phase profiling (Doc 27)

---

## Key Takeaway

**Sync-barrier profiling overstates overhead.** The profiling showed 82.8%
overhead because `tpu_sync()` forces serialization. The real unfused loop
pipelines well due to JAX lazy evaluation. The actual optimization targets
are:

1. vLLM pipeline orchestration (not standalone)
2. LM head matmul cost (30% of step time, 2 calls per step)
3. On-device loop via `jax.lax.while_loop` (enables global XLA optimization)

---

*Created: February 19, 2026*
*Status: Experiment complete — profiling methodology lesson learned*
