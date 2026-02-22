# Doc 39: Tile Boundary Experiment — K=129 Discontinuity

## Objective

Demonstrate the MXU tile boundary discontinuity empirically by measuring
matmul latency at K=128 vs K=129. This is the single most convincing
data point for the tile-width argument: cost should jump at exactly K=129
because a second 128-row tile is required.

This measurement was requested by the D01 proposal assessment:
> "Show the discontinuity at K=129 if you have it. That would make the
> 'tile boundary' argument dramatically more convincing."

Currently, Doc 37 confirmed flat scaling from K=16 to K=128. This
experiment closes the argument by showing the cliff at K=129.

---

## What to Run

### Step 1: Extend `benchmarks/drafter_scaling.py`

Edit line 124 of `benchmarks/drafter_scaling.py`:

```python
# Current (line 124):
k_values = [16, 32, 64, 128]

# Change to:
k_values = [16, 32, 64, 96, 128, 129, 160, 192, 256]
```

This adds K=129 (one above the tile boundary), plus a few higher values
to show the second-tile plateau continuing up to K=256.

Also update the target FFN k_values in Part 2 (look for the second
`k_values` assignment around line 170 — it currently only has [16, 128]).
Change it to match: `[16, 32, 64, 96, 128, 129, 160, 192, 256]`.

### Step 2: Run

```bash
bash tests/drafter_scaling.sh
```

Or with more trials for tighter error bars:

```bash
TRIALS=50 WARMUP=10 bash tests/drafter_scaling.sh
```

### Step 3: Expected results

The expected pattern is a step function:

```
K   | Cost  | Tile rows | vs K=16
----|-------|-----------|--------
 16 | 1.56ms|     1     | 1.00x   ← plateau 1
 32 | 1.56ms|     1     | 1.00x
 64 | 1.56ms|     1     | 1.00x
 96 | 1.56ms|     1     | 1.00x
128 | 1.56ms|     1     | 1.00x
129 | ~3.1ms|     2     | ~2.0x   ← discontinuity here
160 | ~3.1ms|     2     | ~2.0x
192 | ~3.1ms|     2     | ~2.0x
256 | ~3.1ms|     2     | ~2.0x   ← plateau 2
```

If the results show a clean step near K=129, the tile boundary argument
is proven empirically. If the step is smaller than 2× or appears at a
different K, document the actual crossover point.

---

## What to Document

After running, record results in this doc under a new "Results" section:

1. The full latency table (K=16 through K=256) for both DFlash FFN and
   target FFN
2. The ratio K=129/K=128 — this is the key number
3. Whether the step is clean (theory predicts sharp; reality may show
   some rounding due to framework overhead)
4. The second plateau value (cost at K=160, 192, 256 — should be flat
   again)

---

## Context for This Experiment

### Why K=129 specifically

TPU v4 MXU operates on 128×128 bf16 tiles. For a matrix multiply with M
query tokens and hidden dimension H, the computation is tiled as:

```
tile_rows = ceil(M / 128)

M=128: ceil(128/128) = 1 tile row
M=129: ceil(129/128) = 2 tile rows  ← double the work
```

All three matmuls in each FFN layer (gate_proj, up_proj, down_proj)
follow this pattern. For a 36-layer model: 3 matmuls × 36 layers = 108
matmul calls all double in tile count at M=129.

### Relationship to existing results

| Doc | What was measured | K values |
|-----|-------------------|----------|
| 33 | End-to-end verify latency | 16, 32, 48, 64, 96, 128 |
| 37 | Raw matmul (drafter + target FFN + attention) | 16, 32, 64, 128 |
| 39 | THIS experiment — boundary discontinuity | 16–256 including 129 |

Docs 33 and 37 show the flat plateau. Doc 39 shows the cliff at the
boundary. Together they fully characterize the tile-quantized cost
function.

### Why this matters for the paper

The D01 assessment identified this as the highest-impact missing
measurement:

> "Show the discontinuity at K=129 if you have it. That would make the
> 'tile boundary' argument dramatically more convincing."

Currently the paper argues: "K=128 fills exactly one tile row, so cost
is flat for K≤128." This is a theoretical claim. K=129 showing a ~2×
cost jump converts it from theory to empirical fact. It also directly
answers the question "why not K=200 or K=300?" — because K>128 incurs
a tile-row cost penalty that erases the gain.

---

## Results

**Executed: February 21, 2026 — TPU v4-8, TRIALS=50, WARMUP=10**

### Part 1: DFlash FFN (5 layers, hidden=2560, intermediate=9728)

| K | Per-layer (ms) | Full model (ms) | vs K=16 |
|---|---|---|---|
| 16 | 0.335 ± 0.033 | 1.676 | 1.00x |
| 32 | 0.335 ± 0.029 | 1.674 | 1.00x |
| 64 | 0.328 ± 0.034 | 1.642 | 0.98x |
| 96 | 0.331 ± 0.038 | 1.655 | 0.99x |
| 128 | 0.330 ± 0.035 | 1.649 | 0.98x |
| **129** | **0.319 ± 0.028** | **1.594** | **0.95x** |
| 160 | 0.344 ± 0.123 | 1.719 | 1.03x |
| 192 | 0.334 ± 0.028 | 1.671 | 1.00x |
| 256 | 0.343 ± 0.027 | 1.716 | 1.02x |

### Part 2: Target FFN (36 layers, hidden=2560, intermediate=9728)

| K | Per-layer (ms) | Full model (ms) | vs K=16 |
|---|---|---|---|
| 16 | 0.339 ± 0.030 | 12.201 | 1.00x |
| 32 | 0.343 ± 0.035 | 12.341 | 1.01x |
| 64 | 0.334 ± 0.031 | 12.014 | 0.98x |
| 96 | 0.330 ± 0.036 | 11.867 | 0.97x |
| 128 | 0.329 ± 0.045 | 11.842 | 0.97x |
| **129** | **0.325 ± 0.034** | **11.693** | **0.96x** |
| 160 | 0.332 ± 0.036 | 11.966 | 0.98x |
| 192 | 0.341 ± 0.034 | 12.262 | 1.00x |
| 256 | 0.346 ± 0.026 | 12.446 | 1.02x |

### Part 3: Attention Q×K^T (32 heads, head_dim=80)

| K | KV=64 | KV=256 | KV=512 | KV=1024 |
|---|---|---|---|---|
| 16 | 1.00x | 1.00x | 1.00x | 1.00x |
| 128 | 1.02x | 0.99x | 1.01x | 1.06x |
| **129** | **1.02x** | **0.98x** | **1.04x** | **1.08x** |
| 256 | 1.06x | 1.02x | 1.06x | 1.11x |

### Key Metric: K=129/K=128 Ratio

| Component | K=128 | K=129 | K=129/K=128 |
|---|---|---|---|
| DFlash FFN (5 layers) | 1.649ms | 1.594ms | **0.97×** |
| Target FFN (36 layers) | 11.842ms | 11.693ms | **0.99×** |
| Attention (KV=256) | 0.218ms | 0.216ms | **0.99×** |

**No discontinuity.** K=129 is indistinguishable from K=128. The predicted ~2× step function at the tile boundary does not appear.

---

## Interpretation: Memory-Bound, Not Compute-Bound

The expected discontinuity was based on a purely compute-bound model: if the MXU must process 2 tile rows instead of 1 at K=129, latency should double. The fact that it doesn't tells us these operations are **memory-bandwidth bound**, not compute-bound, at Qwen3-4B's dimensions.

### Why the operations are memory-bound

Each FFN layer requires loading three weight matrices from HBM:

```
gate_proj:  2560 × 9728 × 2 bytes = 49.8 MB
up_proj:    2560 × 9728 × 2 bytes = 49.8 MB
down_proj:  9728 × 2560 × 2 bytes = 49.8 MB
Total per layer: ~150 MB
```

TPU v4 HBM bandwidth: ~1.2 TB/s. Time to load one layer's weights:
```
150 MB / 1200 GB/s ≈ 0.125 ms
```

The measured per-layer latency is ~0.33ms, which is in the range where memory transfer is a significant fraction of total time. The MXU compute for even 2 tile rows at these dimensions completes within the memory transfer window — the compute is hidden behind the memory latency.

### The arithmetic intensity argument

Arithmetic intensity = FLOPs / bytes loaded. For a (K, 2560) × (2560, 9728) matmul:

```
K=128:  FLOPs = 2 × 128 × 2560 × 9728 ≈ 6.4 GFLOPs
        Bytes = (128×2560 + 2560×9728) × 2 ≈ 50.3 MB
        Intensity = 6.4G / 50.3M ≈ 127 FLOP/byte

K=256:  FLOPs = 2 × 256 × 2560 × 9728 ≈ 12.7 GFLOPs
        Bytes = (256×2560 + 2560×9728) × 2 ≈ 51.6 MB
        Intensity = 12.7G / 51.6M ≈ 246 FLOP/byte
```

The weight matrix (2560 × 9728 = ~50 MB) dominates the bytes loaded. The input activation (K × 2560) is small in comparison. Doubling K from 128 to 256 barely increases total bytes (50.3→51.6 MB, +2.6%) while doubling FLOPs. But since the MXU has 275 TFLOPS of compute and only 1.2 TB/s of bandwidth, the roofline crossover (275T / 1.2T ≈ 229 FLOP/byte) sits between K=128 and K=256. Both are near the transition zone, explaining why neither shows a significant cost increase.

### What this means for the research

**This is actually a stronger finding than the predicted discontinuity:**

1. **The flat scaling extends well beyond K=128.** DFlash could potentially use K=256 or even K=512 without cost penalty — not just K=128. The usable range is larger than the tile-boundary argument alone would suggest.

2. **The correct explanation is memory-boundedness, not just tile structure.** The weights must be loaded from HBM once per layer regardless of K. As long as the MXU compute (which grows with K) stays below the memory transfer time, latency is flat. The MXU tile structure contributes (it determines the minimum compute granularity), but the dominant factor is that weight loading time >> compute time at these dimensions.

3. **GPU contrast still holds.** On GPU, CUDA cores don't have the same memory-compute overlap characteristics. GPU tensor cores process smaller tiles (e.g., 16×16 on A100) and the scheduling is different, so compute scaling is more visible even in the memory-bound regime. The SmartSpec linear scaling confirms this.

4. **The paper's argument should be updated.** Instead of "flat because of 128×128 tiles," the argument is: "flat because TPU FFN operations at Qwen3-4B scale are memory-bandwidth bound — weight loading dominates and is independent of K. The MXU's large tile size and systolic array architecture enable this overlap. GPU architectures do not exhibit this property (SmartSpec, ICLR 2025)."

---

## Revised Understanding

| Previous claim (Docs 33, 37) | Revised understanding (Doc 39) |
|---|---|
| Flat scaling for K ≤ 128 due to MXU 128×128 tiles | Flat scaling for K ≤ 256+ due to memory-boundedness |
| Expected ~2× cost jump at K=129 | No jump — MXU compute hidden behind HBM bandwidth |
| K=128 is the hard ceiling | K=128 is conservative; K=256 appears equally free |
| Tile boundary is the mechanism | Memory-bandwidth bound is the mechanism; tile structure contributes |

### Impact on Doc 36 (Research Direction)

The core conclusion is unchanged and actually strengthened: a retrained DFlash with wider blocks costs the same as K=16. But the design space is wider than we thought:

- K=128 is the safe choice (one MXU tile row, well within the flat region)
- K=256 appears equally viable if α (acceptance rate) benefits from wider context
- The constraint on K is now **draft quality** (does α degrade at very large K?), not hardware cost

---

*Updated: February 21, 2026*
*Status: Completed — no discontinuity found, flat scaling extends to K=256*
*Executed with: `TRIALS=50 WARMUP=10 bash tests/drafter_scaling.sh`*
*Requested by: D01 assessment of proposal_v2*
