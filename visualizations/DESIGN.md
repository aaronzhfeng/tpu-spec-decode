# Visualization Design — DFlash TPU Speculative Decoding

8 visualizations for the project website, poster, and paper.

---

## Fig 1: K-Flat Property (Hero Figure)

**Story:** TPU verification cost is constant from K=16 to K=1024. GPU rises 1.24x at K=128.

**Layout:** Two panels side-by-side (10" x 4")
- Left: TPU absolute latency (ms) vs K. Flat line at ~1.31ms with error bars. K = {16, 128, 256, 384, 512, 768, 1024}.
- Right: Normalized ratio (K/K=16). TPU flat at ~1.0x. GPU lines at different context lengths rising to 1.24x.

```
 Left Panel: TPU Verification Latency        Right Panel: Ratio (K vs K=16)
 3.0 |                                       2.5 |              GPU L=1024
     |                                           |          .---*
 2.0 |                                       2.0 |      .--'
     |                                           |  .--'   GPU L=512
 1.0 |--*---*---*---*---*---*---*--  1.31ms  1.0 |--*===*===*===*===  TPU (all L)
     |                                           |
 0.0 +--+---+---+---+---+---+---+           0.0 +--+---+---+---+
     16 128 256 384 512 768 1024                 16  32  64  128
                K (tokens)                            K (tokens)
```

**Data:**
- TPU: `results/verify_context_scaling.json` (V5P, L=256)
- GPU: V7 doc hardcoded — K={16,32,64,128} at L={64,256,512,1024}

**Colors:** TPU = blue (#1f77b4), GPU = red (#d62728), lighter shades per L

---

## Fig 2: 9-Dataset Speedup Bars

**Story:** 3.13x average speedup across 9 benchmarks. Math dominates, chat is weakest.

**Layout:** Single panel (10" x 5"), grouped bar chart

```
 6x |  [m500]
    |  |||||
 5x |  |||||
    |  |||||
 4x |  ||||| [ai24]
    |  ||||| |||||  [ai25]
 3x |  ||||| ||||| |||||  [gsm]  [heval] [mbpp]
    |  ||||| ||||| ||||| |||||  |||||  |||||
 2x |  ||||| ||||| ||||| ||||| |||||  |||||          [mtb]
    |  ||||| ||||| ||||| ||||| |||||  ||||| [swe]   |||||  [alp]
 1x |--------------------------------------------- 1.0x baseline ---
    |  math500 aime24 aime25 gsm8k heval  mbpp  swe  mt-b  alp
    |  |--------- math ---------|  |---- code ----|  |-- chat --|
```

**Data:** `results/v5p/standalone_all_benchmarks.csv`
- math500: 4.93x (tau=8.80), aime24: 3.57x (tau=6.48), aime25: 3.39x (tau=6.14), gsm8k: 3.00x (tau=5.40)
- humaneval: 3.17x (tau=5.76), mbpp: 3.27x (tau=6.16), swe-bench: 1.86x (tau=3.35)
- mt-bench: 2.29x (tau=3.87), alpaca: 1.63x (tau=2.86)

**Colors:** Math = green (#2ca02c), Code = blue (#1f77b4), Chat = orange (#ff7f0e)

---

## Fig 3: Tau Ceiling Saturation

**Story:** tau = sum(alpha^k, k=1..K) saturates by K~32-64. Wider blocks only help if alpha improves.

**Layout:** Single panel (8" x 5"), line plot

```
 20 |                              alpha=0.95 ----___________
    |                         .---'
 15 |                    .--'     alpha=0.90 ----__________
    |               .--'     .---'
 10 |          .--'     .---'     alpha=0.80 ----________
    |   * .--'     .---'     .---'
  5 |  *     .---'     .---'          alpha=0.65 ----____
    | *  .---'    .---'          .---'
  0 +-+--+---+----+----+----+---+
    1  4  8  16   32   64  128
              K (block size)
       ^
       Measured tau at K=16 (9 datasets as colored dots)
```

**Data:**
- Theoretical: `tau(alpha, K) = alpha * (1 - alpha^K) / (1 - alpha)`
- Measured: tau from CSV at K=16 (9 values)
- Alpha back-computed: for each dataset, solve `tau = alpha*(1-alpha^16)/(1-alpha)` for alpha

---

## Fig 4: Risk-Free Zone Diagram

**Story:** TPU = entire green zone (any K works). GPU yellow zone shrinks as K grows.

**Layout:** Single panel (7" x 6"), filled contour

```
 1.0 |GGGGGGGGGGGGGGGGG|  G = green (speedup on both)
     |GGGGGGGGGGGGGGGGG|  Y = yellow (TPU only)
 0.8 |GGGGGGGGGGGGGGGGY|  R = red (no speedup)
     |GGGGGGGGGGGGGGYYY|
 0.6 |GGGGGGGGGGGGGYYYR|  GPU boundary curves left
     |GGGGGGGGGGGYYYYRR|
 0.4 |GGGGGGGGGYYYYYYRRR|
     |GGGGGGGYYYYYYYYRRR|
 0.2 |GGGGYYYYYYYYYYYRRR|
     |GYYYYYYYYYYYYYRRRR|
 0.0 +---+---+---+---+--
     16  32  64  96  128
          K (block size)
     alpha
```

**Data:** Analytical. TPU speedup = tau/1.0. GPU speedup = tau / verify_ratio(K). GPU verify ratio from V7: linear interpolation from 1.0 at K=16 to 1.24 at K=128.

---

## Fig 5: Roofline Gap

**Story:** Naive prediction: cost scales 64x from K=16 to K=1024. Reality: 1.00x. XLA pipelining + memory-bound regime.

**Layout:** Single panel (8" x 5"), log-scale line plot

```
 100 |                                  * Predicted (linear scaling)
     |                              *
  10 |                      *
     |              *
     |      *
   1 |--*===*===*===*===*===*===*  Measured (TPU v5p)
     |
 0.1 +--+---+---+---+---+---+---+
     16 128 256 384 512 768 1024
              K (tokens)
```

**Data:** `results/verify_context_scaling.json` (measured). Predicted = 1.31 * K/16.

---

## Fig 6: Step Time Breakdown

**Story:** Verify forward dominates at 59%. Draft is cheap at 12%.

**Layout:** Donut chart (5" x 5")

```
        ,---.---.
      ,'  Verify  `.
    ,'    59%       `.
   |   (10.0 ms)     |
   |      ,---.      |
   |     | 17ms|     |
    `.   `---'  ,'
      `.  Over `.
        `head29%`
          Draft 12%
```

**Data:** Hardcoded from Doc 29 ablation:
- Verify forward: 10.0ms (59%)
- Draft forward + LM head: 2.0ms (12%)
- Overhead (aux proj + host + acceptance): 5.0ms (29%)

---

## Fig 7: Per-Position Acceptance Heatmap

**Story:** Math tasks hold high acceptance across all 16 positions. Chat/general drops off quickly.

**Layout:** Single panel (10" x 5"), seaborn heatmap

```
          Position: 1    4    8    12   16
 math500          [1.0][0.76][0.25][0.13][0.05]  dark green -> light
 aime24           [1.0][0.63][0.18][0.09][0.03]
 aime25           [1.0][0.60][0.17][0.07][0.02]
 gsm8k            [1.0][0.56][0.20][0.10][0.06]
 mbpp             [1.0][0.64][0.20][0.06][0.02]
 humaneval        [1.0][0.55][0.16][0.09][0.04]
 swe-bench        [1.0][0.31][0.07][0.02][0.01]
 mt-bench         [1.0][0.36][0.10][0.04][0.01]
 alpaca           [1.0][0.23][0.06][0.03][0.01]
```

**Data:** `results/v5p/standalone_*.json` -> `summary.acceptance_rate_per_pos`
- 9 datasets x 16 positions
- Sorted by tau (math500 highest at top)

**Colors:** Sequential green colormap (YlGn), vmin=0, vmax=1.0

---

## Fig 8: Real-Time Inference Replay

**Story:** Watch DFlash generate text 3-5x faster than AR, in real time. Tokens appear in bursts.

**Layout:** HTML page, two columns

```
+---------------------------+---------------------------+
| Autoregressive            | DFlash (3.0x faster)      |
| 157 tokens/sec            | 472 tokens/sec            |
|                           |                           |
| The answer is 42.         | The answer is 42. To      |
| To solve this,_           | solve this, we first      |
|                           | need to compute the       |
|                           | value of x given_         |
|                           |                           |
| [=====>          ] 35%    | [===============>  ] 78%  |
|                           |                           |
| Elapsed: 1.2s             | Elapsed: 1.2s             |
+---------------------------+---------------------------+
        [Play] [Pause] [1x] [2x] [5x]
```

**Data:** Synthesized from TPOT:
- Baseline: 1 token every 6.35ms
- DFlash: 16 tokens every ~17ms (burst), with tau=5.4 avg accepted

**Implementation:** Self-contained HTML/CSS/JS, ~250 lines
- Use gsm8k sample for demo text
- Tokens appear one-by-one (AR) vs in bursts of tau tokens (DFlash)
- Speed controls: 1x, 2x, 5x real-time

---

## Priority for Poster

1. Fig 1 (K-flat) — hero, top center
2. Fig 2 (speedup bars) — main results
3. Fig 7 (heatmap) — detailed results
4. Fig 3 (tau ceiling) — theory motivation
5. Fig 6 (breakdown) — supporting
6. Fig 4 (risk-free) — conceptual

## Priority for Website

1. Fig 8 (token replay) — interactive hook
2. Fig 1 (K-flat) — core finding
3. Fig 2 (speedup bars) — results
4. Fig 7 (heatmap) — detail
5. Fig 5 (roofline gap) — technical depth

## Generation

```bash
cd /home/aaronfeng/tpu-spec-decode
python visualizations/visualizations.py --output-dir visualizations/figures
# Or individual:
python visualizations/visualizations.py --figure k-flat
```
