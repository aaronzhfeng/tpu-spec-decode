# Doc 32: Layer-Truncated Verification — Experiment Results

## Objective

Test whether the target model's verification forward pass can be shortened
by using fewer layers. If tau remains high at N<36 layers, early-exit
verification could save significant compute. If tau collapses, verification
genuinely needs all layers.

## Method

Monkey-patched `Qwen3Model.__init__` to capture hidden states at every
layer (all 36). During each verification step:

1. Run the full 36-layer forward pass (ground truth)
2. At each truncation point, take the hidden state after layer N
3. Apply RMSNorm manually, compute logits via the LM head
4. Simulate acceptance using truncated-layer logits
5. Compare simulated tau against ground-truth tau

Truncation points tested: 6, 12, 18, 24, 30, 31, 32, 33, 34, 35, 36.

**Setup**: Qwen3-4B (36 layers) + DFlash drafter (block_size=16), TPU v4-8,
4 samples per dataset, 256 max new tokens, 1 warmup sample.

## Results

### GSM8K (114 verify steps, 1824 positions)

Full model tau: **6.18**

| Layers | Token Match | Sim. Tau | vs Full | Compute Saved |
|--------|-------------|----------|---------|---------------|
| 6 | 0.0% | 1.00 | 16.2% | 83.3% |
| 12 | 0.0% | 1.00 | 16.2% | 66.7% |
| 18 | 3.2% | 1.05 | 17.0% | 50.0% |
| 24 | 17.3% | 1.19 | 19.3% | 33.3% |
| 30 | 49.1% | 2.23 | 36.1% | 16.7% |
| 31 | 54.2% | 2.32 | 37.5% | 13.9% |
| 32 | 56.5% | 2.46 | 39.9% | 11.1% |
| 33 | 67.3% | 3.25 | 52.6% | 8.3% |
| 34 | 76.7% | 4.02 | 65.1% | 5.6% |
| 35 | 86.1% | 4.35 | 70.5% | 2.8% |
| 36 | 100.0% | 6.18 | 100.0% | 0.0% |

### MATH500 (91 verify steps, 1456 positions)

Full model tau: **8.63**

| Layers | Token Match | Sim. Tau | vs Full | Compute Saved |
|--------|-------------|----------|---------|---------------|
| 6 | 0.2% | 1.00 | 11.6% | 83.3% |
| 12 | 0.2% | 1.00 | 11.6% | 66.7% |
| 18 | 3.4% | 1.05 | 12.2% | 50.0% |
| 24 | 22.3% | 1.32 | 15.3% | 33.3% |
| 30 | 51.9% | 2.16 | 25.1% | 16.7% |
| 31 | 52.7% | 2.22 | 25.7% | 13.9% |
| 32 | 58.0% | 2.40 | 27.8% | 11.1% |
| 33 | 69.4% | 3.13 | 36.3% | 8.3% |
| 34 | 79.5% | 4.70 | 54.5% | 5.6% |
| 35 | 87.9% | 6.09 | 70.6% | 2.8% |
| 36 | 100.0% | 8.63 | 100.0% | 0.0% |

---

## Key Findings

### 1. Verification needs ALL layers — no viable truncation point exists

Even skipping **1 layer** (35 of 36, 2.8% compute savings) loses ~30% of
tau on both datasets. The quality-compute tradeoff is catastrophic at every
truncation point.

### 2. The last layer does disproportionate work

Layer 36 flips 13.9% (GSM8K) / 12.1% (MATH500) of argmax decisions
compared to layer 35. This is a massive change for a single transformer
layer — the representation is actively being refined right up to the end.

### 3. Token-level argmax stabilization is gradual, not sudden

The argmax match rate increases roughly linearly from layer 30 to 36:

```
Layer 30: ~50%
Layer 33: ~68%
Layer 35: ~87%
Layer 36: 100%
```

There is no "stabilization cliff" — no layer after which most tokens have
settled. This contradicts SpecEE's premise that tokens stabilize early.
At least for Qwen3-4B, they don't.

### 4. Results are consistent across datasets

The token match percentages are nearly identical between GSM8K and MATH500
at every truncation point (±3%), suggesting this is a model property, not
a data-dependent phenomenon.

### 5. Tau degrades faster than token match suggests

At layer 35, 86-88% of tokens match the final layer's argmax. But tau
retains only ~70% of its value. This is because speculative decoding uses
**prefix matching** — a single wrong token at position k rejects all tokens
after it. A 12-14% per-position error rate compounds to ~30% tau loss.

---

## Implications for Verification Cost Reduction

### Early exit is NOT viable for Qwen3-4B

SpecEE (Xu et al., 2025) reports reducing layers from 32 to ~23 (28%
reduction). Our data shows that for Qwen3-4B, even a 3% layer reduction
(35→36) destroys 30% of acceptance quality. This means:

- SpecEE's approach would need to be validated per-model
- Models with more "redundant" late layers may benefit; Qwen3-4B does not
- The 30% PPL degradation SpecEE reports may be an understatement of the
  actual impact on speculative decoding quality

### The verification bottleneck is real and hard

This experiment was designed to find headroom. It found none. The full
target model forward pass is genuinely necessary for speculative decoding
verification on Qwen3-4B. This means:

1. **Layer truncation**: Not viable (this experiment)
2. **Sparse verification**: Likely also not viable — if removing entire
   layers destroys quality, removing individual neurons within layers
   probably does too
3. **Proxy verification**: The most promising remaining direction. Instead
   of making the target model cheaper, use a different (cheaper) model and
   accept the distribution shift

### What this means for our research direction

The "can we reduce verification cost?" question now has a partial answer:
**not by running fewer layers of the same model.** This pushes toward:

- **Proxy verification** (TriSpec-style) — accept distribution shift,
  measure the quality cost empirically
- **Selective verification** — don't verify all positions, only uncertain
  ones (DFlash confidence as routing signal, Experiment 3 from Doc 31)
- **Amortized verification** — batch multiple draft blocks together to
  improve MXU utilization during verification
- **Theoretical direction** — prove a lower bound that exact-distribution
  verification requires the full target forward pass (our data is
  consistent with this being true)

---

## Relationship to Literature (Doc 31)

| Paper | Their claim | Our evidence |
|-------|-------------|-------------|
| SpecEE | 28% layer reduction viable | Not for Qwen3-4B: even 2.8% reduction loses 30% tau |
| Sparse Verification | Sparsify FFN/attention | If full layers are needed, partial layers likely are too |
| TriSpec | Proxy verifier for 50% of steps | Most promising — doesn't reduce target layers, replaces them |

---

## Reproduction

```bash
# Run on GSM8K
bash tests/layer_truncation.sh

# Run on MATH500
DATASET=math500 bash tests/layer_truncation.sh

# Custom settings
MAX_SAMPLES=8 MAX_NEW_TOKENS=512 bash tests/layer_truncation.sh
```

Benchmark: `benchmarks/layer_truncation.py`
Results: `/dev/shm/dflash-test-outputs/layer_truncation_*.json`

---

*Created: February 20, 2026*
*Status: Experiment complete. Strong negative result — no truncation headroom in Qwen3-4B.*
