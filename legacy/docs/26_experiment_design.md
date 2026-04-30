# Doc 26: Experiment Design — Pipeline Profiling & Iterative Refinement

## Overview

Two experiments to explore research directions 1 and 2 from Doc 25.
Both build on the standalone DFlash benchmark (Doc 22) and require
**zero modifications to tpu-inference**.

---

## Experiment 1: Pipeline Profiling (Direction 1)

### Goal

Produce a fine-grained timing breakdown of every phase in the DFlash
speculative decoding loop on TPU. Identify where time is spent and
quantify the overhead beyond core compute (draft forward + verify forward).

### Hypothesis

The standalone-to-pipeline performance gap (tau 6.67 vs 4.48 in vLLM)
is dominated by non-compute overhead: host-device transfers, context
buffer management, acceptance logic, and cache bookkeeping. Quantifying
these overheads will reveal which phases are worth optimizing and inform
whether XLA-fused execution (Direction 1's core proposal) can help.

### Phases Measured

Each speculative decoding iteration is split into 8 timed phases:

| Phase | What it measures |
|---|---|
| `cache_mgmt` | KV cache crop/reset bookkeeping |
| `ctx_update` | Context buffer update (host numpy ops) |
| `host_device_xfer` | Building noise block + uploading context to device |
| `draft_forward` | Draft model forward pass (4 layers, flash_attention) |
| `draft_sample` | Logits computation + argmax for draft tokens |
| `verify_forward` | Target model forward pass on block (36 layers) |
| `acceptance` | Verification logits + acceptance length computation |
| `aux_projection` | Auxiliary hidden state projection for next iteration |

Each phase is bounded by `jax.effects_barrier()` (TPU sync) to ensure
device computation is complete before timing stops.

### Script

```bash
# Quick test (1 warmup + 3 measured samples, gsm8k)
bash tests/pipeline_profiling.sh

# Full run with JSON output
DATASET=math500 MAX_SAMPLES=8 bash tests/pipeline_profiling.sh --max-new-tokens 256

# All datasets
for ds in gsm8k math500 aime24 aime25; do
  DATASET=$ds OUTPUT_JSON=/output/profiling_${ds}.json \
    bash tests/pipeline_profiling.sh --max-samples 8
done
```

### Expected Output

A per-phase timing table showing mean/median/std/min/max in milliseconds,
plus percentage of total step time. The report also computes:
- **Core compute**: draft_forward + verify_forward
- **Overhead**: everything else
- Overhead as a percentage of total step time

### Success Criteria

The experiment succeeds if it produces actionable data:
1. We can identify which phase(s) dominate overhead
2. The overhead percentage gives a ceiling for how much an XLA-fused
   loop could improve
3. Results are reproducible across runs and datasets

---

## Experiment 2: Iterative Refinement (Direction 2)

### Goal

Test whether running the draft model multiple times on the same block
(feeding predictions back as input) improves draft quality enough to
offset the additional latency.

### Hypothesis

DFlash's non-causal attention means all block positions attend to each
other. In the standard single-pass mode, positions 2-15 only see
mask tokens at neighboring positions. With iterative refinement,
positions can attend to actual predicted tokens, improving cross-position
coherence and raising acceptance rates.

The key question: does the tau improvement outweigh the extra draft
forward passes?

### Mechanism

```
Standard (k=0):
  [tok, mask, mask, ...] -> draft -> sample -> verify

Refinement (k=1):
  [tok, mask, mask, ...] -> draft -> sample = [t1, t2, ..., t15]
  [tok, t1, t2, ..., t14] -> draft -> sample = [t1', t2', ..., t15']
  -> verify [tok, t1', t2', ..., t15']

Refinement (k=2):
  [tok, mask, mask, ...] -> draft -> [t1, ..., t15]
  [tok, t1, ..., t14]    -> draft -> [t1', ..., t15']
  [tok, t1', ..., t14']  -> draft -> [t1'', ..., t15'']
  -> verify [tok, t1'', ..., t15'']
```

### KV Cache Handling for Refinement

This is the critical implementation detail. After the initial draft pass
writes to the KV cache, refinement passes need to rewrite the noise
block portion while preserving the context entries:

1. Record `pre_noise_cache_len` before the initial draft pass
2. For each refinement pass:
   - Set `cache_len = pre_noise_cache_len + actual_ctx_count`
     (this points right after the context, at the start of noise)
   - Set `actual_ctx_count = 0` (context is already in cache)
   - The draft model's `dynamic_update_slice` overwrites the old
     noise K/V with new ones at the same positions
3. The context portion of the cache (positions 0..pre_noise_cache_len +
   actual_ctx_count - 1) remains untouched across refinement passes

### Script

```bash
# Quick test: k=0 and k=1 only, 3 samples
bash tests/iterative_refinement.sh --refinement-steps 0 1 --max-samples 3

# Full sweep: k=0,1,2,3 with baseline, 8 samples
bash tests/iterative_refinement.sh --refinement-steps 0 1 2 3 \
  --run-baseline --max-samples 8 --max-new-tokens 256

# Specific dataset
DATASET=math500 bash tests/iterative_refinement.sh \
  --refinement-steps 0 1 2 3 --run-baseline

# All datasets
for ds in gsm8k math500 aime24 aime25; do
  DATASET=$ds OUTPUT_JSON=/output/refinement_${ds}.json \
    bash tests/iterative_refinement.sh --refinement-steps 0 1 2 3 --run-baseline
done
```

### Expected Output

A comparison table showing for each k:

| k | tau | TPOT(ms) | TPS | Speedup | Draft(ms) | Refine(ms) | Verify(ms) |
|---|-----|----------|-----|---------|-----------|------------|------------|

Plus a relative comparison showing whether each k is a net improvement
(higher TPS) over k=0.

### Success Criteria

There are three possible outcomes, all scientifically valuable:

1. **Positive**: k=1 or k=2 improves tau enough to outweigh latency cost.
   The net TPS is higher than k=0. This is a novel result — no prior
   work has explored multi-step refinement for diffusion-based speculative
   decoding.

2. **Neutral**: Tau improves but latency cost cancels it out. Still
   publishable as a characterization result showing the quality-latency
   Pareto frontier.

3. **Negative**: Refinement doesn't improve tau at all. This establishes
   that single-step denoising is already optimal for the DFlash
   architecture and the non-causal attention is sufficient to capture
   cross-position dependencies in one pass. Also publishable.

### What to Watch For

- **Tau vs k curve**: Does tau increase monotonically with k, plateau,
  or decrease? A plateau would suggest diminishing returns.
- **Per-position acceptance rates**: Does refinement specifically help
  later positions (positions 8-15) more than early positions?
- **Refinement pass latency**: Is the marginal cost of each refinement
  pass equal to the initial draft pass, or cheaper (because context is
  already in cache)?
- **XLA compilation behavior**: Does adding refinement steps trigger
  recompilation or does XLA handle the loop efficiently?

---

## File Structure

```
benchmarks/
  pipeline_profiling.py     # Direction 1: per-phase timing
  iterative_refinement.py   # Direction 2: k-step refinement sweep
  standalone_dflash.py      # Existing baseline benchmark

tests/
  pipeline_profiling.sh     # Docker wrapper for Direction 1
  iterative_refinement.sh   # Docker wrapper for Direction 2

results/
  profiling_*.json          # Direction 1 outputs (after runs)
  refinement_*.json         # Direction 2 outputs (after runs)
```

Both scripts follow the established convention:
- Python scripts in `benchmarks/` import from `tpu_inference` read-only
- Bash wrappers in `tests/` use `docker_exec` from `tests/lib/docker_run.sh`
- JSON results go to `/output/` (mapped to host via Docker volume)

---

## Running Both Experiments

### Quick smoke test (both directions, ~10 min)

```bash
# Direction 1
bash tests/pipeline_profiling.sh --max-samples 3 --max-new-tokens 64

# Direction 2
bash tests/iterative_refinement.sh --refinement-steps 0 1 --max-samples 3 --max-new-tokens 64
```

### Full experiment (both directions, all datasets, ~2-3 hours)

```bash
for ds in gsm8k math500 aime24 aime25; do
  # Direction 1
  DATASET=$ds OUTPUT_JSON=/output/profiling_${ds}.json \
    bash tests/pipeline_profiling.sh --max-samples 8 --max-new-tokens 256

  # Direction 2
  DATASET=$ds OUTPUT_JSON=/output/refinement_${ds}.json \
    bash tests/iterative_refinement.sh --refinement-steps 0 1 2 3 \
    --run-baseline --max-samples 8 --max-new-tokens 256
done
```

---

*Created: February 2026*
*Status: Ready to run*
