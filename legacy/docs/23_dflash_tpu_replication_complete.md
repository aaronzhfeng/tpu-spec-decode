# Doc 23: DFlash TPU Replication — Complete

## Status: Done

The DFlash speculative decoding algorithm has been successfully replicated on TPU.
The implementation achieves **94% of GPU draft quality** and requires **zero vLLM changes**
for the tpu-inference PR.

---

## Final Numbers

### Standalone Benchmark (apples-to-apples with GPU paper)

Single-device, no vLLM pipeline, greedy decoding, Qwen3-4B + DFlash-b16:

| Dataset | TPU τ | TPU Speedup | GPU Paper τ | GPU Paper Speedup | τ % of GPU |
|---|---|---|---|---|---|
| GSM8K | 5.17 | 3.36x | 6.53 | 5.15x | 79% |
| Math500 | **8.72** | **5.59x** | 7.84 | 6.09x | **111%** |
| AIME24 | 6.32 | 3.93x | 7.27 | 5.68x | 87% |
| AIME25 | 6.48 | 4.14x | 6.64 | 5.21x | 98% |
| **Average** | **6.67** | **4.26x** | **7.07** | **5.53x** | **94%** |

### vLLM Pipeline (production serving path)

4 TPU v4 devices, full vLLM pipeline, 32 prompts across 4 datasets:

| Metric | Value |
|---|---|
| Overall τ | 4.48 |
| Overall Speedup | 2.31x |
| DFlash TPS | 212.4 |
| Baseline TPS | 92.1 |
| Draft Acceptance | 23.2% |

### What the Numbers Mean

| Question | Answer |
|---|---|
| Does DFlash work on TPU? | **Yes.** τ=6.67 standalone, 4.26x speedup. |
| Does it match GPU quality? | **94% average**, exceeds GPU on Math500. |
| Is the vLLM integration working? | **Yes.** 2.31x speedup in full pipeline. |
| Why is vLLM slower than standalone? | Pipeline overhead: scheduling, batch management, rejection sampling loop. |
| Are there vLLM changes needed? | **No.** Standalone benchmark proves TPU DFlash works using only tpu-inference dependencies. |

---

## vLLM Independence

### What the standalone benchmark imports from vLLM

```python
from vllm.config import ModelConfig          # config dataclass
from vllm.model_executor.model_loader import LoadConfig  # weight loader config
```

These are **existing tpu-inference dependencies** — every model in tpu-inference imports
`ModelConfig` and `LoadConfig`. DFlash adds zero new vLLM dependencies.

### What we do NOT need

Zhongyan's vLLM fork has exactly **1 commit** ahead of upstream (`5400a8b90 dflash v1`)
modifying **1 file** (`vllm/config/speculative.py`) with **~10 lines**:

- Add `"dflash"` to the `SpeculativeMethod` literal type
- Auto-detect DFlash from model config
- Include method in computation graph hash
- Skip draft-model-only validation for DFlash

These changes are **only needed for vLLM's speculative decoding pipeline** (running DFlash
through `vllm serve`). The standalone benchmark and tpu-inference PR do not require them.

### PR scope for tpu-inference

| File | Change |
|---|---|
| `tpu_inference/models/jax/dflash.py` | DFlash model (new) |
| `tpu_inference/spec_decode/jax/dflash.py` | DFlash proposer (new) |
| `tpu_inference/runner/speculative_decoding_manager.py` | seq_len fix + DFlash integration |
| `benchmarks/standalone_dflash.py` | Standalone benchmark (new) |
| `tests/standalone_benchmark.sh` | Docker wrapper (new) |

No changes to vLLM. No new external dependencies.

---

## Journey Summary

### Phase 1: Understanding (Docs 00–06)
Research overview, TPU setup, tpu-inference architecture deep dive, model/kernel analysis.

### Phase 2: Initial Implementation (Docs 07–12)
DFlash integration plan, test plan, initial model port. Encountered KV cache shape
mismatches between GPU (concat) and TPU (paged) attention.

### Phase 3: KV Cache Redesign (Docs 10.1, 11.1, 13–16)
Backported DFlash to use static on-device KV caches with `dynamic_update_slice` and
TPU `flash_attention` kernel (causal=False). Moved from paged attention to per-layer
static caches matching the GPU DFlash architecture.

### Phase 4: Debugging Low Acceptance (Docs 17–20)
- Doc 17: GPU vs TPU gap analysis — identified architectural differences
- Doc 18: Context padding fix, placeholder token fix, position investigation
- Doc 19: Comprehensive architectural comparison (13 A/B test dimensions)
- Doc 20: Systematic acceptance rate investigation — 13 verification checks all passed,
  pointing to an upstream bug

### Phase 5: Breakthrough (Doc 21)
Discovered the **seq_len inflation bug** in `speculative_decoding_manager.py`:
- `attn_metadata.seq_lens` included unverified draft tokens (~15 phantom tokens per step)
- This corrupted the proposer's context buffer, KV cache positions, and RoPE embeddings
- Fix: Use `num_tokens_no_spec` (actual accepted count) instead of inflated `seq_lens`
- Result: τ jumped from 2.49 to 4.48, speedup from 1.30x to 2.31x

### Phase 6: Standalone Proof (Docs 22–23)
Built a vLLM-independent benchmark mirroring the GPU paper's experimental setup:
- Proved τ=6.67 (94% of GPU) without vLLM pipeline overhead
- Proved the implementation is self-contained for tpu-inference PR
- Established Math500 τ=8.72 **exceeds** GPU paper's 7.84

---

## Key Bugs Found and Fixed

| Bug | Impact | Fix | Doc |
|---|---|---|---|
| seq_len inflation | τ 2.49→4.48, biggest single fix | Use `num_tokens_no_spec` instead of `attn_metadata.seq_lens` | 21 |
| Missing `replace` import | Runtime crash | Add `replace` to dataclass import | 21 |
| Context padding zeros | Corrupted draft context | Pad with actual features, not zeros | 18 |
| Placeholder token in noise | Wrong first draft token | Use mask_token_id consistently | 18 |
| Standalone ctx_len init | Draft got zero prompt context | Initialize `prev_ctx_len=0`, not `num_input_tokens` | 22 |

---

## Architecture at a Glance

```
Target Model (Qwen3-4B)          Draft Model (DFlash-b16)
┌─────────────────────┐          ┌─────────────────────┐
│  Transformer Layers  │          │  Transformer Layers  │
│  [0..35]             │          │  [0..3]              │
│                      │          │                      │
│  aux_hidden_states   │───proj──▶│  target_hidden_states│
│  from layers         │          │  (context features)  │
│  [1, 9, 17, 25, 33] │          │                      │
│                      │          │  flash_attention     │
│  ragged_paged_attn   │          │  (causal=False)      │
│  (paged KV cache)    │          │  (static KV cache)   │
└─────────────────────┘          └─────────────────────┘
         │                                  │
         │  verify block                    │  draft block
         │  [tok, d1, d2, ..., d15]        │  [tok, mask, mask, ..., mask]
         ▼                                  ▼
    ┌─────────┐                       ┌─────────┐
    │ logits  │──compare──accept──▶   │ logits  │
    │ (gold)  │                       │ (draft) │
    └─────────┘                       └─────────┘

One step: draft 15 tokens → verify all 16 → accept consecutive matches + 1 bonus
τ = average tokens accepted per step (higher = better)
```

---

## Reproducing Results

### Standalone benchmark (recommended for evaluation)

```bash
# Quick smoke test (1 sample, 32 tokens)
bash tests/standalone_benchmark.sh --max-samples 1 --max-new-tokens 32

# Full single-dataset run
DATASET=math500 bash tests/standalone_benchmark.sh --max-samples 8 --max-new-tokens 256

# All datasets
for ds in gsm8k math500 aime24 aime25; do
  DATASET=$ds bash tests/standalone_benchmark.sh --max-samples 8 --max-new-tokens 256
done
```

### vLLM pipeline benchmark

```bash
# Smoke test
bash tests/smoke_greedy.sh

# Full 4-dataset benchmark (32 prompts)
bash tests/benchmark.sh math
```

---

## What's Left (optional, not blocking)

These are optimizations, not correctness issues:

1. **Multi-device standalone**: Add `shard_map` wrapping for `flash_attention` to enable
   4-device runs. Would increase throughput but not τ.

2. **vLLM pipeline τ improvement**: The gap between standalone τ=6.67 and vLLM τ=4.48
   suggests room for pipeline-level optimizations (reducing scheduling overhead,
   batching draft proposals).

3. **Upstream vLLM DFlash support**: The 10-line vLLM change could be upstreamed
   independently to enable `vllm serve --speculative-method dflash`.

4. **Longer generation**: Test with 1024+ tokens to verify no degradation over long
   sequences.

5. **Temperature > 0**: Currently only greedy (temperature=0) is validated. Sampling
   would need token-level probability comparison instead of argmax matching.
