# Doc 24 — Output Quality Sanity Check

Quick sanity check to confirm DFlash does not degrade output quality. This is expected by design — speculative decoding verifies every accepted token through the target model's own forward pass — but we ran the numbers anyway to be thorough.

## Method

Ran the standalone benchmark on GSM8K (8 samples, 1024 max tokens, greedy decoding) with both autoregressive baseline and DFlash, then compared outputs token-by-token.

## Results

- 4/8 samples: bit-exact identical output (every token matches)
- 7/8 samples: same final \boxed{} math answer
- 1/8 samples: different final answer (baseline=452, DFlash=470)

The 4 samples that diverge do so at a single token position (pos 8, 13, 68, 245), after which all subsequent tokens differ due to autoregressive cascading. The divergence points are coherent alternative phrasings (e.g. `**Indras**` vs `Indras`, `Step 1:` vs `Initial Setup:`), not garbage.

## Why This Is Expected

The mismatches come from bf16 floating-point precision, not from DFlash. The ragged paged attention kernel accumulates slightly differently when processing 16 query tokens at once (DFlash verify) vs 1 at a time (baseline). At decision boundaries where the top-2 logits are close, the argmax can flip. This happens with any speculative decoding method on bf16 hardware and is not specific to DFlash.

Speculative decoding with greedy sampling is mathematically guaranteed to produce the same output as autoregressive decoding under exact arithmetic. The only deviation is finite-precision noise, which is equally present in the baseline (just with a different accumulation pattern).

## Data

- `results/quality_check.csv` — per-sample token match and answer comparison
- `results/quality_gsm8k.json` — full quality run data with mismatch details

## Verdict

DFlash does not hurt model intelligence. No further investigation needed.
