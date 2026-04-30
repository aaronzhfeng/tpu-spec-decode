# Doc 62 Appendix: Raw Benchmark Logs

Date: 2026-03-15
Platform: TPU v5p-8, Docker vllm/vllm-tpu:latest, flax==0.12.2
Config: --max-model-len 4096, 5 requests, prompt="How many positive whole-number
divisors does 196 have?", max_tokens=128, temperature=0

---

## Baseline (same RoPE positions for context and noise)

Source: `pr_dflash_1/tpu_inference/models/jax/qwen3_dflash.py` (no ctx_positions)

vLLM SpecDecoding metrics log line:
```
(APIServer pid=10) INFO 03-15 16:06:44 [metrics.py:100] SpecDecoding metrics: Mean acceptance length: 3.53, Accepted throughput: 7.34 tokens/s, Drafted throughput: 43.54 tokens/s, Accepted: 455 tokens, Drafted: 2700 tokens, Per-position acceptance rate: 0.722, 0.528, 0.389, 0.278, 0.139, 0.111, 0.083, 0.083, 0.028, 0.028, 0.028, 0.028, 0.028, 0.028, 0.028, Avg Draft acceptance rate: 16.9%
```

## RoPE Fix (separate positions: context at cache_len, noise at block offset)

Source: `pr_dflash_1a/tpu_inference/models/jax/qwen3_dflash.py` (with ctx_positions)

vLLM SpecDecoding metrics log line (after warmup):
```
(APIServer pid=10) INFO 03-15 15:48:34 [metrics.py:100] SpecDecoding metrics: Mean acceptance length: 2.89, Accepted throughput: 2.77 tokens/s, Drafted throughput: 22.00 tokens/s, Accepted: 415 tokens, Drafted: 3300 tokens, Per-position acceptance rate: 0.682, 0.455, 0.318, 0.182, 0.091, 0.068, 0.045, 0.045, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, Avg Draft acceptance rate: 12.6%
```

First request (cold, before warmup):
```
(APIServer pid=10) INFO 03-15 15:46:04 [metrics.py:100] SpecDecoding metrics: Mean acceptance length: 1.78, Accepted throughput: 0.64 tokens/s, Drafted throughput: 12.37 tokens/s, Accepted: 14 tokens, Drafted: 270 tokens, Per-position acceptance rate: 0.444, 0.222, 0.111, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, Avg Draft acceptance rate: 5.2%
```

## Synthetic RoPE comparison (pr-ready/test_rope_positions.py)

```
Output shape: (4, 1, 64)
Max absolute difference:  0.861826
Mean absolute difference: 0.154523
Outputs are DIFFERENT
```
