# Architecture Diagram Draft

ASCII sketch to be replaced with a proper figure later.

```
Target Model (Qwen3-4B)          Draft Model (DFlash-b16)
+---------------------+          +---------------------+
|  Transformer Layers  |          |  Transformer Layers  |
|  [0..35]             |          |  [0..3]              |
|                      |          |                      |
|  aux_hidden_states   |--proj--->|  target_hidden_states|
|  from layers         |          |  (context features)  |
|  [1, 9, 17, 25, 33] |          |                      |
|                      |          |  flash_attention     |
|  ragged_paged_attn   |          |  (causal=False)      |
|  (paged KV cache)    |          |  (static KV cache)   |
+---------------------+          +---------------------+
         |                                  |
    verify block                       draft block
    [tok, d1, ..., d15]          [tok, mask, ..., mask]
         |                                  |
    +---------+                       +---------+
    | logits  |---compare--accept-->  | logits  |
    | (gold)  |                       | (draft) |
    +---------+                       +---------+
```

Each step: draft 15 tokens -> verify all 16 -> accept consecutive matches + 1 bonus token. tau = average tokens accepted per step (higher = better).
