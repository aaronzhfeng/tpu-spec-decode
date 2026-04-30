# Doc 63: PR Review Session Lessons and Recipes

Date: 2026-03-22
Scope: Learnings from addressing kyuyeunk's PR #1868 review (2026-03-13 to 2026-03-16)

---

## Branch structure

The PR is split into three stacked branches in `pr-ready/pr_dflash_1`
(remote: `aaronzhfeng/tpu-inference`):

| Branch | Commit | What it adds |
|--------|--------|-------------|
| `pr_dflash_1` (= `pr_dflash_1a`) | `888650b5` + `82adc7d8` | Model, proposer, tests, inline comments |
| `pr_dflash_1b` | `20ec612c` | Pipeline integration (tpu_runner, kv_cache_manager, spec_decoding_manager, model_loader, qwen3.py) |
| `pr_dflash_1c` | `bdc64e95` | E2e tests + Buildkite CI yml |

All three branch from the same base (`a73366c4`). They are independent
(not stacked on each other), each adding one commit.

### How to combine for testing

Cherry-pick onto 1c (which has the CI changes):
```bash
git checkout pr_dflash_1c
git cherry-pick origin/pr_dflash_1b   # pipeline integration
git cherry-pick 888650b5               # model+proposer (use the commit, not branch HEAD)
```

Do NOT cherry-pick `origin/pr_dflash_1` HEAD if it has extra commits (like
the RoPE revert). Use the specific commit hash.

### Investigation branches

- `pr_dflash_1a` (local branch in pr_dflash_1): has uncommitted RoPE fix.
  DO NOT apply — benchmark showed it degrades performance.
- `pr-ready/pr_dflash_1a/` (cloned directory): same, throwaway copy.
- `pr-ready/pr_dflash_test/` (cloned directory): combined 1a+1b+1c for
  benchmarking. Throwaway.

---

## Docker benchmark recipe

### Working configuration

```bash
sudo docker run -d --rm --privileged --net=host \
  --name dflash_server \
  -v /path/to/combined-tpu-inference:/workspace/tpu_inference \
  -v /path/to/vllm-lkg:/workspace/vllm-lkg \
  -v /dev/shm/hf-cache:/dev/shm/hf-cache \
  -e PYTHONPATH=/workspace/vllm-lkg:/workspace/tpu_inference \
  -e HF_HOME=/dev/shm/hf-cache \
  vllm/vllm-tpu:latest \
  bash -c '
pip install -q flax==0.12.2
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B \
  --trust-remote-code \
  --max-model-len 4096 \
  --speculative-config "{\"model\": \"z-lab/Qwen3-4B-DFlash-b16\", \"num_speculative_tokens\": 15, \"method\": \"dflash\", \"draft_tensor_parallel_size\": 1}" \
  --port 8000'
```

### Critical details

1. **Mount path**: Must mount to `/workspace/tpu_inference` (underscore, not
   hyphen) to override the Docker image's built-in tpu-inference. Otherwise
   Python imports the built-in copy.

2. **PYTHONPATH order**: `vllm-lkg` must come first so it overrides the
   Docker's built-in vLLM (v0.13.0, too new for our code).

3. **flax version**: Must install `flax==0.12.2`. The Docker image has a
   different version. 0.11.1 is too old, 0.12.4 breaks sharding.

4. **--max-model-len 4096**: Full 40960 causes VMEM exhaustion during XLA
   compilation of the DFlash proposer forward pass (exceeds 63.94M vmem
   by ~6M). Reduce to 4096 for benchmarking.

5. **--trust-remote-code**: Required because `z-lab/Qwen3-4B-DFlash-b16`
   has custom code.

### Fixes applied to make it work

**vLLM patches** (applied via `pr-ready/patch_vllm.py` at runtime):
- Add `"dflash"` to `SpeculativeMethod` Literal type
- Add `elif self.method == "dflash": pass` before the `else` clause that
  rejects unknown architectures

**tpu-inference fix** (in `pr_dflash_test` only):
- `spec_decode/jax/dflash.py:85`: `get_model` returns 8 values, not 7.
  Added `_` for the pooler between `compute_logits_fn` and
  `combine_hidden_states_fn`.

---

## What's needed for full vLLM pipeline support

The DFlash speculative decoding method is not registered in upstream vLLM.
A separate PR to vLLM would need to add:

1. **`SpeculativeMethod` Literal**: Add `"dflash"` to the type in
   `vllm/config/speculative.py`

2. **Architecture auto-detection**: Add `elif self.method == "dflash": pass`
   in the method detection chain (before the `else` clause that rejects
   unknown architectures as `draft_model`)

3. **Model registry**: `DFlashDraftModel` needs to be recognized. Currently
   handled by tpu-inference's `model_loader.py` but vLLM validates first.

4. **Engine core**: vLLM v1 engine core has its own speculative decoding
   manager that doesn't know about dflash. The tpu-inference runner
   (`tpu_runner.py`) handles it, but only after the engine core delegates
   to the TPU worker.

---

## Summary of outcomes

| Item | Outcome |
|------|---------|
| Inline comments on `dflash_attention_interface.py` | Committed (`82adc7d8`), pushed |
| PR reply to kyuyeunk | Posted on PR |
| RoPE position investigation | Fix degrades performance, do not apply |
| Can `sharded_flash_attention` replace `dflash_concat_attention` | Possible but not worth the complexity |
| vLLM pipeline benchmark | Working recipe established, baseline acceptance rate 3.53 |
