Sorry about the large PR. The model, proposer, and attention kernel are tightly coupled (proposer calls model forward, model uses the attention kernel), so splitting them further would leave each PR non-functional on its own. All files here are new additions with no changes to existing code, which should make it easier to review.

Broke the original PR down into 3:

1. **This PR (updated):** DFlash model, proposer, and unit tests -- all new files, no existing files modified
2. Pipeline integration -- modifications to existing files (speculative_decoding_manager, kv_cache_manager, qwen3, tpu_runner, model_loader)
3. E2E tests + Buildkite CI

PRs 2 and 3 coming shortly.
