# Questions for Yiming

Copy-paste-ready message below.

---

Hi Yiming,

Hope you're doing well! We had a few questions about the direction of our report and the remaining weeks. Would appreciate your thoughts when you get a chance.

**1. Scope for the next 4 weeks**

Right now we've completed the DFlash port to TPU (tau=6.67, 94% of GPU paper quality). Should we spend the remaining time investigating diffusion language models more broadly on TPU (e.g., Fast-dLLM, FailFast), or would it be better to focus on polishing what we have (cleaning up the tpu-inference PR, running TPU v5 benchmarks, concurrent serving experiments)?

**2. Framing the TPU + diffusion claim**

We originally wrote the report around the idea that "TPU is well-suited for diffusion-based speculative decoding." After doing a thorough literature check, we couldn't find any published evidence supporting this for language models specifically. The closest reference is an image DiT paper (arXiv:2503.00461) showing compute-bound workloads scale better on TPU, but nothing for discrete diffusion LLMs. We've revised the Future Work section to frame this as an open hypothesis rather than an established fact. Is that the right call, or should we remove the diffusion angle from the framing entirely?

**3. Report title direction**

Our current title is: "Toward Diffusion-Accelerated LLM Inference on TPU: Porting and Evaluating Block Diffusion Speculative Decoding"

We're debating between two directions:
- **Option A (grounded):** Focus on what we actually accomplished, e.g., "Porting Block Diffusion Speculative Decoding to TPU: Evaluating DFlash on Google Cloud TPU with JAX"
- **Option B (forward-looking):** Keep the current "Toward..." title, treating the diffusion+TPU intersection as an open research question

Which framing works better for the showcase/submission?

Thanks!
Aaron
