# Doc 25: Research Directions — Speculative Decoding, Diffusion Models, TPU & Efficient Inference

## Context

The DFlash TPU replication (Docs 00–24) accomplished two things: (1) a working port of
diffusion-based speculative decoding to TPU achieving 94% of GPU draft quality, and
(2) deep operational understanding of how speculative decoding, diffusion-style drafting,
KV cache management, and the vLLM serving pipeline interact on TPU hardware.

This document captures research directions that build on that foundation. Directions 1
and 2 are identified as near-term priorities.

---

## What the DFlash Port Uniquely Revealed

These observations emerged from hands-on porting work and are not well-documented in the
existing literature. They motivate the research directions that follow.

### 1. The pipeline overhead problem is massive and underreported

Standalone DFlash achieves tau=6.67 (94% of GPU). Inside the vLLM pipeline, tau drops to
4.48 — a **33% quality loss** from orchestration alone (scheduling, batch management,
rejection sampling loop). Every published speculative decoding paper reports standalone
numbers. The production reality is significantly worse, and nobody has systematically
studied this gap.

### 2. Stateful drafters are fragile in production pipelines

The seq_len inflation bug (Doc 21) was invisible for months because EAGLE-3 (stateless
drafter) was unaffected. DFlash (stateful — persistent context buffer, KV cache across
iterations) broke silently. The bug corrupted three things simultaneously: context buffer
content, KV cache positions, and RoPE embeddings. This reveals an underexplored
reliability dimension: **stateful draft models amplify pipeline bugs that stateless
drafters silently tolerate**.

### 3. TPU naturally favors diffusion-style drafting

The parallel block prediction in DFlash maps directly to TPU's MXU (128x128 bf16 matrix
multiply tiles). Without any TPU-specific optimization, we hit 94% of GPU quality. The
MXU is severely underutilized during standard single-token autoregressive decode (tiny
matrices), but diffusion-style block prediction presents large, regular matrices that
the MXU can saturate. There is likely significant headroom from TPU-aware design.

### 4. Static vs paged KV cache matters for draft models

The DFlash port required redesigning the draft model's KV cache from paged (the
tpu-inference default, optimized for target models) to static with `dynamic_update_slice`.
Paged attention assumes large, long-lived caches; draft models have small, frequently
reset caches. The interaction between cache strategy and draft model architecture is
understudied and has real performance implications.

### 5. Non-causal attention is a first-class requirement for diffusion drafters

DFlash's draft attention is explicitly non-causal — all block positions attend to each
other and to context simultaneously. The entire TPU inference stack (ragged paged
attention) is built around causal attention. Integrating non-causal attention required
using the TPU `flash_attention` kernel with `causal=False` and static caches, bypassing
the standard attention path entirely. Any future diffusion-based drafter on TPU will face
this same architectural constraint.

---

## Direction 1: Closing the Pipeline Gap (Near-Term Priority)

### Problem

Every speculative decoding paper reports standalone numbers. In production serving
frameworks (vLLM, TensorRT-LLM, SGLang), 30-50% of that speedup is lost to scheduling,
batch management, and rejection sampling overhead. This gap is:

- **Large**: tau 6.67 standalone vs 4.48 in vLLM pipeline (our measured data)
- **Universal**: affects all speculative decoding methods, not just DFlash
- **Understudied**: no published work systematically quantifies or addresses it

### Core Idea

Design a speculative decoding execution model that is aware of the draft-verify-accept
cycle from the ground up, rather than bolting speculative decoding onto a serving
framework designed for vanilla autoregressive inference.

### TPU-Specific Opportunities

TPU has properties that GPU-centric pipelines cannot exploit:

1. **XLA compilation**: The entire draft-verify-accept loop can potentially be fused
   into a single compiled graph, eliminating host round-trips between stages. On GPU,
   each stage typically involves separate CUDA kernel launches with host-side orchestration
   between them.

2. **Deterministic execution**: TPU has no equivalent of CUDA stream synchronization
   overhead. Execution timing is predictable, which means the pipeline scheduler can
   make tighter decisions about when to draft vs verify.

3. **Static shapes**: TPU's XLA compiler strongly prefers static tensor shapes. The
   speculative decoding loop has inherently dynamic shapes (variable acceptance length),
   which currently forces recompilation or padding. A pipeline designed around TPU's
   static-shape preference could use fixed-size buffers with masking, avoiding
   recompilation entirely.

4. **Megacore utilization**: TPU v4+ has two MXU cores per chip (Megacore). During
   single-token decode, one core is often idle. A fused spec-decode loop could pipeline
   draft model execution on one core while the target model runs on the other.

### Concrete Experiments

1. **Measure the breakdown**: Profile exactly where time is spent in the vLLM pipeline
   during speculative decoding. Categorize overhead into: host-side scheduling,
   device-host transfers, rejection sampling, shape dynamism / recompilation, idle time
   between stages.

2. **XLA-fused spec decode loop**: Implement the draft-verify-accept cycle as a single
   `jax.lax.while_loop` that stays on-device, eliminating host round-trips. Compare
   tau and wall-clock speedup against the current host-orchestrated pipeline.

3. **Static-shape pipeline**: Replace dynamic acceptance-length handling with fixed-size
   buffers and bitmask-based selection. Measure compilation time reduction and per-step
   latency improvement.

4. **Comparison with GPU pipeline overhead**: Run the same DFlash model on GPU through
   the vLLM pipeline and measure the standalone-vs-pipeline gap there. Quantify whether
   the gap is TPU-specific or universal.

### Expected Outcome

A system-level characterization of speculative decoding pipeline overhead on TPU, plus
a prototype fused execution model that recovers a significant fraction of the
standalone-to-pipeline gap. If the XLA-fused loop recovers even half the gap (tau from
4.48 toward 5.5+), that would be a strong systems contribution.

### Baselines

- Standalone DFlash on TPU: tau=6.67, 4.26x speedup (our Doc 22 numbers)
- vLLM pipeline DFlash on TPU: tau=4.48, 2.31x speedup (our Doc 21 numbers)
- GPU paper standalone: tau=7.07, 5.53x speedup

---

## Direction 2: Iterative Refinement Drafting on TPU (Near-Term Priority)

### Problem

DFlash (and all current diffusion-based drafters) use **one denoising step**: a single
forward pass through the draft model to predict the entire block. Draft quality is
fundamentally limited by what one pass can capture. But adding more denoising steps
would normally slow down drafting, negating the parallelism advantage over autoregressive
drafters.

### Core Idea

On TPU, the MXU is massively underutilized during the single-token target decode phase
(small matrix multiplies on hardware optimized for 128x128 tiles). The draft model's
block prediction presents larger matrices but still underutilizes available compute in
a single pass. The hypothesis: **2-3 refinement passes on the draft block can fit within
the same latency budget as a single pass on GPU**, because:

- The block tensors are already in HBM after the first pass (no reload cost)
- Each refinement pass updates the previous prediction (not starting from noise)
- TPU's XLA compiler can fuse consecutive passes, eliminating launch overhead
- The MXU pipeline stays saturated across refinement steps

### Mechanism

```
Standard DFlash (1 step):
  [noise block] --> draft model --> [predicted tokens] --> verify

Iterative Refinement (k steps):
  [noise block] --> draft model --> [draft v1]
                    draft model --> [draft v2]  (conditioned on v1)
                    draft model --> [draft v3]  (conditioned on v2)
                    --> [predicted tokens] --> verify
```

At each refinement step, the predicted token embeddings from the previous step replace
the noise/mask embeddings at each position. The draft model's non-causal attention allows
all positions to attend to each other's updated predictions, progressively improving
coherence across the block.

### Why This Is Novel

- **Existing diffusion drafters use single-step denoising**: DFlash, SpecDiff-2, FailFast,
  DiffuSpec all use one draft forward pass. None explore multi-step refinement.
- **The reason is latency**: On GPU, each additional pass adds measurable latency. On TPU,
  the XLA fusion and MXU utilization characteristics may change this tradeoff.
- **It is distinct from standard diffusion denoising schedules**: We are not proposing a
  full diffusion process (100s of steps). We propose 2-3 refinement steps on an already
  good single-step prediction, specifically exploiting TPU hardware characteristics.

### Concrete Experiments

1. **Baseline**: Measure per-step draft latency for DFlash on TPU (single forward pass
   through 4 draft layers). Characterize MXU utilization during this pass.

2. **Refinement implementation**: After the first draft pass, feed predicted token
   embeddings back through the draft model (replacing noise tokens), keeping target
   hidden states and context buffer fixed. Measure latency for 1, 2, 3 refinement steps.

3. **Quality vs steps**: For each refinement count, measure tau and per-position acceptance
   rates. The key question: does refinement step 2 improve tau enough to offset its
   latency cost?

4. **XLA fusion**: Implement the multi-step draft as a single fused XLA computation
   (e.g., `jax.lax.fori_loop`). Compare fused vs unfused latency.

5. **Comparison with autoregressive drafters**: If k refinement steps achieve tau close to
   EAGLE-3 but in O(k) parallel passes instead of O(n) sequential passes, that is a
   concrete speedup result.

### Expected Outcome

A characterization of the quality-latency tradeoff for iterative refinement drafting on
TPU. The best case: 2-step refinement improves tau by 10-20% with less than 50% latency
increase (net positive for wall-clock speedup). Even a negative result (refinement doesn't
help) is publishable as it establishes that single-step denoising is optimal for
speculative decoding drafting.

### Connection to Direction 1

Directions 1 and 2 are complementary. Direction 1 reduces pipeline overhead (the gap
between standalone and production). Direction 2 improves standalone draft quality (the
gap between TPU and GPU). Together, they could push TPU speculative decoding performance
significantly beyond the current GPU state-of-the-art.

---

## Direction 3: Hardware-Aware Draft Model Architecture (Future)

### Problem

All existing draft models (EAGLE-3, DFlash) are designed for GPU compute characteristics.
Nobody has designed a draft model architecture specifically for TPU.

### Core Idea

TPU MXU operates on 128x128 bf16 tiles. Design a draft model where:

- Hidden dimensions align with MXU tile sizes (multiples of 128)
- Block size matches TPU's preferred matrix dimensions
- Attention patterns exploit TPU `flash_attention` kernel characteristics

### Why It Matters

From our DFlash port, we observed that the draft model's hidden dimensions (from
GPU-trained checkpoints) don't align with TPU MXU tile boundaries. The XLA compiler
pads to the nearest tile size, wasting compute. A TPU-native architecture would
eliminate this waste.

### Concrete Experiment

Sweep block sizes (8, 16, 32, 64) and hidden dimensions (128, 256, 512, 1024) on TPU,
measuring not just tau but actual wall-clock speedup. Compare with the same sweep on GPU
to identify where optimal architectures diverge.

---

## Direction 4: Hybrid Adaptive Drafting (Future)

### Problem

N-gram drafting is nearly free but low quality. Diffusion drafting is high quality but
costs one forward pass. Fixed strategies waste compute in easy regions and underperform
in hard regions.

### Core Idea

Adaptively switch between drafting strategies based on prediction difficulty:

- **Easy regions** (repetitive code, formulaic math steps): n-gram lookup, zero neural
  compute
- **Hard regions** (novel reasoning, creative text): diffusion draft
- **Difficulty signal**: Target model's entropy or hidden-state variance from the
  auxiliary layers already extracted for DFlash (layers 1, 9, 17, 25, 33)

### TPU Consideration

TPU's static compilation model makes runtime branching expensive. Implementation would
likely run both strategies in parallel and select outputs via masking, rather than
conditionally executing one path.

---

## Direction 5: Disaggregated Speculative Decoding on TPU Pods (Future)

### Problem

Current speculative decoding runs draft and target on the same device(s). The
draft-verify-accept cycle is strictly serial.

### Core Idea

On TPU pods with high-bandwidth ICI (inter-chip interconnect), dedicate some chips to
continuous drafting and others to verification, converting the serial cycle into a
pipeline:

```
Draft chips:   [draft block 1] [draft block 2] [draft block 3] ...
Target chips:         [verify 1]       [verify 2]       [verify 3] ...
```

This could break the fundamental latency bound of speculative decoding (one draft + one
verify per step) by overlapping iterations. tpu-inference already supports multi-host
disaggregated serving, providing infrastructure to build on.

---

## Prioritization

| Direction | Priority | Feasibility | Novelty | Builds On |
|---|---|---|---|---|
| 1. Pipeline gap | **Near-term** | High (have baselines + code) | High (nobody published this) | Docs 21-23 measurements |
| 2. Iterative refinement | **Near-term** | High (modify existing DFlash) | High (no prior work) | DFlash model + standalone benchmark |
| 3. HW-aware architecture | Future | Medium (needs training) | Medium | DFlash port observations |
| 4. Hybrid adaptive | Future | Medium (engineering) | Medium | N-gram + DFlash both available |
| 5. Disaggregated | Future | Lower (multi-host setup) | High | tpu-inference infra |

Directions 1 and 2 are complementary: Direction 1 closes the gap between standalone and
production serving. Direction 2 closes the gap between TPU and GPU standalone quality.
Together they attack the problem from both ends.

---

*Created: February 2026*
*Status: Exploration phase — Directions 1 and 2 prioritized for immediate investigation*
