# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DFlash proposer for speculative decoding on JAX/TPU.

Uses the same proposal path as Eagle3; the draft model is selected by the
model loader (Qwen3DFlashForCausalLM) when method=\"dflash\" and is_draft_model=True.
"""

from tpu_inference.spec_decode.jax.eagle3 import Eagle3Proposer


class DFlashProposer(Eagle3Proposer):
    """Proposer for DFlash speculative decoding.

    Same interface and proposal loop as Eagle3Proposer. The runner and
    model_loader ensure the DFlash draft model (Qwen3DFlashForCausalLM) is
    loaded when method is \"dflash\".
    """

    pass
