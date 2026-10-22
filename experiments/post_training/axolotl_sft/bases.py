# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The three Phase 0 Axolotl base checkpoints, pinned.

Revisions are the HF repo heads at authoring time (2026-10-22); every run manifest
records them. The Nemotron base ships no chat template, so its runs must supply one
explicitly — the published template from the model family, per the train-equals-serve
rule (#7427).
"""

from experiments.post_training.axolotl_sft.launcher import AxolotlBaseModel

QWEN3_30B_A3B_BASE = AxolotlBaseModel(
    name="qwen3-30b-a3b-base",
    hf_id="Qwen/Qwen3-30B-A3B-Base",
    revision="1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9",
    architecture="qwen3_moe",
    tokenizer_has_template=True,
)

QWEN35_35B_A3B_BASE = AxolotlBaseModel(
    name="qwen3.5-35b-a3b-base",
    hf_id="Qwen/Qwen3.5-35B-A3B-Base",
    revision="0f0813072d2358973511097385626f21fcb6d422",
    architecture="qwen3_5_moe",
    tokenizer_has_template=True,
)

NEMOTRON_3_NANO_30B_A3B_BASE = AxolotlBaseModel(
    name="nemotron-3-nano-30b-a3b-base",
    hf_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16",
    revision="e0ac9ee3dfd02be21b5479edfc2f671ed269d0a2",
    architecture="nemotron_h",
    tokenizer_has_template=False,
)

# The base ships no chat template; the published family template comes from the
# reasoning sibling (pinned), per the train-equals-serve rule (#7427).
NEMOTRON_TEMPLATE_SOURCE = (
    "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
    "e5e9932441de940c9a62185c870ea5bcd4cd24e2",
)

AXOLOTL_BASES: dict[str, AxolotlBaseModel] = {
    base.name: base for base in (QWEN3_30B_A3B_BASE, QWEN35_35B_A3B_BASE, NEMOTRON_3_NANO_30B_A3B_BASE)
}
