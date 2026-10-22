# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Axolotl Stage 0 configs (CL12): the same three stages on the three MoE bases.

Each base runs the #8225 curriculum in order — chat, thinking, terminal — with the
same per-example budget as the Snowball run: global batch 64 examples per step
(8 per device x 8 GPUs), one epoch per source, fresh optimizer per stage.
"""

from transformers import AutoTokenizer

from experiments.post_training.axolotl_sft.bases import AXOLOTL_BASES, NEMOTRON_TEMPLATE_SOURCE
from experiments.post_training.axolotl_sft.launcher import AxolotlOptimizer, AxolotlSpec
from experiments.post_training.stage0.templates import preserve_intermediate_reasoning

# #8225: Adam at 5e-6, global batch 64 examples, one epoch per source.
LEARNING_RATE = "5e-6"
GLOBAL_BATCH_EXAMPLES = 64
SEQUENCE_LEN = 32768

STAGE_ORDER = ("wildchat-rewild", "nemotron-science-reasoning", "nemotron-terminal")

_STAGE_SOURCE_FILES = {
    "wildchat-rewild": "wildchat-rewild.jsonl.gz",
    "nemotron-science-reasoning": "nemotron-science-reasoning.jsonl.gz",
    "nemotron-terminal": "nemotron-terminal.jsonl.gz",
}


def _chat_template_for(base) -> str | None:
    """The base's training template with intermediate reasoning preserved."""

    if base.tokenizer_has_template:
        template = AutoTokenizer.from_pretrained(base.hf_id, revision=base.revision).chat_template
    else:
        template_repo, template_revision = NEMOTRON_TEMPLATE_SOURCE
        template = AutoTokenizer.from_pretrained(template_repo, revision=template_revision).chat_template
        if template is None:
            raise ValueError(f"{template_repo}@{template_revision} carries no chat template")
    return preserve_intermediate_reasoning(template)


def stage_specs(base_name: str, *, data_root: str, output_root: str) -> list[AxolotlSpec]:
    """The three chained stage specs for one base, in curriculum order.

    Stage 2 and 3 initialize from the previous stage's exported checkpoint directory
    (the fork's ``base_model`` accepts a local path), keeping the fresh-optimizer
    per-stage rule. Global batch 64 = 8 per device x 8 GPUs x accumulation 1, so the
    per-example budget matches the Snowball Stage 0 run exactly.

    Qwen bases train on their published template with the reasoning gate widened so
    every assistant turn renders its think block (CL12's render check enforces it);
    the Nemotron base ships no template and fetches the pinned published family
    template (``NEMOTRON_TEMPLATE_SOURCE``) at config time.
    """
    base = AXOLOTL_BASES[base_name]
    optimizer = AxolotlOptimizer(learning_rate=LEARNING_RATE, weight_decay=0.0)
    chat_template = _chat_template_for(base)
    specs: list[AxolotlSpec] = []
    init_from: str | None = None
    for stage_index, source in enumerate(STAGE_ORDER):
        stage_output = f"{output_root}/{base_name}/s{stage_index + 1}-{source}"
        specs.append(
            AxolotlSpec(
                name=f"stage0-{base_name}/s{stage_index + 1}-{source}",
                base=base,
                data_manifest=f"{data_root}/{_STAGE_SOURCE_FILES[source]}",
                optimizer=optimizer,
                sequence_len=SEQUENCE_LEN,
                per_device_batch_size=8,
                gradient_accumulation_steps=1,
                epochs=1,
                chat_template=chat_template,
                sample_packing=False,
                init_checkpoint=init_from,
            )
        )
        init_from = stage_output
    return specs
