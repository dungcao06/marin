# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snowball Stage 0 (CL12): the three-stage #8225 curriculum on the 262k base.

Chat, then thinking, then the terminal protocol — each stage initializes from the
previous checkpoint with a fresh optimizer, per the #8225 stage rule. The base is the
current 262k Snowball, `open-athena/snowball-67b-a2b-base-262k-qk175`, pinned. PR
#8172 (the original launcher) was closed unmerged; this config follows
`experiments/sft/configs/snowball_h100.py` against the current `experiments/sft/launcher.py`.
"""

from levanter.optim.config import AdamConfig
from levanter.utils.mesh import MeshConfig
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from marin.experiment.checkpoints import hf_to_levanter

from experiments.post_training.stage0.convert import STAGE_ORDER
from experiments.sft.launcher import (
    ConvertedCheckpointModel,
    DatasetSpec,
    LevanterCheckpointModel,
    SFTSpec,
    resources_from_accelerator,
    sft_step,
)

STAGE0_VERSION = "2026.10.22"
# The design decision log (2026-10-22): the current 262k base, not the 5.7T cooldown.
BASE_HF_ID = "open-athena/snowball-67b-a2b-base-262k-qk175"
BASE_REVISION = "11c9430da3075ef4596f92ee0af142d9d08215c1"
# #8225: 64 H100s, global batch 64 sequences of 32,768 tokens, Adam at 5e-6.
BATCH_SIZE = 64
SEQ_LEN = 32768
LEARNING_RATE = 5e-6
TRAIN_NODES = 8

# The CL12 data manifest's converted sources, in stage order. The manifest paths are
# bound by the pipeline that runs the conversion; these placeholders carry the slugs.
_STAGE_DATASETS: dict[str, DatasetSpec] = {
    "wildchat-rewild": DatasetSpec(
        slug="stage0-wildchat-rewild",
        hf_dataset_id="nyu-dice-lab/wildchat50m-rewild-sft-385700",
        revision="46a5bb5",
        adapter_kwargs={"conversation_column": "conversation"},
        weight=1.0,
    ),
    "nemotron-science-reasoning": DatasetSpec(
        slug="stage0-nemotron-science-reasoning",
        hf_dataset_id="laion/llama-nemotron-science-reasoning-on-canonical-think-full",
        revision="bae881d",
        adapter_kwargs={"conversation_column": "messages"},
        weight=1.0,
    ),
    "nemotron-terminal": DatasetSpec(
        slug="stage0-nemotron-terminal",
        hf_dataset_id="nvidia/Nemotron-Terminal-Corpus",
        revision="a1667c4",
        adapter_kwargs={"conversation_column": "conversations"},
        weight=1.0,
    ),
}


def _stage_optimizer() -> AdamConfig:
    """#8225's optimizer: Adam at 5e-6, fresh at every stage."""
    return AdamConfig(
        learning_rate=LEARNING_RATE,
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
        weight_decay=0.0,
        max_grad_norm=1.0,
        lr_schedule="cosine",
        warmup=0.0,
    )


def _mesh() -> MeshConfig:
    """Snowball shards parameters over expert x (data, context), as in #8225."""
    return MeshConfig(
        axes={"data": 1, "replica": 1, "model": 1, "expert": -1},
        dcn_axes={"context": TRAIN_NODES},
        compute_mapping={
            "batch": ["replica_dcn", "data", "expert"],
            "position": "context",
            "vocab": "model",
        },
    )


def snowball_stage0_steps():
    """The three Stage 0 steps, initialized in sequence; stage 3 is the run's ``B``.

    Each stage trains on exactly one source for one epoch of that source (#8225's
    per-stage schedule), so the steps resolve their counts from the token totals and
    the launcher threads the chat caches as dependencies.
    """

    def stage(name_suffix: str, source: str, model) -> SFTSpec:
        return SFTSpec(
            name=f"checkpoints/stage0-snowball/{name_suffix}",
            version=STAGE0_VERSION,
            model=model,
            chat_template=MARIN_CHAT_TEMPLATE,
            datasets=[_STAGE_DATASETS[source]],
            optimizer=_stage_optimizer(),
            mesh=_mesh(),
            seq_len=SEQ_LEN,
            pack=False,
            batch_size=BATCH_SIZE,
            num_train_epochs=1,
            hf_save_dtype="bfloat16",
            wandb_project="marin_moe_sft",
        )

    base_conversion = hf_to_levanter(
        BASE_HF_ID,
        model_type="snowball",
        hf_revision=BASE_REVISION,
        version=STAGE0_VERSION,
        tokenizer=BASE_HF_ID,
        resources=resources_from_accelerator("8xH100"),
    )
    stage1 = sft_step(
        stage(
            "s1-chat",
            STAGE_ORDER[0],
            ConvertedCheckpointModel(conversion=base_conversion),
        ),
        resources_from_accelerator("8xH100"),
    )
    stage2 = sft_step(
        stage(
            "s2-thinking",
            STAGE_ORDER[1],
            LevanterCheckpointModel(init_from=stage1, model=base_conversion.model, tokenizer_path=BASE_HF_ID),
        ),
        resources_from_accelerator("8xH100"),
    )
    stage3 = sft_step(
        stage(
            "s3-terminal",
            STAGE_ORDER[2],
            LevanterCheckpointModel(init_from=stage2, model=base_conversion.model, tokenizer_path=BASE_HF_ID),
        ),
        resources_from_accelerator("8xH100"),
    )
    return {"s1-chat": stage1, "s2-thinking": stage2, "s3-terminal": stage3}
