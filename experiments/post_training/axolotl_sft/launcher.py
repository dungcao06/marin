# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Axolotl SFT on Iris from the pinned ``marin-community/axolotl`` fork.

The fork pin, the transformers pin, and the supported architectures are explicit;
the pod-side entrypoint clones the fork at the pinned revision, writes the config,
runs ``axolotl`` training, verifies the saved tokenizer embeds the configured chat
template (the fork's ``template_integrity`` contract), and writes a run manifest.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Any

import yaml
from fray.iris_backend import convert_resources, resolve_coscheduling, wrap_multiprocess
from fray.types import ResourceConfig
from iris.client.client import IrisClient
from iris.cluster.types import Entrypoint, EnvironmentSpec
from iris.rpc import job_pb2
from rigging.timing import Duration
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

AXOLOTL_FORK_REPO = "https://github.com/marin-community/axolotl"
# Fork main at the time of this launcher; MARIN_FORK.md records the fork point and
# the ported features (delphi template, template_integrity, mfu, supabase_register).
AXOLOTL_FORK_REVISION = "a208d55050da5794098c183f7ac474e89c3f6859"
AXOLOTL_TRANSFORMERS_PIN = "5.12.1"
# Architectures the fork's transformers pin must ship for the Phase 0 bases.
SUPPORTED_ARCHITECTURES = ("qwen3_moe", "qwen3_5_moe", "nemotron_h")
EXPERT_PARALLEL_INTEGRATION = "integrations/expert_parallel"


@dataclass(frozen=True)
class AxolotlOptimizer:
    """Explicit optimizer settings; the launcher refuses defaults."""

    learning_rate: str  # axolotl YAML form, e.g. "5e-6"
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    lr_scheduler: str = "linear"
    warmup_steps: int = 0

    def __post_init__(self) -> None:
        # axolotl's YAML takes the learning rate as a string like "5e-6"; float() reads it back.
        if float(self.learning_rate) <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate!r}")
        if not 0 <= self.adam_beta1 < 1 or not 0 <= self.adam_beta2 < 1:
            raise ValueError("adam betas must lie in [0, 1)")


@dataclass(frozen=True)
class AxolotlBaseModel:
    """A Phase 0 base checkpoint with its architecture and template provenance."""

    name: str
    hf_id: str
    revision: str
    architecture: str
    # Whether the repo's tokenizer ships a chat template. Bases without one require
    # an explicit template on the spec (the config builder refuses otherwise).
    tokenizer_has_template: bool


@dataclass(frozen=True)
class AxolotlSpec:
    """One Axolotl SFT run: pinned base, data manifest, template, optimizer."""

    name: str
    base: AxolotlBaseModel
    # Path (or glob) of the CL12 data manifest's jsonl files, readable from the pod.
    data_manifest: str
    optimizer: AxolotlOptimizer
    sequence_len: int = 32768
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 64
    epochs: int = 1
    # None trains on the base tokenizer's builtin template. A jinja string trains
    # through a template file the entrypoint materializes next to the config.
    chat_template: str | None = None
    # One conversation per sequence: the Stage 0 budget is matched in examples, and
    # the #6839 masking canary is defined on unpacked sequences.
    sample_packing: bool = False
    expert_parallel: bool = False
    # Initialize weights from this checkpoint path (a previous curriculum stage's
    # export) instead of the base download; the base still proves tokenizer/template.
    init_checkpoint: str | None = None

    def __post_init__(self) -> None:
        if self.base.architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"architecture {self.base.architecture!r} is outside the fork's supported set "
                f"{SUPPORTED_ARCHITECTURES}"
            )
        if self.base.tokenizer_has_template is False and self.chat_template is None:
            raise ValueError(
                f"base {self.base.name!r} ships no chat template; supply an explicit one "
                "(the published template from the model family, per the train-equals-serve rule)"
            )
        if self.sequence_len <= 0 or self.gradient_accumulation_steps <= 0:
            raise ValueError("sequence_len and gradient_accumulation_steps must be positive")
        if self.expert_parallel:
            raise ValueError(
                f"the pinned fork ({AXOLOTL_FORK_REVISION}) ships no {EXPERT_PARALLEL_INTEGRATION}; "
                "run data/tensor parallel on one node, or add expert parallelism to the fork in its "
                "own worktree and PR it first"
            )


def axolotl_config(spec: AxolotlSpec, *, template_path: str | None = None) -> dict[str, Any]:
    """The fork's training YAML for ``spec``.

    ``template_path`` points at the materialized jinja when the spec carries a custom
    template; builtin templates resolve from the base tokenizer at train time.
    ``tokenizer_save_jinja_files: false`` activates the fork's template_integrity
    guard: the saved checkpoint must embed the chat template in
    ``tokenizer_config.json``, which is the serving stack's read path.
    """
    if spec.chat_template is not None and template_path is None:
        raise ValueError("a spec with a custom chat_template needs the materialized template_path")
    config: dict[str, Any] = {
        "base_model": (
            f"{spec.base.hf_id}@{spec.base.revision}" if spec.init_checkpoint is None else spec.init_checkpoint
        ),
        "sequence_len": spec.sequence_len,
        "sample_packing": spec.sample_packing,
        # template_integrity: the saved checkpoint embeds its chat template.
        "tokenizer_save_jinja_files": False,
        "tokenizer_use_fast": True,
        "train_on_inputs": False,
        "datasets": [{"path": spec.data_manifest, "type": "chat_template"}],
        "num_epochs": spec.epochs,
        "per_device_train_batch_size": spec.per_device_batch_size,
        "gradient_accumulation_steps": spec.gradient_accumulation_steps,
        "optimizer": "adamw_torch_fused",
        "learning_rate": spec.optimizer.learning_rate,
        "adam_beta1": spec.optimizer.adam_beta1,
        "adam_beta2": spec.optimizer.adam_beta2,
        "weight_decay": spec.optimizer.weight_decay,
        "lr_scheduler": spec.optimizer.lr_scheduler,
        "warmup_steps": spec.optimizer.warmup_steps,
        "output_dir": spec.name,
        "seed": 42,
        "logging_steps": 1,
    }
    if template_path is not None:
        config["chat_template"] = template_path
    return config


def axolotl_launch_command(
    spec: AxolotlSpec,
    *,
    output_dir: str,
    nodes: int = 1,
    gpus_per_node: int = 8,
    max_steps: int | None = None,
) -> str:
    """The exact shell command a launch runs; the 20-step smoke passes ``max_steps``."""
    overrides = [f"max_steps={max_steps}"] if max_steps is not None else []
    override_args = " ".join(overrides)
    torchrun = f"torchrun --nnodes={nodes} --nproc-per-node={gpus_per_node} " if nodes > 1 else ""
    clone = f"git clone {AXOLOTL_FORK_REPO} && cd axolotl && git checkout {AXOLOTL_FORK_REVISION} && pip install -e ."
    return f"{clone} && cd .. && {torchrun}python -m axolotl.cli.train {output_dir}/axolotl_config.yaml {override_args}"


@dataclass(frozen=True)
class AxolotlLaunchPayload:
    """Everything the pod-side entrypoint needs, JSON-serializable."""

    config: dict[str, Any]
    chat_template: str | None
    fork_revision: str
    output_root: str
    manifest: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True)


def run_axolotl_training(payload_json: str) -> None:
    """Pod-side entrypoint: clone the pinned fork, train, verify the template, manifest.

    The fork is cloned at the pinned revision and installed into the pod environment;
    nothing about the training stack is inherited from the marin image except CUDA
    torch. The saved tokenizer must embed the chat template (``tokenizer_config.json``)
    — the fork's ``template_integrity`` contract — or the job fails after training
    rather than exporting a checkpoint the serving stack cannot read.
    """
    payload = AxolotlLaunchPayload(**json.loads(payload_json))
    workdir = os.environ.get("AXOLOTL_WORKDIR", os.getcwd())
    fork_dir = os.path.join(workdir, "axolotl")
    if not os.path.exists(os.path.join(fork_dir, ".git")):
        subprocess.run(
            ["git", "clone", AXOLOTL_FORK_REPO, fork_dir],
            check=True,
        )
    subprocess.run(["git", "checkout", payload.fork_revision], cwd=fork_dir, check=True)
    subprocess.run(["pip", "install", "-e", "."], cwd=fork_dir, check=True)

    template_path = None
    if payload.chat_template is not None:
        template_path = os.path.join(workdir, "chat_template.jinja")
        with open(template_path, "w") as handle:
            handle.write(payload.chat_template)
    config_path = os.path.join(workdir, "axolotl_config.yaml")
    with open(config_path, "w") as handle:
        yaml.safe_dump(payload.config, handle, sort_keys=False)

    subprocess.run(
        ["python", "-m", "axolotl.cli.train", config_path],
        check=True,
        cwd=workdir,
    )
    _verify_template_integrity(payload, workdir)

    manifest_path = os.path.join(workdir, "axolotl_run_manifest.json")
    with open(manifest_path, "w") as handle:
        json.dump(payload.manifest, handle, indent=2, sort_keys=True)
    logger.info("axolotl run manifest written to {}", manifest_path)


def _verify_template_integrity(payload: AxolotlLaunchPayload, workdir: str) -> None:
    """The fork's template_integrity contract, re-checked outside the save path."""
    output_dir = os.path.join(workdir, payload.config["output_dir"])
    embedded = getattr(AutoTokenizer.from_pretrained(output_dir), "chat_template", None)
    if embedded is None:
        raise ValueError(
            f"saved checkpoint at {output_dir} embeds no chat template; "
            "the serving stack would fail (the template_integrity footgun)"
        )
    if payload.chat_template is not None and embedded != payload.chat_template:
        raise ValueError("the embedded chat template differs from the configured one")


def launch_axolotl(
    spec: AxolotlSpec,
    *,
    endpoint_url: str,
    output_root: str,
    nodes: int = 1,
    gpus_per_node: int = 8,
    priority_band: int = job_pb2.PRIORITY_BAND_INHERIT,
) -> str:
    """Submit one Axolotl SFT job to Iris and return the job id."""
    template_path = None
    if spec.chat_template is not None:
        template_path = os.path.join(output_root, "chat_template.jinja")
        io_mkdir_and_write(template_path, spec.chat_template)
    config = axolotl_config(spec, template_path=template_path)
    manifest = {
        "spec_name": spec.name,
        "base": {"hf_id": spec.base.hf_id, "revision": spec.base.revision, "arch": spec.base.architecture},
        "data_manifest": spec.data_manifest,
        "fork_revision": AXOLOTL_FORK_REVISION,
        "transformers_pin": AXOLOTL_TRANSFORMERS_PIN,
        "optimizer": {
            "learning_rate": spec.optimizer.learning_rate,
            "weight_decay": spec.optimizer.weight_decay,
            "adam_beta1": spec.optimizer.adam_beta1,
            "adam_beta2": spec.optimizer.adam_beta2,
            "lr_scheduler": spec.optimizer.lr_scheduler,
            "warmup_steps": spec.optimizer.warmup_steps,
        },
        "sequence_len": spec.sequence_len,
        "batch": {
            "per_device": spec.per_device_batch_size,
            "gradient_accumulation": spec.gradient_accumulation_steps,
            "nodes": nodes,
            "gpus_per_node": gpus_per_node,
        },
        "epochs": spec.epochs,
        "sample_packing": spec.sample_packing,
        "chat_template": "builtin" if spec.chat_template is None else "custom",
    }
    payload = AxolotlLaunchPayload(
        config=config,
        chat_template=spec.chat_template,
        fork_revision=AXOLOTL_FORK_REVISION,
        output_root=output_root,
        manifest=manifest,
    )

    resources = ResourceConfig.with_gpu(
        "H100",
        count=gpus_per_node,
        cpu=_GPU_HOST_CPU_PER_DEVICE * gpus_per_node,
        ram=f"{_GPU_HOST_RAM_GB_PER_DEVICE * gpus_per_node}g",
        disk=f"{_GPU_HOST_DISK_GB_PER_DEVICE * gpus_per_node}g",
        replicas=nodes,
    )
    native_resources = convert_resources(resources)
    command = Entrypoint(
        command=["python", "-m", "experiments.post_training.axolotl_sft.launcher", "--payload", "launch-payload.json"],
        workdir_files={"launch-payload.json": payload.to_json().encode()},
    )
    marin_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    environment = EnvironmentSpec(
        env_vars={
            "GIT_COMMIT": marin_commit,
            "AXOLOTL_WORKDIR": os.path.join(output_root, spec.name),
            "WANDB_MODE": "disabled",
        },
        extras=["gpu"],
    )
    with IrisClient.remote(endpoint_url) as client:
        job = client.submit(
            wrap_multiprocess(command, native_resources, processes_per_task=1),
            name=f"axolotl-{spec.name}",
            resources=native_resources,
            replicas=nodes,
            environment=environment,
            constraints=(
                convert_resources(resources).constraints
                if hasattr(convert_resources(resources), "constraints")
                else None
            ),
            coscheduling=resolve_coscheduling(resources, nodes),
            scheduling_timeout=Duration.from_hours(24),
            timeout=Duration.from_hours(72),
            max_retries_failure=0,
            priority_band=priority_band,
            existing_job_policy=job_pb2.EXISTING_JOB_POLICY_ERROR,
        )
    logger.info("submitted axolotl run %s as job %s", spec.name, job)
    return str(job)


def io_mkdir_and_write(path: str, content: str) -> None:
    """Write a small text artifact, creating parent directories."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as handle:
        handle.write(content)


_GPU_HOST_CPU_PER_DEVICE = 16
_GPU_HOST_RAM_GB_PER_DEVICE = 220
_GPU_HOST_DISK_GB_PER_DEVICE = 512


def __main__() -> None:  # pragma: no cover - pod entrypoint
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    args = parser.parse_args()
    with open(args.payload) as handle:
        run_axolotl_training(handle.read())


if __name__ == "__main__":
    __main__()
