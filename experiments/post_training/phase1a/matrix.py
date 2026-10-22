# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 1a's run matrix and manifests (WP10).

Thirty-two Phase 1a runs — M1 and M4 over the five task conditions x 3 seeds, plus
one procedure null per method — and the eight Phase 0c LR-sweep runs. Each run is a
MarinSkyRL override table layered on the Snowball OPD base config plus a manifest
recording every fixed parameter, the seed, the data and expert revisions, and the
Marin commit. The M1 overrides satisfy WP1's fixed-completion gates; the M4
overrides ride the existing Snowball OPD config at the Phase 1a fixed parameters.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

# The Phase 1a fixed-parameter table (DESIGN, fixed parameters).
SEQUENCES_PER_STEP = 32
M4_ROLLOUTS_PER_PROMPT = 4
RESPONSE_CAP_TOKENS = 4096
PROMPT_CAP_TOKENS = 32768
M1_LEARNING_RATE = 1e-5
M4_LEARNING_RATE = 1e-5
SEEDS = (0, 1, 2)

CONDITIONS = ("theta_X", "theta_R", "theta_XR", "theta_X_prime", "theta_R_prime")
METHODS = ("M1", "M4")

# Data and expert revisions the manifest must record (pinned by the WP6/WP7 builds).
DATA_REVISIONS = {
    "wildchat_rewild": "46a5bb5",
    "nemotron_science_reasoning": "bae881d",
    "nemotron_terminal": "a1667c4",
}
EXPERT_REVISIONS = {"E_X": "pinned-at-build", "E_R": "pinned-at-build"}
PHASE_1A_STEPS = 100

# Phase 0c: the LR sweep, M1 on X and R at two seeds per point.
LR_SWEEP_POINTS = ("2e-6", "5e-6", "1e-5", "2e-5")
LR_SWEEP_TASKS = ("theta_X", "theta_R")


@dataclass(frozen=True)
class RunSpec:
    """One matrix entry: its identity, the MarinSkyRL override table, and the manifest."""

    name: str
    method: str
    condition: str
    seed: int
    skyrl_overrides: dict
    manifest: dict


def _m1_overrides(condition: str, seed: int, learning_rate: str) -> dict:
    """WP1's fixed-completion path: static rows under the sft loss."""
    return {
        "environment": {"env_class": "fixed_completion"},
        "data": {
            "fixed_completion_format": "text",
            "train_data": [f"{condition}.jsonl"],
        },
        "trainer": {
            "train_batch_size": SEQUENCES_PER_STEP,
            "max_prompt_length": PROMPT_CAP_TOKENS,
            "use_sample_packing": False,
            "algorithm": {
                "policy_loss_type": "sft",
                "advantage_estimator": "uniform",
                "loss_reduction": "token_mean",
                "off_policy_correction": "none",
            },
            "optim": {"lr": learning_rate},
        },
        "generator": {"n_samples_per_prompt": 1, "sampling_params": {"max_generate_length": RESPONSE_CAP_TOKENS}},
        "seed": seed,
    }


def _m4_overrides(condition: str, seed: int, learning_rate: str) -> dict:
    """The OPD path at the Phase 1a fixed parameters, on the Snowball OPD base config.

    Sequences per step count rollouts, so a 32-sequence step is 8 prompts x 4
    rollouts. Packing stays off and every run carries a task-tagged data blend.
    """
    prompts_per_step = SEQUENCES_PER_STEP // M4_ROLLOUTS_PER_PROMPT
    return {
        "environment": {"env_class": "nemotron_ultra"},
        "data": {"train_data": [f"{condition}.jsonl"], "shuffle": True},
        "trainer": {
            "train_batch_size": prompts_per_step,
            "max_prompt_length": PROMPT_CAP_TOKENS,
            "use_sample_packing": False,
            "algorithm": {
                "policy_loss_type": "regular",
                "advantage_estimator": "uniform",
                "loss_reduction": "token_mean",
                "distillation": {"objective": "sampled_reverse_kl"},
                "off_policy_correction": "none",
            },
            "optim": {"lr": learning_rate},
        },
        "generator": {
            "n_samples_per_prompt": M4_ROLLOUTS_PER_PROMPT,
            "sampling_params": {"max_generate_length": RESPONSE_CAP_TOKENS, "temperature": 1.0},
        },
        "seed": seed,
    }


def _matched_fields(overrides: dict) -> dict:
    """The fields every method must agree on for the same condition (control C2)."""
    return {
        "train_batch_size": overrides["trainer"]["train_batch_size"],
        "n_samples_per_prompt": overrides["generator"]["n_samples_per_prompt"],
        "sequences_per_step": overrides["trainer"]["train_batch_size"] * overrides["generator"]["n_samples_per_prompt"],
        "response_cap": overrides["generator"]["sampling_params"]["max_generate_length"],
        "prompt_cap": overrides["trainer"]["max_prompt_length"],
        "packing": overrides["trainer"]["use_sample_packing"],
        "loss_reduction": overrides["trainer"]["algorithm"]["loss_reduction"],
        "advantage_estimator": overrides["trainer"]["algorithm"]["advantage_estimator"],
        "data": overrides["data"]["train_data"],
        "seed": overrides["seed"],
    }


def phase1a_runs(marin_commit: str | None = None) -> list[RunSpec]:
    """The 32 Phase 1a runs: 2 methods x 5 conditions x 3 seeds, plus 2 procedure nulls."""
    commit = marin_commit if marin_commit is not None else _marin_commit()
    runs: list[RunSpec] = []
    for method in METHODS:
        builder = _m1_overrides if method == "M1" else _m4_overrides
        for condition in CONDITIONS:
            for seed in SEEDS:
                overrides = builder(condition, seed, M1_LEARNING_RATE)
                runs.append(
                    RunSpec(
                        name=f"phase1a-{method.lower()}-{condition}-seed{seed}",
                        method=method,
                        condition=condition,
                        seed=seed,
                        skyrl_overrides=overrides,
                        manifest=_manifest(method, condition, seed, overrides, commit, procedure_null=False),
                    )
                )
        # The procedure null: the method's full pipeline with no learning signal —
        # M1 trains on an empty schedule, M4 runs rollouts with the update disabled.
        overrides = builder("theta_X", 0, M1_LEARNING_RATE)
        overrides["trainer"]["max_steps"] = 0 if method == "M1" else overrides["trainer"].get("max_steps", 0)
        overrides["procedure_null"] = True
        runs.append(
            RunSpec(
                name=f"phase1a-{method.lower()}-procedure-null",
                method=method,
                condition="null",
                seed=0,
                skyrl_overrides=overrides,
                manifest=_manifest(method, "null", 0, overrides, commit, procedure_null=True),
            )
        )
    return runs


def phase0c_lr_sweep(marin_commit: str | None = None) -> list[RunSpec]:
    """The eight Phase 0c LR-sweep runs: four LRs x two single-task conditions, M1."""
    commit = marin_commit if marin_commit is not None else _marin_commit()
    runs: list[RunSpec] = []
    for learning_rate in LR_SWEEP_POINTS:
        for condition in LR_SWEEP_TASKS:
            overrides = _m1_overrides(condition, seed=0, learning_rate=learning_rate)
            runs.append(
                RunSpec(
                    name=f"phase0c-lr-{learning_rate}-{condition}",
                    method="M1",
                    condition=condition,
                    seed=0,
                    skyrl_overrides=overrides,
                    manifest=_manifest("M1", condition, 0, overrides, commit, procedure_null=False),
                )
            )
    return runs


def _manifest(method: str, condition: str, seed: int, overrides: dict, commit: str, *, procedure_null: bool) -> dict:
    return {
        "method": method,
        "condition": condition,
        "seed": seed,
        "procedure_null": procedure_null,
        "fixed_parameters": {
            "sequences_per_step": SEQUENCES_PER_STEP,
            "m4_rollouts_per_prompt": M4_ROLLOUTS_PER_PROMPT,
            "response_cap_tokens": RESPONSE_CAP_TOKENS,
            "prompt_cap_tokens": PROMPT_CAP_TOKENS,
            "packing": False,
            "learning_rate": overrides["trainer"]["optim"]["lr"],
            "steps": PHASE_1A_STEPS,
        },
        "data_revisions": DATA_REVISIONS,
        "expert_revisions": EXPERT_REVISIONS,
        "marin_commit": commit,
        "skyrl_overrides": overrides,
    }


def _marin_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
