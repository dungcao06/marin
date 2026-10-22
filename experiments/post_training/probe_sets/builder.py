# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The probe-set assembly: prompts, manifests, and the disjointness checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from experiments.post_training.task_data.code_x import PROBE_SPLIT
from experiments.post_training.task_data.math_r import normalize_problem

# Set sizes from the DESIGN's probe-set table.
P_CHAT_SIZE = 1500
P_THINK_SIZE = 300
P_AGENT_SIZE = 150
P_TASK_SIZE = 1000
T_TOKEN_BUDGET = 2_000_000
T_DOMAINS = 10

PROBE_SET_NAMES = ("P_chat", "P_think", "P_agent", "P_X", "P_R", "T")


@dataclass(frozen=True)
class ProbeSetSpec:
    """One probe set's build contract: sources at pinned revisions and its size."""

    name: str
    size: int
    # Loader returning the source rows (problem/prompt dicts); injectable for tests.
    load: Callable[[], list[dict[str, Any]]]
    revision: str


def _hash_rows(rows: list[dict[str, Any]]) -> str:
    hasher = hashlib.sha256()
    for row in rows:
        hasher.update(json.dumps(row, sort_keys=True).encode())
    return hasher.hexdigest()


def build_probe_sets(specs: dict[str, ProbeSetSpec], *, stage0_norm_texts: set[str] | None = None) -> dict[str, Any]:
    """Assemble every probe set and return the manifest.

    ``stage0_norm_texts`` carries Stage 0's normalized prompt texts; a probe prompt
    colliding with one is dropped — probe prompts must be disjoint from Stage 0.
    Each set is truncated to its spec size after the exclusion.
    """
    stage0_norm_texts = stage0_norm_texts or set()
    manifest: dict[str, Any] = {"sets": {}}
    for name in PROBE_SET_NAMES:
        spec = specs.get(name)
        if spec is None:
            raise ValueError(f"probe set {name!r} has no spec")
        rows = spec.load()
        if stage0_norm_texts:
            rows = [
                row
                for row in rows
                if normalize_prompt_of(row) not in stage0_norm_texts
            ]
        rows = rows[: spec.size]
        manifest["sets"][name] = {
            "revision": spec.revision,
            "rows": len(rows),
            "sha256": _hash_rows(rows),
            "prompts": rows,
        }
    return manifest


def normalize_prompt_of(row: dict[str, Any]) -> str:
    """The normalized prompt text of one probe row, whatever its field is named."""
    text = row.get("prompt") or row.get("problem") or row.get("task") or ""
    return normalize_problem(str(text))


def probe_manifest(manifest: dict[str, Any], output_path: str) -> None:
    """Write the probe-set manifest beside the prompts."""
    serializable = {
        name: {key: value for key, value in entry.items() if key != "prompts"}
        for name, entry in manifest["sets"].items()
    }
    Path(output_path).write_text(json.dumps({"sets": serializable}, indent=2, sort_keys=True))


def probe_prompts_from_tasks(tasks: list, *, task_field: str) -> list[dict[str, Any]]:
    """The held-out (probe-split) task rows as probe prompts, for P_X and P_R."""
    return [
        {"prompt": getattr(task, task_field), "task_id": task.task_id, "split": task.split}
        for task in tasks
        if task.split == PROBE_SPLIT
    ]
