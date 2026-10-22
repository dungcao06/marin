# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unified Stage 0 data (CL12): the three #8225 datasets in one message-list format.

Every source is converted at its pinned revision to message lists whose assistant
turns carry the family's canonical think markers verbatim (`<|start_think|>...` for
the science reasoning source, `<think>...</think>` for terminal) and whose tool calls
stay structured, so Levanter (Grug) and Axolotl read the same files and render them
through their own chat templates. Reasoning is explicit in the text stream; CL12's
render check enforces that it survives every template. A manifest records row counts
and SHA-256 hashes per source.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

Message = dict[str, Any]


@dataclass(frozen=True)
class SourceSpec:
    """One Stage 0 source at its pinned revision, with its row adapter."""

    name: str
    repo: str
    revision: str
    config: str | None
    adapt: Callable[[dict], list[Message]]


def adapt_wildchat_rewild(row: dict) -> list[Message]:
    """``conversation`` turns carry plain user/assistant content."""
    return [{"role": turn["role"], "content": turn["content"]} for turn in row["conversation"]]


def adapt_nemotron_science(row: dict) -> list[Message]:
    """Assistant bodies embed `<|start_think|>...<|end_think|>`; kept verbatim so
    every chat template renders the reasoning span."""
    return [{"role": turn["role"], "content": turn["content"]} for turn in row["messages"]]


def adapt_nemotron_terminal(row: dict) -> list[Message]:
    """Terminal trajectories: assistant turns embed `<think>...</think>`; the
    environment's command outputs stay in the user turns verbatim."""
    return [{"role": turn["role"], "content": turn["content"]} for turn in row["conversations"]]


# The three #8225 Stage 0 sources at their pinned revisions (DESIGN, Stage 0 data).
STAGE0_SOURCES = (
    SourceSpec(
        name="wildchat-rewild",
        repo="nyu-dice-lab/wildchat50m-rewild-sft-385700",
        revision="46a5bb5",
        config=None,
        adapt=adapt_wildchat_rewild,
    ),
    SourceSpec(
        name="nemotron-science-reasoning",
        repo="laion/llama-nemotron-science-reasoning-on-canonical-think-full",
        revision="bae881d",
        config=None,
        adapt=adapt_nemotron_science,
    ),
    SourceSpec(
        name="nemotron-terminal",
        repo="nvidia/Nemotron-Terminal-Corpus",
        revision="a1667c4",
        config="dataset_adapters",
        adapt=adapt_nemotron_terminal,
    ),
)

# Stage order is #8225's curriculum: chat, then thinking, then terminal protocol.
STAGE_ORDER = ("wildchat-rewild", "nemotron-science-reasoning", "nemotron-terminal")


def _sha256_of(handle) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1 << 20), b""):
        digest.update(chunk)
    return digest.hexdigest()


def convert_source(
    spec: SourceSpec,
    output_dir: str,
    *,
    max_rows: int | None = None,
    load_dataset=None,
) -> dict[str, Any]:
    """Convert one source to gzipped message lists and return its manifest entry.

    ``load_dataset`` is injectable for tests. Rows are written streaming so the
    226k-row terminal source never sits in memory whole.
    """
    if load_dataset is None:
        from datasets import load_dataset  # noqa: PLC0415 - deferred: heavy at import time

    dataset = load_dataset(spec.repo, spec.config, revision=spec.revision, split="train", streaming=True)
    rows_written = 0
    output_path = f"{output_dir}/{spec.name}.jsonl.gz"
    hasher = hashlib.sha256()
    with gzip.open(output_path + ".tmp", "wt") as handle:
        for row in dataset:
            messages = spec.adapt(row)
            line = json.dumps({"source": spec.name, "messages": messages}, sort_keys=True) + "\n"
            handle.write(line)
            hasher.update(line.encode())
            rows_written += 1
            if max_rows is not None and rows_written >= max_rows:
                break
    os_replace(output_path + ".tmp", output_path)
    return {
        "name": spec.name,
        "repo": spec.repo,
        "revision": spec.revision,
        "config": spec.config,
        "rows": rows_written,
        "sha256": hasher.hexdigest(),
        "path": output_path,
    }


def os_replace(source: str, target: str) -> None:
    os.replace(source, target)


def build_manifest(output_dir: str, *, max_rows: int | None = None, load_dataset=None) -> dict[str, Any]:
    """Convert every source in stage order and write ``manifest.json``."""
    entries = []
    for spec in sorted(STAGE0_SOURCES, key=lambda source: STAGE_ORDER.index(source.name)):
        entries.append(convert_source(spec, output_dir, max_rows=max_rows, load_dataset=load_dataset))
    manifest = {
        "stage_order": list(STAGE_ORDER),
        "sources": entries,
        "total_rows": sum(entry["rows"] for entry in entries),
        "unified_format": {
            "messages": "list of {role, content, tool_calls?}",
            "reasoning": "assistant content carries the family's canonical think markers verbatim",
        },
    }
    with open(f"{output_dir}/manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest
