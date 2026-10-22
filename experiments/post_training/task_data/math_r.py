# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The R task family: long-thinking competition math with checkable final answers.

Problems are drawn from an open competition-math pool at a pinned revision (an
explicit build input, recorded in the run manifest). The pools reserved for
``P_think`` — AIME, HMMT and OlympiadBench — are excluded, and every remaining
problem is deduplicated against them by normalized problem text. Grading goes
through the math verifier MarinSkyRL already uses (``marin-verifyit[math]``).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from verifyit.modes.grade_math import grade_math_text_candidate
from verifyit.spec import MathSpec

# Pools reserved for the P_think probe set; R excludes them by source tag and by
# normalized problem text.
P_THINK_SOURCES = ("aime", "hmmt", "olympiadbench")

TRAIN_SPLIT = "train"
PROBE_SPLIT = "probe"


@dataclass(frozen=True)
class RTask:
    """One competition-math task with its checkable final answer and split."""

    task_id: str
    problem: str
    answer: str
    source: str
    split: str


def normalize_problem(text: str) -> str:
    """Lowercase, collapse whitespace, and strip punctuation/LaTeX markup so that
    re-typeset duplicates of one problem collide."""
    stripped = re.sub(r"\\[a-zA-Z]+", " ", text)
    return " ".join("".join(c if c.isalnum() else " " for c in stripped.lower()).split())


def is_reserved_for_probe(source: str) -> bool:
    """Whether a problem's source tag names a P_think-reserved pool."""
    return any(reserved in source.lower() for reserved in P_THINK_SOURCES)


def draw_r_tasks(
    pool_rows: list[dict],
    *,
    seed: int,
    probe_fraction: float = 0.1,
    p_think_pool_path=None,
) -> list[RTask]:
    """Draw verified R tasks from pool rows of ``{problem, answer, source}``.

    Excludes P_think-reserved sources, deduplicates against the P_think pool's
    normalized problem texts when that file is supplied, requires a nonempty
    final answer, and assigns disjoint seeded splits.
    """
    p_think_texts: set[str] = set()
    if p_think_pool_path is not None:
        path = Path(p_think_pool_path)
        if path.exists():
            p_think_texts = {
                normalize_problem(row.get("problem", ""))
                for line in path.read_text().splitlines()
                if (row := json_row(line))
            }
    tasks: list[RTask] = []
    seen: set[str] = set()
    for row in pool_rows:
        source = str(row.get("source", ""))
        if is_reserved_for_probe(source):
            continue
        problem = str(row.get("problem", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not problem or not answer:
            continue
        normalized = normalize_problem(problem)
        if normalized in seen or normalized in p_think_texts:
            continue
        seen.add(normalized)
        digest = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        split = PROBE_SPLIT if (int(digest[:8], 16) % 10_000) < probe_fraction * 10_000 else TRAIN_SPLIT
        tasks.append(RTask(task_id=f"r-{digest}", problem=problem, answer=answer, source=source, split=split))
    return tasks


def json_row(line: str) -> dict | None:
    line = line.strip()
    return json.loads(line) if line else None


def verify_final_answer(problem_answer: str, completion: str) -> bool:
    """Grade one completion's final answer against the task's stored answer with the
    math verifier MarinSkyRL uses."""
    reward = grade_math_text_candidate(MathSpec(expected=problem_answer), completion)
    return reward.reward > 0


def write_split(tasks: list[RTask], output_dir_path) -> dict[str, int]:
    """Write train and probe jsonl files; returns the per-split counts."""
    output_dir = Path(output_dir_path)
    counts = {TRAIN_SPLIT: 0, PROBE_SPLIT: 0}
    for split in (TRAIN_SPLIT, PROBE_SPLIT):
        with (output_dir / f"r_{split}.jsonl").open("w") as handle:
            for task in tasks:
                if task.split != split:
                    continue
                handle.write(
                    json.dumps(
                        {"task_id": task.task_id, "problem": task.problem, "answer": task.answer,
                         "source": task.source, "split": task.split},
                        sort_keys=True,
                    )
                    + "\n"
                )
                counts[split] += 1
    if counts[TRAIN_SPLIT] == 0 or counts[PROBE_SPLIT] == 0:
        raise ValueError(f"splits must both be nonempty, got {counts}")
    return counts
