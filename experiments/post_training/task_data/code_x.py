# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The X task family: CruxEval-style Python input/output prediction, graded by
sandboxed execution (reusing the #8225-era grading primitives).

Functions are synthesized from seeded parameterized templates, so no public CruxEval
prompt can enter the set; a normalized-text exclusion against the public CruxEval
prompts still runs at build time and records its count. Train/probe splits are
assigned by seeded hash and are disjoint by construction.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

from experiments.post_training.curriculum_sft.code_tasks import (
    literals_equal,
    trace_value,
)

# Prediction directions, as in CruxEval.
DIRECTION_OUTPUT = "output"  # given code + input, predict the return value
DIRECTION_INPUT = "input"  # given code + output, find an input that produces it

TRAIN_SPLIT = "train"
PROBE_SPLIT = "probe"


@dataclass(frozen=True)
class XTask:
    """One graded prediction task with its ground truth and split."""

    task_id: str
    direction: str
    code: str
    call: str
    answer: str
    split: str


def _normalize_prompt(text: str) -> str:
    """Lowercase, collapse whitespace, and strip punctuation for exclusion checks."""
    return " ".join("".join(character if character.isalnum() else " " for character in text.lower()).split())


def _render_function(template: tuple[str, ...], value: int) -> str:
    """One template family member: a short pure function over ints."""
    return "\n".join(line.format(k=value) for line in template)


_TEMPLATES: tuple[tuple[str, ...], ...] = (
    (
        "def f(x):",
        "    total = 0",
        "    for step in range(x % {k} + 3):",
        "        total += step * {k}",
        "    return total % {k}",
    ),
    (
        "def f(x):",
        "    seen = set()",
        "    value = x % {k}",
        "    while value not in seen and len(seen) < {k}:",
        "        seen.add(value)",
        "        value = (value * 7 + {k}) % {k}",
        "    return len(seen)",
    ),
    (
        "def f(x):",
        "    digits = [int(d) for d in str(abs(x))]",
        "    return sum(d * {k} for d in digits) % 97",
    ),
)


def synthesize_x_tasks(*, count: int, seed: int, template_pool=None) -> list[XTask]:
    """Synthesize ``count`` execution-graded prediction tasks from the templates.

    Each candidate function is executed on random inputs twice (determinism-checked)
    through the curriculum's trace harness; tasks whose execution errors, times out,
    produces non-literal values, or is nondeterministic are dropped, so every shipped
    task has an execution-verified ground truth.
    """
    templates = template_pool if template_pool is not None else _TEMPLATES
    generator = random.Random(seed)
    tasks: list[XTask] = []
    attempts = 0
    while len(tasks) < count and attempts < count * 10:
        attempts += 1
        template = templates[attempts % len(templates)]
        parameter = generator.randint(2, 12)
        code = _render_function(template, parameter)
        argument = generator.randint(-10_000, 10_000)
        call = f"f({argument})"
        answer, _rejection = trace_value(code, call, timeout=10.0)
        if answer is None:
            continue
        direction = DIRECTION_OUTPUT if generator.random() < 0.5 else DIRECTION_INPUT
        digest = hashlib.sha256(f"{code}\n{call}".encode()).hexdigest()[:16]
        tasks.append(
            XTask(
                task_id=f"x-{digest}",
                direction=direction,
                code=code,
                call=call,
                answer=answer,
                split=PROBE_SPLIT if int(digest[:8], 16) % 100 < 10 else TRAIN_SPLIT,
            )
        )
    if len(tasks) < count:
        raise ValueError(f"template pool produced only {len(tasks)} of {count} tasks; widen the pool")
    return tasks


def verify_input_prediction(code: str, answer: str, candidate_input: str, *, timeout: float = 10.0) -> bool:
    """Grade an input prediction: execute the code on the candidate input and
    compare the resulting value with the reference output as literals."""
    literal, _rejection = trace_value(code, candidate_input, timeout=timeout)
    if literal is None:
        return False
    return literals_equal(answer, literal)


def exclude_public_cruxeval(tasks: list[XTask], cruxeval_prompts_path: Path) -> tuple[list[XTask], int]:
    """Drop tasks whose normalized text collides with a public CruxEval prompt.

    Returns the kept tasks and the exclusion count, which the build manifest records.
    """
    if not cruxeval_prompts_path.exists():
        return tasks, 0
    blocklist = {
        _normalize_prompt(json.loads(line)["prompt"])
        for line in cruxeval_prompts_path.read_text().splitlines()
        if line.strip()
    }
    kept = [task for task in tasks if _normalize_prompt(task.code) not in blocklist]
    return kept, len(tasks) - len(kept)


def write_split(tasks: list[XTask], output_dir: Path) -> dict[str, int]:
    """Write train and probe jsonl files; returns the per-split counts."""
    counts = {TRAIN_SPLIT: 0, PROBE_SPLIT: 0}
    for split in (TRAIN_SPLIT, PROBE_SPLIT):
        with (output_dir / f"x_{split}.jsonl").open("w") as handle:
            for task in tasks:
                if task.split != split:
                    continue
                handle.write(json.dumps(task.__dict__, sort_keys=True) + "\n")
                counts[split] += 1
    if counts[TRAIN_SPLIT] == 0 or counts[PROBE_SPLIT] == 0:
        raise ValueError(f"splits must both be nonempty, got {counts}")
    return counts
