# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Expert completion sampling for the M1 datasets (WP1's fixed-completion rows).

One served expert (E_X or E_R behind an OpenAI-compatible endpoint) samples
completions for a task file; X completions are graded by sandboxed execution and R
completions by the math verifier. Only graded-correct completions become M1 rows, so
the fixed data is verified-expert data by construction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from experiments.post_training.task_data.code_x import verify_input_prediction
from experiments.post_training.task_data.math_r import verify_final_answer


@dataclass(frozen=True)
class CompletionRow:
    """One fixed-completion training row: prompt messages plus the completion."""

    task_id: str
    data_source: str
    prompt: list[dict[str, str]]
    completion: str
    graded_correct: bool


def sample_completions(
    tasks: list[dict[str, Any]],
    *,
    data_source: str,
    complete: Callable[[list[dict[str, str]]], str],
    grade: Callable[[dict[str, Any], str], bool],
) -> list[CompletionRow]:
    """Sample one completion per task through ``complete`` and grade each one.

    ``grade`` closes over the family's verifier (sandboxed execution for X, the math
    verifier for R); the graded flag lands on the row so the M1 dataset builder can
    filter.
    """
    rows: list[CompletionRow] = []
    for task in tasks:
        prompt = [{"role": "user", "content": task["prompt"]}]
        completion = complete(prompt)
        rows.append(
            CompletionRow(
                task_id=str(task.get("task_id", "")),
                data_source=data_source,
                prompt=prompt,
                completion=completion,
                graded_correct=grade(task, completion),
            )
        )
    return rows


def grade_x_task(task: dict[str, Any], completion: str) -> bool:
    """Grade an X completion: the completion is a candidate input for the task's
    call; executing the code on it must reproduce the reference output."""
    return verify_input_prediction(task["code"], task["answer"], completion.strip())


def grade_r_task(task: dict[str, Any], completion: str) -> bool:
    """Grade an R completion: its final answer must verify against the stored answer."""
    return verify_final_answer(task["answer"], completion)


def write_m1_rows(rows: list[CompletionRow], output_path: str, *, graded_only: bool = True) -> dict[str, int]:
    """Write fixed-completion rows as jsonl; returns kept/dropped counts.

    ``graded_only`` drops unverified completions — the M1 objective trains on
    verified-expert data only.
    """
    kept = 0
    dropped = 0
    with open(output_path, "w") as handle:
        for row in rows:
            if graded_only and not row.graded_correct:
                dropped += 1
                continue
            handle.write(
                json.dumps(
                    {
                        "prompt": row.prompt,
                        "completion": row.completion,
                        "data_source": row.data_source,
                        "task_id": row.task_id,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            kept += 1
    return {"kept": kept, "dropped": dropped}
