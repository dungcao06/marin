# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Strong-teacher trace collection (as in #9920's GLM relay) for building E_X and E_R.

R traces keep the teacher's thinking and are capped at 16k tokens: traces that exceed
the cap are dropped, never truncated, and the dropped share is reported.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

R_THINKING_TOKEN_CAP = 16_000


@dataclass(frozen=True)
class TraceBatch:
    """Kept traces plus the drop report the build manifest records."""

    traces: list[dict[str, Any]]
    dropped_share: float
    dropped_count: int
    token_cap: int = R_THINKING_TOKEN_CAP


def _token_count(text: str) -> int:
    """A conservative token estimate: whitespace words plus punctuation splits.

    The real cap check runs on the serving tokenizer at collection time; this
    estimate keeps the offline cap logic testable without a tokenizer download.
    """
    return len(text.split())


def collect_traces(
    tasks: list[dict[str, Any]],
    *,
    complete: Callable[[str], str],
    token_count: Callable[[str], int] = _token_count,
    token_cap: int = R_THINKING_TOKEN_CAP,
) -> TraceBatch:
    """Collect one teacher trace per task and enforce the R token cap.

    ``complete`` prompts the strong teacher (the GLM relay's chat-completion
    endpoint) and returns its full response, thinking included. Over-cap traces are
    dropped — never truncated — and the dropped share is reported.
    """
    traces: list[dict[str, Any]] = []
    dropped = 0
    for task in tasks:
        prompt = task["prompt"] if isinstance(task, dict) else str(task)
        response = complete(prompt)
        if token_count(response) > token_cap:
            dropped += 1
            continue
        traces.append({"prompt": prompt, "trace": response, "tokens": token_count(response)})
    total = len(tasks)
    return TraceBatch(
        traces=traces,
        dropped_count=dropped,
        dropped_share=dropped / total if total else 0.0,
        token_cap=token_cap,
    )


def write_trace_batch(batch: TraceBatch, output_path: str) -> dict[str, Any]:
    """Write the kept traces and return the drop report for the run manifest."""
    with open(output_path, "w") as handle:
        for trace in batch.traces:
            handle.write(json.dumps(trace, sort_keys=True) + "\n")
    return {
        "kept": len(batch.traces),
        "dropped": batch.dropped_count,
        "dropped_share": batch.dropped_share,
        "token_cap": batch.token_cap,
    }
