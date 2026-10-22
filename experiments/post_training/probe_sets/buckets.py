# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""P_agent KL buckets: token-level tags over recorded agentic trajectories.

Every recorded assistant turn carries bucket tags the analysis aggregates: reasoning,
tool name, tool arguments, and the final message, each with its turn index. Tool
observations are fixed context — never scored, so they carry no bucket.
"""

from __future__ import annotations

from typing import Any

BUCKET_REASONING = "reasoning"
BUCKET_TOOL_NAME = "tool_name"
BUCKET_TOOL_ARGUMENTS = "tool_arguments"
BUCKET_FINAL_MESSAGE = "final_message"
BUCKETS = (BUCKET_REASONING, BUCKET_TOOL_NAME, BUCKET_TOOL_ARGUMENTS, BUCKET_FINAL_MESSAGE)


def tag_trajectory(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tag one recorded trajectory turn by turn.

    Input messages follow the unified protocol: assistant turns may carry
    ``reasoning_content`` and structured ``tool_calls``; user turns are tool
    observations. Output rows carry the turn index, the role, the bucket name, and
    the bucket's text. The final assistant turn's plain content is the final
    message; earlier assistant content is ordinary assistant text (bucketed as the
    final message is, since it is model-generated output tokens).
    """
    assistant_turns = [index for index, message in enumerate(messages) if message.get("role") == "assistant"]
    last_assistant = assistant_turns[-1] if assistant_turns else None
    rows: list[dict[str, Any]] = []
    for turn_index, message in enumerate(messages):
        role = message.get("role")
        if role == "assistant":
            reasoning = message.get("reasoning_content")
            if reasoning:
                rows.append(_row(turn_index, BUCKET_REASONING, reasoning))
            for call in message.get("tool_calls") or []:
                function = call.get("function", {})
                rows.append(_row(turn_index, BUCKET_TOOL_NAME, str(function.get("name", ""))))
                rows.append(_row(turn_index, BUCKET_TOOL_ARGUMENTS, str(function.get("arguments", ""))))
            content = message.get("content") or ""
            if content:
                bucket = BUCKET_FINAL_MESSAGE if turn_index == last_assistant else BUCKET_FINAL_MESSAGE
                rows.append(_row(turn_index, bucket, content))
        # user/system turns are fixed context: no bucket, no scoring.
    return rows


def _row(turn_index: int, bucket: str, text: str) -> dict[str, Any]:
    return {"turn_index": turn_index, "bucket": bucket, "text": text}
