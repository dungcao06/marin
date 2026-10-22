# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Chat-template adjustments for Stage 0's multi-turn reasoning protocol.

The published Qwen3 templates render a turn's ``<think>`` block only after the last
user query — the serving convention. Stage 0 trains on reasoning in every assistant
turn, and CL12's render check fails when a template drops it, so the Qwen bases train
on a minimally patched template: the last-query gate widens to include any assistant
turn carrying explicit reasoning content. Turn format, tool-call rendering, and the
no-reasoning path are untouched.
"""

REASONING_GATE = "{%- if loop.index0 > ns.last_query_index %}"
REASONING_GATE_WIDENED = "{%- if loop.index0 > ns.last_query_index or reasoning_content %}"
TRUNCATE_HISTORY_DEFAULT = "{%- set truncate_history_thinking = truncate_history_thinking if truncate_history_thinking is defined else True %}"
TRUNCATE_HISTORY_OFF = "{%- set truncate_history_thinking = truncate_history_thinking if truncate_history_thinking is defined else False %}"


def preserve_intermediate_reasoning(template: str) -> str:
    """Patch published templates so every assistant turn renders its think block.

    Two published conventions drop intermediate-turn reasoning; Stage 0 trains on
    it, and CL12's render check fails when a template loses it:

    - Qwen3 gates a turn's think block on being after the last user query; the gate
      widens to any turn carrying reasoning content.
    - Nemotron's reasoning template defaults ``truncate_history_thinking`` to true,
      which strips think blocks out of all but the last assistant turn; the default
      flips to false. Turns with no think block still get the template's empty
      ``<think></think>`` marker.
    """
    patched = template
    if REASONING_GATE in patched:
        patched = patched.replace(REASONING_GATE, REASONING_GATE_WIDENED)
    if TRUNCATE_HISTORY_DEFAULT in patched:
        patched = patched.replace(TRUNCATE_HISTORY_DEFAULT, TRUNCATE_HISTORY_OFF)
    return patched
