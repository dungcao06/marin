# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe-set builder tests: assembly, disjointness, hashes, and the bucket tagger."""

import json

import pytest

from experiments.post_training.probe_sets.buckets import (
    BUCKET_FINAL_MESSAGE,
    BUCKET_REASONING,
    BUCKET_TOOL_ARGUMENTS,
    BUCKET_TOOL_NAME,
    tag_trajectory,
)
from experiments.post_training.probe_sets.builder import (
    P_AGENT_SIZE,
    P_CHAT_SIZE,
    P_THINK_SIZE,
    PROBE_SET_NAMES,
    ProbeSetSpec,
    build_probe_sets,
    probe_manifest,
)


def _spec(name: str, rows: list[dict]) -> ProbeSetSpec:
    return ProbeSetSpec(name=name, size=min(len(rows), 3), load=lambda rows=rows: list(rows), revision="feedface")


def test_build_probe_sets_requires_every_set_and_truncates_to_size():
    def loader_for(rows):
        return lambda: rows

    specs = {
        "P_chat": ProbeSetSpec("P_chat", 2, loader_for([{"prompt": f"chat {i}"} for i in range(5)]), "r1"),
        "P_think": ProbeSetSpec("P_think", 1, loader_for([{"problem": f"math {i}"} for i in range(2)]), "r2"),
        "P_agent": ProbeSetSpec("P_agent", 1, loader_for([{"task": f"swe {i}"} for i in range(2)]), "r3"),
        "P_X": ProbeSetSpec("P_X", 1, loader_for([{"prompt": f"x {i}"} for i in range(2)]), "r4"),
        "P_R": ProbeSetSpec("P_R", 1, loader_for([{"prompt": f"r {i}"} for i in range(2)]), "r5"),
        "T": ProbeSetSpec("T", 1, loader_for([{"prompt": f"text {i}"} for i in range(2)]), "r6"),
    }
    manifest = build_probe_sets(specs)
    assert set(manifest["sets"]) == set(PROBE_SET_NAMES)
    assert manifest["sets"]["P_chat"]["rows"] == 2
    assert manifest["sets"]["P_think"]["rows"] == 1
    for name, entry in manifest["sets"].items():
        assert len(entry["sha256"]) == 64
    with pytest.raises(ValueError, match="no spec"):
        build_probe_sets({name: specs[name] for name in ("P_chat",)})


def test_probe_prompts_must_be_disjoint_from_stage0():
    def loader():
        return [{"prompt": "the stage zero prompt"}, {"prompt": "a fresh probe prompt"}]

    specs = {
        "P_chat": ProbeSetSpec("P_chat", 10, loader, "r1"),
        "P_think": ProbeSetSpec("P_think", 10, loader, "r2"),
        "P_agent": ProbeSetSpec("P_agent", 10, loader, "r3"),
        "P_X": ProbeSetSpec("P_X", 10, loader, "r4"),
        "P_R": ProbeSetSpec("P_R", 10, loader, "r5"),
        "T": ProbeSetSpec("T", 10, loader, "r6"),
    }
    manifest = build_probe_sets(specs, stage0_norm_texts={"the stage zero prompt"})
    assert manifest["sets"]["P_chat"]["rows"] == 1


def test_manifest_writes_counts_and_hashes_without_the_prompt_bodies(tmp_path):
    specs = {
        "P_chat": ProbeSetSpec("P_chat", 1, lambda: [{"prompt": "chat"}], "r1"),
        "P_think": ProbeSetSpec("P_think", 1, lambda: [{"problem": "math"}], "r2"),
        "P_agent": ProbeSetSpec("P_agent", 1, lambda: [{"task": "swe"}], "r3"),
        "P_X": ProbeSetSpec("P_X", 1, lambda: [{"prompt": "x"}], "r4"),
        "P_R": ProbeSetSpec("P_R", 1, lambda: [{"prompt": "r"}], "r5"),
        "T": ProbeSetSpec("T", 1, lambda: [{"prompt": "text"}], "r6"),
    }
    manifest = build_probe_sets(specs)
    output = tmp_path / "manifest.json"
    probe_manifest(manifest, str(output))
    written = json.loads(output.read_text())
    assert "prompts" not in written["sets"]["P_chat"]
    assert written["sets"]["P_chat"]["rows"] == 1


def test_bucket_tagger_covers_reasoning_tool_name_arguments_and_final_message():
    messages = [
        {"role": "user", "content": "fix the bug"},
        {
            "role": "assistant",
            "reasoning_content": "I should look at the tests.",
            "content": "Checking the tests.",
            "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "pytest -x"}}}],
        },
        {"role": "user", "content": "New Terminal Output: 1 failed"},
        {"role": "assistant", "content": "Fixed by removing the off-by-one."},
    ]
    rows = tag_trajectory(messages)
    buckets = {row["bucket"] for row in rows}
    assert buckets == {
        BUCKET_REASONING,
        BUCKET_TOOL_NAME,
        BUCKET_TOOL_ARGUMENTS,
        BUCKET_FINAL_MESSAGE,
    }
    by_bucket = {row["bucket"]: row for row in rows if row["bucket"] in (BUCKET_REASONING, BUCKET_TOOL_NAME)}
    assert by_bucket[BUCKET_REASONING]["text"] == "I should look at the tests."
    assert by_bucket[BUCKET_TOOL_NAME]["text"] == "bash"
    assert by_bucket[BUCKET_REASONING]["turn_index"] == 1
    final = [row for row in rows if row["bucket"] == BUCKET_FINAL_MESSAGE and row["turn_index"] == 3]
    assert final[0]["text"] == "Fixed by removing the off-by-one."


def test_bucket_tagger_records_turn_indexes_and_ignores_context():
    messages = [
        {"role": "user", "content": "observation only"},
        {"role": "assistant", "content": "first turn"},
        {"role": "user", "content": "another observation"},
        {"role": "assistant", "content": "second turn"},
    ]
    rows = tag_trajectory(messages)
    assert all(row["bucket"] == BUCKET_FINAL_MESSAGE for row in rows)
    assert [row["turn_index"] for row in rows] == [1, 3]
    # The fixed-context user turns never enter a bucket.
    assert all("observation" not in row["text"] for row in rows)


def test_probe_sizes_match_the_design_table():
    assert (P_CHAT_SIZE, P_THINK_SIZE, P_AGENT_SIZE) == (1500, 300, 150)
