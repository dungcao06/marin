# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage 0 tests: the unified-format adapters, the render check, and both configs."""

import gzip
import json
import os

import pytest

from experiments.post_training.stage0.convert import (
    STAGE_ORDER,
    SourceSpec,
    adapt_nemotron_science,
    adapt_nemotron_terminal,
    adapt_wildchat_rewild,
    build_manifest,
    convert_source,
)
from experiments.post_training.stage0.templates import preserve_intermediate_reasoning

# --- adapters: source schemas -> the unified message-list format ---


def test_wildchat_adapter_passes_conversation_through():
    row = {"conversation": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
    messages = adapt_wildchat_rewild(row)
    assert messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_science_adapter_keeps_think_markers_verbatim():
    row = {
        "messages": [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": "<|start_think|>\nwork it out\n<|end_think|>\nThe answer is 4.",
            },
        ]
    }
    messages = adapt_nemotron_science(row)
    assert messages[1]["content"] == "<|start_think|>\nwork it out\n<|end_think|>\nThe answer is 4."


def test_terminal_adapter_keeps_think_blocks_and_observations_verbatim():
    row = {
        "conversations": [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "<think>\nplan\n</think>run ls"},
            {"role": "user", "content": "New Terminal Output: files"},
        ]
    }
    messages = adapt_nemotron_terminal(row)
    assert messages[1]["content"] == "<think>\nplan\n</think>run ls"
    assert messages[2] == {"role": "user", "content": "New Terminal Output: files"}


def test_reasoning_gate_patch_renders_intermediate_think_blocks():
    """The published Qwen3 gate renders a turn's think block only after the last user
    query; the widened gate renders every think-bearing assistant turn."""
    builtin = (
        "{%- if loop.index0 > ns.last_query_index %}"
        "{%- if loop.last or reasoning_content %}[think]{{ reasoning_content }}[{{ content }}]"
        "{%- else %}[plain {{ content }}]{%- endif %}"
        "{%- else %}[plain {{ content }}]{%- endif %}"
    )
    patched = preserve_intermediate_reasoning(builtin)
    assert "or reasoning_content" in patched
    unchanged = preserve_intermediate_reasoning("no gate here")
    assert unchanged == "no gate here"


def test_convert_source_writes_rows_and_a_manifest_entry(tmp_path):
    rows = iter(
        [
            {"messages": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]},
            {"messages": [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]},
        ]
    )

    def fake_load_dataset(repo, config, revision, split, streaming):
        assert (repo, revision, split, streaming) == (
            "laion/llama-nemotron-science-reasoning-on-canonical-think-full",
            "bae881d",
            "train",
            True,
        )
        return rows

    spec = SourceSpec(
        name="nemotron-science-reasoning",
        repo="laion/llama-nemotron-science-reasoning-on-canonical-think-full",
        revision="bae881d",
        config=None,
        adapt=adapt_nemotron_science,
    )
    entry = convert_source(spec, str(tmp_path), load_dataset=fake_load_dataset)
    assert entry["rows"] == 2
    assert len(entry["sha256"]) == 64
    with gzip.open(tmp_path / "nemotron-science-reasoning.jsonl.gz", "rt") as handle:
        lines = [json.loads(line) for line in handle]
    assert len(lines) == 2
    assert lines[0]["source"] == "nemotron-science-reasoning"


def test_manifest_records_row_counts_and_hashes_in_stage_order(tmp_path):
    specs = {
        name: SourceSpec(
            name=name, repo=f"example/{name}", revision="feedface", config=None, adapt=adapt_wildchat_rewild
        )
        for name in STAGE_ORDER
    }
    calls: list[str] = []

    def fake_load_dataset(repo, config, revision, split, streaming):
        name = repo.split("/")[-1]
        calls.append(name)

        def iterator():
            for index in range(3):
                yield {"conversation": [{"role": "user", "content": f"q{index}"}]}

        return iterator()

    originals = {spec.name: spec for spec in specs.values()}
    import experiments.post_training.stage0.convert as convert_module  # noqa: PLC0415 - monkeypatched by this test

    original_sources = convert_module.STAGE0_SOURCES
    convert_module.STAGE0_SOURCES = tuple(originals[name] for name in STAGE_ORDER)
    try:
        manifest = build_manifest(str(tmp_path), load_dataset=fake_load_dataset)
    finally:
        convert_module.STAGE0_SOURCES = original_sources
    assert manifest["stage_order"] == list(STAGE_ORDER)
    assert [entry["name"] for entry in manifest["sources"]] == list(STAGE_ORDER)
    assert manifest["total_rows"] == 9
    assert calls == list(STAGE_ORDER), "sources convert in curriculum order"


# --- render check (CL12): 100 rows per source through each of the four templates ---


def _synthetic_rows(source_name: str, count: int = 100) -> list[dict]:
    """Unified-format rows with the protocol's reasoning and tool-call structure."""
    rows = []
    for index in range(count):
        if source_name == "wildchat-rewild":
            messages = [
                {"role": "user", "content": f"chat question {index}"},
                {"role": "assistant", "content": f"chat answer {index}"},
            ]
        elif source_name == "nemotron-science-reasoning":
            messages = [
                {"role": "user", "content": f"science question {index}"},
                {
                    "role": "assistant",
                    "content": f"<|start_think|>science reasoning {index}<|end_think|>answer {index}",
                },
            ]
        else:
            messages = [
                {"role": "user", "content": f"terminal task {index}"},
                {
                    "role": "assistant",
                    "content": f"<think>terminal reasoning {index}</think>ls output {index}",
                    "tool_calls": [{"function": {"name": "bash", "arguments": {"command": f"ls {index}"}}}],
                },
                {"role": "user", "content": f"New Terminal Output: out{index}"},
            ]
        rows.append({"source": source_name, "messages": messages})
    return rows


@pytest.mark.parametrize("source_name", list(STAGE_ORDER))
def test_reasoning_and_tool_call_spans_survive_every_chat_template(source_name):
    """100 rows per source, rendered through each of the four chat templates.

    The Grug template renders through a tokenizer that carries its special tokens;
    the three MoE bases render through their published templates. Reasoning content
    and structured tool calls must survive every render.
    """
    transformers = pytest.importorskip("transformers")
    AutoTokenizer = transformers.AutoTokenizer
    if not _hf_reachable():
        pytest.skip("HF hub unreachable")

    from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE  # noqa: PLC0415 - config-context import

    rows = _synthetic_rows(source_name)
    assert len(rows) == 100

    # The exact templates the Stage 0 configs train with.
    from experiments.post_training.axolotl_sft.bases import AXOLOTL_BASES  # noqa: PLC0415
    from experiments.post_training.stage0.axolotl_stage0 import _chat_template_for  # noqa: PLC0415 - same stack

    templates = {}
    grug_tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct")
    grug_tokenizer.chat_template = MARIN_CHAT_TEMPLATE
    templates["grug"] = grug_tokenizer
    for template_name, base_name in (
        ("qwen3_moe", "qwen3-30b-a3b-base"),
        ("qwen3_5_moe", "qwen3.5-35b-a3b-base"),
        ("nemotron_h", "nemotron-3-nano-30b-a3b-base"),
    ):
        template = _chat_template_for(AXOLOTL_BASES[base_name])
        assert template is not None
        tokenizer = AutoTokenizer.from_pretrained(
            AXOLOTL_BASES[base_name].hf_id, revision=AXOLOTL_BASES[base_name].revision
        )
        tokenizer.chat_template = template
        templates[template_name] = tokenizer

    for template_name, tokenizer in templates.items():
        for row in rows:
            rendered = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
            normalized = " ".join(rendered.split())
            assistant_messages = [m for m in row["messages"] if m["role"] == "assistant"]
            for message in assistant_messages:
                # Templates may reformat the think block's whitespace but must keep
                # the reasoning text and the answer.
                for span in _think_spans(message["content"]):
                    assert " ".join(span.split()) in normalized, (template_name, span[:60])
                for call in message.get("tool_calls", []):
                    assert call["function"]["name"] in normalized, (template_name, call["function"]["name"])


def _think_spans(content: str) -> list[str]:
    """The reasoning span and the answer of one think-marked assistant body."""
    for open_tag, close_tag in (("<think>", "</think>"), ("<|start_think|>", "<|end_think|>")):
        start = content.find(open_tag)
        if start == -1:
            continue
        end = content.find(close_tag, start)
        if end == -1:
            break
        reasoning = content[start + len(open_tag) : end]
        answer = content[end + len(close_tag) :]
        return [reasoning, answer]
    return [content]


def _hf_reachable() -> bool:
    return os.environ.get("HF_HUB_OFFLINE") != "1"


# --- the two trainer configs ---


def test_snowball_stage0_chains_three_stages_with_the_fixed_parameters():
    from experiments.post_training.stage0.snowball_stage0 import (  # noqa: PLC0415 - reads the pinned HF config at import
        BASE_HF_ID,
        BATCH_SIZE,
        SEQ_LEN,
        snowball_stage0_steps,
    )

    assert (BASE_HF_ID, BATCH_SIZE, SEQ_LEN) == (
        "open-athena/snowball-67b-a2b-base-262k-qk175",
        64,
        32768,
    )
    try:
        steps = snowball_stage0_steps()
    except Exception as error:
        pytest.skip(f"HF config for {BASE_HF_ID} unavailable: {error}")
    assert list(steps) == ["s1-chat", "s2-thinking", "s3-terminal"]
    # Stage 2 initializes from stage 1's checkpoint; stage 3 from stage 2's.
    for earlier, later in (("s1-chat", "s2-thinking"), ("s2-thinking", "s3-terminal")):
        assert any(dep.name == steps[earlier].name for dep in steps[later].deps)


def test_axolotl_stage0_chains_three_stages_with_matched_budget():
    from experiments.post_training.axolotl_sft.launcher import axolotl_config  # noqa: PLC0415
    from experiments.post_training.stage0.axolotl_stage0 import (  # noqa: PLC0415 - pulls the launcher's iris stack
        LEARNING_RATE,
        SEQUENCE_LEN,
        STAGE_ORDER,
        stage_specs,
    )

    base_name = "qwen3-30b-a3b-base"
    specs = stage_specs(base_name, data_root="/data", output_root="/out")
    assert [spec.base.architecture for spec in specs] == ["qwen3_moe"] * 3
    assert [spec.name.split("/")[-1] for spec in specs] == [
        f"s{index + 1}-{source}" for index, source in enumerate(STAGE_ORDER)
    ]
    for index, spec in enumerate(specs):
        assert spec.sequence_len == SEQUENCE_LEN
        assert spec.optimizer.learning_rate == LEARNING_RATE
        assert spec.epochs == 1
        # Global batch 64 examples = 8 per device x 8 GPUs, matching Snowball.
        assert spec.per_device_batch_size * 8 * spec.gradient_accumulation_steps == 64
        config = axolotl_config(spec, template_path="/tmp/template.jinja" if spec.chat_template else None)
        if index == 0:
            assert config["base_model"].startswith("Qwen/Qwen3-30B-A3B-Base@")
        else:
            previous_output = f"/out/{base_name}/s{index}-{STAGE_ORDER[index - 1]}"
            assert config["base_model"] == previous_output
