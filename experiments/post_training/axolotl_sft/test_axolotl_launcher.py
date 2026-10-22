# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the Axolotl launcher: config build, pins, and the masking canary."""

import os

import pytest

from experiments.post_training.axolotl_sft.bases import (
    AXOLOTL_BASES,
    NEMOTRON_3_NANO_30B_A3B_BASE,
    QWEN3_30B_A3B_BASE,
    QWEN35_35B_A3B_BASE,
)
from experiments.post_training.axolotl_sft.launcher import (
    AXOLOTL_FORK_REVISION,
    AXOLOTL_TRANSFORMERS_PIN,
    SUPPORTED_ARCHITECTURES,
    AxolotlOptimizer,
    AxolotlSpec,
    axolotl_config,
    axolotl_launch_command,
)

STAGE0_OPTIMIZER = AxolotlOptimizer(learning_rate="5e-6", weight_decay=0.0)
# The #6839 canary conversation: reasoning content, structured tool calls, a tool
# response, and a final assistant turn — the Phase 0 protocol's exact shape.
CANARY_CONVERSATION = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the weather in Paris?"},
    {
        "role": "assistant",
        "reasoning_content": "The user wants weather. I should call the tool.",
        "content": "Let me check.",
        "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}],
    },
    {"role": "tool", "content": '{"temp_c": 18, "cond": "cloudy"}'},
    {"role": "assistant", "content": "It's 18C and cloudy in Paris."},
]

_CANARY_MARKERS = (
    "What is the weather in Paris?",
    "The user wants weather. I should call the tool.",
    "get_weather",
    "Paris",
    "It's 18C and cloudy in Paris.",
)


def _spec(base, **overrides) -> AxolotlSpec:
    kwargs = {
        "name": f"stage0-{base.name}",
        "base": base,
        "data_manifest": "gs://example/stage0/data-*.jsonl",
        "optimizer": STAGE0_OPTIMIZER,
        "sequence_len": 32768,
        "epochs": 1,
    }
    if base.tokenizer_has_template is False:
        kwargs["chat_template"] = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    kwargs.update(overrides)
    return AxolotlSpec(**kwargs)


def test_config_builds_for_each_base_with_the_integrity_guard():
    for base in AXOLOTL_BASES.values():
        spec = _spec(base)
        config = axolotl_config(spec, template_path="/workspace/chat_template.jinja" if spec.chat_template else None)
        assert config["base_model"] == f"{base.hf_id}@{base.revision}"
        assert config["sequence_len"] == 32768
        assert config["tokenizer_save_jinja_files"] is False, "template_integrity guard must be on"
        assert config["train_on_inputs"] is False
        assert config["optimizer"] == "adamw_torch_fused"
        assert config["learning_rate"] == "5e-6"
        assert config["datasets"] == [{"path": "gs://example/stage0/data-*.jsonl", "type": "chat_template"}]


def test_nemotron_requires_an_explicit_template():
    with pytest.raises(ValueError, match="ships no chat template"):
        AxolotlSpec(
            name="stage0-nemotron",
            base=NEMOTRON_3_NANO_30B_A3B_BASE,
            data_manifest="gs://example/stage0/data-*.jsonl",
            optimizer=STAGE0_OPTIMIZER,
        )


def test_custom_template_requires_the_materialized_path():
    spec = _spec(QWEN3_30B_A3B_BASE, chat_template="{% for m in messages %}{{ m['content'] }}{% endfor %}")
    with pytest.raises(ValueError, match="template_path"):
        axolotl_config(spec)
    config = axolotl_config(spec, template_path="/workspace/chat_template.jinja")
    assert config["chat_template"] == "/workspace/chat_template.jinja"


def test_config_builds_for_nemotron_with_a_supplied_template():
    spec = _spec(NEMOTRON_3_NANO_30B_A3B_BASE)
    config = axolotl_config(spec, template_path="/workspace/chat_template.jinja")
    assert config["chat_template"] == "/workspace/chat_template.jinja"
    assert config["tokenizer_save_jinja_files"] is False


def test_expert_parallel_is_rejected_until_the_fork_ships_it():
    with pytest.raises(ValueError, match="expert"):
        _spec(QWEN3_30B_A3B_BASE, expert_parallel=True)


def test_fork_and_transformer_pins_recorded():
    assert AXOLOTL_TRANSFORMERS_PIN == "5.12.1"
    assert set(SUPPORTED_ARCHITECTURES) == {"qwen3_moe", "qwen3_5_moe", "nemotron_h"}
    architectures = {base.architecture for base in AXOLOTL_BASES.values()}
    assert architectures == set(SUPPORTED_ARCHITECTURES)


def test_transformers_pin_supports_the_three_architectures():
    """The fork pins transformers==5.12.1 (MARIN_FORK.md); that release must ship
    ``qwen3_5_moe`` and ``nemotron_h`` for the Phase 0 bases to load."""
    transformers = pytest.importorskip("transformers")
    version = tuple(int(part) for part in transformers.__version__.split(".")[:2])
    if version < (5, 12):
        pytest.skip(f"local transformers {transformers.__version__} predates the fork's 5.12.1 pin")
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES  # noqa: PLC0415 - optional dep guard

    for architecture in ("qwen3_5_moe", "nemotron_h"):
        assert architecture in CONFIG_MAPPING_NAMES, architecture


def test_twenty_step_smoke_is_a_launch_command_with_the_pinned_revision():
    spec = _spec(QWEN3_30B_A3B_BASE)
    command = axolotl_launch_command(spec, output_dir="/tmp/smoke", max_steps=20)
    assert "git clone" in command and AXOLOTL_FORK_REVISION in command
    assert "max_steps=20" in command
    assert "axolotl.cli.train" in command


def test_optimizer_rejects_silent_defaults():
    with pytest.raises(ValueError, match="positive"):
        AxolotlOptimizer(learning_rate="0")
    with pytest.raises(ValueError, match=r"betas"):
        AxolotlOptimizer(learning_rate="5e-6", adam_beta1=1.0)


def _hf_reachable() -> bool:
    return os.environ.get("HF_HUB_OFFLINE") != "1"


@pytest.mark.parametrize("base", [QWEN3_30B_A3B_BASE, QWEN35_35B_A3B_BASE, NEMOTRON_3_NANO_30B_A3B_BASE])
def test_masking_canary_per_chat_template(base):
    """The #6839 canary, per base template: the supervised region is exactly the
    assistant turns — user, system, and tool-response tokens stay out of the loss.

    Nemotron's base ships no template, so the canary runs on an explicit one, the
    same contract the launcher enforces at config time.
    """
    transformers = pytest.importorskip("transformers")
    AutoTokenizer = transformers.AutoTokenizer

    template = None
    if base.tokenizer_has_template:
        try:
            tokenizer = AutoTokenizer.from_pretrained(base.hf_id, revision=base.revision)
        except Exception as error:
            pytest.skip(f"could not load {base.hf_id}: {error}")
    else:
        template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}User: {{ message['content'] }}\n"
            "{% elif message['role'] == 'assistant' %}Assistant:"
            "{% if message['reasoning_content'] %} {{ message['reasoning_content'] }}{% endif %}"
            " {{ message['content'] }}"
            "{% if message['tool_calls'] %}{% for call in message['tool_calls'] %}"
            " {{ call['function']['name'] }} {{ call['function']['arguments'] }}{% endfor %}{% endif %}\n"
            "{% else %}Tool: {{ message['content'] }}\n{% endif %}"
            "{% endfor %}"
        )
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Base", revision=QWEN3_30B_A3B_BASE.revision)
        tokenizer.chat_template = template

    full_render = tokenizer.apply_chat_template(CANARY_CONVERSATION, tokenize=False, add_generation_prompt=False)
    for marker in _CANARY_MARKERS:
        assert marker in full_render, f"{base.name}: reasoning/tool spans must survive the template"

    # Incremental rendering, the way axolotl derives per-role token spans for
    # train_on_inputs=false: each message owns the tokens its render adds. Templates
    # that reject a system-only render fold the system message into the user span.
    spans: list[tuple[str, list[int]]] = []
    previous_prefix_len = 0
    for index in range(len(CANARY_CONVERSATION)):
        try:
            prefix = tokenizer.apply_chat_template(
                CANARY_CONVERSATION[: index + 1], tokenize=False, add_generation_prompt=False
            )
        except Exception:
            continue
        new_text = prefix[previous_prefix_len:]
        if not new_text:
            continue
        new_ids = tokenizer(new_text, add_special_tokens=False)["input_ids"]
        spans.append((CANARY_CONVERSATION[index]["role"], new_ids))
        previous_prefix_len = len(prefix)

    # The spans must positionally partition the full render (each message owns the
    # tokens its render added, no gaps or overlaps).
    full_ids = tokenizer(full_render, add_special_tokens=False)["input_ids"]
    span_total = sum(len(ids) for _, ids in spans)
    assert span_total == len(full_ids), f"{base.name}: spans cover {span_total} of {len(full_ids)} rendered tokens"

    supervised_ids = [token for role, ids in spans if role == "assistant" for token in ids]
    unsupervised_ids = [token for role, ids in spans if role != "assistant" for token in ids]
    supervised_text = tokenizer.decode(supervised_ids)
    assert "The user wants weather. I should call the tool." in supervised_text
    assert "It's 18C and cloudy in Paris." in supervised_text
    assert "What is the weather in Paris?" not in supervised_text
    assert "You are a helpful assistant." not in supervised_text
    assert '{"temp_c": 18, "cond": "cloudy"}' not in supervised_text
    assert supervised_ids and unsupervised_ids
