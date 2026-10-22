# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task-data tests: X synthesis + sandboxed grading, R exclusion/dedup, trace caps,
and the M1 row format."""

import json

import pytest

from experiments.post_training.task_data.code_x import (
    DIRECTION_INPUT,
    PROBE_SPLIT,
    TRAIN_SPLIT,
    exclude_public_cruxeval,
    synthesize_x_tasks,
    verify_input_prediction,
)
from experiments.post_training.task_data.code_x import (
    write_split as write_x_split,
)
from experiments.post_training.task_data.expert_outputs import (
    grade_r_task,
    sample_completions,
    write_m1_rows,
)
from experiments.post_training.task_data.expert_traces import collect_traces, write_trace_batch
from experiments.post_training.task_data.math_r import (
    draw_r_tasks,
    is_reserved_for_probe,
    normalize_problem,
    verify_final_answer,
)
from experiments.post_training.task_data.math_r import (
    write_split as write_r_split,
)


def test_synth_x_tasks_are_execution_graded_and_split_disjointly():
    tasks = synthesize_x_tasks(count=6, seed=7)
    assert len(tasks) == 6
    for task in tasks:
        assert task.answer, "every task ships an execution-verified ground truth"
        if task.direction == DIRECTION_INPUT:
            assert verify_input_prediction(
                task.code, task.answer, task.call
            ), "the reference call must reproduce the reference output"
    splits = {task.split for task in tasks}
    assert splits <= {TRAIN_SPLIT, PROBE_SPLIT}


def test_cruxeval_exclusion_drops_colliding_prompts(tmp_path):
    tasks = synthesize_x_tasks(count=4, seed=3)
    blocklist = tmp_path / "cruxeval_prompts.jsonl"
    blocklist.write_text(json.dumps({"prompt": tasks[0].code}) + "\n")
    kept, excluded = exclude_public_cruxeval(tasks, blocklist)
    assert excluded == 1 and len(kept) == 3
    # No blocklist file: nothing is excluded, and the count says so.
    kept_none, excluded_none = exclude_public_cruxeval(tasks, tmp_path / "missing.jsonl")
    assert (kept_none, excluded_none) == (tasks, 0)


def test_write_split_requires_both_splits_nonempty(tmp_path):
    tasks = synthesize_x_tasks(count=8, seed=11)
    counts = write_x_split(tasks, tmp_path)
    assert counts[TRAIN_SPLIT] > 0 and counts[PROBE_SPLIT] > 0
    assert counts[TRAIN_SPLIT] + counts[PROBE_SPLIT] == len(tasks)
    one_side = [task for task in tasks if task.split == tasks[0].split]
    with pytest.raises(ValueError, match="nonempty"):
        write_x_split(one_side, tmp_path)


def test_reserved_sources_and_text_dedup_exclude_p_think_problems():
    assert is_reserved_for_probe("aime2024")
    assert is_reserved_for_probe("OlympiadBench")
    assert not is_reserved_for_probe("math_train")
    pool = [
        {"problem": "Compute 2+2.", "answer": "4", "source": "math_train"},
        {"problem": "Compute  2 + 2!", "answer": "4", "source": "math_train"},  # normalizes to the same text
        {"problem": "AIME problem", "answer": "1", "source": "aime2024"},
        {"problem": "", "answer": "3", "source": "math_train"},
    ]
    p_think = None
    tasks = draw_r_tasks(pool, seed=1, p_think_pool_path=p_think)
    assert [task.problem for task in tasks] == ["Compute 2+2."]


def test_r_dedup_against_a_p_think_pool_file(tmp_path):
    pool_path = tmp_path / "p_think_pool.jsonl"
    pool_path.write_text(json.dumps({"problem": "The_reserved problem!"}) + "\n")
    pool = [
        {"problem": "the RESERVED problem", "answer": "7", "source": "math_train"},
        {"problem": "another problem", "answer": "8", "source": "math_train"},
    ]
    tasks = draw_r_tasks(pool, seed=1, p_think_pool_path=pool_path)
    assert [task.problem for task in tasks] == ["another problem"]


def test_r_final_answer_verification_uses_the_math_verifier():
    assert verify_final_answer("4", "The answer is $\\boxed{4}$.")
    assert not verify_final_answer("4", "The answer is $\\boxed{5}$.")


def test_r_write_split(tmp_path):
    pool = [
        {"problem": f"problem {index} plus one", "answer": str(index), "source": "math_train"} for index in range(40)
    ]
    tasks = draw_r_tasks(pool, seed=5, probe_fraction=0.2)
    counts = write_r_split(tasks, tmp_path)
    assert counts["train"] + counts["probe"] == len(tasks)
    assert counts["probe"] > 0


def test_trace_collection_drops_over_cap_and_reports_the_share():
    def complete(prompt: str) -> str:
        return "short trace" if "small" in prompt else "word " * 20_000

    tasks = [{"prompt": "small task"}, {"prompt": "big task"}, {"prompt": "another small task"}]
    batch = collect_traces(tasks, complete=complete, token_cap=100)
    assert len(batch.traces) == 2
    assert batch.dropped_count == 1
    assert batch.dropped_share == pytest.approx(1 / 3)


def test_trace_writer_records_the_drop_report(tmp_path):
    tasks = [{"prompt": "small"}]
    batch = collect_traces(tasks, complete=lambda prompt: "ok trace", token_cap=100)
    report = write_trace_batch(batch, str(tmp_path / "traces.jsonl"))
    assert report == {"kept": 1, "dropped": 0, "dropped_share": 0.0, "token_cap": 100}
    lines = (tmp_path / "traces.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["trace"] == "ok trace"


def test_m1_rows_carry_prompt_completion_and_task_tag_and_filter_ungraded(tmp_path):
    tasks = [
        {"task_id": "r-1", "prompt": "problem 1", "answer": "4"},
        {"task_id": "r-2", "prompt": "problem 2", "answer": "9"},
    ]

    def complete(prompt_messages):
        return "The answer is $\\boxed{4}$." if "problem 1" in prompt_messages[0]["content"] else "wrong"

    rows = sample_completions(tasks, data_source="task_R", complete=complete, grade=grade_r_task)
    assert [row.graded_correct for row in rows] == [True, False]
    counts = write_m1_rows(rows, str(tmp_path / "m1.jsonl"), graded_only=True)
    assert counts == {"kept": 1, "dropped": 1}
    line = json.loads((tmp_path / "m1.jsonl").read_text().splitlines()[0])
    assert line["data_source"] == "task_R"
    assert line["prompt"] == [{"role": "user", "content": "problem 1"}]
    assert "boxed{4}" in line["completion"]


def test_normalize_problem_collides_retypeset_duplicates():
    assert normalize_problem("Compute  2 + 2!") == normalize_problem("compute 2 + 2")
    # LaTeX commands strip out entirely, so re-typeset markup still collides.
    assert normalize_problem("\\frac{1}{2}") == normalize_problem("{1}{2}")
