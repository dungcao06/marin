# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from taskcompendium.grading import grade_answer
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import ConversationTrace, GradingAttempt, TaskSpec, TextMessage

from experiments.evaluation.think_budget_calibration.mulcarry import (
    MULCARRY_MASTER_SEED,
    MulCarryConfig,
    MulCarryItem,
    direct_product_carry_pair_density,
    mulcarry_item,
    mulcarry_task,
)


def _grade(task: TaskSpec, answer: str) -> GradeResult:
    attempt = GradingAttempt(
        conversation=ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=answer)))
    )
    return grade_answer(task, attempt)


def test_mulcarry_task_is_byte_identical_for_the_published_seed():
    config = MulCarryConfig(n_digits=12, carry_density=0.5, seed=MULCARRY_MASTER_SEED)

    assert mulcarry_item(config, index=17) == MulCarryItem(
        left=119875155815,
        right=999185099780,
        answer=119777469524153822220700,
        observed_carry_density=0.5,
    )
    first = mulcarry_task(config, index=17).model_dump_json()
    repeated = mulcarry_task(config, index=17).model_dump_json()

    assert first == repeated


def test_mulcarry_task_identity_changes_with_every_generation_parameter():
    base = MulCarryConfig(n_digits=12, carry_density=0.5, seed=MULCARRY_MASTER_SEED)
    identities = {
        mulcarry_task(base, index=17).id,
        mulcarry_task(MulCarryConfig(n_digits=12, carry_density=0.5, seed="other-seed"), index=17).id,
        mulcarry_task(MulCarryConfig(n_digits=13, carry_density=0.5, seed=MULCARRY_MASTER_SEED), index=17).id,
        mulcarry_task(MulCarryConfig(n_digits=12, carry_density=0.9, seed=MULCARRY_MASTER_SEED), index=17).id,
        mulcarry_task(base, index=18).id,
    }

    assert len(identities) == 5


def test_direct_product_carry_density_matches_a_hand_worked_example():
    # For 159 x 246, the 5 and 9 each carry against all three right-hand digits.
    assert direct_product_carry_pair_density(159, 246) == 6 / 9


def test_mulcarry_density_changes_carry_work_without_changing_operand_length():
    n_digits = 20
    observed = []

    for target in (0.1, 0.5, 0.9):
        item = mulcarry_item(
            MulCarryConfig(n_digits=n_digits, carry_density=target, seed=MULCARRY_MASTER_SEED),
            index=3,
        )
        assert len(str(item.left)) == len(str(item.right)) == n_digits
        assert item.answer == item.left * item.right
        assert abs(item.observed_carry_density - target) <= 1 / n_digits
        observed.append(item.observed_carry_density)

    assert observed == sorted(observed)


def test_mulcarry_task_round_trip_grades_the_exact_product():
    config = MulCarryConfig(n_digits=8, carry_density=0.5, seed=MULCARRY_MASTER_SEED)
    item = mulcarry_item(config, index=9)
    task = TaskSpec.model_validate_json(mulcarry_task(config, index=9).model_dump_json())

    correct = _grade(task, str(item.answer))
    wrong = _grade(task, str(item.answer + 1))

    assert (correct.status, correct.reward) == (Outcome.GRADED, 1.0)
    assert (wrong.status, wrong.reward) == (Outcome.GRADED, 0.0)
