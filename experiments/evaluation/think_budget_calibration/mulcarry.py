# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic multiplication tasks with controlled direct-product carry density.

The density is the fraction of digit pairs whose direct product is at least ten. It
does not count propagated carries or the addition of partial products.
"""

import hashlib
from dataclasses import dataclass

from taskcompendium.grader import verifyit_package
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    ResourceGroups,
    Source,
    TaskSpec,
    TextMessage,
)
from verifyit.spec import NumericSpec

MULCARRY_MASTER_SEED = "marin-mulcarry-v1"
MULCARRY_GENERATOR_REVISION = "v1"
MULCARRY_DATASET = "marin/think-budget-calibration/mulcarry"


@dataclass(frozen=True)
class MulCarryConfig:
    """Parameters that identify a reproducible MulCarry item family.

    ``carry_density`` targets the fraction of cross-operand digit pairs whose
    direct product is at least ten.
    """

    n_digits: int
    carry_density: float
    seed: str

    def __post_init__(self) -> None:
        if self.n_digits < 1:
            raise ValueError("n_digits must be positive")
        if not 0 <= self.carry_density <= 1:
            raise ValueError("carry_density must be between zero and one")
        if not self.seed:
            raise ValueError("seed must be nonempty")


@dataclass(frozen=True)
class MulCarryItem:
    """One generated multiplication problem and its exact answer."""

    left: int
    right: int
    answer: int
    observed_carry_density: float


def _high_digit_counts(n_digits: int, carry_density: float) -> tuple[int, int]:
    """Choose high-digit counts whose pair density is closest to the target."""
    _, _, left_count, right_count = min(
        (
            abs((left * right) / (n_digits * n_digits) - carry_density),
            abs(left - right),
            left,
            right,
        )
        for left in range(n_digits + 1)
        for right in range(n_digits + 1)
    )
    return left_count, right_count


def _digest(seed_material: str, label: str) -> bytes:
    return hashlib.sha256(f"{seed_material}:{label}".encode()).digest()


def _operand(n_digits: int, high_count: int, seed_material: str) -> int:
    ranked_positions = sorted(range(n_digits), key=lambda position: _digest(seed_material, f"rank:{position}"))
    high_positions = set(ranked_positions[:high_count])
    digits = []
    for position in range(n_digits):
        draw = _digest(seed_material, f"digit:{position}")[0]
        digits.append(5 + draw % 5 if position in high_positions else draw % 2)
    if digits[0] == 0:
        digits[0] = 1
    return int("".join(map(str, digits)))


def direct_product_carry_pair_density(left: int, right: int) -> float:
    """Return the fraction of cross-operand digit products that reach ten."""
    left_digits = tuple(map(int, str(left)))
    right_digits = tuple(map(int, str(right)))
    carry_pairs = sum(a * b >= 10 for a in left_digits for b in right_digits)
    return carry_pairs / (len(left_digits) * len(right_digits))


def mulcarry_item(config: MulCarryConfig, index: int) -> MulCarryItem:
    """Generate one item from its family parameters and zero-based index."""
    if index < 0:
        raise ValueError("index must be nonnegative")

    density = repr(config.carry_density)
    seed_material = f"{config.seed}:{config.n_digits}:{density}:{index}"
    left_high, right_high = _high_digit_counts(config.n_digits, config.carry_density)
    left = _operand(config.n_digits, left_high, f"{seed_material}:left")
    right = _operand(config.n_digits, right_high, f"{seed_material}:right")
    return MulCarryItem(
        left=left,
        right=right,
        answer=left * right,
        observed_carry_density=direct_product_carry_pair_density(left, right),
    )


def mulcarry_task(config: MulCarryConfig, index: int) -> TaskSpec:
    """Build a serializable task with an exact in-process numeric grader."""
    item = mulcarry_item(config, index)
    density = repr(config.carry_density)
    density_key = density.replace(".", "p")
    seed_key = hashlib.sha256(config.seed.encode()).hexdigest()[:12]
    grader = verifyit_package(NumericSpec(expected=str(item.answer), tolerance_abs=0.0, tolerance_rel=0.0))
    return TaskSpec(
        id=f"mulcarry-{seed_key}-{config.n_digits}-{density_key}-{index}",
        context=ConversationInput(
            events=(
                TextMessage(
                    role="user",
                    content=f"Multiply {item.left} by {item.right}. Reply with the exact integer product.",
                ),
            )
        ),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.NUMBER,
        answer_format=PlainText(),
        grader=grader.grader,
        source=Source(
            dataset=MULCARRY_DATASET,
            revision=MULCARRY_GENERATOR_REVISION,
            row=f"{config.seed}:{config.n_digits}:{density}:{index}",
            importer_revision=MULCARRY_GENERATOR_REVISION,
        ),
        resources=ResourceGroups(verifier=grader.resources),
        tags=(
            "reasoning",
            "think-budget-calibration",
            "mulcarry",
            f"digits-{config.n_digits}",
            f"carry-density-{density_key}",
            "carry-definition-direct-product-pairs",
        ),
    )
