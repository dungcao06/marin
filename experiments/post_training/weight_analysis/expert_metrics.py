# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 1a weight analyses: per-expert deltas, conflict, additivity, seed ceiling."""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch

# Routed-expert parameter names: model.layers.<i>.mlp.experts.<j>.<proj>.weight
_EXPERT_INDEX = re.compile(r"mlp\.experts\.(\d+)\.")


def _expert_of(name: str) -> int | None:
    match = _EXPERT_INDEX.search(name)
    return int(match.group(1)) if match else None


@dataclass(frozen=True)
class ExpertNorms:
    """Per-expert delta norms plus the Gini coefficient of their distribution."""

    norms: dict[int, float]
    gini: float


def per_expert_norms(delta: dict[str, torch.Tensor]) -> ExpertNorms:
    """The routed-expert delta norm per expert index and the Gini coefficient.

    Gini runs over per-expert energies: 0 when every expert moves equally, and
    approaching 1 when a handful of experts carry the whole update.
    """
    energies: dict[int, float] = {}
    for name, tensor in delta.items():
        expert = _expert_of(name)
        if expert is not None:
            energies[expert] = energies.get(expert, 0.0) + float(tensor.double().pow(2).sum())
    norms = {expert: energy**0.5 for expert, energy in sorted(energies.items())}
    if len(norms) < 2:
        return ExpertNorms(norms=norms, gini=0.0)
    values = sorted(norms.values())
    count = len(values)
    total = sum(values)
    gini = sum((2 * index - count - 1) * value for index, value in enumerate(values, start=1))
    return ExpertNorms(norms=norms, gini=gini / (count * total))


def per_expert_conflict(
    delta_a: dict[str, torch.Tensor],
    delta_b: dict[str, torch.Tensor],
) -> dict[int, float]:
    """Per-expert cosine between two task deltas: the conflict measure.

    An expert whose two task updates point in opposite directions (negative cosine)
    is where training both tasks fights itself.
    """
    energies_a: dict[int, list[torch.Tensor]] = {}
    energies_b: dict[int, list[torch.Tensor]] = {}
    for name, tensor in delta_a.items():
        expert = _expert_of(name)
        if expert is not None:
            energies_a.setdefault(expert, []).append(tensor.double().reshape(-1))
    for name, tensor in delta_b.items():
        expert = _expert_of(name)
        if expert is not None:
            energies_b.setdefault(expert, []).append(tensor.double().reshape(-1))
    conflict: dict[int, float] = {}
    for expert, flat_a in energies_a.items():
        flat_b = energies_b.get(expert)
        if flat_b is None:
            continue
        vector_a = torch.cat(flat_a)
        vector_b = torch.cat(flat_b)
        denominator = vector_a.norm() * vector_b.norm()
        conflict[expert] = float(vector_a.dot(vector_b) / denominator) if denominator else 0.0
    return conflict


def additivity(
    delta_joint: dict[str, torch.Tensor],
    delta_a: dict[str, torch.Tensor],
    delta_b: dict[str, torch.Tensor],
) -> float:
    """How additive two single-task updates are inside the joint update.

    Returns ‖Δ_joint − Δ_a − Δ_b‖ / (‖Δ_a‖ + ‖Δ_b‖): 0 when the joint delta is
    exactly the sum of the singles, 1 when the interaction energy equals the total
    single-task energy.
    """
    interaction = {}
    for name in delta_joint:
        interaction[name] = delta_joint[name] - delta_a.get(name, 0) - delta_b.get(name, 0)

    def norm(tensors: dict[str, torch.Tensor]) -> float:
        return math_sqrt(sum(float(tensor.double().pow(2).sum()) for tensor in tensors.values()))

    denominator = norm(delta_a) + norm(delta_b)
    if denominator == 0:
        return 0.0
    return norm(interaction) / denominator


def math_sqrt(value: float) -> float:
    return value**0.5


def seed_ceiling_normalize(value: float, seed_ceiling: float) -> float:
    """Express a metric against the seed ceiling: the seed-to-seed delta norm.

    A delta no larger than the ceiling is indistinguishable from re-running the same
    recipe with another seed; the design treats the ceiling as the noise floor.
    """
    if seed_ceiling <= 0:
        raise ValueError("seed ceiling must be positive; run the two-seed null first")
    return value / seed_ceiling
