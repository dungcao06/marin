# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Delta statistics over Grug component groups, from safetensors checkpoints."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from safetensors.torch import load_file

# Grug component groups, by HF-layout tensor-name pattern. The catch-all group is
# reported so unmatched tensors are never silently dropped.
COMPONENT_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("attention", ("self_attn.",)),
    ("routed_experts", ("mlp.experts.",)),
    ("shared_expert", ("shared_expert",)),
    ("router", ("mlp.gate", "router", "expert_bias")),
    ("embeddings_lm_head", ("embed_tokens", "lm_head")),
    ("norms_and_gates", ("norm", "_gate")),
)


def component_of(name: str) -> str:
    """The Grug component group a tensor name belongs to."""
    for group, patterns in COMPONENT_GROUPS:
        if any(pattern in name for pattern in patterns):
            return group
    return "other"


def load_tensors(path: str | Path) -> dict[str, torch.Tensor]:
    """Load one checkpoint's tensors, widened to fp32 (WP2's exports are fp32; bf16
    sources widen losslessly)."""
    tensors = load_file(str(path))
    return {name: value.float() for name, value in tensors.items()}


def deltas(checkpoint: dict[str, torch.Tensor], base: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Per-tensor deltas against the student B, fp32, on the shared tensor names."""
    missing = sorted(set(base) - set(checkpoint))
    if missing:
        raise ValueError(f"checkpoint is missing {len(missing)} base tensors, e.g. {missing[:3]}")
    return {name: checkpoint[name] - base[name] for name in sorted(set(base) & set(checkpoint))}


def _group_by_component(tensors: dict[str, torch.Tensor]) -> dict[str, list[torch.Tensor]]:
    grouped: dict[str, list[torch.Tensor]] = {}
    for name, tensor in tensors.items():
        grouped.setdefault(component_of(name), []).append(tensor)
    return grouped


def relative_norm(delta: dict[str, torch.Tensor], base: dict[str, torch.Tensor]) -> dict[str, float]:
    """‖Δ‖/‖B‖ per component and overall, in Frobenius norm."""
    delta_groups = _group_by_component(delta)
    base_groups = _group_by_component(base)
    result = {"overall": _norm(delta.values()) / max(_norm(base.values()), 1e-30)}
    for group in sorted(set(delta_groups) | set(base_groups)):
        result[group] = _norm(delta_groups.get(group, [])) / max(_norm(base_groups.get(group, [])), 1e-30)
    return result


def _norm(tensors) -> float:
    total = 0.0
    for tensor in tensors:
        total += float(tensor.double().pow(2).sum())
    return math.sqrt(total)


def cosine(delta_a: dict[str, torch.Tensor], delta_b: dict[str, torch.Tensor]) -> dict[str, float]:
    """Delta-direction cosine per component and overall; 0 when either side is zero."""
    groups_a = _group_by_component(delta_a)
    groups_b = _group_by_component(delta_b)
    result = {"overall": _flat_cosine(list(delta_a.values()), list(delta_b.values()))}
    for group in sorted(set(groups_a) | set(groups_b)):
        result[group] = _flat_cosine(groups_a.get(group, []), groups_b.get(group, []))
    return result


def _flat_cosine(left: list[torch.Tensor], right: list[torch.Tensor]) -> float:
    left_flat = torch.cat([tensor.double().reshape(-1) for tensor in left]) if left else None
    right_flat = torch.cat([tensor.double().reshape(-1) for tensor in right]) if right else None
    if left_flat is None or right_flat is None or left_flat.numel() != right_flat.numel():
        return 0.0
    denominator = left_flat.norm() * right_flat.norm()
    if denominator == 0:
        return 0.0
    return float(left_flat.dot(right_flat) / denominator)


def sign_agreement(delta_a: dict[str, torch.Tensor], delta_b: dict[str, torch.Tensor]) -> float:
    """Sign agreement on weights both deltas changed, as a share of those weights."""
    both_changed = 0
    agreeing = 0
    for name in sorted(set(delta_a) & set(delta_b)):
        a = delta_a[name].double()
        b = delta_b[name].double()
        mask = (a != 0) & (b != 0)
        both_changed += int(mask.sum())
        agreeing += int((torch.sign(a[mask]) == torch.sign(b[mask])).sum())
    return agreeing / both_changed if both_changed else 0.0


def subspace_overlap(
    delta: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    *,
    seed: int = 0,
) -> dict[str, float]:
    """Per-matrix share of ``delta``'s energy inside ``reference``'s top singular
    directions, where k holds 50% of the reference matrix's energy — with the random
    k-dim subspace and the energy-matched permuted-reference controls.

    Only 2-D matrices participate; vectors and scalars are skipped.
    """
    generator = torch.Generator().manual_seed(seed)
    overlaps: list[float] = []
    random_controls: list[float] = []
    permuted_controls: list[float] = []
    for name in sorted(set(delta) & set(reference)):
        matrix_delta = delta[name]
        matrix_reference = reference[name]
        if matrix_delta.ndim != 2 or min(matrix_delta.shape) < 2:
            continue
        k = _half_energy_rank(matrix_reference)
        if k <= 0:
            continue
        basis = _top_right_singular_basis(matrix_reference, k)
        overlaps.append(_energy_inside(matrix_delta, basis))
        random_basis = torch.randn(basis.shape[0], k, generator=generator, dtype=torch.float64)
        random_basis, _ = torch.linalg.qr(random_basis)
        random_controls.append(_energy_inside(matrix_delta, random_basis))
        permuted = matrix_reference.flatten()[torch.randperm(matrix_reference.numel(), generator=generator)]
        permuted = permuted.reshape(matrix_reference.shape)
        permuted_basis = _top_right_singular_basis(permuted, k)
        permuted_controls.append(_energy_inside(matrix_delta, permuted_basis))
    if not overlaps:
        return {"overlap": 0.0, "random_control": 0.0, "permuted_control": 0.0, "matrices": 0.0}
    return {
        "overlap": sum(overlaps) / len(overlaps),
        "random_control": sum(random_controls) / len(random_controls),
        "permuted_control": sum(permuted_controls) / len(permuted_controls),
        "matrices": float(len(overlaps)),
    }


def _half_energy_rank(matrix: torch.Tensor) -> int:
    squared = matrix.double().pow(2)
    total = float(squared.sum())
    if total <= 0:
        return 0
    singular_values = torch.linalg.svdvals(matrix.double())
    cumulative = torch.cumsum(singular_values.square(), dim=0)
    return int((cumulative < total / 2).sum()) + 1


def _top_right_singular_basis(matrix: torch.Tensor, k: int) -> torch.Tensor:
    """The first k right singular directions, as a [features, k] basis."""
    result = torch.linalg.svd(matrix.double(), full_matrices=False)
    return result.Vh.T[:, :k]


def _energy_inside(matrix: torch.Tensor, basis: torch.Tensor) -> float:
    total = float(matrix.double().pow(2).sum())
    if total <= 0:
        return 0.0
    projected = matrix.double() @ basis
    return float(projected.pow(2).sum() / total)
