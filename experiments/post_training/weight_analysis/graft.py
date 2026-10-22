# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The grafting builder: ``B + Δ·1[S]`` checkpoints for Grug component groups."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

from experiments.post_training.weight_analysis.delta_stats import component_of


def graft(
    base: dict[str, torch.Tensor],
    delta: dict[str, torch.Tensor],
    groups: list[str],
    *,
    alpha: float = 1.0,
) -> dict[str, torch.Tensor]:
    """``B + alpha·Δ`` restricted to the named component groups.

    Tensors outside ``groups`` come back byte-identical to the base; tensors inside
    are widened to fp32, shifted, and cast back to the base dtype. Tensors present
    only in the delta (a group the base lacks) are an error — grafting never
    invents weights.
    """
    unknown = sorted(set(groups) - {group for group, _ in _KNOWN_GROUPS()})
    if unknown:
        raise ValueError(f"unknown component groups {unknown}; known: {[g for g, _ in _KNOWN_GROUPS()]}")
    grafted: dict[str, torch.Tensor] = {}
    for name, base_tensor in base.items():
        if component_of(name) in groups and name in delta:
            shifted = base_tensor.float() + alpha * delta[name].float()
            grafted[name] = shifted.to(base_tensor.dtype)
        else:
            grafted[name] = base_tensor
    return grafted


def _KNOWN_GROUPS() -> tuple[tuple[str, tuple[str, ...]], ...]:
    from experiments.post_training.weight_analysis.delta_stats import COMPONENT_GROUPS

    return COMPONENT_GROUPS


def write_checkpoint(tensors: dict[str, torch.Tensor], output_path: str | Path) -> None:
    """Write one safetensors checkpoint."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output_path))
