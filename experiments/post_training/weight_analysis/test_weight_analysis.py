# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Weight-analysis tests against hand-computed deltas on synthetic checkpoints."""

import pytest
import torch

from experiments.post_training.weight_analysis.delta_stats import (
    component_of,
    cosine,
    deltas,
    relative_norm,
    sign_agreement,
    subspace_overlap,
)
from experiments.post_training.weight_analysis.expert_metrics import (
    additivity,
    per_expert_conflict,
    per_expert_norms,
    seed_ceiling_normalize,
)
from experiments.post_training.weight_analysis.graft import graft


def _checkpoint(spec: dict[str, list[float]]) -> dict[str, torch.Tensor]:
    return {name: torch.tensor(values, dtype=torch.float64) for name, values in spec.items()}


BASE = _checkpoint(
    {
        "model.layers.0.self_attn.q_proj.weight": [1.0, 2.0, 3.0, 4.0],
        "model.layers.0.mlp.experts.0.gate_proj.weight": [1.0, 1.0, 1.0, 1.0],
        "model.layers.0.mlp.experts.1.gate_proj.weight": [2.0, 2.0, 2.0, 2.0],
        "model.layers.0.mlp.shared_expert.gate_proj.weight": [3.0, 3.0, 3.0, 3.0],
        "model.layers.0.mlp.gate.weight": [1.0, 0.0, 0.0, 0.0],
        "model.layers.0.input_layernorm.weight": [1.0, 1.0, 1.0, 1.0],
        "lm_head.weight": [0.5, 0.5, 0.5, 0.5],
    }
)


def test_component_classification_covers_the_grug_groups():
    assert component_of("model.layers.0.self_attn.q_proj.weight") == "attention"
    assert component_of("model.layers.0.mlp.experts.3.gate_proj.weight") == "routed_experts"
    assert component_of("model.layers.0.mlp.shared_expert.gate_proj.weight") == "shared_expert"
    assert component_of("model.layers.0.mlp.gate.weight") == "router"
    assert component_of("model.layers.0.mlp.router.expert_bias") == "router"
    assert component_of("lm_head.weight") == "embeddings_lm_head"
    assert component_of("model.layers.0.input_layernorm.weight") == "norms_and_gates"


def test_relative_norm_is_per_component_and_zero_for_identical_checkpoints():
    delta = deltas(BASE, BASE)
    norms = relative_norm(delta, BASE)
    assert norms["overall"] == 0.0
    changed = dict(BASE)
    changed["model.layers.0.self_attn.q_proj.weight"] = torch.tensor([2.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    delta_changed = deltas(changed, BASE)
    norms_changed = relative_norm(delta_changed, BASE)
    # ‖Δ‖ = 1 over the attention component; the component's base q_proj is [1,2,3,4].
    assert norms_changed["attention"] == pytest.approx(1.0 / (30**0.5))
    assert norms_changed["routed_experts"] == 0.0


def test_cosine_and_sign_agreement_on_hand_computed_deltas():
    base = _checkpoint({"w": [1.0, 0.0]})
    plus = _checkpoint({"w": [2.0, 0.0]})  # Δ = +x
    minus = _checkpoint({"w": [-1.0, 0.0]})  # Δ = -2x
    delta_plus = deltas(plus, base)
    delta_minus = deltas(minus, base)
    assert cosine(delta_plus, delta_minus)["overall"] == pytest.approx(-1.0)
    assert sign_agreement(delta_plus, delta_minus) == pytest.approx(0.0)
    assert sign_agreement(delta_plus, delta_plus) == pytest.approx(1.0)


def test_subspace_overlap_of_self_exceeds_half_energy_and_beats_the_controls():
    torch.manual_seed(0)
    reference = torch.randn(16, 8, dtype=torch.float64)
    delta = {"a.matrix": reference.clone()}
    ref = {"a.matrix": reference.clone()}
    result = subspace_overlap(delta, ref)
    # k holds 50% of the reference's energy, so a delta overlapping itself captures
    # at least half its energy in those directions — far above the random control.
    assert result["overlap"] > 0.5
    assert result["overlap"] > result["random_control"]
    # The permuted control destroys the reference's row space.
    assert result["permuted_control"] < result["overlap"]
    assert result["matrices"] == 1.0


def test_per_expert_norms_and_gini():
    delta = {
        "model.layers.0.mlp.experts.0.gate_proj.weight": torch.tensor([3.0, 0.0], dtype=torch.float64),
        "model.layers.0.mlp.experts.1.gate_proj.weight": torch.tensor([1.0, 0.0], dtype=torch.float64),
        "model.layers.0.self_attn.q_proj.weight": torch.tensor([9.0, 9.0], dtype=torch.float64),
    }
    norms = per_expert_norms(delta)
    assert set(norms.norms) == {0, 1}
    assert norms.norms[0] == pytest.approx(3.0)
    assert norms.norms[1] == pytest.approx(1.0)
    # Attention tensors carry no expert index and never enter the expert stats.
    assert 0.0 < norms.gini < 1.0


def test_uniform_expert_deltas_have_zero_gini():
    delta = {
        f"model.layers.0.mlp.experts.{expert}.gate_proj.weight": torch.tensor([1.0, 1.0], dtype=torch.float64)
        for expert in range(4)
    }
    assert per_expert_norms(delta).gini == pytest.approx(0.0)


def test_per_expert_conflict_is_negative_for_opposed_updates():
    experts_zero = "model.layers.0.mlp.experts.0.gate_proj.weight"
    delta_a = {experts_zero: torch.tensor([1.0], dtype=torch.float64)}
    delta_b_same = {experts_zero: torch.tensor([2.0], dtype=torch.float64)}
    delta_b_opposed = {experts_zero: torch.tensor([-2.0], dtype=torch.float64)}
    assert per_expert_conflict(delta_a, delta_b_same)[0] == pytest.approx(1.0)
    assert per_expert_conflict(delta_a, delta_b_opposed)[0] == pytest.approx(-1.0)


def test_additivity_is_zero_for_exactly_additive_updates():
    base = _checkpoint({"w": [0.0, 0.0]})
    joint = _checkpoint({"w": [1.0, 2.0]})
    a = _checkpoint({"w": [1.0, 0.0]})
    b = _checkpoint({"w": [0.0, 2.0]})
    assert additivity(deltas(joint, base), deltas(a, base), deltas(b, base)) == pytest.approx(0.0)
    noisy = _checkpoint({"w": [1.0, 3.0]})
    assert additivity(deltas(noisy, base), deltas(a, base), deltas(b, base)) > 0.0


def test_seed_ceiling_normalization_rejects_an_unrun_ceiling():
    assert seed_ceiling_normalize(1.0, 2.0) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="two-seed null"):
        seed_ceiling_normalize(1.0, 0.0)


def test_graft_shifts_only_the_named_groups_and_keeps_dtypes():
    base = {
        "model.layers.0.self_attn.q_proj.weight": torch.tensor([1.0, 1.0], dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.0.gate_proj.weight": torch.tensor([5.0, 5.0], dtype=torch.bfloat16),
    }
    delta = {
        "model.layers.0.self_attn.q_proj.weight": torch.tensor([1.0, 0.0], dtype=torch.float64),
        "model.layers.0.mlp.experts.0.gate_proj.weight": torch.tensor([0.0, 1.0], dtype=torch.float64),
    }
    grafted = graft(base, delta, groups=["attention"], alpha=2.0)
    assert grafted["model.layers.0.self_attn.q_proj.weight"].dtype == torch.bfloat16
    assert grafted["model.layers.0.self_attn.q_proj.weight"].float().tolist() == [3.0, 1.0]
    # Outside the named groups the base comes back untouched.
    assert grafted["model.layers.0.mlp.experts.0.gate_proj.weight"].float().tolist() == [5.0, 5.0]


def test_graft_rejects_unknown_groups():
    with pytest.raises(ValueError, match="unknown component groups"):
        graft({}, {}, groups=["brain_stem"])
