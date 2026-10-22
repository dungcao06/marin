# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 1a matrix tests: the run count, the M1/M4 field agreement, and manifests."""

from experiments.post_training.phase1a.matrix import (
    CONDITIONS,
    LR_SWEEP_POINTS,
    LR_SWEEP_TASKS,
    M4_ROLLOUTS_PER_PROMPT,
    METHODS,
    SEEDS,
    SEQUENCES_PER_STEP,
    _matched_fields,
    phase0c_lr_sweep,
    phase1a_runs,
)


def test_matrix_holds_thirty_two_runs_with_the_design_shape():
    runs = phase1a_runs(marin_commit="test-commit")
    assert len(runs) == 32
    trained = [run for run in runs if not run.manifest["procedure_null"]]
    nulls = [run for run in runs if run.manifest["procedure_null"]]
    assert len(trained) == 30
    assert len(nulls) == 2
    assert {(run.method, run.condition, run.seed) for run in trained} == {
        (method, condition, seed) for method in METHODS for condition in CONDITIONS for seed in SEEDS
    }
    assert {null.method for null in nulls} == set(METHODS)


def test_every_run_manifest_records_the_fixed_parameters_and_revisions():
    runs = phase1a_runs(marin_commit="test-commit")
    for run in runs:
        manifest = run.manifest
        assert manifest["marin_commit"] == "test-commit"
        assert manifest["fixed_parameters"]["sequences_per_step"] == SEQUENCES_PER_STEP
        assert manifest["fixed_parameters"]["m4_rollouts_per_prompt"] == M4_ROLLOUTS_PER_PROMPT
        assert manifest["fixed_parameters"]["response_cap_tokens"] == 4096
        assert manifest["fixed_parameters"]["packing"] is False
        assert manifest["data_revisions"]["wildchat_rewild"] == "46a5bb5"
        assert set(manifest["expert_revisions"]) == {"E_X", "E_R"}


def test_m1_and_m4_pairs_agree_on_every_matched_field():
    """Control C2: for one condition and seed, both methods draw the same budget."""
    runs = phase1a_runs(marin_commit="test-commit")
    by_identity = {(run.method, run.condition, run.seed): run for run in runs if not run.manifest["procedure_null"]}
    for condition in CONDITIONS:
        for seed in SEEDS:
            m1 = _matched_fields(by_identity[("M1", condition, seed)].skyrl_overrides)
            m4 = _matched_fields(by_identity[("M4", condition, seed)].skyrl_overrides)
            # The methods disagree only on how the step composes: M1 draws 32 fixed
            # rows, M4 draws 8 prompts x 4 rollouts; the products and caps match.
            assert m1.pop("n_samples_per_prompt") == 1
            assert m4.pop("n_samples_per_prompt") == M4_ROLLOUTS_PER_PROMPT
            assert m1.pop("train_batch_size") == 32
            assert m4.pop("train_batch_size") == 8
            assert m1 == m4
            assert m1["sequences_per_step"] == SEQUENCES_PER_STEP


def test_lr_sweep_holds_eight_runs_across_both_tasks():
    runs = phase0c_lr_sweep(marin_commit="test-commit")
    assert len(runs) == len(LR_SWEEP_POINTS) * len(LR_SWEEP_TASKS) == 8
    assert {run.skyrl_overrides["trainer"]["optim"]["lr"] for run in runs} == set(LR_SWEEP_POINTS)
    assert all(run.method == "M1" for run in runs)
    lrs_by_task = {
        condition: {run.skyrl_overrides["trainer"]["optim"]["lr"] for run in runs if run.condition == condition}
        for condition in LR_SWEEP_TASKS
    }
    assert all(points == set(LR_SWEEP_POINTS) for points in lrs_by_task.values())
