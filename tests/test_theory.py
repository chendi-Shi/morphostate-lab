"""Numerical witnesses for finite-set statements, not statistical coverage tests."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from morphostate.budgeted import BudgetedLabelSet
from morphostate.data import load_dataset
from morphostate.geometry import load_graphs
from morphostate.stress import make_stress_case
from morphostate.theory import (
    baseline_membership,
    diagnose_means,
    diagnose_predictions,
    support_dual_certificate,
)


def _set(cost_budget=None, background_budget=0):
    return BudgetedLabelSet(sparse.csr_matrix([[1, 1, 1]], dtype=bool), [2.], np.array([0]), 1,
        edge_cost=[0., 1., 0.], max_total_cost=cost_budget,
        background_index=2, max_background_mass=background_budget)


def test_support_dual_certificate_preserves_negative_equality_and_positive_budget_weights():
    labels = _set(cost_budget=.4)
    coefficients = np.array([[-1.], [2.], [-4.]])
    certificate = support_dual_certificate(labels, coefficients)
    assert certificate["primal_value"] == pytest.approx(-.8)
    assert certificate["dual_value"] == pytest.approx(-.8)
    assert abs(certificate["duality_gap"]) <= certificate["tolerance"]
    assert certificate["equality_dual"][0] == pytest.approx(-1.)
    assert certificate["budget_dual"][0] == pytest.approx(3.)
    assert min(certificate["budget_dual"]) >= -certificate["tolerance"]
    assert certificate["minimum_dual_slack"] >= -certificate["tolerance"]
    assert certificate["primal_feasibility"]["maximum_mass_residual"] < 1e-7


def test_support_dual_without_budgets_matches_rowwise_solver():
    labels = _set(background_budget=None)
    coefficients = np.array([[-2.], [-3.], [-1.]])
    certificate = support_dual_certificate(labels, coefficients)
    assert certificate["budget_dual"] == []
    assert certificate["primal_value"] == labels.support_function(coefficients) == -2.
    assert certificate["equality_dual"] == [-1.]


def test_zero_duality_gap_alone_does_not_certify_an_infeasible_primal(monkeypatch):
    labels = _set(background_budget=None)
    fake = SimpleNamespace(success=True, x=np.zeros(3),
                           eqlin=SimpleNamespace(marginals=np.zeros(1)),
                           ineqlin=SimpleNamespace(marginals=np.zeros(0)))
    monkeypatch.setattr("morphostate.theory.linprog", lambda *args, **kwargs: fake)
    with pytest.raises(RuntimeError, match="mass conservation"):
        support_dual_certificate(labels, np.zeros((3, 1)))


def test_linf_membership_leaves_background_unconstrained_except_its_budget():
    labels = _set(background_budget=None)
    feasible = baseline_membership(labels, [[.5], [.25]])
    assert feasible["baseline_feasible_within_tolerance"]
    assert feasible["linf_residual"] == pytest.approx(0)
    np.testing.assert_allclose(feasible["witness_counts"], [[.5], [.25]])
    # The remaining 1.25 molecular mass belongs to background, not to an
    # implicitly imposed zero background label.
    assert feasible["witness_counts"].sum() < labels.mass.sum()
    infeasible = baseline_membership(labels, [[3.], [0.]])
    assert not infeasible["baseline_feasible_within_tolerance"]
    assert infeasible["linf_residual"] == pytest.approx(1.)
    blocked = baseline_membership(_set(background_budget=0), [[.5], [.25]])
    assert not blocked["baseline_feasible_within_tolerance"]
    assert blocked["linf_residual"] == pytest.approx(.625)


def test_projected_membership_can_hold_while_full_count_membership_fails():
    labels = _set(background_budget=0)
    baseline = np.array([[2.], [2.]])  # four desired counts, but only two observed
    full = baseline_membership(labels, baseline)
    contrast = np.array([[1.], [-1.]])
    projected = baseline_membership(labels, baseline, projection=contrast)
    assert not full["baseline_feasible_within_tolerance"]
    assert full["linf_residual"] == pytest.approx(1.)
    assert projected["baseline_feasible_within_tolerance"]
    assert projected["projection_dimension"] == 1
    assert projected["distance_space"] == "projected_contrasts"
    np.testing.assert_allclose(contrast.T @ (projected["witness_counts"] - baseline), 0., atol=1e-8)
    assert projected["witness_full_count_linf_residual"] > 0
    # Signed column scaling is caller-defined; it scales the reported moment
    # residual, not the underlying feasible ownership set.
    mass_axis = np.array([[2.], [2.]])
    missing_mass = baseline_membership(labels, baseline, projection=mass_axis)
    assert missing_mass["linf_residual"] == pytest.approx(4.)


@pytest.mark.parametrize("projection", [np.ones((3, 1)), np.ones((2, 0)), [[1., np.nan], [0., 1.]]])
def test_projected_membership_rejects_invalid_axes_and_values(projection):
    with pytest.raises(ValueError, match="projection"):
        baseline_membership(_set(), [[1.], [1.]], projection=projection)


def test_same_marginal_box_is_an_outer_relaxation_and_feasible_baseline_gives_kl_bound():
    labels = _set(background_budget=0)
    candidate, baseline = np.array([[.5], [.5]]), np.array([[1.], [1.]])
    result, adversarial, witness = diagnose_means(labels, candidate, baseline)
    assert result["membership"]["baseline_feasible_within_tolerance"]
    assert result["joint_relative_risk"] >= result["poisson_kl_baseline_to_candidate"] - 1e-8
    assert result["marginal_box_relative_risk"] > result["joint_relative_risk"]
    assert result["box_minus_joint"] == pytest.approx(np.log(2))
    assert adversarial.sum() == pytest.approx(2)
    np.testing.assert_allclose(witness, baseline)
    assert result["membership_distance_risk_correction"] == pytest.approx(0)
    same, _, _ = diagnose_means(labels, baseline, baseline)
    assert same["joint_relative_risk"] == 0.
    assert same["poisson_kl_baseline_to_candidate"] == 0.


def test_near_zero_membership_does_not_incorrectly_assert_exact_kl_collapse():
    labels = BudgetedLabelSet(sparse.csr_matrix([[1, 0, 0], [0, 1, 0]], dtype=bool),
        [1., 1.], np.array([0, 0]), 1, background_index=2)
    baseline = np.full((2, 1), 1. + 5e-8)
    result, _, _ = diagnose_means(labels, np.full((2, 1), .1), baseline)
    assert result["membership"]["baseline_feasible_within_tolerance"]
    # This baseline is mathematically outside the singleton set despite the
    # numerical flag. An uncorrected exact-membership claim would be false.
    assert result["joint_relative_risk"] < result["poisson_kl_baseline_to_candidate"] - 1e-8
    assert result["membership_distance_risk_correction"] > 1e-8
    assert result["joint_relative_risk"] >= result["poisson_kl_minus_distance_correction"] - 1e-8


@pytest.mark.parametrize("allow_null", [True, False])
def test_constructive_stress_reports_both_null_feasible_and_excluded_cases(tmp_path, allow_null):
    make_stress_case(tmp_path / "case", donors=3, pairs_per_donor=2, state_strength=.5,
                     state_prevalence=.5, ambiguity=.6, allow_null=allow_null)
    blocks = load_graphs(tmp_path / "case" / "graphs")
    with np.load(tmp_path / "case" / "truth.npz") as truth:
        positions = truth["cell_rows_block00000"]
        candidate, baseline = truth["true_counts"][positions], truth["null_counts"][positions]
    result, _, _ = diagnose_means(blocks[0].label_set, candidate, baseline)
    assert result["membership"]["baseline_feasible_within_tolerance"] is allow_null
    if allow_null:
        assert result["joint_relative_risk"] >= result["poisson_kl_baseline_to_candidate"] - 1e-7
        assert result["joint_relative_risk"] > 0
    else:
        assert result["joint_relative_risk"] < 0
        assert result["membership"]["linf_residual"] > 0


def test_prediction_roundtrip_preserves_axes_exposure_and_conditional_gene_policy(tmp_path):
    make_stress_case(tmp_path / "case", donors=3, pairs_per_donor=2)
    data = load_dataset(tmp_path / "case" / "data")
    with np.load(tmp_path / "case" / "truth.npz") as truth:
        # Reorder rows and subset/reorder genes. Reference library=40 means
        # exposure=2 and exported rates must be half the target count means.
        rows = np.arange(len(truth["cell_ids"]))[::-1]
        genes = np.array([1, 0])
        np.savez(tmp_path / "predictions.npz", cell_ids=truth["cell_ids"][rows], gene_ids=truth["gene_ids"][genes],
                 candidate=truth["true_counts"][np.ix_(rows, genes)] / 2,
                 baseline=truth["null_counts"][np.ix_(rows, genes)] / 2)
    results = diagnose_predictions(tmp_path / "case" / "data", tmp_path / "case" / "graphs",
        tmp_path / "predictions.npz", tmp_path / "diagnostics", 40., "candidate", "baseline")
    assert len(results) == 3
    for result in results:
        assert result["n_genes"] == 2
        assert result["gene_subset_policy"] == "condition_removed_molecules_on_observed_owners"
        assert result["membership"]["baseline_feasible_within_tolerance"]
    saved = json.loads((tmp_path / "diagnostics" / "diagnostics.json").read_text())
    assert saved["reference_library"] == 40.
    assert saved["model"] == "candidate"
    with np.load(tmp_path / "diagnostics" / "witnesses.npz") as witnesses:
        np.testing.assert_array_equal(witnesses["gene_ids"], [data.genes[1], data.genes[0]])
        assert witnesses["block00000_adversarial_labels"].shape == (5, 2)
        assert witnesses["block00000_closest_baseline_labels"].shape == (4, 2)


def test_prediction_diagnostics_reject_partial_coupled_blocks(tmp_path):
    make_stress_case(tmp_path / "case", donors=3, pairs_per_donor=2)
    with np.load(tmp_path / "case" / "truth.npz") as truth:
        np.savez(tmp_path / "partial.npz", cell_ids=truth["cell_ids"][1:], gene_ids=truth["gene_ids"],
                 joint_relative=truth["true_counts"][1:], covariate_nb=truth["null_counts"][1:])
    with pytest.raises(ValueError, match="complete graph blocks"):
        diagnose_predictions(tmp_path / "case" / "data", tmp_path / "case" / "graphs",
            tmp_path / "partial.npz", tmp_path / "diagnostics", 80.)
    assert not (tmp_path / "diagnostics").exists()


def test_relative_diagnostics_reject_nonpositive_means():
    with pytest.raises(ValueError, match="strictly positive"):
        diagnose_means(_set(), np.array([[0.], [1.]]), np.ones((2, 1)))
