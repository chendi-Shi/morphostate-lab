"""Mechanism, leakage, deployment and complete-variant smoke checks."""

import copy
import json

import numpy as np
import pytest
import torch
import yaml
from scipy import sparse

from morphostate.budgeted import BudgetedLabelSet
from morphostate.data import load_dataset, save_dataset
from morphostate.geometry import make_synthetic_graphs, save_graphs
from morphostate.synthetic import make_synthetic
from morphostate.weak_training import (
    METHODS,
    _aggregate_groups,
    _aggregate_loss,
    _core_labels,
    _validate_graph_alignment,
    poisson_assignment_loss,
    predict_weak,
    run_weak_experiment,
)


def _label_set(budget=None, bg_budget=None):
    # Two real owners and an explicit background; observed hard owner is cell 0.
    support = sparse.csr_matrix([[1, 1, 1]], dtype=bool)
    return BudgetedLabelSet(support, [2.0], np.array([0]), 1,
                            edge_cost=[0.0, 1.0, 0.0], max_total_cost=budget,
                            background_index=2, max_background_mass=bg_budget)


def test_fixed_poisson_gradient_and_origin_are_correct():
    graph = _label_set()
    log_mu = torch.tensor([[0.2], [-0.3]], dtype=torch.float64, requires_grad=True)
    baseline = torch.zeros_like(log_mu)
    fixed = graph.fixed_labels(np.array([0]))
    loss, _ = poisson_assignment_loss(log_mu, baseline, graph, "fixed", fixed_labels=fixed)
    loss.backward()
    np.testing.assert_allclose(log_mu.grad, (np.exp(log_mu.detach().numpy()) - fixed[:-1]) / 2)
    zero = torch.zeros((2, 1), requires_grad=True)
    loss, _ = poisson_assignment_loss(zero, zero.detach(), graph, "joint_relative")
    assert loss.item() == 0.0
    loss.backward()
    assert torch.isfinite(zero.grad).all()


def test_joint_and_interval_use_same_marginals_but_different_conservation():
    graph = _label_set(bg_budget=0)
    log_mu = torch.zeros((2, 1), dtype=torch.float64, requires_grad=True)
    baseline = torch.ones_like(log_mu)
    bounds = graph.count_bounds(exact=True)
    joint, _ = poisson_assignment_loss(log_mu, baseline, graph, "joint_relative")
    interval, _ = poisson_assignment_loss(log_mu, baseline, graph, "interval_relative", bounds=bounds)
    assert float(interval - joint) == pytest.approx(1.0)
    # Joint target mass is exactly 2, while the outer box can duplicate it to 4.
    joint.backward()
    assert float(log_mu.grad.sum()) == pytest.approx(0.0)


def test_budget_limits_worst_target_and_relaxation_cannot_reduce_supremum():
    tight = _label_set(budget=0.2, bg_budget=0)
    loose = _label_set(budget=None, bg_budget=0)
    mu = torch.tensor([[0.0], [-2.0]], dtype=torch.float64, requires_grad=True)
    base = torch.zeros_like(mu)
    first, _ = poisson_assignment_loss(mu, base, tight, "joint_relative")
    second, _ = poisson_assignment_loss(mu, base, loose, "joint_relative")
    assert second > first
    first.backward()
    expected_target = np.array([[1.8], [0.2]])
    np.testing.assert_allclose(mu.grad, (np.exp(mu.detach().numpy()) - expected_target) / 2)


def test_optimistic_and_absolute_conventions_match_manual_linear_solutions():
    graph = _label_set(budget=0.5, bg_budget=0)
    mu = torch.tensor([[0.3], [-0.7]], dtype=torch.float64, requires_grad=True)
    base = torch.tensor([[0.1], [0.2]], dtype=torch.float64)
    optimistic, _ = poisson_assignment_loss(mu, base, graph, "optimistic_relative")
    relative, _ = poisson_assignment_loss(mu, base, graph, "joint_relative")
    assert optimistic <= relative
    coeff = np.vstack([-mu.detach().numpy(), [[0.0]]])
    support, _ = graph.maximize(coeff)
    absolute, _ = poisson_assignment_loss(mu, base, graph, "joint_absolute")
    assert float(absolute) == pytest.approx((float(mu.exp().sum()) + support) / 2)


def test_random_assignment_is_seeded_feasible_vertex():
    graph = _label_set(budget=0.5, bg_budget=0.1)
    mu = torch.zeros((2, 1), dtype=torch.float64, requires_grad=True)
    first, _ = poisson_assignment_loss(mu, mu.detach(), graph, "random_assignment",
                                       rng=np.random.default_rng(7))
    first.backward()
    target = mu.detach().exp().numpy() - mu.grad.numpy() * 2
    assert target[1, 0] <= 0.5 + 1e-7
    assert 1.9 - 1e-7 <= target.sum() <= 2.0 + 1e-7
    other = torch.zeros((2, 1), dtype=torch.float64, requires_grad=True)
    second, _ = poisson_assignment_loss(other, other.detach(), graph, "random_assignment",
                                        rng=np.random.default_rng(7))
    second.backward()
    np.testing.assert_allclose(other.grad, mu.grad)


def test_core_subset_does_not_count_ambiguous_molecules_as_certain():
    graph = BudgetedLabelSet(sparse.csr_matrix([[1, 0, 0], [1, 1, 1]], dtype=bool),
        [3, 5], np.array([0, 0]), 1, background_index=2)
    np.testing.assert_array_equal(_core_labels(graph), [[3], [0], [0]])


def test_aggregate_background_is_explicit_and_sum_loss_matches():
    graph = _label_set()
    assert _aggregate_groups(graph, "exclude") == []
    groups = _aggregate_groups(graph, "fixed_observed")
    np.testing.assert_array_equal(groups[0], [0, 1])
    mu = torch.tensor([[np.log(2)], [np.log(3)]], dtype=torch.float64, requires_grad=True)
    loss = _aggregate_loss(mu, graph.fixed_labels(np.array([0])), groups)
    assert float(loss) == pytest.approx((5 - 2 * np.log(5)) / 2)
    loss.backward()
    assert torch.isfinite(mu.grad).all()


def _fixture(tmp_path, methods=None, remove_covariates=False):
    data_path = tmp_path / "data"
    make_synthetic(data_path, donors=8, cells_per_donor=4, genes=4, features=4)
    graphs = make_synthetic_graphs(data_path, tmp_path / "graphs", ambiguous_fraction=0.5)
    cfg = {"data": "data", "graphs": "graphs", "output": "run", "seed": 7,
           "preprocess": {"max_dense_elements": 10000},
           "training": {"epochs": 2, "max_steps": 4, "patience": 2, "batch_size": 16,
                        "lr": 0.01, "device": "cpu"},
           "weak": {"methods": list(methods or ["fixed"]), "ranks": [2], "hidden_dims": [0],
                    "blocks_per_step": 2, "ensemble_members": 2, "remove_covariates": remove_covariates,
                    "max_bound_queries": 1000},
           "evaluation": {"bootstrap_repetitions": 10}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return load_dataset(data_path), graphs, cfg, path


def test_complete_method_matrix_runs_and_ensemble_budget_is_shared(tmp_path):
    torch.set_num_threads(1)
    data, graphs, config, path = _fixture(tmp_path, METHODS)
    summary = run_weak_experiment(path)
    assert set(summary["models"]) == set(METHODS) | {"covariate_nb"}
    assert "core_only" not in summary["evaluation_groups"]["primary"]
    assert not summary["model_metadata"]["core_only"]["eligible_for_primary_ranking"]
    selection = json.loads((tmp_path / "run" / "selection.json").read_text())
    for method in METHODS:
        candidate = selection[method]["candidates"][0]
        assert candidate["actual_total_optimizer_steps"] <= 4
        assert sum(member["configured_step_budget"] for member in candidate["members"]) == 4
        assert (tmp_path / "run" / f"{method}.pt").exists()
        assert (tmp_path / "run" / f"{method}_per_donor.csv").exists()
        for member in candidate["members"]:
            assert member["restored_best_validation_checkpoint"]
            assert member["inner_solve_seconds"] >= 0
    donors = json.loads((tmp_path / "run" / "split_donors.json").read_text())
    assert not set(donors["train"]) & set(donors["test"])
    assert not set(donors["val"]) & set(donors["test"])


@pytest.mark.parametrize("method,remove_covariates", [("joint_relative", False), ("ensemble", True)])
def test_checkpoint_inference_matches_test_predictions_without_rna(tmp_path, method, remove_covariates):
    data, _, _, path = _fixture(tmp_path, [method], remove_covariates)
    run_weak_experiment(path)
    with np.load(tmp_path / "run" / "test_predictions.npz") as stored:
        ids, expected = stored["cell_ids"].copy(), stored[method].copy()
    lookup = {cell: i for i, cell in enumerate(data.cells.cell_id)}
    rows = np.array([lookup[cell] for cell in ids])
    np.savez(tmp_path / "features.npz", cell_ids=ids, features=data.features[rows])
    np.savez(tmp_path / "covariates.npz", cell_ids=ids[::-1], covariates=data.covariates[rows[::-1]])
    (tmp_path / "data" / "counts.npz").unlink()
    predict_weak(tmp_path / "run" / f"{method}.pt", tmp_path / "features.npz",
                 tmp_path / "covariates.npz", tmp_path / "prediction.npz")
    with np.load(tmp_path / "prediction.npz") as actual:
        np.testing.assert_array_equal(actual["cell_ids"], ids)
        np.testing.assert_allclose(actual["rates"], expected, rtol=1e-5, atol=1e-6)


def test_graph_alignment_rejects_cross_patient_and_count_changes(tmp_path):
    data, graphs, _, _ = _fixture(tmp_path)
    bad = copy.copy(graphs[0])
    bad.donor_id = graphs[1].donor_id
    with pytest.raises(ValueError, match="donor"):
        _validate_graph_alignment([bad, *graphs[1:]], data)
    with pytest.raises(ValueError, match="more than one"):
        _validate_graph_alignment([graphs[0], *graphs], data)
    with pytest.raises(ValueError, match="cover every"):
        _validate_graph_alignment(graphs[:-1], data)
    data.counts.data[0] += 1
    with pytest.raises(ValueError, match="counts differ"):
        _validate_graph_alignment(graphs, data)


def test_gene_subset_is_explicitly_conditioned_and_train_only(tmp_path):
    _, _, config, path = _fixture(tmp_path)
    config["preprocess"].update(max_genes=3, min_train_cells=1)
    path.write_text(yaml.safe_dump(config))
    run_weak_experiment(path)
    provenance = json.loads((tmp_path / "run" / "provenance.json").read_text())
    assert provenance["n_genes"] == 3
    assert provenance["gene_subset_conditions_removed_molecules_on_fixed_owners"]


def test_rejects_arbitrary_cell_cap_and_oracle_input(tmp_path):
    data, _, config, path = _fixture(tmp_path)
    config["preprocess"]["max_cells_per_donor"] = 3
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="entire graph blocks"):
        run_weak_experiment(path)
    del config["preprocess"]["max_cells_per_donor"]
    path.write_text(yaml.safe_dump(config))
    data.provenance["feature_source"] = "rna_oracle"
    save_dataset(data, tmp_path / "data")
    with pytest.raises(ValueError, match="RNA-derived features"):
        run_weak_experiment(path)


def test_test_label_changes_do_not_change_selected_model_or_preprocessing(tmp_path):
    data, _, config, path = _fixture(tmp_path)
    config["weak"]["ranks"] = [1, 2]
    path.write_text(yaml.safe_dump(config))
    run_weak_experiment(path)
    donor_split = json.loads((tmp_path / "run" / "split_donors.json").read_text())
    test_rows = data.cells.donor_id.isin(donor_split["test"]).to_numpy()
    # Change only test outcomes, and regenerate matching test ownership data.
    multiplier = np.ones(len(data.cells))
    multiplier[test_rows] = 7
    data.counts = data.counts.multiply(multiplier[:, None]).tocsr()
    save_dataset(data, tmp_path / "data")
    save_graphs(make_synthetic_graphs(data), tmp_path / "graphs_changed")
    config["graphs"], config["output"] = "graphs_changed", "second"
    path.write_text(yaml.safe_dump(config))
    run_weak_experiment(path)
    first = torch.load(tmp_path / "run" / "fixed.pt", weights_only=True)
    second = torch.load(tmp_path / "second" / "fixed.pt", weights_only=True)
    assert first["selected"] == second["selected"]
    assert first["genes"] == second["genes"]
    for key in ("x_mean", "x_scale", "c_mean", "c_scale", "cov_to_image"):
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    for key in first["state_dict"]:
        torch.testing.assert_close(first["state_dict"][key], second["state_dict"][key], rtol=0, atol=0)


def test_predict_rejects_bad_output_suffix():
    with pytest.raises(ValueError, match="end in .npz"):
        predict_weak("unused", "unused", "unused", "no_suffix")


def test_ambiguity_training_preserves_the_entire_frozen_baseline(tmp_path, monkeypatch):
    import morphostate.weak_training as module
    _, _, _, path = _fixture(tmp_path, ["joint_relative"])
    original_fit = module.fit_nb_baseline
    snapshot = {}

    def capture(*args, **kwargs):
        model, history = original_fit(*args, **kwargs)
        snapshot.update({name: value.detach().clone() for name, value in model.state_dict().items()})
        return model, history

    monkeypatch.setattr(module, "fit_nb_baseline", capture)
    run_weak_experiment(path)
    stored = torch.load(tmp_path / "run" / "joint_relative.pt", weights_only=True)
    for name, value in snapshot.items():
        torch.testing.assert_close(stored["state_dict"]["baseline." + name], value, rtol=0, atol=0)
    torch.testing.assert_close(stored["state_dict"]["raw_dispersion"], snapshot["raw_dispersion"], rtol=0, atol=0)


def test_no_geometry_ablation_is_reproducible_and_does_not_mutate_graphs(tmp_path):
    data, graphs, _, path = _fixture(tmp_path, ["no_geometry", "no_background_budget"])
    expected = [(block.label_set.max_total_cost, block.label_set.max_background_mass) for block in graphs]
    run_weak_experiment(path)
    from morphostate.geometry import load_graphs
    reloaded = load_graphs(tmp_path / "graphs")
    assert [(block.label_set.max_total_cost, block.label_set.max_background_mass) for block in reloaded] == expected


def test_empty_core_evidence_fails_instead_of_assuming_all_ambiguous_rna_is_zero(tmp_path):
    _, _, config, path = _fixture(tmp_path, ["core_only"])
    make_synthetic_graphs(tmp_path / "data", tmp_path / "all_ambiguous", ambiguous_fraction=1.0)
    config["graphs"] = "all_ambiguous"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="no structurally single-owner"):
        run_weak_experiment(path)
