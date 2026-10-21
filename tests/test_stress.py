import json

import numpy as np
import pandas as pd
import pytest

from morphostate.data import load_dataset
from morphostate.geometry import load_graphs
from morphostate.stress import make_stress_case


def test_constructive_null_is_feasible_and_true_state_has_worse_relative_risk(tmp_path):
    result = make_stress_case(tmp_path / "case", donors=3, pairs_per_donor=2, state_prevalence=.5)
    assert all(row["null_feasible"] for row in result["blocks"])
    assert all(row["state_relative_objective"] > row["baseline_relative_objective"] for row in result["blocks"])
    assert all(row["state_relative_objective"] + 1e-8 >= row["null_witness_kl_lower_bound"] > 0
               for row in result["blocks"])
    data, blocks = load_dataset(tmp_path / "case" / "data"), load_graphs(tmp_path / "case" / "graphs")
    with np.load(tmp_path / "case" / "truth.npz", allow_pickle=False) as truth:
        np.testing.assert_array_equal(data.counts.toarray(), truth["true_counts"])
        np.testing.assert_array_equal(truth["true_counts"].sum(1), 80)
        np.testing.assert_array_equal(truth["null_counts"].sum(1), 80)
        assert np.any(truth["true_counts"] != truth["null_counts"])
        for number, block in enumerate(blocks):
            rows = truth[f"cell_rows_block{number:05d}"]
            null_labels = block.label_set.fixed_labels(truth[f"null_owner_block{number:05d}"])
            np.testing.assert_array_equal(null_labels[:-1], truth["null_counts"][rows])
    labels = pd.read_csv(tmp_path / "case" / "state_labels.csv")
    assert set(labels.cell_type) == {"type_A", "type_B"}
    np.testing.assert_array_equal(labels.label, labels.latent_state_score > 0)


def test_disallowing_null_reverses_the_constructive_preference(tmp_path):
    result = make_stress_case(tmp_path / "case", donors=3, pairs_per_donor=2,
                              state_prevalence=1, allow_null=False)
    assert all(not row["null_feasible"] for row in result["blocks"])
    assert all(row["state_relative_objective"] < 0 for row in result["blocks"])
    blocks = load_graphs(tmp_path / "case" / "graphs")
    with np.load(tmp_path / "case" / "truth.npz", allow_pickle=False) as truth:
        for number, block in enumerate(blocks):
            with pytest.raises(ValueError, match="budget"):
                block.label_set.fixed_labels(truth[f"null_owner_block{number:05d}"])


def test_zero_state_yields_equal_baseline_and_observed_counts(tmp_path):
    result = make_stress_case(tmp_path / "zero", donors=3, pairs_per_donor=2, state_strength=0, ambiguity=0)
    assert result["realized_positive_cell_prevalence"] == 0
    assert all(row["state_relative_objective"] == pytest.approx(0) for row in result["blocks"])
    with np.load(tmp_path / "zero" / "truth.npz") as truth:
        np.testing.assert_array_equal(truth["null_counts"], truth["true_counts"])


def test_stress_rejects_unrepresentable_ambiguity_and_existing_output(tmp_path):
    with pytest.raises(ValueError, match="ambiguity is too small"):
        make_stress_case(tmp_path / "impossible", donors=3, pairs_per_donor=2, ambiguity=.01)
    assert not (tmp_path / "impossible").exists()
    make_stress_case(tmp_path / "case", donors=3, pairs_per_donor=2)
    with pytest.raises(FileExistsError):
        make_stress_case(tmp_path / "case", donors=3, pairs_per_donor=2)


def test_synthetic_state_evaluation_recovers_predeclared_program(tmp_path):
    from morphostate.state_evaluation import evaluate_states
    make_stress_case(tmp_path / "case", donors=3, pairs_per_donor=2, state_prevalence=1)
    results = evaluate_states(tmp_path / "case" / "truth.npz", tmp_path / "case" / "state_labels.csv",
                               tmp_path / "case" / "programs.json", tmp_path / "evaluation", "synthetic")
    np.testing.assert_allclose(results.auroc, 1)
    np.testing.assert_allclose(results.average_precision, 1)


def test_stress_is_seed_reproducible(tmp_path):
    first = make_stress_case(tmp_path / "a", donors=3, pairs_per_donor=2, seed=34)
    second = make_stress_case(tmp_path / "b", donors=3, pairs_per_donor=2, seed=34)
    assert first == second
    assert json.loads((tmp_path / "a" / "programs.json").read_text()) == {
        "positive_state": {"STATE_UP": 1, "STATE_DOWN": -1}}
    with np.load(tmp_path / "a" / "truth.npz") as a, np.load(tmp_path / "b" / "truth.npz") as b:
        assert set(a.files) == set(b.files)
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
