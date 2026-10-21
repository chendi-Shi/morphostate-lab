import numpy as np
import pandas as pd
import pytest

from morphostate.selective import evaluate_bound_risk


def test_unresolved_and_empty_acceptance_are_reported_without_cherry_picking(tmp_path):
    bounds = pd.DataFrame({"contrast": ["a", "b", "c"], "lower": [2., -2., -5.],
        "upper": [3., 2., -2.], "fixed_assignment_value": [2., 1., -3.], "n_donors": [1, 1, 1],
        "donor_id": ["d1", "d1", "d2"]})
    truth = pd.DataFrame({"contrast": ["c", "b", "a"], "donor_id": ["d2", "d1", "d1"], "truth": [2., 1., 2.5]})
    bounds.to_csv(tmp_path / "bounds.csv", index=False)
    truth.to_csv(tmp_path / "truth.csv", index=False)
    curve = evaluate_bound_risk(tmp_path / "bounds.csv", tmp_path / "truth.csv", tmp_path / "out", [0, 1, 4], "synthetic")
    assert np.isnan(curve.loc[0, "risk"])
    assert curve.loc[0, "n_donors_with_accepted_contrasts"] == 0
    assert curve.loc[1, "coverage"] == .25  # one of two in d1; none in d2
    assert curve.loc[4, "risk"] == .5
    assert curve.loc[4, "coverage"] == .75
    truth.loc[0, "truth"] = 0
    truth.to_csv(tmp_path / "truth.csv", index=False)
    with pytest.raises(ValueError, match="Zero-truth"):
        evaluate_bound_risk(tmp_path / "bounds.csv", tmp_path / "truth.csv", tmp_path / "bad", [1], "synthetic")


def _files(tmp_path):
    bounds = pd.DataFrame({"contrast": ["a", "b"], "donor_id": ["d1", "d1"],
        "lower": [1., -3.], "upper": [2., -1.], "fixed_assignment_value": [1.5, -2.], "n_donors": [1, 1]})
    truth = pd.DataFrame({"contrast": ["a", "b"], "donor_id": ["d1", "d1"], "truth": [1.5, -2.]})
    bounds.to_csv(tmp_path / "bounds.csv", index=False)
    truth.to_csv(tmp_path / "truth.csv", index=False)
    return bounds, truth


def test_missing_difficult_truth_or_relabeling_patient_cannot_improve_curve(tmp_path):
    _, truth = _files(tmp_path)
    truth.iloc[:1].to_csv(tmp_path / "truth.csv", index=False)
    with pytest.raises(ValueError, match="cover every bound"):
        evaluate_bound_risk(tmp_path / "bounds.csv", tmp_path / "truth.csv", tmp_path / "out", [1], "synthetic")
    truth.loc[1, "donor_id"] = "invented_second_patient"
    truth.to_csv(tmp_path / "truth.csv", index=False)
    with pytest.raises(ValueError, match="graph-derived donor_id"):
        evaluate_bound_risk(tmp_path / "bounds.csv", tmp_path / "truth.csv", tmp_path / "out", [1], "synthetic")


def test_infeasible_fixed_label_and_multi_donor_contrast_are_rejected(tmp_path):
    bounds, _ = _files(tmp_path)
    bounds.loc[0, "fixed_assignment_value"] = 9.
    bounds.to_csv(tmp_path / "bounds.csv", index=False)
    with pytest.raises(ValueError, match="inside its feasible"):
        evaluate_bound_risk(tmp_path / "bounds.csv", tmp_path / "truth.csv", tmp_path / "out", [1], "synthetic")
    bounds.loc[0, "fixed_assignment_value"] = 1.5
    bounds.loc[0, "n_donors"] = 2
    bounds.to_csv(tmp_path / "bounds.csv", index=False)
    with pytest.raises(ValueError, match="one donor per contrast"):
        evaluate_bound_risk(tmp_path / "bounds.csv", tmp_path / "truth.csv", tmp_path / "out", [1], "synthetic")


def test_changing_truth_changes_risk_but_never_selection(tmp_path):
    _, truth = _files(tmp_path)
    first = evaluate_bound_risk(tmp_path / "bounds.csv", tmp_path / "truth.csv", tmp_path / "first", [0, 1, 2], "synthetic")
    truth["truth"] = -truth["truth"]
    truth.to_csv(tmp_path / "truth.csv", index=False)
    second = evaluate_bound_risk(tmp_path / "bounds.csv", tmp_path / "truth.csv", tmp_path / "second", [0, 1, 2], "synthetic")
    np.testing.assert_array_equal(first.coverage, second.coverage)
    np.testing.assert_array_equal(first.accepted_contrasts, second.accepted_contrasts)
    assert first.loc[2, "risk"] == 0
    assert second.loc[2, "risk"] == 1
    assert first.loc[2, "accepted_contrast_fixed_risk"] == 0
