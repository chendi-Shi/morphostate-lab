import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from morphostate.budgeted import BudgetedLabelSet
from morphostate.geometry import GraphBlock, save_graphs
from morphostate.state_evaluation import binary_scores, evaluate_contrasts, evaluate_states


def test_auc_and_ap_handle_ties_and_single_class():
    assert binary_scores([0, 1, 0, 1], [1, 1, 1, 1])["auroc"] == .5
    assert binary_scores([0, 1, 0, 1], [1, 1, 1, 1])["average_precision"] == .5
    assert binary_scores([0, 0, 1, 1], [0, 0, 1, 1])["auroc"] == 1
    assert binary_scores([0, 0], [1, 2])["auroc"] is None


def test_independent_state_evaluation_does_not_fit_test_labels(tmp_path):
    pred, labels, programs = tmp_path / "pred.npz", tmp_path / "labels.csv", tmp_path / "programs.json"
    np.savez(pred, cell_ids=np.array(["a", "b", "c", "d"]), gene_ids=np.array(["g"]), rates=np.array([[1], [2], [8], [10]]))
    pd.DataFrame({"cell_id": ["d", "c", "b", "a"], "donor_id": ["p"] * 4,
                  "cell_type": ["epithelial"] * 4, "state": ["cycling"] * 4, "label": [1, 1, 0, 0]}).to_csv(labels, index=False)
    programs.write_text(json.dumps({"cycling": {"g": 1}}))
    result = evaluate_states(pred, labels, programs, tmp_path / "out", "protein")
    assert result.auroc.iloc[0] == 1
    with pytest.raises(ValueError, match="independent"):
        evaluate_states(pred, labels, programs, tmp_path / "bad", "rna")


def _state_files(tmp_path):
    pred, labels, programs = tmp_path / "pred.npz", tmp_path / "labels.csv", tmp_path / "programs.json"
    np.savez(pred, cell_ids=np.array(["a", "b", "c", "d"]), gene_ids=np.array(["g"]), rates=np.array([[1], [2], [8], [10]]))
    frame = pd.DataFrame({"cell_id": ["a", "b", "c"], "donor_id": ["p", "p", "q"],
                          "cell_type": ["epithelial"] * 3, "state": ["cycling"] * 3, "label": [0, 1, 1]})
    frame.to_csv(labels, index=False)
    programs.write_text(json.dumps({"cycling": {"g": 1}}))
    return pred, labels, programs, frame


def test_state_evaluation_reports_labeled_and_evaluable_coverage(tmp_path):
    pred, labels, programs, _ = _state_files(tmp_path)
    evaluate_states(pred, labels, programs, tmp_path / "out", "protein")
    meta = json.loads((tmp_path / "out" / "provenance.json").read_text())
    assert meta["labeled_cell_fraction"] == .75
    assert meta["independent_donors"] == 2
    macro = pd.read_csv(tmp_path / "out" / "state_macro.csv")
    assert macro.n_donors.iloc[0] == 2
    assert macro.n_evaluable_donors.iloc[0] == 1


def test_cell_cannot_be_counted_as_two_patients_across_state_labels(tmp_path):
    pred, labels, programs, frame = _state_files(tmp_path)
    extra = frame.iloc[[0]].copy()
    extra["state"], extra["donor_id"] = "other", "fake_patient"
    pd.concat([frame, extra]).to_csv(labels, index=False)
    with pytest.raises(ValueError, match="cannot change donor_id"):
        evaluate_states(pred, labels, programs, tmp_path / "out", "protein")


def test_state_validation_rejects_empty_labels_and_malformed_programs(tmp_path):
    pred, labels, programs, frame = _state_files(tmp_path)
    frame.iloc[:0].to_csv(labels, index=False)
    with pytest.raises(ValueError, match="complete"):
        evaluate_states(pred, labels, programs, tmp_path / "out", "protein")
    frame.to_csv(labels, index=False)
    programs.write_text("[]")
    with pytest.raises(ValueError, match="nonempty mapping"):
        evaluate_states(pred, labels, programs, tmp_path / "out", "protein")


def test_budgeted_state_contrast_bounds_include_fixed_assignment(tmp_path):
    graph = BudgetedLabelSet(sparse.csr_matrix([[1, 1, 1]], dtype=bool), [2.], np.array([0]), 1,
                            edge_cost=[0., 1., 0.], max_total_cost=.5, background_index=2,
                            max_background_mass=0)
    save_graphs([GraphBlock("slide", "patient", ["a", "b"], ["g"], graph, np.array([0]))], tmp_path / "graphs")
    coefficients = tmp_path / "coefficients.csv"
    pd.DataFrame({"contrast": ["state", "state"], "cell_id": ["a", "b"],
                  "gene_id": ["g", "g"], "coefficient": [1., -1.]}).to_csv(coefficients, index=False)
    result = evaluate_contrasts(tmp_path / "graphs", coefficients, tmp_path / "bounds.csv")
    assert result[0]["lower"] == pytest.approx(1.)
    assert result[0]["upper"] == pytest.approx(2.)
    assert result[0]["fixed_assignment_value"] == pytest.approx(2.)
    assert result[0]["sign"] == "positive"
    assert result[0]["n_donors"] == 1
    assert result[0]["donor_id"] == "patient"
    with pytest.raises(ValueError, match="new .csv"):
        evaluate_contrasts(tmp_path / "graphs", coefficients, tmp_path / "bounds.csv")
