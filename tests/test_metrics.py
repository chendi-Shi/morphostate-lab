import numpy as np
import pandas as pd
from scipy import sparse

from morphostate.experiment import fit_scaler, select_genes
from morphostate.metrics import (
    bootstrap_donor_gain,
    donor_metrics,
    mean_gene_pcc,
    poisson_deviance,
    within_donor_type_permutation,
)


def test_deviance_zero_and_zero_count_finite():
    y = np.array([[0., 2.], [4., 1.]])
    assert np.allclose(poisson_deviance(y, y), 0, atol=1e-8)
    assert np.isfinite(poisson_deviance(y, np.ones_like(y))).all()


def test_float32_constant_predictions_have_undefined_correlation():
    y = np.arange(120, dtype=np.float32).reshape(40, 3)
    prediction = np.tile(np.array([0.1234567, 1.234567, 4.567891], dtype=np.float32), (40, 1))
    assert mean_gene_pcc(y, prediction) is None


def test_bootstrap_uses_donors_not_cells():
    base = pd.DataFrame({"donor_id": ["A", "B"], "poisson_deviance": [10., 2.]})
    candidate = pd.DataFrame({"donor_id": ["B", "A"], "poisson_deviance": [3., 7.]})
    out = bootstrap_donor_gain(base, candidate, repetitions=50)
    assert out["mean_deviance_gain"] == 1.0
    assert out["positive_donors"] == 1


def test_permutation_cannot_cross_donor_or_cell_type():
    cells = pd.DataFrame({"donor_id": ["A"] * 10 + ["B"] * 10,
                          "cell_type": ["T", "M"] * 10})
    indices = within_donor_type_permutation(cells, seed=7)
    assert np.array_equal(np.sort(indices), np.arange(20))
    assert np.array_equal(cells.to_numpy(), cells.iloc[indices].to_numpy())


def test_gene_filter_only_sees_training_counts():
    train = sparse.csr_matrix([[1, 0, 1], [1, 0, 8], [1, 0, 2]])
    indices = select_genes(train, np.ones(3), max_genes=1, min_cells=2)
    assert indices.tolist() == [2]
    mean, scale = fit_scaler(np.array([[0., 2.], [2., 2.]]))
    assert np.array_equal(mean, [1., 2.])
    assert np.array_equal(scale, [1., 1.])


def test_within_type_removes_pure_type_mean_signal():
    y = np.array([[1, 8], [1, 8], [8, 1], [8, 1]])
    cells = pd.DataFrame({"donor_id": ["A"] * 4, "cell_type": ["T", "T", "M", "M"]})
    out = donor_metrics(y, y, np.ones(4), cells)
    assert out.gene_pcc.iloc[0] == 1
    assert pd.isna(out.within_type_pcc.iloc[0])


def test_unknown_types_have_no_within_type_score_even_with_perfect_prediction():
    y = np.array([[1., 8.], [2., 6.], [8., 1.], [4., 2.]])
    cells = pd.DataFrame({"donor_id": ["A"] * 4,
                          "cell_type": ["unknown", "Unassigned", "N/A", " unannotated "]})
    out = donor_metrics(y, y, np.ones(4), cells).iloc[0]
    assert np.isclose(out.gene_pcc, 1)
    assert pd.isna(out.within_type_pcc)
    assert out.known_cell_type_fraction == 0
    assert out.within_type_evaluable_fraction == 0


def test_within_type_score_excludes_unknown_and_singleton_types():
    # The known repeated type is predicted correctly. Unknown cells are
    # deliberately anticorrelated, and the singleton cannot define a state.
    y = np.array([[1., 8.], [2., 6.], [4., 2.], [10., 1.], [1., 10.], [20., 20.]])
    pred = y.copy()
    pred[3:5] = pred[3:5][::-1]
    pred[5] = [100., 1.]
    cells = pd.DataFrame({"donor_id": ["A"] * 6,
                          "cell_type": ["T", "T", "T", "unknown", "unknown", "M"]})
    out = donor_metrics(y, pred, np.ones(6), cells).iloc[0]
    assert np.isclose(out.within_type_pcc, 1)
    assert out.gene_pcc < 1
    assert np.isclose(out.known_cell_type_fraction, 4 / 6)
    assert out.within_type_evaluable_fraction == 0.5


def test_within_type_minimum_count_is_per_donor_not_global():
    y = np.array([[1., 8.], [2., 6.], [4., 2.], [10., 1.]])
    cells = pd.DataFrame({"donor_id": ["A", "A", "B", "B"],
                          "cell_type": ["T", "M", "T", "M"]})
    out = donor_metrics(y, y, np.ones(4), cells)
    assert out.within_type_pcc.isna().all()
    assert (out.known_cell_type_fraction == 1).all()
    assert (out.within_type_evaluable_fraction == 0).all()


def test_permutation_keeps_unknown_cells_fixed_with_noncontiguous_known_rows():
    cells = pd.DataFrame({"donor_id": ["A"] * 7,
                          "cell_type": ["T", "unknown", "T", "NA", "T", "Unassigned", "M"]},
                         index=np.arange(10, 17))
    indices = within_donor_type_permutation(cells, seed=7)
    np.testing.assert_array_equal(indices[[1, 3, 5, 6]], [1, 3, 5, 6])
    np.testing.assert_array_equal(np.sort(indices[[0, 2, 4]]), [0, 2, 4])
    np.testing.assert_array_equal(np.sort(indices), np.arange(7))
