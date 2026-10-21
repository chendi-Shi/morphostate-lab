import itertools

import numpy as np
import pytest
from scipy import sparse

from morphostate.ambiguity import MoleculeLabelSet


def test_one_shared_transcript_does_not_duplicate_mass():
    labels = MoleculeLabelSet(sparse.csr_matrix([[1, 1]]), [1], [0], n_genes=1)
    lower, upper = labels.count_bounds()
    np.testing.assert_array_equal(lower, [[0], [0]])
    np.testing.assert_array_equal(upper, [[1], [1]])
    coefficients = np.ones((2, 1))
    value, argmax = labels.maximize(coefficients)
    assert value == 1
    assert argmax.sum() == 1
    assert (coefficients * upper).sum() == 2  # Impossible joint label from interval box.
    np.testing.assert_array_equal(argmax, [[1], [0]])  # Deterministic tie rule.


def test_exact_contrast_identifiability_and_single_owner_counts():
    labels = MoleculeLabelSet(
        sparse.csr_matrix([[1, 1], [0, 1]]), [2, 3], [0, 0], n_genes=1
    )
    assert labels.contrast_bounds([[1], [1]]) == (5.0, 5.0)
    assert labels.contrast_bounds([[1], [-1]]) == (-5.0, -1.0)
    lower, upper = labels.count_bounds()
    np.testing.assert_array_equal(lower, [[0], [3]])
    np.testing.assert_array_equal(upper, [[2], [5]])


def test_linear_support_matches_exhaustive_assignment_and_preserves_each_gene():
    support = sparse.csr_matrix([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    mass = np.array([2.0, 3.0, 0.5])
    genes = np.array([0, 1, 0])
    labels = MoleculeLabelSet(support, mass, genes, n_genes=2)
    coefficients = np.array([[2.0, -3.0], [-1.0, 4.0], [3.0, 1.0]])
    all_values = []
    owner_sets = [support.getrow(row).indices for row in range(3)]
    for owners in itertools.product(*owner_sets):
        candidate = np.zeros((3, 2))
        np.add.at(candidate, (owners, genes), mass)
        all_values.append((candidate * coefficients).sum())
    value, maximizing = labels.maximize(coefficients)
    assert value == max(all_values)
    assert labels.support_function(coefficients) == value
    assert labels.contrast_bounds(coefficients) == (min(all_values), max(all_values))
    np.testing.assert_allclose(maximizing.sum(axis=0), np.bincount(genes, weights=mass, minlength=2))
    np.testing.assert_array_equal(labels.argmax_labels(coefficients), maximizing)


def test_permutations_and_row_aggregation_preserve_feasible_support():
    support = sparse.csr_matrix([[1, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 0]])
    mass = np.array([2.0, 3.0, 0.5, 1.5])
    genes = np.array([0, 1, 0, 1])
    coefficients = np.array([[2.0, -3.0], [-1.0, 4.0], [3.0, 1.0]])
    labels = MoleculeLabelSet(support, mass, genes, n_genes=2)
    row_order, cell_order = np.array([2, 0, 3, 1]), np.array([2, 0, 1])
    permuted = MoleculeLabelSet(
        support[row_order][:, cell_order], mass[row_order], genes[row_order], n_genes=2
    )
    value, argmax = labels.maximize(coefficients)
    permuted_value, permuted_argmax = permuted.maximize(coefficients[cell_order])
    assert permuted_value == value
    np.testing.assert_array_equal(permuted_argmax, argmax[cell_order])  # No coefficient ties here.
    aggregated = labels.aggregate()
    assert aggregated.n_observations == 3
    assert aggregated.contrast_bounds(coefficients) == labels.contrast_bounds(coefficients)
    for actual, expected in zip(aggregated.count_bounds(), labels.count_bounds()):
        np.testing.assert_array_equal(actual, expected)


def test_zero_mass_unused_genes_and_background_column_are_supported():
    labels = MoleculeLabelSet(
        sparse.csr_matrix([[1, 0, 1], [0, 1, 0]]), [0, 3], [0, 1], n_genes=3
    )
    # Column 2 can be designated background by the caller; no implicit discard.
    coefficients = np.array([[0, 1, 0], [0, 2, 0], [100, 3, 0]])
    value, argmax = labels.maximize(coefficients)
    assert value == 6
    np.testing.assert_array_equal(argmax.sum(axis=0), [0, 3, 0])


@pytest.mark.parametrize(
    "support,mass,genes,n_genes,match",
    [
        (sparse.csr_matrix([[0, 0]]), [1], [0], 1, "candidate owner"),
        (sparse.csr_matrix([[0.5, 1]]), [1], [0], 1, "zero/one"),
        (sparse.csr_matrix([[1, 0]]), [-1], [0], 1, "nonnegative"),
        (sparse.csr_matrix([[1, 0]]), [np.nan], [0], 1, "finite"),
        (sparse.csr_matrix([[1, 0]]), [1, 2], [0], 1, "matching"),
        (sparse.csr_matrix([[1, 0]]), [1], [0.0], 1, "integer"),
        (sparse.csr_matrix([[1, 0]]), [1], [1], 1, "entries"),
        (sparse.csr_matrix([[1, 0]]), [1], [0], 0, "positive integer"),
    ],
)
def test_invalid_label_sets_are_rejected(support, mass, genes, n_genes, match):
    with pytest.raises(ValueError, match=match):
        MoleculeLabelSet(support, mass, genes, n_genes)


def test_dense_support_and_invalid_coefficients_are_rejected():
    with pytest.raises(TypeError, match="CSR"):
        MoleculeLabelSet(np.array([[1, 1]]), [1], [0], 1)
    labels = MoleculeLabelSet(sparse.csr_matrix([[1, 1]]), [1], [0], 1)
    for coefficients in (np.ones(2), np.ones((2, 2)), [[np.nan], [1]]):
        with pytest.raises(ValueError, match="coefficients"):
            labels.support_function(coefficients)
