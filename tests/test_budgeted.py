import itertools

import numpy as np
import pytest
from scipy import sparse

from morphostate.budgeted import BudgetedLabelSet


def graph(mass=(1.0, 1.0), genes=(0, 1), cost=0.5, background=0.2):
    return BudgetedLabelSet(sparse.csr_matrix(np.ones((2, 3), dtype=bool)), mass, np.array(genes), 2,
                            edge_cost=np.array([0, 1, 0, 0, 1, 0]), max_total_cost=cost,
                            background_index=2, max_background_mass=background)


def test_budgeted_fractional_optimum_and_conservation():
    labels = graph()
    coefficients = np.array([[0, 0], [2, 2], [1, 1]])
    value, y = labels.maximize(coefficients)
    assert value == pytest.approx(1.2)
    np.testing.assert_allclose(y.sum(axis=0), [1, 1])
    assert y[1].sum() == pytest.approx(0.5)
    assert y[2].sum() == pytest.approx(0.2)
    assert value == pytest.approx(np.sum(coefficients * y))


def test_exact_marginals_are_tighter_than_independent_box():
    labels = graph(background=0)
    lower, upper = labels.count_bounds()
    loose_lower, loose_upper = labels.count_bounds(exact=False)
    np.testing.assert_allclose(lower[0], [0.5, 0.5])
    np.testing.assert_allclose(upper[1], [0.5, 0.5])
    np.testing.assert_allclose(upper[2], 0)
    np.testing.assert_allclose(loose_upper, 1)
    np.testing.assert_allclose(loose_lower, 0)
    with pytest.raises(ValueError, match="max_queries"):
        labels.count_bounds(max_queries=1)


def test_integral_budget_lp_matches_exhaustive_owner_assignments():
    labels = graph(cost=1, background=0)
    coefficients = np.array([[1.3, -0.2], [0.3, 2.1], [5, 5]])
    values = []
    for owners in itertools.product(range(2), repeat=2):
        if sum(owners) <= 1:
            values.append(sum(coefficients[owner, gene] for gene, owner in enumerate(owners)))
    assert labels.support_function(coefficients) == pytest.approx(max(values))
    lo, hi = labels.contrast_bounds(coefficients)
    assert lo == pytest.approx(min(values))
    assert hi == pytest.approx(max(values))


def test_marginal_box_does_not_preserve_joint_constraint():
    labels = graph(cost=0.5, background=0)
    _, upper = labels.count_bounds()
    coefficients = np.array([[0, 0], [1, 1], [0, 0]])
    assert np.sum(coefficients * upper) == pytest.approx(1)
    assert labels.support_function(coefficients) == pytest.approx(0.5)


def test_background_and_geometry_can_be_jointly_infeasible():
    with pytest.raises(ValueError, match="infeasible"):
        BudgetedLabelSet(sparse.csr_matrix([[True, True]]), np.array([1.0]), np.array([0]), 1,
                         edge_cost=[1, 0], background_index=1,
                         max_total_cost=0.4, max_background_mass=0.5)


def test_fixed_owner_validation_includes_budgets_and_support():
    labels = graph()
    np.testing.assert_allclose(labels.fixed_labels(np.array([0, 0])), [[1, 1], [0, 0], [0, 0]])
    with pytest.raises(ValueError, match="budget"):
        labels.fixed_labels(np.array([1, 1]))
    assert labels.fixed_labels(np.array([1, 1]), require_feasible=False).sum() == 2
    sparse_labels = BudgetedLabelSet(sparse.csr_matrix([[True, False]]), [1], np.array([0]), 1)
    with pytest.raises(ValueError, match="outside"):
        sparse_labels.fixed_labels(np.array([1]))


def test_unbudgeted_fast_path_matches_base_support():
    from morphostate.ambiguity import MoleculeLabelSet
    matrix = sparse.csr_matrix([[1, 1, 0], [0, 1, 1]], dtype=bool)
    labels = BudgetedLabelSet(matrix, [2, 3], np.array([0, 1]), 2, background_index=2)
    base = MoleculeLabelSet(matrix, [2, 3], np.array([0, 1]), 2)
    coefficient = np.array([[1, 2], [3, 1], [-1, 4]])
    a, ay = labels.maximize(coefficient)
    b, by = base.maximize(coefficient)
    assert a == b
    np.testing.assert_array_equal(ay, by)
    compressed = labels.aggregate()
    assert compressed.support_function(coefficient) == pytest.approx(a)
    assert compressed.n_observations == labels.n_observations


def test_budgeted_aggregation_preserves_support_exact_marginals_and_constraints():
    support = sparse.csr_matrix([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 0, 1]], dtype=bool)
    original = BudgetedLabelSet(support, [1.25, .75, 2., .5], np.array([0, 0, 1, 0]), 2,
        edge_cost=[0, 1, .2, 0, 1, .2, 0, 1, .2, 0, .3], max_total_cost=.9,
        background_index=2, max_background_mass=.4)
    compressed = original.aggregate()
    assert compressed.n_observations == 3
    assert compressed.max_total_cost == original.max_total_cost
    assert compressed.max_background_mass == original.max_background_mass
    assert compressed.background_index == original.background_index
    np.testing.assert_allclose(np.bincount(compressed.gene_index, weights=compressed.mass),
                               np.bincount(original.gene_index, weights=original.mass))
    generator = np.random.default_rng(73)
    for _ in range(12):
        coefficients = generator.normal(size=(3, 2))
        assert compressed.support_function(coefficients) == pytest.approx(original.support_function(coefficients))
        np.testing.assert_allclose(compressed.contrast_bounds(coefficients), original.contrast_bounds(coefficients), atol=1e-8)
    original_bounds, compressed_bounds = original.count_bounds(), compressed.count_bounds()
    np.testing.assert_allclose(compressed_bounds[0], original_bounds[0], atol=1e-8)
    np.testing.assert_allclose(compressed_bounds[1], original_bounds[1], atol=1e-8)
    # Compression is deterministic, independent of observation row ordering.
    permutation = np.array([2, 1, 3, 0])
    costs_by_row = [original.edge_cost[original.candidate_owners.indptr[row]:original.candidate_owners.indptr[row + 1]]
                    for row in permutation]
    permuted = BudgetedLabelSet(support[permutation].tocsr(), original.mass[permutation],
        original.gene_index[permutation], 2, edge_cost=np.concatenate(costs_by_row),
        max_total_cost=.9, background_index=2, max_background_mass=.4).aggregate()
    np.testing.assert_array_equal(permuted.candidate_owners.toarray(), compressed.candidate_owners.toarray())
    np.testing.assert_array_equal(permuted.mass, compressed.mass)
    np.testing.assert_array_equal(permuted.edge_cost, compressed.edge_cost)


def test_aggregation_never_merges_different_cost_or_gene_vectors():
    support = sparse.csr_matrix(np.ones((4, 3), dtype=bool))
    original = BudgetedLabelSet(support, [1., 1., 1., 1.], np.array([0, 0, 0, 1]), 2,
        edge_cost=[0, 1, 0, 0, 2, 0, 0, 1, 1, 0, 1, 0],
        max_total_cost=.5, background_index=2, max_background_mass=0)
    compressed = original.aggregate()
    assert compressed.n_observations == 4
    for coefficient in (np.array([[0, 0], [1, 1], [0, 0]]), np.array([[1, -1], [0, 2], [5, 5]])):
        assert compressed.support_function(coefficient) == pytest.approx(original.support_function(coefficient))


def test_aggregation_handles_zero_mass_and_preserves_unbudgeted_labels():
    original = BudgetedLabelSet(sparse.csr_matrix([[1, 1], [1, 1], [1, 1]], dtype=bool),
                                [0., 0., 2.], np.array([0, 0, 1]), 2,
                                edge_cost=[0, 0, 0, 0, 0, 0], background_index=1)
    compressed = original.aggregate()
    assert compressed.n_observations == 2
    np.testing.assert_array_equal(compressed.mass, [0., 2.])
    np.testing.assert_array_equal(compressed.count_bounds()[0], original.count_bounds()[0])
    np.testing.assert_array_equal(compressed.count_bounds()[1], original.count_bounds()[1])


def test_cost_alignment_and_memory_guards():
    unsorted = sparse.csr_matrix(([True, True], [1, 0], [0, 2]), shape=(1, 2))
    with pytest.raises(ValueError, match="canonical"):
        BudgetedLabelSet(unsorted, [1], np.array([0]), 1, edge_cost=[1, 0])
    matrix = sparse.csr_matrix([[True, True]])
    for costs in ([1], [-1, 0], [np.nan, 0]):
        with pytest.raises(ValueError, match="edge_cost"):
            BudgetedLabelSet(matrix, [1], np.array([0]), 1, edge_cost=costs)
    with pytest.raises(ValueError, match="max_edges"):
        BudgetedLabelSet(matrix, [1], np.array([0]), 1, max_edges=1)
    with pytest.raises(ValueError, match="max_label_entries"):
        BudgetedLabelSet(matrix, [1], np.array([0]), 2, max_label_entries=1)


def test_failed_solver_is_not_silently_replaced(monkeypatch):
    from types import SimpleNamespace

    import morphostate.budgeted as module
    labels = graph()
    monkeypatch.setattr(module, "linprog", lambda *a, **k: SimpleNamespace(success=False, status=4,
                                                                         message="numerical failure"))
    with pytest.raises(RuntimeError, match="numerical failure"):
        labels.maximize(np.ones((3, 2)))
