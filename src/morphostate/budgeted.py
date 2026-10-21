"""Sparse LP optimization of molecule ownership with shared physical budgets.

Assignments are divisible mass, not integer molecule assignments. Shared budgets
can produce fractional vertices. Bounds are conditional on the declared graph
and budgets; this module does not provide statistical coverage or infer budgets.
"""

from __future__ import annotations

from collections import defaultdict
from math import fsum

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .ambiguity import MoleculeLabelSet


class BudgetedLabelSet(MoleculeLabelSet):
    """Conserved ownership mass subject to displacement/background budgets.

    ``edge_cost`` follows the exact CSR data order. To prevent accidental cost
    misalignment, the input CSR must be canonical, sorted and contain no zeros.
    Variables are allocated mass per edge; ``max_total_cost`` bounds their
    cost-weighted sum and ``max_background_mass`` bounds mass in one explicit
    background column. Omitting both budgets uses the independent-row solver.
    """

    def __init__(self, candidate_owners, mass, gene_index, n_genes, *, edge_cost=None,
                 max_total_cost=None, background_index=None, max_background_mass=None,
                 max_edges=2_000_000, max_label_entries=10_000_000):
        if (not sparse.issparse(candidate_owners) or candidate_owners.format != "csr"
                or not candidate_owners.has_canonical_format
                or not candidate_owners.has_sorted_indices
                or (candidate_owners.data == 0).any()):
            raise ValueError("candidate_owners must be canonical sorted CSR without explicit zeros")
        if max_edges < 1 or candidate_owners.nnz > max_edges:
            raise ValueError("Candidate graph exceeds max_edges; create smaller explicit blocks")
        super().__init__(candidate_owners, mass, gene_index, n_genes)
        if max_label_entries < 1 or self.n_cells * self.n_genes > max_label_entries:
            raise ValueError("Dense label matrix exceeds max_label_entries; reduce pilot block size")
        costs = np.zeros(self.candidate_owners.nnz) if edge_cost is None else np.asarray(edge_cost)
        if (costs.shape != (self.candidate_owners.nnz,) or not np.issubdtype(costs.dtype, np.number)
                or np.iscomplexobj(costs) or not np.isfinite(costs).all() or (costs < 0).any()):
            raise ValueError("edge_cost must be a finite nonnegative vector in CSR edge order")
        self.edge_cost = np.array(costs, dtype=np.float64, copy=True)
        self.edge_cost.flags.writeable = False
        for name, value in (("max_total_cost", max_total_cost),
                            ("max_background_mass", max_background_mass)):
            if value is not None and (not np.isscalar(value) or isinstance(value, (bool, complex))
                                      or not np.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be a finite nonnegative scalar or None")
        if background_index is not None and (
            not isinstance(background_index, (int, np.integer)) or isinstance(background_index, bool)
            or not 0 <= background_index < self.n_cells
        ):
            raise ValueError("background_index must be a valid integer cell index")
        if max_background_mass is not None and background_index is None:
            raise ValueError("max_background_mass requires an explicit background_index")
        self.max_total_cost = None if max_total_cost is None else float(max_total_cost)
        self.background_index = None if background_index is None else int(background_index)
        self.max_background_mass = None if max_background_mass is None else float(max_background_mass)
        self.max_edges = int(max_edges)
        self.max_label_entries = int(max_label_entries)
        self._has_budgets = self.max_total_cost is not None or self.max_background_mass is not None
        self._a_eq = sparse.csr_matrix((np.ones(self.candidate_owners.nnz), self._edge_rows,
                                       np.arange(self.candidate_owners.nnz + 1)),
                                      shape=(self.candidate_owners.nnz, self.n_observations)).T.tocsr()
        constraint_rows, bounds = [], []
        if self.max_total_cost is not None:
            constraint_rows.append(sparse.csr_matrix(self.edge_cost[None, :]))
            bounds.append(self.max_total_cost)
        if self.max_background_mass is not None:
            bg = (self.candidate_owners.indices == self.background_index).astype(np.float64)
            constraint_rows.append(sparse.csr_matrix(bg[None, :]))
            bounds.append(self.max_background_mass)
        self._a_ub = sparse.vstack(constraint_rows, format="csr") if constraint_rows else None
        self._b_ub = np.array(bounds) if bounds else None
        if self._has_budgets:
            self._solve(np.zeros(self.candidate_owners.nnz))

    def _solve(self, edge_coefficients):
        result = linprog(-np.asarray(edge_coefficients), A_ub=self._a_ub, b_ub=self._b_ub,
                         A_eq=self._a_eq, b_eq=self.mass, bounds=(0, None), method="highs")
        if not result.success:
            if result.status == 2:
                raise ValueError(f"Ownership budgets are infeasible: {result.message}")
            raise RuntimeError(f"Ownership LP failed (status {result.status}): {result.message}")
        allocation = np.asarray(result.x)
        tolerance = 1e-7 * (1.0 + np.max(self.mass, initial=0))
        if (not np.isfinite(allocation).all() or allocation.min(initial=0) < -tolerance
                or np.max(np.abs(self._a_eq @ allocation - self.mass), initial=0) > tolerance):
            raise RuntimeError("Ownership LP returned an invalid mass-conservation solution")
        if self._a_ub is not None:
            allowed = 1e-7 * (1 + np.abs(self._b_ub))
            if np.any(self._a_ub @ allocation - self._b_ub > allowed):
                raise RuntimeError("Ownership LP returned a budget-violating solution")
        return allocation

    def maximize(self, coefficients):
        values = self._coefficients(coefficients)
        if not self._has_budgets:
            return super().maximize(values)
        edge_coefficients = values[self.candidate_owners.indices, self.gene_index[self._edge_rows]]
        allocation = self._solve(edge_coefficients)
        labels = np.zeros((self.n_cells, self.n_genes))
        np.add.at(labels, (self.candidate_owners.indices, self.gene_index[self._edge_rows]), allocation)
        objective = float(np.dot(edge_coefficients, allocation))
        if not np.isfinite(objective):
            raise FloatingPointError("Ownership LP objective overflowed float64")
        return objective, labels

    def support_function(self, coefficients):
        return self.maximize(coefficients)[0]

    def count_bounds(self, *, exact=True, max_queries=10_000):
        """Return coordinate bounds; exact=True solves all nontrivial marginals.

        With budgets, ``exact=False`` returns the explicitly looser independent
        graph bounds. The query guard fails rather than silently using those
        outer bounds, so joint-versus-interval comparisons can match marginals.
        """
        lower, upper = super().count_bounds()
        if not self._has_budgets or not exact:
            return lower, upper
        coordinates = np.argwhere(upper > lower)
        if not isinstance(max_queries, (int, np.integer)) or max_queries < 0:
            raise ValueError("max_queries must be a nonnegative integer")
        if 2 * len(coordinates) > max_queries:
            raise ValueError(f"Exact count bounds need {2 * len(coordinates)} LP queries, "
                             f"exceeding max_queries={max_queries}")
        coefficients = np.zeros_like(lower)
        for cell, gene in coordinates:
            coefficients[cell, gene] = 1
            lo, hi = self.contrast_bounds(coefficients)
            lower[cell, gene], upper[cell, gene] = max(0.0, lo), max(0.0, hi)
            coefficients[cell, gene] = 0
        return lower, upper

    def fixed_labels(self, fixed_owner, *, require_feasible=True):
        """Map a declared segmentation to labels, checking graph and budgets."""
        owners = np.asarray(fixed_owner)
        if (owners.shape != (self.n_observations,) or not np.issubdtype(owners.dtype, np.integer)
                or (owners < 0).any() or (owners >= self.n_cells).any()):
            raise ValueError("fixed_owner must contain one valid integer owner per observation")
        matched = self.candidate_owners.indices == owners[self._edge_rows]
        if not np.all(np.add.reduceat(matched.astype(int), self.candidate_owners.indptr[:-1]) == 1):
            raise ValueError("fixed_owner contains an owner outside the candidate graph")
        assignment = np.zeros(self.candidate_owners.nnz)
        assignment[matched] = self.mass
        if require_feasible and self._a_ub is not None:
            if np.any(self._a_ub @ assignment > self._b_ub + 1e-7 * (1 + self._b_ub)):
                raise ValueError("fixed_owner violates the declared geometry/background budget")
        labels = np.zeros((self.n_cells, self.n_genes))
        np.add.at(labels, (owners, self.gene_index), self.mass)
        return labels

    def copy_with_budgets(self, *, max_total_cost=None, max_background_mass=None):
        return BudgetedLabelSet(self.candidate_owners, self.mass, self.gene_index, self.n_genes,
                                edge_cost=self.edge_cost, max_total_cost=max_total_cost,
                                background_index=self.background_index,
                                max_background_mass=max_background_mass, max_edges=self.max_edges,
                                max_label_entries=self.max_label_entries)

    def aggregate(self):
        """Exactly compress rows with identical gene, owners AND edge costs.

        This preserves the divisible-mass feasible label set and all shared
        linear budgets. A merged allocation can be split back in proportion to
        the original row masses; zero-total rows allocate zero. It does not
        preserve a one-owner-per-original-row integer assignment constraint,
        nor does it produce a new fixed_owner vector for a GraphBlock.
        """
        groups = defaultdict(list)
        for row in range(self.n_observations):
            start, stop = self.candidate_owners.indptr[row:row + 2]
            key = (int(self.gene_index[row]),
                   tuple(int(owner) for owner in self.candidate_owners.indices[start:stop]),
                   tuple(float(cost) for cost in self.edge_cost[start:stop]))
            groups[key].append(float(self.mass[row]))
        keys = sorted(groups)
        indices, indptr, costs, masses = [], [0], [], []
        for key in keys:
            indices.extend(key[1])
            costs.extend(key[2])
            indptr.append(len(indices))
            try:
                total = fsum(groups[key])
            except OverflowError as exc:
                raise FloatingPointError("Aggregated molecular mass overflows float64") from exc
            if not np.isfinite(total):
                raise FloatingPointError("Aggregated molecular mass overflows float64")
            masses.append(total)
        support = sparse.csr_matrix((np.ones(len(indices), dtype=bool), indices, indptr),
                                    shape=(len(keys), self.n_cells))
        return BudgetedLabelSet(support, masses, np.asarray([key[0] for key in keys], dtype=np.int64),
                                self.n_genes, edge_cost=costs, max_total_cost=self.max_total_cost,
                                background_index=self.background_index,
                                max_background_mass=self.max_background_mass,
                                max_edges=self.max_edges, max_label_entries=self.max_label_entries)
