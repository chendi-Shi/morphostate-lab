"""Exact, limited baseline for ambiguity in transcript-to-cell ownership.

Each observation has one gene, nonnegative mass, and a nonempty set of candidate
owners. Its mass may be distributed among those owners, but cannot be duplicated
or discarded. Observations are otherwise independent: this module has NO shared
displacement budget, cell capacity, learned segmentation, or geometric coupling.

The resulting *cell labels* are coupled by mass conservation, despite independent
observation assignments. Coordinatewise count intervals are an outer box, not
the joint feasible set. Linear extrema are computed exactly by assigning each
observation to the candidate with the largest coefficient. This is a baseline
and numerical prototype, not a proposed novel algorithm or a coverage guarantee.

Any background compartment must be an explicit ordinary candidate column. Bounds
are conditional on the supplied candidate sets containing the relevant owners;
they do not establish that those sets are scientifically correct. Multiple
segmentations of the same molecules must not be treated as independent data.
"""

from __future__ import annotations

from collections import defaultdict
from math import fsum
from numbers import Integral

import numpy as np
from scipy import sparse


class MoleculeLabelSet:
    """A product of owner simplices mapped to cell-by-gene count labels.

    Parameters
    ----------
    candidate_owners:
        Binary SciPy CSR matrix, observations by cells. Explicit zero entries
        are removed. Every row must have at least one candidate owner.
    mass:
        Finite nonnegative vector, one mass per observation. A row may represent
        one molecule or an aggregation of molecules with IDENTICAL gene and
        candidate-owner sets. Noninteger mass is allowed for this convex model.
    gene_index:
        Integer vector identifying the gene of each observation.
    n_genes:
        Positive integer defining the complete coefficient/label gene axis.

    Linear optimization is also exact over unsplit molecule assignments: a
    maximizing vertex always assigns an entire row's mass to one candidate.
    These methods do not infer unobserved transcripts or native expression.
    """

    def __init__(self, candidate_owners, mass, gene_index, n_genes: int) -> None:
        if not sparse.issparse(candidate_owners) or candidate_owners.format != "csr":
            raise TypeError("candidate_owners must be a binary SciPy CSR matrix or array")
        if len(candidate_owners.shape) != 2 or min(candidate_owners.shape) < 1:
            raise ValueError("candidate_owners must have nonzero observation and cell axes")
        if (
            not np.issubdtype(candidate_owners.dtype, np.number)
            and candidate_owners.dtype != np.dtype(bool)
        ) or np.issubdtype(candidate_owners.dtype, np.complexfloating):
            raise ValueError("candidate_owners must contain real binary entries")
        support = sparse.csr_matrix(candidate_owners, copy=True)
        support.check_format(full_check=True)
        if not np.isfinite(support.data).all() or not np.isin(support.data, [0, 1]).all():
            raise ValueError("candidate_owners must contain only finite zero/one entries")
        support.eliminate_zeros()
        support.sum_duplicates()
        support.data = np.ones(support.nnz, dtype=bool)
        support.sort_indices()
        owner_counts = np.diff(support.indptr)
        if (owner_counts == 0).any():
            raise ValueError("Every observation must have at least one candidate owner")
        if not isinstance(n_genes, Integral) or isinstance(n_genes, bool) or n_genes < 1:
            raise ValueError("n_genes must be a positive integer")

        mass_array = np.asarray(mass)
        if (
            mass_array.shape != (support.shape[0],)
            or not np.issubdtype(mass_array.dtype, np.number)
            or np.issubdtype(mass_array.dtype, np.complexfloating)
        ):
            raise ValueError("mass must be a real vector matching the observation rows")
        mass_array = np.array(mass_array, dtype=np.float64, copy=True)
        if not np.isfinite(mass_array).all() or (mass_array < 0).any():
            raise ValueError("mass must be finite and nonnegative")

        genes = np.asarray(gene_index)
        if genes.shape != (support.shape[0],) or not np.issubdtype(genes.dtype, np.integer):
            raise ValueError("gene_index must be an integer vector matching observation rows")
        if (genes < 0).any() or (genes >= n_genes).any():
            raise ValueError("gene_index entries must lie in [0, n_genes)")

        self.candidate_owners = support
        self.mass = mass_array
        self.gene_index = genes.astype(np.int64, copy=True)
        self.n_genes = int(n_genes)
        self.n_observations, self.n_cells = support.shape
        self._owner_counts = owner_counts
        self._edge_rows = np.repeat(np.arange(self.n_observations), owner_counts)
        # Prevent accidental mutation of an already validated instance's arrays.
        for array in (
            support.data, support.indices, support.indptr, self.mass,
            self.gene_index, self._owner_counts, self._edge_rows,
        ):
            array.flags.writeable = False

    def _coefficients(self, coefficients) -> np.ndarray:
        values = np.asarray(coefficients)
        if (
            values.shape != (self.n_cells, self.n_genes)
            or not np.issubdtype(values.dtype, np.number)
            or np.issubdtype(values.dtype, np.complexfloating)
        ):
            raise ValueError("coefficients must be a real cells-by-genes matrix")
        values = np.asarray(values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("coefficients must be finite")
        return values

    def _row_maxima(self, coefficients) -> tuple[np.ndarray, np.ndarray]:
        values = self._coefficients(coefficients)
        edges = values[
            self.candidate_owners.indices,
            self.gene_index[self._edge_rows],
        ]
        maxima = np.maximum.reduceat(edges, self.candidate_owners.indptr[:-1])
        return edges, maxima

    def support_function(self, coefficients) -> float:
        """Return ``max_Y sum(coefficients * Y)`` over jointly feasible labels."""
        _, maxima = self._row_maxima(coefficients)
        value = float(np.dot(self.mass, maxima))
        if not np.isfinite(value):
            raise FloatingPointError("Linear support value overflowed float64")
        return value

    def maximize(self, coefficients) -> tuple[float, np.ndarray]:
        """Return the exact linear maximum and one feasible maximizing label.

        Labels are a dense cells-by-genes matrix. Ties select the smallest cell
        column index. The value is invariant to cell permutations; a tied argmax
        need not select the same physical cell after renumbering.
        """
        edges, maxima = self._row_maxima(coefficients)
        is_maximum = edges == maxima[self._edge_rows]
        candidates = np.where(is_maximum, self.candidate_owners.indices, self.n_cells)
        owners = np.minimum.reduceat(candidates, self.candidate_owners.indptr[:-1])
        labels = np.zeros((self.n_cells, self.n_genes), dtype=np.float64)
        with np.errstate(over="raise", invalid="raise"):
            np.add.at(labels, (owners, self.gene_index), self.mass)
        value = float(np.dot(self.mass, maxima))
        if not np.isfinite(value):
            raise FloatingPointError("Linear support value overflowed float64")
        return value, labels

    def argmax_labels(self, coefficients) -> np.ndarray:
        """Return one feasible cell-by-gene label matrix attaining the maximum."""
        return self.maximize(coefficients)[1]

    def count_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return exact coordinatewise lower and upper count bounds.

        The lower bound includes only observations with one possible owner. The
        upper bound includes all observations that could belong to that cell.
        Different coordinates' extrema generally cannot be attained together.
        """
        lower = np.zeros((self.n_cells, self.n_genes), dtype=np.float64)
        upper = np.zeros_like(lower)
        single = self._owner_counts == 1
        single_owners = self.candidate_owners.indices[self.candidate_owners.indptr[:-1][single]]
        with np.errstate(over="raise", invalid="raise"):
            np.add.at(lower, (single_owners, self.gene_index[single]), self.mass[single])
            np.add.at(
                upper,
                (self.candidate_owners.indices, self.gene_index[self._edge_rows]),
                self.mass[self._edge_rows],
            )
        return lower, upper

    def contrast_bounds(self, coefficients) -> tuple[float, float]:
        """Return sharp minimum/maximum of a specified linear contrast.

        Equal endpoints establish identifiability only within this supplied
        feasible-set model; they are not statistical confidence intervals.
        """
        values = self._coefficients(coefficients)
        return -self.support_function(-values), self.support_function(values)

    def aggregate(self) -> MoleculeLabelSet:
        """Combine rows with identical genes and candidate sets, summing mass.

        This preserves the feasible label set for divisible mass and every
        linear support value. It does not combine differing candidate sets.
        """
        groups = defaultdict(list)
        for row in range(self.n_observations):
            start, stop = self.candidate_owners.indptr[row : row + 2]
            key = (int(self.gene_index[row]), tuple(self.candidate_owners.indices[start:stop]))
            groups[key].append(float(self.mass[row]))
        keys = sorted(groups)
        indices = []
        indptr = [0]
        for _, candidates in keys:
            indices.extend(candidates)
            indptr.append(len(indices))
        support = sparse.csr_matrix(
            (np.ones(len(indices), dtype=bool), indices, indptr),
            shape=(len(keys), self.n_cells),
        )
        masses = np.array([fsum(groups[key]) for key in keys], dtype=np.float64)
        genes = np.array([key[0] for key in keys], dtype=np.int64)
        return MoleculeLabelSet(support, masses, genes, self.n_genes)
