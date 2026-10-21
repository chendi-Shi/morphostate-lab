"""Clearly synthetic plumbing data; not evidence for a biological hypothesis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from .data import Dataset, save_dataset


def make_synthetic(output, seed=7, donors=12, cells_per_donor=80, genes=32, features=12) -> None:
    for name, value, minimum in (
        ("donors", donors, 3), ("cells_per_donor", cells_per_donor, 4),
        ("genes", genes, 4), ("features", features, 4),
    ):
        if not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    rng = np.random.default_rng(seed)
    n = donors * cells_per_donor
    n_types = 3
    donor = np.repeat(np.arange(donors), cells_per_donor)
    type_index = np.empty(n, dtype=int)
    for d in range(donors):
        # Donors differ in composition; there is no cell-level random split.
        probabilities = rng.dirichlet(np.full(n_types, 4.0))
        type_index[donor == d] = rng.choice(n_types, cells_per_donor, p=probabilities)
    identity = np.eye(n_types)[type_index]
    state = rng.normal(size=(n, 2))
    state += rng.normal(scale=0.2, size=(donors, 2))[donor]
    neighborhood = rng.beta(2, 2, size=n)
    covariates = np.column_stack([identity, neighborhood]).astype(np.float32)
    image = rng.normal(scale=0.45, size=(n, features))
    image[:, 0] += 1.5 * state[:, 0]
    image[:, 1] += 1.5 * state[:, 1]
    image[:, 2:4] += identity[:, :2] * 2.0
    image += rng.normal(scale=0.12, size=(donors, features))[donor]
    type_loading = rng.normal(scale=0.65, size=(n_types, genes))
    state_loading = rng.normal(scale=0.3, size=(2, genes))
    log_rate = 1.35 + identity @ type_loading + state @ state_loading
    log_rate += neighborhood[:, None] * rng.normal(scale=0.2, size=genes)
    log_rate += rng.normal(scale=0.1, size=(donors, genes))[donor]
    depth = rng.lognormal(mean=0, sigma=0.25, size=n)
    mean = np.exp(np.clip(log_rate, -2, 5)) * depth[:, None]
    dispersion = 8.0
    counts = rng.negative_binomial(dispersion, dispersion / (dispersion + mean)).astype(np.int64)
    # Resample the very rare all-zero row instead of silently changing its counts.
    empty = counts.sum(axis=1) == 0
    while empty.any():
        counts[empty] = rng.negative_binomial(dispersion, dispersion / (dispersion + mean[empty]))
        empty = counts.sum(axis=1) == 0
    dataset = Dataset(
        counts=sparse.csr_matrix(counts), features=image.astype(np.float32), covariates=covariates,
        cells=pd.DataFrame({
            "cell_id": [f"synthetic_d{d:02d}_cell{i:06d}" for i, d in enumerate(donor)],
            "donor_id": [f"synthetic_d{d:02d}" for d in donor],
            "slide_id": [f"synthetic_d{d:02d}_slide" for d in donor],
            "cell_type": [f"synthetic_type{t}" for t in type_index],
        }),
        genes=[f"synthetic_gene_{g:04d}" for g in range(genes)],
        provenance={
            "dataset_id": "synthetic-plumbing-only", "feature_source": "synthetic",
            "covariate_source": "synthetic", "seed": int(seed),
            "description": "Simulated negative-binomial counts with cell identity, composition, within-type image signal, donor shift, and depth variation. Not biological evidence.",
            "covariate_columns": ["synthetic_type0", "synthetic_type1", "synthetic_type2", "synthetic_neighborhood"],
            "cell_type_role": "Evaluation label; synthetic identity is explicitly present in synthetic covariates.",
        },
    )
    save_dataset(dataset, output)
