"""Validated, unnormalized data interchange for patient-held-out experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


CELL_COLUMNS = ("cell_id", "donor_id", "slide_id", "cell_type")
SOURCE_TYPES = {"image", "rna_oracle", "synthetic"}


@dataclass
class Dataset:
    counts: sparse.csr_matrix
    features: np.ndarray
    covariates: np.ndarray
    cells: pd.DataFrame
    genes: list[str]
    provenance: dict


def _check_identifiers(values, name: str, *, unique: bool = False) -> list[str]:
    result = list(values)
    if any(not isinstance(v, str) or not v.strip() or v != v.strip() for v in result):
        raise ValueError(f"{name} must contain nonempty, whitespace-trimmed strings")
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


def validate_dataset(dataset: Dataset) -> None:
    """Check alignment and numerical validity without fitting any preprocessing."""
    if not sparse.issparse(dataset.counts) or dataset.counts.ndim != 2:
        raise ValueError("counts must be a two-dimensional scipy sparse matrix")
    counts = dataset.counts
    if counts.shape[0] == 0 or counts.shape[1] == 0:
        raise ValueError("counts must contain at least one cell and one gene")
    values = counts.data
    if (not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values)
            or not np.isfinite(values).all() or (values < 0).any()
            or not np.equal(values, np.floor(values)).all()):
        raise ValueError("counts must be finite nonnegative integers")
    if (np.asarray(counts.sum(axis=1)).ravel() <= 0).any():
        raise ValueError("zero-total cells are not supported; filter them explicitly before import")
    for name, array in (("features", dataset.features), ("covariates", dataset.covariates)):
        if not isinstance(array, np.ndarray) or array.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional numpy array")
        if array.shape[0] != counts.shape[0]:
            raise ValueError(f"{name} row count does not match counts")
        if (not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array)
                or not np.isfinite(array).all()):
            raise ValueError(f"{name} must contain finite real numbers")
    if dataset.features.shape[1] == 0 or dataset.covariates.shape[1] == 0:
        raise ValueError("features and covariates must each have at least one column; use a constant covariate for intercept-only")
    if not isinstance(dataset.cells, pd.DataFrame) or len(dataset.cells) != counts.shape[0]:
        raise ValueError("cells must be a DataFrame with one row per count row")
    missing = set(CELL_COLUMNS) - set(dataset.cells.columns)
    if missing:
        raise ValueError(f"cells is missing required columns: {sorted(missing)}")
    for column in CELL_COLUMNS:
        _check_identifiers(dataset.cells[column], column, unique=column == "cell_id")
    _check_identifiers(dataset.genes, "gene_id", unique=True)
    if len(dataset.genes) != counts.shape[1]:
        raise ValueError("gene_id length does not match count columns")
    if not isinstance(dataset.provenance, dict):
        raise ValueError("provenance must be a dictionary")
    for key in ("feature_source", "covariate_source"):
        if dataset.provenance.get(key) not in SOURCE_TYPES:
            raise ValueError(f"provenance.{key} must be one of {sorted(SOURCE_TYPES)}")
    _check_identifiers([dataset.provenance.get("dataset_id")], "provenance.dataset_id")
    try:
        json.dumps(dataset.provenance, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("provenance must be JSON-serializable without NaN/Infinity") from exc


def load_dataset(path: str | Path) -> Dataset:
    """Read a prepared directory; never use all cells to fit normalization."""
    path = Path(path)
    counts = sparse.load_npz(path / "counts.npz").tocsr()
    cells = pd.read_csv(path / "cells.csv", dtype=str, keep_default_na=False)
    gene_frame = pd.read_csv(path / "genes.csv", dtype=str, keep_default_na=False)
    if "gene_id" not in gene_frame:
        raise ValueError("genes.csv must contain a gene_id column")
    with (path / "provenance.json").open(encoding="utf-8") as handle:
        provenance = json.load(handle)
    dataset = Dataset(
        counts=counts,
        features=np.load(path / "features.npy", allow_pickle=False),
        covariates=np.load(path / "covariates.npy", allow_pickle=False),
        cells=cells,
        genes=gene_frame["gene_id"].tolist(),
        provenance=provenance,
    )
    validate_dataset(dataset)
    return dataset


def save_dataset(dataset: Dataset, path: str | Path) -> None:
    """Validate before writing. IDs describe the exact row order in every array."""
    validate_dataset(dataset)
    # Conversion must also remain finite at the stored float32 precision.
    with np.errstate(over="ignore"):
        features = dataset.features.astype(np.float32)
        covariates = dataset.covariates.astype(np.float32)
    if not np.isfinite(features).all() or not np.isfinite(covariates).all():
        raise ValueError("features/covariates exceed the finite float32 range")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    counts = dataset.counts.tocsr(copy=True)
    counts.sum_duplicates()
    counts.eliminate_zeros()
    sparse.save_npz(path / "counts.npz", counts)
    np.save(path / "features.npy", features, allow_pickle=False)
    np.save(path / "covariates.npy", covariates, allow_pickle=False)
    dataset.cells.to_csv(path / "cells.csv", index=False)
    pd.DataFrame({"gene_id": dataset.genes}).to_csv(path / "genes.csv", index=False)
    with (path / "provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset.provenance, handle, indent=2, allow_nan=False, ensure_ascii=False)
        handle.write("\n")
