"""Explicit local imports; no patient data are downloaded automatically."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .data import Dataset, _check_identifiers, load_dataset, save_dataset


def _strings(values, name: str) -> list[str]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional string array")
    decoded = [value.decode("utf-8") if isinstance(value, bytes) else value for value in array.tolist()]
    return _check_identifiers(decoded, name, unique=True)


def read_feature_metadata(feature_path):
    """Require an explicit encoder identity, rather than inferring it from width."""
    path = Path(feature_path).with_suffix(".json")
    if not path.exists():
        raise ValueError(f"Missing feature metadata sidecar: {path}; record source and encoder weights/transform")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    encoder = metadata.get("feature_encoder")
    if metadata.get("feature_source") != "image" or not isinstance(encoder, dict):
        raise ValueError("Feature metadata requires feature_source=image and a feature_encoder object")
    if not encoder.get("model") or not encoder.get("weights_sha256") or not encoder.get("transform"):
        raise ValueError("feature_encoder must record model, weights_sha256 and transform")
    return metadata


def _align_frame(frame: pd.DataFrame, ids: list[str], name: str) -> pd.DataFrame:
    if "cell_id" not in frame:
        raise ValueError(f"{name} must contain cell_id")
    supplied = _check_identifiers(frame["cell_id"], f"{name} cell_id", unique=True)
    missing = set(ids) - set(supplied)
    extra = set(supplied) - set(ids)
    if missing or extra:
        raise ValueError(f"{name} cell IDs differ from matrix: {len(missing)} missing, {len(extra)} extra")
    return frame.set_index("cell_id", drop=False).loc[ids].reset_index(drop=True)


def _aligned_npz(path: str | Path, ids: list[str], key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as bundle:
        if "cell_ids" not in bundle or key not in bundle:
            raise ValueError(f"{path} must contain cell_ids and {key}")
        supplied = _strings(bundle["cell_ids"], f"{key} cell_ids")
        array = bundle[key]
        if array.ndim != 2 or array.shape[0] != len(supplied):
            raise ValueError(f"{key} must be a matrix with one row per cell_ids entry")
        index = _align_frame(pd.DataFrame({"cell_id": supplied, "row": np.arange(len(supplied))}), ids, key)
        return np.asarray(array[index["row"].to_numpy()])


def _read_xenium_h5(path: str | Path):
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Xenium H5 import requires h5py") from exc
    with h5py.File(path, "r") as handle:
        if "matrix" not in handle:
            raise ValueError("expected a standard 10x/Xenium H5 with /matrix")
        matrix = handle["matrix"]
        required = ("data", "indices", "indptr", "shape", "barcodes", "features/id", "features/feature_type")
        missing = [key for key in required if key not in matrix]
        if missing:
            raise ValueError(f"Xenium H5 missing matrix fields: {missing}")
        shape = tuple(int(value) for value in matrix["shape"][:])
        if len(shape) != 2:
            raise ValueError("Xenium matrix shape must have two dimensions")
        gene_ids = _strings(matrix["features/id"][:], "H5 feature IDs")
        barcodes = _strings(matrix["barcodes"][:], "H5 barcodes")
        types_raw = matrix["features/feature_type"][:]
        types = np.asarray([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in types_raw])
        if shape != (len(gene_ids), len(barcodes)) or len(types) != len(gene_ids):
            raise ValueError("Xenium shape, feature IDs, types, and barcodes disagree")
        matrix_csc = sparse.csc_matrix(
            (matrix["data"][:], matrix["indices"][:], matrix["indptr"][:]), shape=shape,
        )
        matrix_csc.check_format(full_check=True)
        keep = types == "Gene Expression"
        if not keep.any():
            raise ValueError("Xenium H5 has no Gene Expression features")
        counts = matrix_csc[keep, :].T.tocsr()
        genes = np.asarray(gene_ids)[keep].tolist()
        return counts, genes, barcodes, int((~keep).sum())


def import_xenium(
    matrix_path, cells_path, feature_path, covariate_path, donor_id, slide_id, output,
    covariate_source, dataset_id="custom",
) -> None:
    """Import one donor/slide with exact ID joins and explicit covariate origin.

    ``feature_path`` contains ``cell_ids`` and ``features``; ``covariate_path``
    contains ``cell_ids`` and ``covariates``. Both use original Xenium cell IDs.
    An import represents one donor; first subset multi-donor TMA data explicitly.
    """
    _check_identifiers([donor_id], "donor_id")
    _check_identifiers([slide_id], "slide_id")
    if covariate_source not in {"image", "rna_oracle"}:
        raise ValueError("real Xenium covariate_source must be image or rna_oracle")
    counts, genes, ids, excluded = _read_xenium_h5(matrix_path)
    cells_path = Path(cells_path)
    if cells_path.suffix.lower() == ".parquet":
        cells = pd.read_parquet(cells_path)
    else:
        cells = pd.read_csv(cells_path, dtype=str, keep_default_na=False)
    cells = _align_frame(cells, ids, "cells metadata")
    for column, value in (("donor_id", donor_id), ("slide_id", slide_id)):
        if column in cells and not (cells[column] == value).all():
            raise ValueError(f"cells metadata {column} conflicts with the supplied {column}; subset mixed-donor data explicitly")
        cells[column] = value
    if "cell_type" not in cells:
        cells["cell_type"] = "unknown"
    cells["source_cell_id"] = ids
    cells["cell_id"] = [f"{slide_id}::{cell_id}" for cell_id in ids]
    features = _aligned_npz(feature_path, ids, "features")
    covariates = _aligned_npz(covariate_path, ids, "covariates")
    feature_metadata = read_feature_metadata(feature_path)
    dataset = Dataset(
        counts=counts, features=features, covariates=covariates, cells=cells, genes=genes,
        provenance={
            "dataset_id": dataset_id,
            "feature_source": "image", "covariate_source": covariate_source,
            "feature_encoder": feature_metadata["feature_encoder"],
            "importer": "xenium-10x-h5-v1",
            "source_files": {"matrix": Path(matrix_path).name, "cells": cells_path.name,
                             "features": Path(feature_path).name, "covariates": Path(covariate_path).name},
            "count_units": "transcript counts", "excluded_non_gene_features": excluded,
            "cell_type_role": "Evaluation-only annotation; never automatically added to model covariates.",
            "alignment": "Exact raw cell_id joins to H5 barcode order; final IDs are slide_id::source_cell_id.",
        },
    )
    save_dataset(dataset, output)


def merge_datasets(inputs, output) -> None:
    """Concatenate cells only; never intersect or zero-fill different gene panels."""
    inputs = list(inputs)
    if not inputs:
        raise ValueError("at least one prepared input directory is required")
    datasets = [load_dataset(path) for path in inputs]
    first = datasets[0]
    for index, dataset in enumerate(datasets[1:], start=1):
        if dataset.genes != first.genes:
            raise ValueError(f"gene panel/order mismatch in input {index}; explicitly harmonize panels outside merge")
        for field in ("features", "covariates"):
            if getattr(dataset, field).shape[1] != getattr(first, field).shape[1]:
                raise ValueError(f"{field} dimension mismatch in input {index}")
        for field in ("feature_source", "covariate_source", "feature_encoder", "covariate_columns"):
            if dataset.provenance.get(field) != first.provenance.get(field):
                raise ValueError(f"{field} mismatch in input {index}")
    provenance = {
        "dataset_id": "merged:" + "+".join(dict.fromkeys(ds.provenance["dataset_id"] for ds in datasets)),
        "feature_source": first.provenance["feature_source"],
        "covariate_source": first.provenance["covariate_source"],
        "merge": "Identical ordered gene panels; concatenated cells without normalization.",
        "source_datasets": [ds.provenance for ds in datasets],
        "identity_warning": "Donor IDs must already be harmonized across cohorts. Merge cannot discover aliases for the same person.",
    }
    for field in ("feature_encoder", "covariate_columns"):
        if field in first.provenance:
            provenance[field] = first.provenance[field]
    combined = Dataset(
        counts=sparse.vstack([ds.counts for ds in datasets], format="csr"),
        features=np.concatenate([ds.features for ds in datasets]),
        covariates=np.concatenate([ds.covariates for ds in datasets]),
        cells=pd.concat([ds.cells for ds in datasets], ignore_index=True),
        genes=list(first.genes), provenance=provenance,
    )
    save_dataset(combined, output)
