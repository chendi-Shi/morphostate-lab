"""Tiny end-to-end runs exercise data, selection, reporting and deployment."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from morphostate.data import load_dataset, save_dataset
from morphostate.experiment import predict_from_checkpoint, run_experiment
from morphostate.synthetic import make_synthetic


def _small_experiment(tmp_path):
    data_dir = tmp_path / "data"
    make_synthetic(data_dir, seed=31, donors=8, cells_per_donor=8, genes=6, features=4)
    data = load_dataset(data_dir)
    # A legitimate intercept-only covariate baseline has undefined PCC for
    # every donor, including after centering by cell type.
    data.covariates[:] = 1
    data.cells["cell_type"] = "one_type"
    save_dataset(data, data_dir)
    config = {
        "data": "data", "output": "run", "seed": 31,
        "preprocess": {"max_genes": 6, "min_train_cells": 1, "max_cells_per_donor": 8},
        "training": {"epochs": 2, "patience": 2, "batch_size": 16, "ranks": [2],
                     "lr": 0.01, "hidden_dim": 0, "device": "cpu"},
        "evaluation": {"bootstrap_repetitions": 10, "permutation_repeats": 1},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return data, data_dir, config_path


def test_run_preserves_undefined_metrics_and_checkpoint_predicts_without_rna(tmp_path):
    data, data_dir, config_path = _small_experiment(tmp_path)
    summary = run_experiment(config_path)
    output = tmp_path / "run"
    for model in ("covariate_nb", "covariate_ridge"):
        assert summary["models"][model]["gene_pcc"] is None
        assert summary["models"][model]["within_type_pcc"] is None
    assert "undefined" in (output / "report.md").read_text(encoding="utf-8")
    stored = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert stored["models"]["covariate_nb"]["within_type_pcc"] is None
    split = pd.read_csv(output / "split.csv")
    test_ids = split.loc[split["split"] == "test", "cell_id"].tolist()
    lookup = {cell: index for index, cell in enumerate(data.cells.cell_id)}
    indices = np.array([lookup[cell] for cell in test_ids])
    features_path, covariates_path = tmp_path / "images.npz", tmp_path / "covariates.npz"
    np.savez(features_path, cell_ids=np.asarray(test_ids), features=data.features[indices])
    # Deliberately reverse the covariate file; prediction must join by ID.
    np.savez(covariates_path, cell_ids=np.asarray(test_ids[::-1]), covariates=data.covariates[indices[::-1]])
    (data_dir / "counts.npz").unlink()
    prediction_path = tmp_path / "prediction.npz"
    predict_from_checkpoint(output / "conditional_nb.pt", features_path, covariates_path, prediction_path)
    with np.load(output / "test_predictions.npz") as expected, np.load(prediction_path) as predicted:
        np.testing.assert_array_equal(predicted["cell_ids"], expected["cell_ids"])
        np.testing.assert_array_equal(predicted["gene_ids"], expected["gene_ids"])
        np.testing.assert_allclose(predicted["rates"], expected["conditional_nb"], rtol=1e-5, atol=1e-6)
    metadata = json.loads(prediction_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["mode"] == "synthetic"


def test_run_rejects_rna_derived_image_features_before_training(tmp_path):
    data, data_dir, config_path = _small_experiment(tmp_path)
    data.provenance["feature_source"] = "rna_oracle"
    save_dataset(data, data_dir)
    with pytest.raises(ValueError, match="RNA-derived features"):
        run_experiment(config_path)
    assert not (tmp_path / "run").exists()


def test_run_requires_explicit_oracle_opt_in(tmp_path):
    data, data_dir, config_path = _small_experiment(tmp_path)
    data.provenance["covariate_source"] = "rna_oracle"
    save_dataset(data, data_dir)
    with pytest.raises(ValueError, match="allow_oracle"):
        run_experiment(config_path)
    assert not (tmp_path / "run").exists()
