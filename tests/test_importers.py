import json
from dataclasses import replace

import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from morphostate.data import load_dataset, save_dataset
from morphostate.importers import import_xenium, merge_datasets


@pytest.fixture
def xenium_inputs(tmp_path):
    matrix_path = tmp_path / "cell_feature_matrix.h5"
    matrix = sparse.csc_matrix([[1, 0, 3], [0, 2, 4], [9, 9, 9]])
    with h5py.File(matrix_path, "w") as handle:
        group = handle.create_group("matrix")
        for name, data in {"data": matrix.data, "indices": matrix.indices,
                           "indptr": matrix.indptr, "shape": matrix.shape,
                           "barcodes": np.array(["a", "b", "c"], dtype="S")}.items():
            group.create_dataset(name, data=data)
        features = group.create_group("features")
        features.create_dataset("id", data=np.array(["g1", "g2", "control"], dtype="S"))
        features.create_dataset("feature_type", data=np.array(["Gene Expression", "Gene Expression", "Negative Control Probe"], dtype="S"))
    cells_path = tmp_path / "cells.csv"
    pd.DataFrame({"cell_id": ["c", "a", "b"], "cell_type": ["C", "A", "B"],
                  "x_centroid": [30, 10, 20]}).to_csv(cells_path, index=False)
    feature_path = tmp_path / "features.npz"
    np.savez(feature_path, cell_ids=np.array(["b", "c", "a"]), features=np.array([[20, 21], [30, 31], [10, 11]]))
    feature_path.with_suffix(".json").write_text(json.dumps({"feature_source": "image", "feature_encoder": {
        "model": "test_fixture", "weights_sha256": "fixture-only", "transform": {"size": 4}}}), encoding="utf-8")
    covariate_path = tmp_path / "covariates.npz"
    np.savez(covariate_path, cell_ids=np.array(["c", "b", "a"]), covariates=np.array([[3], [2], [1]]))
    return dict(matrix_path=matrix_path, cells_path=cells_path, feature_path=feature_path,
                covariate_path=covariate_path, donor_id="donor1", slide_id="slide1",
                covariate_source="image", dataset_id="fixture")


def test_import_joins_every_input_by_cell_id_and_excludes_controls(xenium_inputs, tmp_path):
    output = tmp_path / "prepared"
    import_xenium(**xenium_inputs, output=output)
    result = load_dataset(output)
    assert result.genes == ["g1", "g2"]
    assert result.cells["cell_id"].tolist() == ["slide1::a", "slide1::b", "slide1::c"]
    assert result.cells["cell_type"].tolist() == ["A", "B", "C"]
    assert result.cells["x_centroid"].tolist() == ["10", "20", "30"]
    np.testing.assert_array_equal(result.counts.toarray(), [[1, 0], [0, 2], [3, 4]])
    np.testing.assert_array_equal(result.features, [[10, 11], [20, 21], [30, 31]])
    np.testing.assert_array_equal(result.covariates, [[1], [2], [3]])
    assert result.provenance["excluded_non_gene_features"] == 1
    assert result.provenance["feature_encoder"]["model"] == "test_fixture"


def test_merge_rejects_different_encoder_weights(xenium_inputs, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    import_xenium(**xenium_inputs, output=first)
    sidecar = xenium_inputs["feature_path"].with_suffix(".json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata["feature_encoder"]["weights_sha256"] = "different-weights"
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    import_xenium(**{**xenium_inputs, "slide_id": "slide2"}, output=second)
    with pytest.raises(ValueError, match="feature_encoder mismatch"):
        merge_datasets([first, second], tmp_path / "merged")


def test_import_requires_encoder_metadata(xenium_inputs, tmp_path):
    xenium_inputs["feature_path"].with_suffix(".json").unlink()
    with pytest.raises(ValueError, match="Missing feature metadata"):
        import_xenium(**xenium_inputs, output=tmp_path / "bad")


@pytest.mark.parametrize("bad_ids", [["a", "a", "c"], ["a", "b", "x"], ["a", "b"], ["a", "b", "c", "x"]])
def test_import_rejects_duplicate_missing_and_extra_feature_ids(xenium_inputs, tmp_path, bad_ids):
    np.savez(xenium_inputs["feature_path"], cell_ids=np.array(bad_ids), features=np.ones((len(bad_ids), 2)))
    with pytest.raises(ValueError, match="unique|cell IDs differ"):
        import_xenium(**xenium_inputs, output=tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_import_rejects_metadata_id_mismatch(xenium_inputs, tmp_path):
    pd.DataFrame({"cell_id": ["a", "b", "x"]}).to_csv(xenium_inputs["cells_path"], index=False)
    with pytest.raises(ValueError, match="cells metadata cell IDs differ"):
        import_xenium(**xenium_inputs, output=tmp_path / "bad")


def test_import_rejects_mixed_donor_metadata(xenium_inputs, tmp_path):
    pd.DataFrame({"cell_id": ["a", "b", "c"], "donor_id": ["donor1", "donor1", "donor2"]}).to_csv(xenium_inputs["cells_path"], index=False)
    with pytest.raises(ValueError, match="conflicts"):
        import_xenium(**xenium_inputs, output=tmp_path / "bad")


def test_import_marks_missing_cell_type_unknown_and_preserves_oracle_flag(xenium_inputs, tmp_path):
    pd.DataFrame({"cell_id": ["a", "b", "c"]}).to_csv(xenium_inputs["cells_path"], index=False)
    import_xenium(**{**xenium_inputs, "covariate_source": "rna_oracle"}, output=tmp_path / "prepared")
    result = load_dataset(tmp_path / "prepared")
    assert set(result.cells["cell_type"]) == {"unknown"}
    assert result.provenance["covariate_source"] == "rna_oracle"


def test_merge_rejects_different_gene_panels_or_order(xenium_inputs, tmp_path):
    first = tmp_path / "one"
    import_xenium(**xenium_inputs, output=first)
    dataset = load_dataset(first)
    second = tmp_path / "two"
    save_dataset(replace(dataset, genes=list(reversed(dataset.genes))), second)
    with pytest.raises(ValueError, match="panel/order mismatch"):
        merge_datasets([first, second], tmp_path / "merged")


def test_merge_rejects_duplicate_cells(xenium_inputs, tmp_path):
    first = tmp_path / "one"
    import_xenium(**xenium_inputs, output=first)
    with pytest.raises(ValueError, match="cell_id must be unique"):
        merge_datasets([first, first], tmp_path / "merged")


def test_merge_rejects_covariate_source_mixing(xenium_inputs, tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"
    import_xenium(**xenium_inputs, output=first)
    import_xenium(**{**xenium_inputs, "slide_id": "slide2", "covariate_source": "rna_oracle"}, output=second)
    with pytest.raises(ValueError, match="covariate_source mismatch"):
        merge_datasets([first, second], tmp_path / "merged")


def test_merge_preserves_multiple_slides_for_same_donor(xenium_inputs, tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"
    import_xenium(**xenium_inputs, output=first)
    import_xenium(**{**xenium_inputs, "slide_id": "slide2"}, output=second)
    merge_datasets([first, second], tmp_path / "merged")
    result = load_dataset(tmp_path / "merged")
    assert result.counts.shape == (6, 2)
    assert result.cells["cell_id"].is_unique
    assert result.cells["donor_id"].nunique() == 1
    assert result.cells["slide_id"].nunique() == 2
