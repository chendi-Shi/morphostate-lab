from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from morphostate.data import Dataset, load_dataset, save_dataset, validate_dataset
from morphostate.splits import make_donor_split
from morphostate.synthetic import make_synthetic


@pytest.fixture
def dataset():
    return Dataset(
        counts=sparse.csr_matrix([[1, 0], [0, 2], [3, 1], [1, 1], [0, 3], [4, 2]]),
        features=np.arange(18, dtype=np.float32).reshape(6, 3),
        covariates=np.eye(2, dtype=np.float32)[[0, 1, 0, 1, 0, 1]],
        cells=pd.DataFrame({
            "cell_id": [f"c{i}" for i in range(6)],
            "donor_id": ["d0", "d0", "d1", "d1", "d2", "d2"],
            "slide_id": [f"slide{i}" for i in range(6)],
            "cell_type": ["A", "B"] * 3,
        }),
        genes=["g1", "g2"],
        provenance={"dataset_id": "fixture", "feature_source": "image", "covariate_source": "image"},
    )


def test_round_trip_is_lossless_and_does_not_normalize(dataset, tmp_path):
    save_dataset(dataset, tmp_path)
    loaded = load_dataset(tmp_path)
    np.testing.assert_array_equal(loaded.counts.toarray(), dataset.counts.toarray())
    np.testing.assert_array_equal(loaded.features, dataset.features)
    np.testing.assert_array_equal(loaded.covariates, dataset.covariates)
    pd.testing.assert_frame_equal(loaded.cells, dataset.cells)
    assert loaded.genes == dataset.genes
    assert loaded.provenance == dataset.provenance


@pytest.mark.parametrize("bad", [-1, 0.5, np.nan, np.inf])
def test_rejects_invalid_counts(dataset, bad):
    counts = dataset.counts.astype(float)
    counts.data[0] = bad
    with pytest.raises(ValueError, match="finite nonnegative integers"):
        validate_dataset(replace(dataset, counts=counts))


def test_rejects_zero_total_cell(dataset):
    counts = dataset.counts.toarray()
    counts[0] = 0
    with pytest.raises(ValueError, match="zero-total"):
        validate_dataset(replace(dataset, counts=sparse.csr_matrix(counts)))


def test_rejects_duplicate_and_missing_ids(dataset):
    cells = dataset.cells.copy()
    cells.loc[1, "cell_id"] = cells.loc[0, "cell_id"]
    with pytest.raises(ValueError, match="unique"):
        validate_dataset(replace(dataset, cells=cells))
    cells = dataset.cells.copy()
    cells.loc[0, "donor_id"] = ""
    with pytest.raises(ValueError, match="nonempty"):
        validate_dataset(replace(dataset, cells=cells))
    with pytest.raises(ValueError, match="unique"):
        validate_dataset(replace(dataset, genes=["g1", "g1"]))


def test_rejects_misaligned_or_nonfinite_features(dataset):
    with pytest.raises(ValueError, match="row count"):
        validate_dataset(replace(dataset, features=dataset.features[:-1]))
    features = dataset.features.copy()
    features[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite real"):
        validate_dataset(replace(dataset, features=features))


def test_rejects_unknown_provenance(dataset):
    with pytest.raises(ValueError, match="covariate_source"):
        validate_dataset(replace(dataset, provenance={**dataset.provenance, "covariate_source": "unspecified"}))


def test_float32_overflow_is_rejected_before_writing(dataset, tmp_path):
    features = dataset.features.astype(np.float64)
    features[0, 0] = 1e100
    with pytest.raises(ValueError, match="float32"):
        save_dataset(replace(dataset, features=features), tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_donor_split_keeps_all_slides_together_and_is_reproducible(dataset):
    result = make_donor_split(dataset.cells, seed=17)
    repeated = make_donor_split(dataset.cells, seed=17)
    seen = set()
    for name in ("train", "val", "test"):
        np.testing.assert_array_equal(result[name], repeated[name])
        donors = set(dataset.cells.iloc[result[name]]["donor_id"])
        assert donors.isdisjoint(seen)
        seen |= donors
        assert len(result[name]) == 2
    np.testing.assert_array_equal(np.sort(np.concatenate(list(result.values()))), np.arange(6))


@pytest.mark.parametrize("explicit,pattern", [
    ({"train": ["d0", "d1"], "val": ["d1"], "test": ["d2"]}, "overlap"),
    ({"train": ["d0"], "val": ["missing"], "test": ["d2"]}, "unknown donors"),
    ({"train": ["d0"], "val": ["d1"], "test": []}, "must not be empty"),
    ({"train": ["d0", "d0"], "val": ["d1"], "test": ["d2"]}, "unique"),
    ({"train": ["d0"], "test": ["d2"]}, "exactly"),
])
def test_explicit_split_rejects_leakage_and_unknowns(dataset, explicit, pattern):
    with pytest.raises(ValueError, match=pattern):
        make_donor_split(dataset.cells, explicit=explicit)


def test_explicit_split_requires_every_donor():
    cells = pd.DataFrame({"donor_id": ["a", "b", "c", "d"]})
    with pytest.raises(ValueError, match="missing donors.*d"):
        make_donor_split(cells, explicit={"train": ["a"], "val": ["b"], "test": ["c"]})


def test_explicit_split_uses_positions_not_dataframe_index(dataset):
    cells = dataset.cells.copy()
    cells.index = [10, 20, 30, 40, 50, 60]
    result = make_donor_split(cells, explicit={"train": ["d2"], "val": ["d0"], "test": ["d1"]})
    np.testing.assert_array_equal(result["train"], [4, 5])


def test_synthetic_is_reproducible_and_explicitly_labelled(tmp_path):
    for name in ("one", "two"):
        make_synthetic(tmp_path / name, seed=42, donors=4, cells_per_donor=12, genes=8, features=6)
    one, two = (load_dataset(tmp_path / name) for name in ("one", "two"))
    np.testing.assert_array_equal(one.counts.toarray(), two.counts.toarray())
    np.testing.assert_array_equal(one.features, two.features)
    assert len(one.cells["donor_id"].unique()) == 4
    assert one.provenance["covariate_source"] == "synthetic"
    assert one.provenance["feature_source"] == "synthetic"
    assert "plumbing-only" in one.provenance["dataset_id"]
    assert all(one.cells.groupby("donor_id")["cell_type"].nunique() > 1)
