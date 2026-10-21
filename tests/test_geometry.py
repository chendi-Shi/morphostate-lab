import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from morphostate.data import Dataset, save_dataset
from morphostate.geometry import (
    construct_geometry_graphs,
    load_graphs,
    make_synthetic_graphs,
    save_graphs,
    subset_genes,
    without_budgets,
)


def fixture_data(tmp_path):
    cells = pd.DataFrame({"cell_id": ["s::a", "s::b"], "source_cell_id": ["a", "b"],
                          "donor_id": ["d", "d"], "slide_id": ["s", "s"], "cell_type": ["T", "T"],
                          "x_centroid": [0.0, 2.0], "y_centroid": [0.0, 0.0]})
    dataset = Dataset(sparse.csr_matrix([[20, 10], [10, 20]]), np.ones((2, 2)), np.ones((2, 1)),
                      cells, ["G0", "G1"], {"dataset_id": "test", "feature_source": "synthetic",
                                           "covariate_source": "synthetic"})
    path = tmp_path / "data"
    save_dataset(dataset, path)
    table = pd.DataFrame({"slide_id": ["s"] * 6, "x": [0, 1, 1, 2, 1, 0], "y": [0] * 6,
                          "gene_id": ["G0", "G1", "G0", "G1", "G0", "control"],
                          "cell_id": ["a", "a", "b", "b", "", "a"],
                          "mass": [20, 10, 10, 20, 1, 2],
                          "is_core": [True, False, False, True, False, False]})
    transcript_path = tmp_path / "transcripts.csv"
    table.to_csv(transcript_path, index=False)
    return dataset, path, table, transcript_path


def test_registered_geometry_conservation_and_roundtrip(tmp_path):
    dataset, path, _, transcript_path = fixture_data(tmp_path)
    blocks = construct_geometry_graphs(path, transcript_path, tmp_path / "graphs", radius=1.1,
                                        geometry_budget=20, background_fraction=0.1,
                                        coordinate_units="micrometre")
    assert len(blocks) == 1
    block = blocks[0]
    np.testing.assert_array_equal(block.label_set.fixed_labels(block.fixed_owner)[:-1], dataset.counts.toarray())
    assert block.label_set.fixed_labels(block.fixed_owner)[-1, 0] == 1
    assert block.metadata["unknown_gene_rows_filtered"] == 1
    assert block.metadata["contains_independent_nuclear_core_annotation"]
    assert (block.label_set._owner_counts == 1).sum() == 2
    loaded = load_graphs(tmp_path / "graphs")[0]
    coefficient = np.arange(6).reshape(3, 2)
    assert loaded.label_set.support_function(coefficient) == pytest.approx(block.label_set.support_function(coefficient))
    assert loaded.metadata == block.metadata
    with pytest.raises(FileExistsError):
        save_graphs(blocks, tmp_path / "graphs")


def test_geometry_rejects_radius_or_count_mismatch(tmp_path):
    _, path, table, transcript_path = fixture_data(tmp_path)
    with pytest.raises(ValueError, match="outside radius"):
        construct_geometry_graphs(path, transcript_path, tmp_path / "bad_radius", 0.5,
                                   coordinate_units="micrometre")
    table.loc[0, "mass"] -= 1
    table.to_csv(transcript_path, index=False)
    with pytest.raises(ValueError, match="disagree"):
        construct_geometry_graphs(path, transcript_path, tmp_path / "bad_counts", 1.1,
                                   coordinate_units="micrometre")


def test_explicit_units_unknown_owner_and_memory_guards(tmp_path):
    _, path, table, transcript_path = fixture_data(tmp_path)
    with pytest.raises(ValueError, match="coordinate_units"):
        construct_geometry_graphs(path, transcript_path, tmp_path / "bad_units", 1.1)
    with pytest.raises(ValueError, match="max_observations"):
        construct_geometry_graphs(path, transcript_path, tmp_path / "too_big", 1.1,
                                   coordinate_units="um", max_observations=2)
    table.loc[0, "cell_id"] = "missing"
    table.to_csv(transcript_path, index=False)
    with pytest.raises(ValueError, match="Unknown original"):
        construct_geometry_graphs(path, transcript_path, tmp_path / "bad_id", 1.1,
                                   coordinate_units="um")


def test_no_core_inference_without_annotation(tmp_path):
    _, path, table, transcript_path = fixture_data(tmp_path)
    table.drop(columns="is_core").to_csv(transcript_path, index=False)
    block = construct_geometry_graphs(path, transcript_path, tmp_path / "no_core", 1.1,
                                       coordinate_units="um")[0]
    assert not (block.label_set._owner_counts == 1).any()
    assert not block.metadata["contains_independent_nuclear_core_annotation"]


def test_core_background_is_rejected(tmp_path):
    _, path, table, transcript_path = fixture_data(tmp_path)
    table.loc[4, "is_core"] = True
    table.to_csv(transcript_path, index=False)
    with pytest.raises(ValueError, match="real-cell"):
        construct_geometry_graphs(path, transcript_path, tmp_path / "bad_core", 1.1,
                                   coordinate_units="um")


def test_synthetic_graph_preserves_observed_counts_and_has_core(tmp_path):
    dataset, path, _, _ = fixture_data(tmp_path)
    blocks = make_synthetic_graphs(path, tmp_path / "synthetic", seed=21)
    block = blocks[0]
    np.testing.assert_array_equal(block.label_set.fixed_labels(block.fixed_owner)[:-1], dataset.counts.toarray())
    assert (block.label_set._owner_counts == 1).any()
    assert (block.label_set._owner_counts > 1).any()
    again = make_synthetic_graphs(path, seed=21)[0]
    np.testing.assert_array_equal(block.label_set.mass, again.label_set.mass)
    np.testing.assert_array_equal(block.label_set.candidate_owners.indices, again.label_set.candidate_owners.indices)


def test_gene_subset_conditions_removed_budget_and_reorders(tmp_path):
    _, path, _, transcript_path = fixture_data(tmp_path)
    block = construct_geometry_graphs(path, transcript_path, tmp_path / "graphs", 1.1,
                                       geometry_budget=20, background_fraction=0.1,
                                       coordinate_units="um")[0]
    subset = subset_genes(block, ["G1"])
    assert subset.gene_ids == ["G1"]
    assert subset.label_set.max_total_cost == pytest.approx(20 - 10 / 1.1**2)
    assert subset.label_set.max_background_mass == pytest.approx(6.1 - 1)
    np.testing.assert_array_equal(subset.label_set.fixed_labels(subset.fixed_owner)[:-1, 0], [10, 20])
    assert "condition_removed" in subset.metadata["gene_subset_policy"]
    reordered = subset_genes(block, ["G1", "G0"])
    np.testing.assert_array_equal(reordered.label_set.fixed_labels(reordered.fixed_owner)[:-1], [[10, 20], [20, 10]])
    unbounded = without_budgets(block, geometry=True, background=False)
    assert unbounded.label_set.max_total_cost is None
    assert unbounded.label_set.max_background_mass == block.label_set.max_background_mass


def test_multidonor_slide_rejected(tmp_path):
    dataset, path, _, _ = fixture_data(tmp_path)
    dataset.cells.loc[1, "donor_id"] = "other"
    save_dataset(dataset, path)
    with pytest.raises(ValueError, match="multiple donors"):
        make_synthetic_graphs(path)


def test_artifact_rejects_path_traversal_and_duplicate_cells(tmp_path):
    _, path, _, _ = fixture_data(tmp_path)
    blocks = make_synthetic_graphs(path, tmp_path / "graphs")
    with pytest.raises(ValueError, match="unique IDs"):
        save_graphs([blocks[0], blocks[0]], tmp_path / "duplicates")
    manifest_path = tmp_path / "graphs" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["blocks"][0]["file"] = "../escape.npz"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="basename"):
        load_graphs(tmp_path / "graphs")
