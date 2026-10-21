import numpy as np
import pandas as pd
import pytest

from morphostate.covariates import check_covariate_provenance, neighborhood_composition


def test_neighbors_never_cross_slide_or_include_self():
    cells = pd.DataFrame({"slide_id": ["a", "a", "b", "b"], "x_centroid": [0, 1, 0, 1], "y_centroid": [0] * 4})
    probabilities = np.eye(4)
    result = neighborhood_composition(probabilities, cells, neighbors=1)
    np.testing.assert_array_equal(result, probabilities[[1, 0, 3, 2]])


def test_fitted_covariates_cannot_be_reused_across_incompatible_folds():
    with pytest.raises(ValueError, match="outside this training fold"):
        check_covariate_provenance({"covariate_fit_donors": ["test"]}, {"train": ["train"], "val": ["val"], "test": ["test"]})
    with pytest.raises(ValueError, match="selected using"):
        check_covariate_provenance({"covariate_fit_donors": ["train"], "covariate_selection_donors": ["test"]},
                                   {"train": ["train"], "val": ["val"], "test": ["test"]})


def test_neighbors_respect_donors_and_tied_centroids():
    cells = pd.DataFrame({"slide_id": ["shared"] * 4, "donor_id": ["a", "a", "b", "b"],
                          "x_centroid": [0] * 4, "y_centroid": [0] * 4})
    probabilities = np.eye(4)
    np.testing.assert_array_equal(neighborhood_composition(probabilities, cells, neighbors=1),
                                  probabilities[[1, 0, 3, 2]])


@pytest.mark.parametrize("probabilities,neighbors", [([[0.2, 0.2]], 1), ([[-1, 2]], 1), ([[0, 1]], 1.5)])
def test_neighbors_reject_invalid_probability_inputs(probabilities, neighbors):
    cells = pd.DataFrame({"slide_id": ["s"], "x_centroid": [0], "y_centroid": [0]})
    with pytest.raises(ValueError):
        neighborhood_composition(probabilities, cells, neighbors)


def test_type_classifier_end_to_end_and_test_label_invariance(tmp_path):
    import json

    import torch
    from scipy import sparse

    from morphostate.covariates import build_image_covariates
    from morphostate.data import Dataset, load_dataset, save_dataset

    torch.set_num_threads(1)
    rng = np.random.default_rng(123)
    n = 36
    types = np.tile(["immune", "tumor"], n // 2)
    signal = np.where(types == "immune", -2.0, 2.0)
    features = np.column_stack([signal + rng.normal(0, .1, n), rng.normal(size=n)]).astype(np.float32)
    donors = np.repeat(["train_a", "train_b", "validation", "test"], 9)
    cells = pd.DataFrame({"cell_id": [f"cell{i}" for i in range(n)], "donor_id": donors,
                          "slide_id": donors, "cell_type": types,
                          "x_centroid": np.tile(np.arange(9), 4), "y_centroid": np.zeros(n)})
    dataset = Dataset(sparse.csr_matrix(np.ones((n, 2), dtype=int)), features, np.ones((n, 1)),
                      cells, ["G1", "G2"], {"dataset_id": "classifier_test", "feature_source": "synthetic",
                                           "covariate_source": "synthetic"})
    source = tmp_path / "data"
    save_dataset(dataset, source)
    split = {"train": ["train_a", "train_b"], "val": ["validation"], "test": ["test"]}
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    output = tmp_path / "fit"
    result = build_image_covariates(source, split_path, output, neighbors=2, epochs=80, patience=15, seed=8)
    persisted = load_dataset(output)
    assert persisted.covariates.shape == (n, 4)
    np.testing.assert_allclose(persisted.covariates[:, :2].sum(1), 1, atol=1e-6)
    np.testing.assert_allclose(persisted.covariates[:, 2:].sum(1), 1, atol=1e-6)
    checkpoint = torch.load(output / "type_classifier.pt", weights_only=True)
    assert checkpoint["classes"] == ["immune", "tumor"]
    predicted = np.asarray(checkpoint["classes"])[result.covariates[:, :2].argmax(1)]
    assert (predicted[donors == "test"] == types[donors == "test"]).mean() > .8
    check_covariate_provenance(persisted.provenance, split)
    np.testing.assert_allclose(checkpoint["mean"].numpy(), features[np.isin(donors, split["train"])].mean(0), rtol=1e-5)

    # Replace every test label by a new class. Even class vocabulary, selected
    # epoch, model parameters and all output covariates must remain unchanged.
    changed = load_dataset(source)
    changed.cells.loc[changed.cells.donor_id == "test", "cell_type"] = "never_seen_secret_label"
    save_dataset(changed, tmp_path / "changed")
    rerun = build_image_covariates(tmp_path / "changed", split_path, tmp_path / "fit_changed",
                                   neighbors=2, epochs=80, patience=15, seed=8)
    np.testing.assert_array_equal(rerun.covariates, result.covariates)
    assert (output / "type_classifier_history.json").read_text() == (tmp_path / "fit_changed" / "type_classifier_history.json").read_text()
    altered_checkpoint = torch.load(tmp_path / "fit_changed" / "type_classifier.pt", weights_only=True)
    assert altered_checkpoint["classes"] == checkpoint["classes"]
    for key in checkpoint["state_dict"]:
        assert torch.equal(checkpoint["state_dict"][key], altered_checkpoint["state_dict"][key])
