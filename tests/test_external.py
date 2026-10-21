import numpy as np
import pytest

from morphostate.external import align_external_predictions


def test_external_aligns_ids_and_reference_units(tmp_path):
    path = tmp_path / "prediction.npz"
    np.savez(path, cell_ids=np.array(["b", "a"]), gene_ids=np.array(["y", "x"]),
             rates=np.array([[40., 30.], [20., 10.]]))
    aligned = align_external_predictions(path, ["a", "b"], ["x", "y"], 10000, 1000)
    np.testing.assert_allclose(aligned, [[1, 2], [3, 4]])


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf")])
def test_external_rejects_invalid_predictions(tmp_path, bad):
    path = tmp_path / "prediction.npz"
    np.savez(path, cell_ids=np.array(["a"]), gene_ids=np.array(["x"]), rates=np.array([[bad]]))
    with pytest.raises(ValueError, match="finite nonnegative"):
        align_external_predictions(path, ["a"], ["x"], 1000, 1000)


def test_external_rejects_missing_test_cells(tmp_path):
    path = tmp_path / "prediction.npz"
    np.savez(path, cell_ids=np.array(["a"]), gene_ids=np.array(["x"]), rates=np.ones((1, 1)))
    with pytest.raises(ValueError, match="exactly"):
        align_external_predictions(path, ["a", "b"], ["x"], 1000, 1000)
