import json

import numpy as np
import pandas as pd
import pytest

from morphostate.imaging import fit_affine_landmarks, transform_points


def test_affine_fit_never_uses_validation_landmarks(tmp_path):
    source = tmp_path / "landmarks.csv"
    pd.DataFrame({"source_x": [0, 1, 0, 1], "source_y": [0, 0, 1, 1],
                  "target_x": [3, 5, 3, 100], "target_y": [7, 7, 10, 100],
                  "split": ["train"] * 3 + ["validation"]}).to_csv(source, index=False)
    result = fit_affine_landmarks(source, tmp_path / "affine.json")
    np.testing.assert_allclose(transform_points([[1, 1]], result["source_to_he_level0"]), [[5, 10]])
    assert result["landmark_errors"]["validation"]["max_error_pixels"] > 100
    with pytest.raises(ValueError, match="exceeds"):
        fit_affine_landmarks(source, tmp_path / "bad.json", residual_limit=5)


def test_registered_crop_uses_transform_and_preserves_ids(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    from morphostate.imaging import extract_registered_patches
    image = image_module.new("RGB", (20, 20), (120, 50, 80))
    image.save(tmp_path / "he.png")
    pd.DataFrame({"cell_id": ["raw::cell"], "x_centroid": [5], "y_centroid": [5]}).to_csv(tmp_path / "cells.csv", index=False)
    (tmp_path / "affine.json").write_text(json.dumps({"source_to_he_level0": [[1, 0, 5], [0, 1, 5], [0, 0, 1]]}))
    result = extract_registered_patches(tmp_path / "he.png", tmp_path / "cells.csv", tmp_path / "affine.json", tmp_path / "patches", 8)
    manifest = pd.read_csv(result)
    assert manifest.he_x.iloc[0] == 10
    assert manifest.cell_id.iloc[0] == "raw::cell"


def test_registration_rejects_nan_validation_limit_and_singular_matrix(tmp_path):
    source = tmp_path / "landmarks.csv"
    pd.DataFrame({"source_x": [0, 1, 0, 1], "source_y": [0, 0, 1, 1],
                  "target_x": [0, 1, 0, 1], "target_y": [0, 0, 1, 1],
                  "split": ["train"] * 3 + ["validation"]}).to_csv(source, index=False)
    with pytest.raises(ValueError, match="positive residual limit"):
        fit_affine_landmarks(source, tmp_path / "bad.json", residual_limit=np.nan)
    with pytest.raises(ValueError, match="invertible"):
        transform_points([[0, 0]], [[1, 0, 0], [0, 0, 0], [0, 0, 1]])


def test_rgba_crops_composite_transparency_and_edges_white(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    from morphostate.imaging import extract_registered_patches
    image = image_module.new("RGBA", (4, 4), (0, 0, 0, 0))
    image.putpixel((0, 0), (100, 20, 30, 255))
    image.save(tmp_path / "he.png")
    pd.DataFrame({"cell_id": ["NA"], "x_centroid": [0], "y_centroid": [0]}).to_csv(tmp_path / "cells.csv", index=False)
    (tmp_path / "affine.json").write_text(json.dumps({"source_to_he_level0": np.eye(3).tolist()}))
    with pytest.raises(ValueError, match="cross image borders"):
        extract_registered_patches(tmp_path / "he.png", tmp_path / "cells.csv", tmp_path / "affine.json",
                                     tmp_path / "error", 4)
    manifest_path = extract_registered_patches(tmp_path / "he.png", tmp_path / "cells.csv", tmp_path / "affine.json",
                                                tmp_path / "white", 4, edge_policy="white")
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    assert manifest.cell_id.tolist() == ["NA"]
    with image_module.open(manifest_path.parent / manifest.patch_path.iloc[0]) as patch:
        assert patch.mode == "RGB"
        assert patch.getpixel((2, 2)) == (100, 20, 30)
        assert patch.getpixel((3, 3)) == (255, 255, 255)
        assert patch.getpixel((0, 0)) == (255, 255, 255)


def test_single_image_cannot_silently_crop_multiple_slides(tmp_path):
    pytest.importorskip("PIL.Image")
    from morphostate.imaging import extract_registered_patches
    pd.DataFrame({"cell_id": ["a", "b"], "x_centroid": [3, 3], "y_centroid": [3, 3],
                  "slide_id": ["first", "second"]}).to_csv(tmp_path / "cells.csv", index=False)
    with pytest.raises(ValueError, match="one declared slide"):
        extract_registered_patches(tmp_path / "not_opened.png", tmp_path / "cells.csv",
                                     tmp_path / "not_opened.json", tmp_path / "patches", 4)
