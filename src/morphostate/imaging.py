"""Landmark registration and reproducible H&E crop preparation.

Landmark fitting is not automatic biological registration. Input x/y coordinates
and landmark source coordinates must use the same units; target coordinates are
H&E level-zero pixels. Validation landmarks never enter the fit.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import _check_identifiers


def fit_affine_landmarks(landmarks_path, output, residual_limit=None):
    points = pd.read_csv(landmarks_path)
    columns = ["source_x", "source_y", "target_x", "target_y"]
    if not set(columns) <= set(points) or not np.isfinite(points[columns].to_numpy()).all():
        raise ValueError("Landmarks require finite source_x,source_y,target_x,target_y")
    subset = points["split"] if "split" in points else pd.Series(["train"] * len(points))
    if not subset.isin(["train", "validation"]).all():
        raise ValueError("Landmark split must be train or validation")
    train = subset.to_numpy() == "train"
    design = np.column_stack([points.source_x, points.source_y, np.ones(len(points))])
    targets = points[["target_x", "target_y"]].to_numpy(dtype=float)
    if train.sum() < 3 or np.linalg.matrix_rank(design[train]) != 3:
        raise ValueError("Need >=3 non-collinear training landmarks")
    coefficients = np.linalg.lstsq(design[train], targets[train], rcond=None)[0]
    transform = np.eye(3)
    transform[:2] = coefficients.T
    if abs(np.linalg.det(transform[:2, :2])) < 1e-12:
        raise ValueError("Estimated affine transform is singular")
    residuals = np.linalg.norm(design @ coefficients - targets, axis=1)
    stats = {part: {"n": int(mask.sum()), "rmse_pixels": float(np.sqrt(np.mean(residuals[mask] ** 2))),
                    "max_error_pixels": float(residuals[mask].max())} if mask.any() else None
             for part, mask in [("train", train), ("validation", ~train)]}
    if residual_limit is not None:
        if not np.isfinite(residual_limit) or residual_limit <= 0 or not (~train).any():
            raise ValueError("A positive residual limit requires independent validation landmarks")
        if stats["validation"]["max_error_pixels"] > residual_limit:
            raise ValueError("Independent registration validation exceeds the declared error limit")
    result = {"source_to_he_level0": transform.tolist(), "landmark_errors": stats,
              "note": "Affine landmarks only; training residuals do not establish independent registration accuracy."}
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def transform_points(xy, matrix):
    xy, matrix = np.asarray(xy, dtype=float), np.asarray(matrix, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or matrix.shape != (3, 3):
        raise ValueError("Expected n-by-2 points and a 3-by-3 transform")
    if not np.isfinite(xy).all() or not np.isfinite(matrix).all() or not np.allclose(matrix[2], [0, 0, 1]):
        raise ValueError("Only finite affine transforms are supported")
    if np.linalg.matrix_rank(matrix[:2, :2]) < 2:
        raise ValueError("Affine transform must be invertible")
    transformed = np.column_stack([xy, np.ones(len(xy))]) @ matrix[:2].T
    if not np.isfinite(transformed).all():
        raise ValueError("Transformed coordinates overflowed; verify units and transform")
    return transformed


def extract_registered_patches(image_path, cells_path, transform_path, output, patch_pixels=224,
                               backend="pillow", edge_policy="error"):
    """Crop centers in H&E coordinates; edge padding requires an explicit policy.

    For pyramidal WSI use optional OpenSlide and its native runtime. Pillow's
    image-size protection remains enabled. This never guesses channel or units.
    """
    from PIL import Image
    if not isinstance(patch_pixels, int) or patch_pixels < 2 or edge_policy not in {"error", "white"}:
        raise ValueError("Invalid patch size or edge policy")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    cells = pd.read_csv(cells_path, dtype={"cell_id": str}, keep_default_na=False)
    if not {"cell_id", "x_centroid", "y_centroid"} <= set(cells) or cells.cell_id.isna().any() or cells.cell_id.duplicated().any() or not len(cells):
        raise ValueError("Cell CSV requires unique IDs and x_centroid,y_centroid")
    _check_identifiers(cells.cell_id, "cell_id", unique=True)
    if "slide_id" in cells and cells.slide_id.nunique() != 1:
        raise ValueError("One H&E image can crop only one declared slide; split the cell table by slide")
    saved = json.loads(Path(transform_path).read_text(encoding="utf-8"))
    coordinates = transform_points(cells[["x_centroid", "y_centroid"]].to_numpy(), saved["source_to_he_level0"])
    if (np.abs(coordinates) > np.iinfo(np.int64).max / 4).any():
        raise ValueError("Transformed coordinates exceed the integer crop range; verify coordinate units")
    if backend == "openslide":
        import openslide
        handle = openslide.OpenSlide(str(image_path))
        width, height = handle.dimensions
    elif backend == "pillow":
        handle = Image.open(image_path)
        if handle.mode not in {"RGB", "RGBA"}:
            handle.close()
            raise ValueError("H&E input must have explicit RGB/RGBA channels; fluorescence is not H&E")
        width, height = handle.size
    else:
        raise ValueError("backend must be pillow or openslide")
    starts = np.floor(coordinates - patch_pixels / 2).astype(int)
    clipped = (starts[:, 0] < 0) | (starts[:, 1] < 0) | (starts[:, 0] + patch_pixels > width) | (starts[:, 1] + patch_pixels > height)
    if edge_policy == "error" and clipped.any():
        handle.close()
        raise ValueError(f"{clipped.sum()} patches cross image borders; explicitly filter cells or use --edge-policy white")
    output.mkdir(parents=True, exist_ok=True)
    (output / "patches").mkdir(exist_ok=True)
    manifest = []
    try:
        for i, (x, y) in enumerate(starts):
            if backend == "openslide":
                rgba = handle.read_region((int(x), int(y)), 0, (patch_pixels, patch_pixels)).convert("RGBA")
                patch = Image.new("RGBA", rgba.size, "white")
                patch.alpha_composite(rgba)
                patch = patch.convert("RGB")
            else:
                patch = Image.new("RGB", (patch_pixels, patch_pixels), "white")
                box = (max(0, x), max(0, y), min(width, x + patch_pixels), min(height, y + patch_pixels))
                if box[2] > box[0] and box[3] > box[1]:
                    crop = handle.crop(box)
                    if crop.mode == "RGBA":
                        # Transparent source pixels represent missing tissue;
                        # converting directly to RGB would turn them black.
                        opaque = Image.new("RGBA", crop.size, "white")
                        opaque.alpha_composite(crop)
                        crop = opaque.convert("RGB")
                    else:
                        crop = crop.convert("RGB")
                    patch.paste(crop, (max(0, -x), max(0, -y)))
            name = f"patches/{i:08d}.png"
            patch.save(output / name)
            manifest.append({"cell_id": cells.cell_id.iloc[i], "patch_path": name,
                             "he_x": float(coordinates[i, 0]), "he_y": float(coordinates[i, 1]), "padded": bool(clipped[i])})
    finally:
        handle.close()
    pd.DataFrame(manifest).to_csv(output / "patch_manifest.csv", index=False)
    (output / "provenance.json").write_text(json.dumps({"image": str(Path(image_path).resolve()), "transform": saved,
        "patch_pixels_level0": patch_pixels, "backend": backend, "edge_policy": edge_policy,
        "note": "Record source coordinate units and image microns-per-pixel separately; this tool does not infer physical scale or confirm H&E identity."}, indent=2), encoding="utf-8")
    return output / "patch_manifest.csv"
