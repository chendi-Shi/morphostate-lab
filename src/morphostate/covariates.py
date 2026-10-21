"""Train a coarse image-only type predictor and construct spatial composition.

Training labels may come from paired RNA, but held-out labels never enter input
construction. Covariates must be regenerated inside each outer training fold.
"""

import copy
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .data import _check_identifiers, load_dataset, save_dataset
from .metrics import known_cell_type_mask
from .splits import make_donor_split


def check_covariate_provenance(provenance, split_donors):
    """Fail when a fitted nuisance classifier was trained using held-out people."""
    for key in ("covariate_fit_donors", "covariate_selection_donors"):
        values = provenance.get(key, [])
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"{key} must be an explicit list of donor IDs")
        _check_identifiers(values, key, unique=True)
    fitted = set(provenance.get("covariate_fit_donors", []))
    selected = set(provenance.get("covariate_selection_donors", []))
    if not fitted <= set(split_donors["train"]):
        raise ValueError("Image covariate predictor used donors outside this training fold; regenerate covariates per fold")
    if selected & set(split_donors["test"]):
        raise ValueError("Image covariate predictor was selected using current test donors")


def neighborhood_composition(probabilities, cells, neighbors=12):
    """Average other cells on the same slide and donor, with a singleton fallback.

    A donor/slide group containing one cell has no neighborhood information;
    its own type probabilities are returned explicitly as the fallback.
    """
    if (not isinstance(neighbors, (int, np.integer)) or isinstance(neighbors, bool)
            or neighbors < 1 or not {"x_centroid", "y_centroid", "slide_id"} <= set(cells)):
        raise ValueError("Need positive neighbors and spatial cell coordinates/slide_id")
    values = np.asarray(probabilities, dtype=float)
    xy = cells[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
    if (values.ndim != 2 or values.shape[1] == 0 or len(values) != len(cells)
            or not np.isfinite(values).all() or not np.isfinite(xy).all()
            or (values < 0).any() or not np.allclose(values.sum(axis=1), 1, atol=1e-6)):
        raise ValueError("Invalid probabilities or coordinates")
    _check_identifiers(cells.slide_id, "slide_id")
    group_columns = ["slide_id"]
    if "donor_id" in cells:
        _check_identifiers(cells.donor_id, "donor_id")
        group_columns.append("donor_id")
    result = np.zeros_like(values)
    for indices in cells.groupby(group_columns, sort=True).indices.values():
        indices = np.asarray(indices)
        if len(indices) == 1:
            result[indices] = values[indices]
            continue
        _, nearest = cKDTree(xy[indices]).query(xy[indices], k=min(neighbors + 1, len(indices)))
        for local, neighbor_ids in enumerate(nearest):
            # Exclude the exact cell index, including when centers are tied.
            chosen = np.asarray(neighbor_ids)[np.asarray(neighbor_ids) != local][:neighbors]
            result[indices[local]] = values[indices[chosen]].mean(axis=0)
    return result


def build_image_covariates(dataset_path, split_path, output, neighbors=12, epochs=100, patience=15, seed=7):
    import torch
    from torch import nn

    from .training import set_seed
    data = load_dataset(dataset_path)
    if data.provenance["feature_source"] == "rna_oracle":
        raise ValueError("Cannot train an image covariate predictor on RNA-derived features")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    explicit = json.loads(Path(split_path).read_text(encoding="utf-8"))
    split = make_donor_split(data.cells, explicit=explicit)
    labels = data.cells.cell_type.to_numpy(dtype=str)
    train = split["train"][known_cell_type_mask(labels[split["train"]])]
    # Plain Python strings keep the checkpoint compatible with weights_only
    # loading; NumPy scalar strings otherwise require unsafe pickle globals.
    classes = sorted({str(label) for label in labels[train]})
    if (len(classes) < 2 or not isinstance(epochs, int) or not isinstance(patience, int)
            or epochs < 1 or patience < 1):
        raise ValueError("Need >=2 known coarse training types and positive training settings")
    lookup = {label: i for i, label in enumerate(classes)}
    validation = np.array([i for i in split["val"] if labels[i] in lookup], dtype=int)
    if not len(validation):
        raise ValueError("No validation annotations belong to training classes")
    mean = data.features[train].mean(0, dtype=np.float64)
    scale = data.features[train].std(0, dtype=np.float64)
    scale[scale < 1e-6] = 1
    x = torch.tensor((data.features - mean) / scale, dtype=torch.float32)
    if not torch.isfinite(x).all():
        raise ValueError("Normalized image features exceed finite float32 range")
    y_train = torch.tensor([lookup[labels[i]] for i in train])
    y_val = torch.tensor([lookup[labels[i]] for i in validation])
    train_donors = data.cells.iloc[train].donor_id
    val_donors = data.cells.iloc[validation].donor_id
    w = torch.tensor(1. / train_donors.map(train_donors.value_counts()).to_numpy(), dtype=torch.float32)
    w = w / w.sum()
    w_val = torch.tensor(1. / val_donors.map(val_donors.value_counts()).to_numpy(), dtype=torch.float32)
    w_val = w_val / w_val.sum()
    set_seed(seed)
    model = nn.Linear(x.shape[1], len(classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.001)
    best, best_score, bad, history = None, float("inf"), 0, []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = (nn.functional.cross_entropy(model(x[train]), y_train, reduction="none") * w).sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("Type classifier training loss is nonfinite")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            score = float((nn.functional.cross_entropy(model(x[validation]), y_val, reduction="none") * w_val).sum())
        if not np.isfinite(score):
            raise FloatingPointError("Type classifier validation loss is nonfinite")
        history.append({"epoch": epoch + 1, "validation_cross_entropy": score})
        if score < best_score:
            best, best_score, bad = copy.deepcopy(model.state_dict()), score, 0
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(best)
    with torch.no_grad():
        probabilities = model(x).softmax(1).numpy()
    neighborhood = neighborhood_composition(probabilities, data.cells, neighbors)
    data.covariates = np.column_stack([probabilities, neighborhood]).astype(np.float32)
    data.provenance.update(covariate_source="synthetic" if data.provenance["feature_source"] == "synthetic" else "image",
        covariate_fit_donors=explicit["train"], covariate_selection_donors=explicit["val"],
        covariate_columns=[f"type_probability:{label}" for label in classes] + [f"neighborhood_probability:{label}" for label in classes],
        covariate_method="Train-donor-only linear image classifier; same-slide same-donor k-neighbor average excluding self; singleton groups use self probabilities",
        covariate_label_note="Only coarse type annotations on training/validation donors used. Test labels not used. Regenerate for every outer fold.")
    save_dataset(data, output)
    torch.save({"state_dict": best, "mean": torch.from_numpy(mean), "scale": torch.from_numpy(scale),
                "classes": classes, "neighbors": neighbors, "fit_donors": explicit["train"]}, output / "type_classifier.pt")
    (output / "type_classifier_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return data
