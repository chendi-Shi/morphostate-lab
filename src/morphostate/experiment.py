"""One locked donor split, train-only preprocessing, validation-only selection."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .data import load_dataset
from .metrics import bootstrap_donor_gain, donor_metrics, within_donor_type_permutation
from .models import RidgePredictor, model_from_config
from .splits import make_donor_split
from .training import fit_conditional_nb, fit_nb_baseline


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def fit_scaler(values):
    mean = values.mean(axis=0).astype(np.float32)
    scale = values.std(axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1
    return mean, scale


def donor_weights(cells):
    donors = cells.donor_id.astype(str)
    return (1.0 / donors.map(donors.value_counts()).to_numpy(dtype=np.float64)).astype(np.float32)


def select_genes(train_counts, exposure, max_genes, min_cells):
    if max_genes < 1 or min_cells < 1:
        raise ValueError("max_genes and min_train_cells must be positive")
    eligible = np.flatnonzero(np.asarray((train_counts > 0).sum(axis=0)).ravel() >= min_cells)
    if len(eligible) == 0:
        raise ValueError("No genes survive the TRAIN-only detection filter")
    # Sparse moments: no full-transcriptome densification for HVG selection.
    normalized = train_counts[:, eligible].multiply(1 / exposure[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    mean = np.asarray(normalized.mean(axis=0)).ravel()
    variance = np.asarray(normalized.power(2).mean(axis=0)).ravel() - mean**2
    order = np.argsort(-variance, kind="stable")[:max_genes]
    return np.sort(eligible[order])


@torch.no_grad()
def predict_nb(model, x, c, batch_size=1024):
    model.cpu().eval()
    predictions = []
    for start in range(0, len(x), batch_size):
        xx = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32)
        cc = torch.as_tensor(c[start:start + batch_size], dtype=torch.float32)
        predictions.append(model.log_rate(xx, cc).clamp(-20, 20).exp().numpy())
    return np.concatenate(predictions, axis=0)


def run_experiment(config_path, output_override=None):
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_path = (config_path.parent / config["data"]).resolve()
    output = Path(output_override).resolve() if output_override else (config_path.parent / config["output"]).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}. Choose a fresh --output directory.")
    data = load_dataset(data_path)
    source = data.provenance["covariate_source"]
    feature_source = data.provenance.get("feature_source", "unknown")
    if feature_source == "rna_oracle":
        raise ValueError("RNA-derived features cannot serve as histology image features")
    if source == "rna_oracle" and not config.get("allow_oracle", False):
        raise ValueError("RNA-derived covariates require allow_oracle: true; results are ORACLE only")
    seed = int(config.get("seed", 7))
    prep = config.get("preprocess", {})
    rng = np.random.default_rng(seed)
    cap = int(prep.get("max_cells_per_donor", 2000))
    if cap < 1:
        raise ValueError("max_cells_per_donor must be positive")
    chosen = []
    for _, indices in data.cells.groupby("donor_id", sort=True).indices.items():
        chosen.extend(rng.choice(indices, size=min(cap, len(indices)), replace=False).tolist())
    chosen = np.sort(chosen)
    cells = data.cells.iloc[chosen].reset_index(drop=True)
    counts = data.counts[chosen]
    x_raw, c_raw = data.features[chosen], data.covariates[chosen]
    split_cfg = config.get("split", {})
    split = make_donor_split(cells, seed=seed, **split_cfg)
    tr, va, te = split["train"], split["val"], split["test"]
    library = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    median_library = float(np.median(library[tr]))
    exposure = library / median_library
    gene_indices = select_genes(counts[tr], exposure[tr], int(prep.get("max_genes", 256)),
                               int(prep.get("min_train_cells", 10)))
    if len(cells) * len(gene_indices) > int(prep.get("max_dense_elements", 40000000)):
        raise MemoryError("Pilot dense count limit exceeded. Reduce max_cells_per_donor or max_genes.")
    y = counts[:, gene_indices].toarray().astype(np.float32)
    genes = [data.genes[i] for i in gene_indices]
    x_mean, x_scale = fit_scaler(x_raw[tr])
    c_mean, c_scale = fit_scaler(c_raw[tr])
    x = ((x_raw - x_mean) / x_scale).astype(np.float32)
    c = ((c_raw - c_mean) / c_scale).astype(np.float32)
    # Linear partialling-out is a transparent baseline, not proof of independence.
    design = np.column_stack([c[tr], np.ones(len(tr))])
    cov_to_image = np.linalg.lstsq(design, x[tr], rcond=1e-6)[0].astype(np.float32)
    x_cond = x - np.column_stack([c, np.ones(len(c))]) @ cov_to_image
    x_cond = x_cond.astype(np.float32)
    output.mkdir(parents=True, exist_ok=True)
    run_mode = "oracle" if source == "rna_oracle" else ("synthetic" if source == "synthetic" else "image_covariates")
    write_json(output / "config.json", config)
    write_json(output / "provenance.json", {
        "mode": run_mode, "dataset": data.provenance, "seed": seed,
        "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
        "dataset_path": str(data_path), "n_cells": len(cells), "n_genes": len(genes),
        "median_train_library": median_library,
        "input_fingerprints": {name: hashlib.sha256((data_path / name).read_bytes()).hexdigest()
                               for name in ("cells.csv", "genes.csv", "provenance.json")},
        "note": "Metadata hashes only; large count/image arrays are not hashed. No clinical validation.",
    })
    split_rows = cells[["cell_id", "donor_id", "slide_id", "cell_type"]].copy()
    split_rows["split"] = ""
    for name, idx in split.items():
        split_rows.loc[idx, "split"] = name
    split_rows.to_csv(output / "split.csv", index=False)
    write_json(output / "split_donors.json", {name: sorted(cells.iloc[idx].donor_id.unique().tolist())
                                             for name, idx in split.items()})
    training = dict(config.get("training", {}))
    training.update(seed=seed, sample_weights=donor_weights(cells.iloc[tr]),
                    val_sample_weights=donor_weights(cells.iloc[va]))
    print(f"Mode={run_mode}; train/val/test cells={len(tr)}/{len(va)}/{len(te)}; genes={len(genes)}", flush=True)
    baseline, history = fit_nb_baseline(c[tr], y[tr], exposure[tr], c[va], y[va], exposure[va], training)
    write_json(output / "history_covariate_nb.json", history)
    ranks = list(dict.fromkeys(int(r) for r in training.get("ranks", [4])))
    if not ranks or any(r < 1 or r > min(x.shape[1], y.shape[1]) for r in ranks):
        raise ValueError("Candidate ranks must be nonempty, positive, and <= min(feature_dim, n_genes)")
    candidates = []
    best, best_score, best_rank = None, float("inf"), None
    for rank in ranks:
        params = dict(training, rank=rank)
        model, history = fit_conditional_nb(x_cond[tr], c[tr], y[tr], exposure[tr],
                                           x_cond[va], c[va], y[va], exposure[va], baseline, params)
        score = model.training_summary["best_val_nll"]
        candidates.append({"rank": rank, "validation_donor_weighted_nb_nll": score})
        write_json(output / f"history_rank_{rank}.json", history)
        print(f"rank={rank}, validation NB NLL={score:.5f}", flush=True)
        if score < best_score:
            best, best_score, best_rank = model, score, rank
    write_json(output / "selection.json", {"candidates": candidates, "selected_rank": best_rank,
                                          "criterion": "validation donor-weighted NB NLL; test never selects"})
    torch.save({"model_config": best.model_config(), "state_dict": best.state_dict(),
                "x_mean": torch.from_numpy(x_mean), "x_scale": torch.from_numpy(x_scale),
                "c_mean": torch.from_numpy(c_mean), "c_scale": torch.from_numpy(c_scale),
                "cov_to_image": torch.from_numpy(cov_to_image), "genes": genes,
                "median_train_library": median_library, "mode": run_mode,
                "covariate_source": source, "feature_source": feature_source,
                "feature_encoder": data.provenance.get("feature_encoder"),
                "selected_rank": best_rank}, output / "conditional_nb.pt")
    # Only the selected conditional candidate is evaluated on test patients.
    rates = {"covariate_nb": predict_nb(baseline, x[te], c[te]),
             "conditional_nb": predict_nb(best, x_cond[te], c[te])}
    log_y = np.log1p(y[tr] / exposure[tr, None])
    for mode in ("covariate", "image", "residual"):
        ridge = RidgePredictor(mode=mode, alpha=float(training.get("ridge_alpha", 10)))
        ridge.fit(x[tr], c[tr], log_y)
        rates[f"{mode}_ridge"] = np.expm1(np.clip(ridge.predict(x[te], c[te]), 0, 20))
        ridge_arrays = {f"{part}_{key}": val for part, comp in ridge.components_.items() for key, val in comp.items()}
        np.savez_compressed(output / f"{mode}_ridge_parameters.npz", **ridge_arrays)
    evaluation = config.get("evaluation", {})
    frames, summary = {}, {"mode": run_mode, "selected_rank": best_rank, "models": {}}
    for name, rate in rates.items():
        frame = donor_metrics(y[te], rate, exposure[te], cells.iloc[te])
        frames[name] = frame
        frame.to_csv(output / f"{name}_per_donor.csv", index=False)
        # A correlation may be undefined for every donor (for example a
        # type-only baseline after within-type centering). Pandas then gives
        # that column object dtype. Preserve its key and report None instead
        # of silently dropping it and crashing during report generation.
        metric_names = ("poisson_deviance", "log_rate_mse", "gene_pcc", "within_type_pcc",
                        "known_cell_type_fraction", "within_type_evaluable_fraction")
        means = {key: pd.to_numeric(frame[key], errors="coerce").mean() for key in metric_names}
        summary["models"][name] = {k: None if pd.isna(v) else float(v) for k, v in means.items()}
    summary["conditional_vs_covariate_nb"] = bootstrap_donor_gain(
        frames["covariate_nb"], frames["conditional_nb"], seed,
        int(evaluation.get("bootstrap_repetitions", 1000)))
    controls = []
    for repetition in range(int(evaluation.get("permutation_repeats", 3))):
        permutation = within_donor_type_permutation(cells.iloc[te], seed + 1000 + repetition)
        # Recombine permuted raw image features with original covariates.
        perm_x = x[te][permutation] - np.column_stack([c[te], np.ones(len(te))]) @ cov_to_image
        rate = predict_nb(best, perm_x.astype(np.float32), c[te])
        frame = donor_metrics(y[te], rate, exposure[te], cells.iloc[te])
        controls.append(float(frame.poisson_deviance.mean()))
    summary["negative_control"] = {"shuffled_image_deviance": controls,
                                   "note": "Shuffle within known donor/type groups; unknown and singleton types unchanged. Not an exact conditional test or p-value."}
    write_json(output / "metrics.json", summary)
    np.savez_compressed(output / "test_predictions.npz", cell_ids=cells.iloc[te].cell_id.to_numpy(dtype=str),
                        gene_ids=np.asarray(genes), **rates)
    lines = ["# Experiment report", "", f"Mode: **{run_mode}**. Selected rank: {best_rank}.", "",
             "All metrics below are donor-macro averages on the locked test split.", "",
             "| Model | Poisson deviance ↓ | Log-rate MSE ↓ | Within-type PCC ↑ |", "|---|---:|---:|---:|"]
    for name, values in summary["models"].items():
        pcc = values["within_type_pcc"]
        pcc_text = "undefined" if pcc is None else f"{pcc:.4f}"
        lines.append(f"| {name} | {values['poisson_deviance']:.4f} | {values['log_rate_mse']:.4f} | {pcc_text} |")
    lines.extend(["", "Synthetic results validate software plumbing only. An oracle run uses RNA-derived covariates.",
                  "Within-type centering is descriptive; it does not fully control neighborhood composition.",
                  "Inspect per-donor CSVs and provenance before interpreting pooled scores."])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved report and checkpoint to {output}", flush=True)
    return summary


def predict_from_checkpoint(checkpoint, feature_path, covariate_path, output, allow_oracle=False):
    """Image + image-derived covariates only. No count file or exposure required."""
    target = Path(output)
    if target.suffix != ".npz":
        raise ValueError("Prediction output must end in .npz")
    if target.exists() or target.with_suffix(".json").exists():
        raise FileExistsError(target)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["mode"] == "oracle" and not allow_oracle:
        raise ValueError("Oracle checkpoint: pass --allow-oracle for a labeled research-only prediction")
    if saved.get("feature_source") == "image" and saved.get("feature_encoder") is not None:
        from .importers import read_feature_metadata
        if read_feature_metadata(feature_path)["feature_encoder"] != saved["feature_encoder"]:
            raise ValueError("Prediction encoder weights/transform differ from training")
    with np.load(feature_path, allow_pickle=False) as f, np.load(covariate_path, allow_pickle=False) as c:
        from .importers import _strings
        ids = np.asarray(_strings(f["cell_ids"], "feature cell_ids"))
        other = np.asarray(_strings(c["cell_ids"], "covariate cell_ids"))
        if len(set(ids)) != len(ids) or len(set(other)) != len(other) or set(ids) != set(other):
            raise ValueError("Feature and covariate cell IDs must be unique and identical sets")
        if c["covariates"].ndim != 2 or len(c["covariates"]) != len(other):
            raise ValueError("Covariate rows must match cell IDs")
        lookup = {key: index for index, key in enumerate(other)}
        x, cov = f["features"].astype(np.float32), c["covariates"][[lookup[i] for i in ids]].astype(np.float32)
    if x.ndim != 2 or cov.ndim != 2 or len(x) != len(ids) or not len(ids):
        raise ValueError("Invalid or empty prediction feature arrays")
    if x.shape[1] != len(saved["x_mean"]) or cov.shape[1] != len(saved["c_mean"]):
        raise ValueError("Feature/covariate dimensions differ from the trained checkpoint")
    if not np.isfinite(x).all() or not np.isfinite(cov).all():
        raise ValueError("Non-finite prediction inputs")
    x = (x - saved["x_mean"].numpy()) / saved["x_scale"].numpy()
    cov = (cov - saved["c_mean"].numpy()) / saved["c_scale"].numpy()
    x = (x - np.column_stack([cov, np.ones(len(cov))]) @ saved["cov_to_image"].numpy()).astype(np.float32)
    model = model_from_config(saved["model_config"])
    model.load_state_dict(saved["state_dict"])
    rate = predict_nb(model, x, cov)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, cell_ids=ids, gene_ids=np.asarray(saved["genes"]), rates=rate)
    write_json(target.with_suffix(".json"), {"mode": saved["mode"], "checkpoint": str(checkpoint),
        "reference_library": saved["median_train_library"],
        "note": "Standardized expected counts at train reference exposure. No test RNA consumed."})
