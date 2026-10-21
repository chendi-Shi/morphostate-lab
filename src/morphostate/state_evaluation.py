"""Independent state-label evaluation and conditional-on-geometry contrast bounds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def _nonempty_identifiers(values, name):
    values = np.asarray(values)
    if (values.ndim != 1 or not len(values)
            or any(not isinstance(value, str) or not value.strip() or value.strip() != value for value in values)):
        raise ValueError(f"{name} must be nonempty trimmed identifier strings")
    return values


def binary_scores(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.shape != labels.shape or not len(labels):
        raise ValueError("Labels and scores must be matching nonempty vectors")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Labels must be binary and scores finite")
    positive = int(labels.sum())
    negative = len(labels) - positive
    if not positive or not negative:
        return {"auroc": None, "average_precision": None, "n_positive": positive, "n_negative": negative}
    auc = (rankdata(scores)[labels == 1].sum() - positive * (positive + 1) / 2) / (positive * negative)
    order = np.argsort(-scores, kind="stable")
    cumulative = np.cumsum(labels[order])
    # Threshold entire ties together; order within a tie cannot change AP.
    endpoints = np.r_[np.flatnonzero(np.diff(scores[order]) != 0), len(scores) - 1]
    recall = cumulative[endpoints] / positive
    precision = cumulative[endpoints] / (endpoints + 1)
    ap = np.sum(np.diff(np.r_[0, recall]) * precision)
    return {"auroc": float(auc), "average_precision": float(ap), "n_positive": positive, "n_negative": negative}


def evaluate_states(predictions, labels_path, programs_path, output, label_source, rate_key="rates"):
    """Use predeclared signed gene programs; do not tune scores on test labels.

    Label CSV requires cell_id,donor_id,cell_type,state,label. Programs JSON maps
    state names to {gene_id: signed_weight}. Only independently supplied labels
    are accepted; provenance claims still require investigator verification.
    """
    if label_source not in {"protein", "if", "expert", "synthetic"}:
        raise ValueError("State labels must come from independent protein/IF/expert evidence or synthetic truth")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    labels = pd.read_csv(labels_path, dtype={"cell_id": str, "donor_id": str, "cell_type": str, "state": str})
    required = {"cell_id", "donor_id", "cell_type", "state", "label"}
    if labels.empty or not required <= set(labels) or labels[list(required)].isna().any().any():
        raise ValueError("State labels require complete cell_id,donor_id,cell_type,state,label")
    for column in ("cell_id", "donor_id", "cell_type", "state"):
        _nonempty_identifiers(labels[column].to_numpy(), column)
    if labels.duplicated(["cell_id", "state"]).any() or not np.isin(labels.label, [0, 1]).all():
        raise ValueError("Labels must be binary and unique per cell/state")
    if (labels.groupby("cell_id")[["donor_id", "cell_type"]].nunique() > 1).any().any():
        raise ValueError("A cell cannot change donor_id or cell_type between state annotations")
    programs = json.loads(Path(programs_path).read_text(encoding="utf-8"))
    if not isinstance(programs, dict) or not programs:
        raise ValueError("Programs must be a nonempty mapping of states to signed gene weights")
    with np.load(predictions, allow_pickle=False) as bundle:
        ids, genes = bundle["cell_ids"].astype(str), bundle["gene_ids"].astype(str)
        rates = np.asarray(bundle[rate_key], dtype=float)
    _nonempty_identifiers(ids, "Prediction cell IDs")
    _nonempty_identifiers(genes, "Prediction gene IDs")
    if len(set(ids)) != len(ids) or len(set(genes)) != len(genes) or rates.shape != (len(ids), len(genes)):
        raise ValueError("Prediction IDs or dimensions are invalid")
    if not np.isfinite(rates).all() or (rates < 0).any():
        raise ValueError("Rates must be finite and nonnegative")
    if not set(labels.cell_id) <= set(ids):
        raise ValueError("Independent label cells missing from predictions")
    index, gene_index = {x: i for i, x in enumerate(ids)}, {x: i for i, x in enumerate(genes)}
    scores = labels.copy()
    scores["prediction_score"] = np.nan
    for state, rows in scores.groupby("state").groups.items():
        program = programs.get(state)
        if not isinstance(program, dict) or not program or not set(program) <= set(genes):
            raise ValueError(f"State {state} needs a nonempty predeclared program of measured genes")
        try:
            weights = np.array(list(program.values()), dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Program weights must be numeric") from exc
        if not np.isfinite(weights).all() or not np.any(weights):
            raise ValueError("Program weights must be finite and nonzero")
        positions = [index[x] for x in scores.loc[rows, "cell_id"]]
        columns = [gene_index[x] for x in program]
        scores.loc[rows, "prediction_score"] = np.log1p(rates[np.ix_(positions, columns)]) @ weights / np.abs(weights).sum()
    results = []
    for (donor, cell_type, state), frame in scores.groupby(["donor_id", "cell_type", "state"], sort=True):
        row = {"donor_id": donor, "cell_type": cell_type, "state": state,
               **binary_scores(frame.label.to_numpy(), frame.prediction_score.to_numpy())}
        row["mean_score_difference"] = (float(frame.loc[frame.label == 1, "prediction_score"].mean() -
                                               frame.loc[frame.label == 0, "prediction_score"].mean())
                                        if frame.label.nunique() == 2 else None)
        results.append(row)
    result = pd.DataFrame(results)
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / "cell_scores.csv", index=False)
    result.to_csv(output / "state_per_donor_type.csv", index=False)
    groups = result.groupby(["cell_type", "state"])
    macro = groups[["auroc", "average_precision", "mean_score_difference"]].mean()
    macro["n_donors"] = groups.donor_id.nunique()
    macro["n_evaluable_donors"] = groups.auroc.count()
    macro["n_positive"] = groups.n_positive.sum()
    macro["n_negative"] = groups.n_negative.sum()
    macro.to_csv(output / "state_macro.csv")
    (output / "provenance.json").write_text(json.dumps({"label_source": label_source, "programs": programs,
        "program_sha256": hashlib.sha256(Path(programs_path).read_bytes()).hexdigest(),
        "prediction_cells": len(ids), "independently_labeled_cells": labels.cell_id.nunique(),
        "labeled_cell_fraction": float(labels.cell_id.nunique() / len(ids)),
        "independent_donors": labels.donor_id.nunique(),
        "note": "Predeclared log-rate gene program; no score/threshold fitting. Labels are not per-gene RNA truth. Independently audit masks and provenance."}, indent=2), encoding="utf-8")
    return result


def evaluate_contrasts(graphs_path, coefficients_path, output):
    """Sharp linear bounds under supplied block budgets, not confidence intervals."""
    from .geometry import load_graphs
    target = Path(output)
    if target.suffix != ".csv" or target.exists() or target.with_suffix(".json").exists():
        raise ValueError("Contrast output must be a new .csv file")
    blocks = load_graphs(graphs_path)
    table = pd.read_csv(coefficients_path, dtype={"contrast": str, "cell_id": str, "gene_id": str})
    required = {"contrast", "cell_id", "gene_id", "coefficient"}
    if table.empty or not required <= set(table) or table[list(required)].isna().any().any():
        raise ValueError("Contrast CSV requires contrast,cell_id,gene_id,coefficient")
    for column in ("contrast", "cell_id", "gene_id"):
        _nonempty_identifiers(table[column].to_numpy(), column)
    table["coefficient"] = pd.to_numeric(table["coefficient"], errors="raise")
    if not np.isfinite(table.coefficient).all():
        raise ValueError("Contrast coefficients must be finite")
    all_cells = {cell for block in blocks for cell in block.cell_ids}
    if not set(table.cell_id) <= all_cells:
        raise ValueError("Contrast refers to unknown cells")
    results = []
    for name, frame in table.groupby("contrast", sort=True):
        lower, upper, fixed = 0., 0., 0.
        used, donors, used_blocks = 0, set(), 0
        for block in blocks:
            local = frame[frame.cell_id.isin(block.cell_ids)]
            if not len(local):
                continue
            cells = {x: i for i, x in enumerate(block.cell_ids)}
            genes = {x: i for i, x in enumerate(block.gene_ids)}
            if not set(local.gene_id) <= set(genes):
                raise ValueError("Contrast includes unmeasured genes")
            coefficients = np.zeros((block.label_set.n_cells, block.label_set.n_genes))
            for row in local.itertuples():
                coefficients[cells[row.cell_id], genes[row.gene_id]] += row.coefficient
            lo, hi = block.label_set.contrast_bounds(coefficients)
            lower, upper = lower + lo, upper + hi
            labels = block.label_set.fixed_labels(block.fixed_owner)
            fixed += float(np.sum(coefficients * labels))
            used += len(local)
            donors.add(block.donor_id)
            used_blocks += 1
        if used != len(frame):
            raise ValueError("Cell IDs overlap graph blocks or contrast was not fully evaluated")
        tolerance = 1e-7 * (1 + abs(lower) + abs(upper) + abs(fixed))
        if lower > upper + tolerance or not lower - tolerance <= fixed <= upper + tolerance:
            raise RuntimeError("Computed contrast bounds do not contain their feasible fixed assignment")
        results.append({"contrast": name, "lower": lower, "upper": upper, "width": max(0., upper - lower),
                        "fixed_assignment_value": fixed, "n_donors": len(donors), "n_blocks": used_blocks,
                        "donor_id": next(iter(donors)) if len(donors) == 1 else "",
                        "numerical_tolerance": tolerance,
                        "sign": "positive" if lower > tolerance else "negative" if upper < -tolerance else "unresolved"})
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(target, index=False)
    target.with_suffix(".json").write_text(json.dumps({"note": "Bounds conditional on owner sets and budgets containing reality. Not sampling confidence intervals or a proof of recovered biological state."}), encoding="utf-8")
    return results
