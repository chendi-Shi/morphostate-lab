"""Prespecified width thresholds for contrast coverage/risk, without tuning on truth."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def evaluate_bound_risk(bounds_path, truth_path, output, thresholds, truth_source):
    """Truth CSV: contrast,donor_id,truth (same linear contrast and count units).

    Each contrast must belong to one donor. Selection uses only a sign-resolved
    bound and a prespecified maximum width, never its truth/error. Ground truth
    must be simulation truth or an independently measured numeric contrast.
    """
    if truth_source not in {"synthetic", "independent_assay"}:
        raise ValueError("Truth must be synthetic or an independently measured same-unit contrast")
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim != 1 or not len(thresholds) or not np.isfinite(thresholds).all() or (thresholds < 0).any():
        raise ValueError("Prespecify nonnegative finite width thresholds")
    if len(np.unique(thresholds)) != len(thresholds):
        raise ValueError("Duplicate thresholds")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    bounds = pd.read_csv(bounds_path, dtype={"contrast": str, "donor_id": str})
    truth = pd.read_csv(truth_path, dtype={"contrast": str, "donor_id": str})
    if not {"contrast", "donor_id", "lower", "upper", "fixed_assignment_value", "n_donors"} <= set(bounds):
        raise ValueError("Use the output of contrast-bounds with per-donor contrasts")
    if not {"contrast", "donor_id", "truth"} <= set(truth) or truth.empty:
        raise ValueError("Truth CSV needs contrast,donor_id,truth")
    for frame, columns in ((bounds, ["contrast", "donor_id"]), (truth, ["contrast", "donor_id"])):
        if frame[columns].isna().any().any() or any((frame[c].str.strip() != frame[c]).any()
                or (frame[c] == "").any() for c in columns):
            raise ValueError("Identifiers must be nonempty trimmed strings")
    if bounds.contrast.duplicated().any() or truth.contrast.duplicated().any():
        raise ValueError("Contrast IDs must be unique")
    if set(bounds.contrast) != set(truth.contrast) or not (bounds.n_donors == 1).all():
        raise ValueError("Truth must cover every bound, with one donor per contrast")
    table = bounds.merge(truth[["contrast", "donor_id", "truth"]], on="contrast", validate="one_to_one",
                         suffixes=("", "_truth"))
    if not table.donor_id.eq(table.donor_id_truth).all():
        raise ValueError("Truth donor_id must match the graph-derived donor_id of every contrast")
    table = table.drop(columns="donor_id_truth")
    for column in ("lower", "upper", "fixed_assignment_value", "truth"):
        table[column] = pd.to_numeric(table[column], errors="raise")
    values = table[["lower", "upper", "fixed_assignment_value", "truth"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (table.lower > table.upper).any():
        raise ValueError("Invalid finite contrast bounds/truth")
    if (table.truth == 0).any():
        raise ValueError("Zero-truth contrasts have no binary sign; define a separate null-detection protocol")
    tolerance = 1e-7 * (1 + np.abs(table.lower) + np.abs(table.upper) + np.abs(table.fixed_assignment_value))
    if ((table.fixed_assignment_value < table.lower - tolerance)
            | (table.fixed_assignment_value > table.upper + tolerance)).any():
        raise ValueError("Fixed assignment must lie inside its feasible contrast bounds")
    table["width"] = table.upper - table.lower
    table["sign_resolved"] = (table.lower > tolerance) | (table.upper < -tolerance)
    table["bound_sign"] = np.where(table.lower > tolerance, 1, np.where(table.upper < -tolerance, -1, 0))
    table["sign_error"] = np.where(table.sign_resolved,
                                   (table.bound_sign != np.sign(table.truth)).astype(float), np.nan)
    table["fixed_sign_error"] = (np.sign(table.fixed_assignment_value) != np.sign(table.truth)).astype(float)
    table["truth_in_bounds"] = (table.truth >= table.lower - tolerance) & (table.truth <= table.upper + tolerance)
    rows = []
    for threshold in sorted(thresholds):
        selected = table.sign_resolved & (table.width <= threshold)
        for donor, frame in table.groupby("donor_id", sort=True):
            accepted = frame[selected.loc[frame.index]]
            row = {"threshold": float(threshold), "donor_id": donor, "total": len(frame), "accepted": len(accepted),
                   "coverage": len(accepted) / len(frame),
                   "risk": float(accepted.sign_error.mean()) if len(accepted) else None,
                   "accepted_contrast_fixed_risk": float(accepted.fixed_sign_error.mean()) if len(accepted) else None,
                   "all_contrast_fixed_risk": float(frame.fixed_sign_error.mean()),
                   "truth_containment": float(frame.truth_in_bounds.mean())}
            rows.append(row)
    per_donor = pd.DataFrame(rows)
    grouped = per_donor.groupby("threshold", sort=True)
    curve = grouped[["coverage", "risk", "accepted_contrast_fixed_risk", "all_contrast_fixed_risk", "truth_containment"]].mean()
    curve["n_donors"] = grouped.donor_id.nunique()
    curve["n_donors_with_accepted_contrasts"] = grouped.risk.count()
    curve["accepted_contrasts"] = grouped.accepted.sum()
    curve["total_contrasts"] = grouped.total.sum()
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "all_contrasts.csv", index=False)
    per_donor.to_csv(output / "coverage_risk_per_donor.csv", index=False)
    curve.to_csv(output / "coverage_risk.csv")
    (output / "protocol.json").write_text(json.dumps({"truth_source": truth_source,
        "thresholds": thresholds.tolist(), "selection": "sign-resolved bound and width <= prespecified threshold",
        "bounds_sha256": hashlib.sha256(Path(bounds_path).read_bytes()).hexdigest(),
        "truth_sha256": hashlib.sha256(Path(truth_path).read_bytes()).hexdigest(),
        "thresholds_sha256": hashlib.sha256(json.dumps(thresholds.tolist()).encode()).hexdigest(),
        "risk_denominator": "Only donors with at least one accepted contrast; within each such donor, average over its accepted contrasts",
        "coverage_denominator": "All supplied donors, including donors with zero accepted contrasts",
        "note": "Descriptive donor-macro risk among donors with accepted contrasts. Read coverage and accepted-donor count together. Code never optimizes thresholds against outcomes; investigators must establish their prior declaration. Feasible-set bounds are not statistical confidence intervals."}, indent=2), encoding="utf-8")
    return curve
