"""Metrics keep the patient, not the cell, as the replication unit."""

import numpy as np
import pandas as pd

UNKNOWN_CELL_TYPES = {"", "unknown", "unassigned", "unannotated", "na", "n/a", "none", "nan", "<na>", "missing"}


def known_cell_type_mask(labels):
    """Recognize missing annotations without interpreting a named type as RNA input."""
    return np.asarray([str(label).strip().lower() not in UNKNOWN_CELL_TYPES for label in labels])


def poisson_deviance(y, mu):
    """Per-cell mean Poisson deviance; a common metric across all models.

    This is not an assertion that the counts are Poisson. All models use the
    same observed evaluation exposure; deployment predicts standardized rates.
    """
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    if y.ndim != 2 or y.shape != mu.shape or np.any(y < 0) or not np.isfinite(y).all() or not np.isfinite(mu).all() or np.any(mu < 0):
        raise ValueError("Invalid shapes, counts, or predictions for deviance")
    mu = np.maximum(mu, 1e-10)
    term = np.zeros_like(y)
    positive = y > 0
    term[positive] = y[positive] * np.log(y[positive] / mu[positive])
    return np.maximum(2 * (term - y + mu), 0).mean(axis=1)


def mean_gene_pcc(y, pred):
    a, b = np.asarray(y, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    # Center after conversion: float32 means can manufacture tiny variance in
    # an exactly constant predictor and turn an undefined PCC into a number.
    a, b = a - a.mean(axis=0), b - b.mean(axis=0)
    denom = np.sqrt(np.sum(a * a, axis=0) * np.sum(b * b, axis=0))
    usable = denom > 1e-12
    if not np.any(usable):
        return None
    return float(np.mean(np.sum(a[:, usable] * b[:, usable], axis=0) / denom[usable]))


def group_center(values, groups):
    """Evaluation-only centering; never supplies labels to the predictor."""
    result = np.asarray(values, dtype=np.float64).copy()
    for group in np.unique(groups):
        mask = groups == group
        result[mask] -= result[mask].mean(axis=0)
    return result


def donor_metrics(counts, rates, exposures, cells):
    """Return donor metrics, excluding missing type labels from state evaluation.

    ``within_type_pcc`` pools cells after centering within each donor/type; it
    is not an equally weighted average over types. Only known types represented
    by at least two cells in that donor contribute. Neither centering nor these
    evaluation annotations are fed to the predictor.
    """
    observed = np.log1p(counts / exposures[:, None])
    predicted = np.log1p(np.maximum(rates, 0))
    deviance = poisson_deviance(counts, rates * exposures[:, None])
    rows = []
    donors = cells.donor_id.astype(str).to_numpy()
    types = cells.cell_type.astype(str).to_numpy()
    for donor in sorted(np.unique(donors)):
        mask = donors == donor
        donor_types = types[mask]
        known = known_cell_type_mask(donor_types)
        frequencies = pd.Series(donor_types[known]).value_counts()
        eligible_types = frequencies.index[frequencies >= 2].to_numpy()
        eligible = known & np.isin(donor_types, eligible_types)
        within_pcc = None
        if eligible.sum() >= 2:
            within_y = group_center(observed[mask][eligible], donor_types[eligible])
            within_p = group_center(predicted[mask][eligible], donor_types[eligible])
            within_pcc = mean_gene_pcc(within_y, within_p)
        rows.append({
            "donor_id": donor,
            "n_cells": int(mask.sum()),
            "known_cell_type_fraction": float(known.mean()),
            "within_type_evaluable_fraction": float(eligible.mean()),
            "poisson_deviance": float(deviance[mask].mean()),
            "log_rate_mse": float(np.mean((observed[mask] - predicted[mask]) ** 2)),
            "gene_pcc": mean_gene_pcc(observed[mask], predicted[mask]),
            "within_type_pcc": within_pcc,
        })
    return pd.DataFrame(rows)


def bootstrap_donor_gain(baseline, candidate, seed=7, repetitions=1000):
    if repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    merged = baseline[["donor_id", "poisson_deviance"]].merge(
        candidate[["donor_id", "poisson_deviance"]], on="donor_id", validate="one_to_one",
        suffixes=("_baseline", "_candidate"))
    if len(merged) != len(baseline) or len(merged) != len(candidate):
        raise ValueError("Paired donor bootstrap requires identical donors")
    gains = (merged.poisson_deviance_baseline - merged.poisson_deviance_candidate).to_numpy()
    if not len(gains):
        raise ValueError("No patients to evaluate")
    output = {"mean_deviance_gain": float(gains.mean()), "n_donors": len(gains),
              "positive_donors": int(np.sum(gains > 0)),
              "warning": "Few donors: intervals are descriptive, not population guarantees." if len(gains) < 10 else None}
    if len(gains) >= 2:
        rng = np.random.default_rng(seed)
        samples = rng.choice(gains, size=(repetitions, len(gains)), replace=True).mean(axis=1)
        output["bootstrap_95_percent_interval"] = np.quantile(samples, [0.025, 0.975]).tolist()
    else:
        output["bootstrap_95_percent_interval"] = None
    return output


def within_donor_type_permutation(cells, seed):
    """Nuisance-preserving negative control, NOT an exact conditional test.

    Neighborhood composition and spatial autocorrelation need not be preserved.
    Cells with unknown/unassigned type labels remain unchanged; they cannot
    support a within-type negative control. ``known_cell_type_fraction`` in
    the donor metric table reports annotation coverage. Singleton types also
    remain unchanged naturally.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(cells))
    frame = cells.reset_index(drop=True)
    known = known_cell_type_mask(frame["cell_type"])
    for rows in frame.loc[known].groupby(["donor_id", "cell_type"], sort=True).groups.values():
        rows = np.asarray(rows, dtype=int)
        indices[rows] = rng.permutation(rows)
    return indices

