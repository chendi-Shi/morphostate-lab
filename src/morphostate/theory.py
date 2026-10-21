"""Numerical checks of the conditional finite-sample statements in docs/theory.md."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from scipy.special import kl_div

from .geometry import load_graphs, subset_genes


def _check_primal_assignment(label_set, allocation):
    """Validate a numerical witness independently of its objective value."""
    allocation = np.asarray(allocation, dtype=float)
    if allocation.shape != (label_set.candidate_owners.nnz,) or not np.isfinite(allocation).all():
        raise RuntimeError("Certificate has invalid primal allocation dimensions or values")
    mass_tolerance = 1e-7 * (1 + np.abs(label_set.mass))
    mass_residual = np.abs(label_set._a_eq @ allocation - label_set.mass)
    nonnegative_tolerance = 1e-7 * (1 + np.max(label_set.mass, initial=0))
    if (allocation < -nonnegative_tolerance).any() or (mass_residual > mass_tolerance).any():
        raise RuntimeError("Certificate primal witness violates nonnegativity or molecular mass conservation")
    budget_violation = 0.0
    if label_set._a_ub is not None:
        excess = label_set._a_ub @ allocation - label_set._b_ub
        budget_violation = float(np.max(excess, initial=0))
        if (excess > 1e-7 * (1 + np.abs(label_set._b_ub))).any():
            raise RuntimeError("Certificate primal witness violates a geometry/background budget")
    return {"maximum_mass_residual": float(np.max(mass_residual, initial=0)),
            "maximum_budget_violation": budget_violation,
            "minimum_allocated_mass": float(np.min(allocation))}


def support_dual_certificate(label_set, coefficients):
    """Primal/dual numerical witnesses, not an exact-arithmetic proof."""
    values = label_set._coefficients(coefficients)
    objective = values[label_set.candidate_owners.indices, label_set.gene_index[label_set._edge_rows]]
    result = linprog(-objective, A_eq=label_set._a_eq, b_eq=label_set.mass,
                     A_ub=label_set._a_ub, b_ub=label_set._b_ub, bounds=(0, None), method="highs")
    if not result.success:
        raise RuntimeError(f"Certificate LP failed: {result.message}")
    primal_check = _check_primal_assignment(label_set, result.x)
    equality_dual = -np.asarray(result.eqlin.marginals)
    budget_dual = -np.asarray(result.ineqlin.marginals)
    slack = label_set._a_eq.T @ equality_dual - objective
    dual_value = float(label_set.mass @ equality_dual)
    if label_set._a_ub is not None:
        slack += label_set._a_ub.T @ budget_dual
        dual_value += float(label_set._b_ub @ budget_dual)
    primal_value = float(objective @ result.x)
    if not np.isfinite([primal_value, dual_value]).all() or not np.isfinite(slack).all():
        raise RuntimeError("Numerical primal/dual certificate contains non-finite values")
    tolerance = 1e-7 * (1 + abs(primal_value) + abs(dual_value))
    gap = dual_value - primal_value
    if (abs(gap) > tolerance or np.min(slack, initial=0) < -tolerance
            or np.min(budget_dual, initial=0) < -tolerance):
        raise RuntimeError("Numerical primal/dual certificate fails its tolerance")
    return {"primal_value": primal_value, "dual_value": dual_value, "duality_gap": gap,
            "minimum_dual_slack": float(np.min(slack)), "tolerance": tolerance,
            "equality_dual": equality_dual.tolist(), "budget_dual": budget_dual.tolist(),
            "primal_feasibility": primal_check}


def baseline_membership(label_set, baseline, projection=None):
    """Minimize L-infinity distance to real-cell labels, letting background vary.

    With ``projection`` of shape (n_real*n_genes, k), instead minimize the
    L-infinity norm of projection.T @ (Y_real.ravel() - baseline.ravel()).
    Flattening is cell-major, then gene. Columns may be signed; loss weights or
    normalization must be incorporated by the caller. This tests moment-image
    membership, NOT full membership or membership of a neural network class.
    A projected match can occur while the full baseline is infeasible.

    Positive residual excludes a baseline numerically. Near-zero residual only
    establishes membership up to solver tolerance, not an exact logical test.
    """
    baseline = np.asarray(baseline, dtype=float)
    if label_set.n_cells < 2 or label_set.background_index != label_set.n_cells - 1:
        raise ValueError("Diagnostics require an explicit last background column")
    if baseline.shape != (label_set.n_cells - 1, label_set.n_genes) or not np.isfinite(baseline).all() or (baseline < 0).any():
        raise ValueError("Baseline must match finite nonnegative real-cell counts")
    n_edges = label_set.candidate_owners.nnz
    n_labels = baseline.size
    real = label_set.candidate_owners.indices != label_set.background_index
    label_rows = label_set.candidate_owners.indices[real] * label_set.n_genes + label_set.gene_index[label_set._edge_rows[real]]
    mapping = sparse.coo_matrix((np.ones(real.sum()), (label_rows, np.flatnonzero(real))),
                                 shape=(n_labels, n_edges)).tocsr()
    target, objective_mapping = baseline.ravel(), mapping
    if projection is not None:
        projection = np.asarray(projection, dtype=float)
        if (projection.ndim != 2 or projection.shape[0] != n_labels or projection.shape[1] < 1
                or not np.isfinite(projection).all()):
            raise ValueError("projection must be a finite (n_real*n_genes, positive k) matrix")
        objective_mapping = sparse.csr_matrix(projection.T) @ mapping
        target = projection.T @ target
        if not np.isfinite(target).all() or not np.isfinite(objective_mapping.data).all():
            raise ValueError("Projected membership inputs overflowed")
    dimension = len(target)
    minus_one = sparse.csr_matrix(-np.ones((dimension, 1)))
    constraints = [sparse.hstack([objective_mapping, minus_one]), sparse.hstack([-objective_mapping, minus_one])]
    rhs = [target, -target]
    if label_set._a_ub is not None:
        constraints.append(sparse.hstack([label_set._a_ub, sparse.csr_matrix((len(label_set._b_ub), 1))]))
        rhs.append(label_set._b_ub)
    cost = np.r_[np.zeros(n_edges), 1.]
    result = linprog(cost, A_ub=sparse.vstack(constraints).tocsr(), b_ub=np.concatenate(rhs),
        A_eq=sparse.hstack([label_set._a_eq, sparse.csr_matrix((label_set.n_observations, 1))]).tocsr(),
        b_eq=label_set.mass, bounds=(0, None), method="highs")
    if not result.success:
        raise RuntimeError(f"Membership LP failed: {result.message}")
    primal_check = _check_primal_assignment(label_set, result.x[:-1])
    tolerance = float(1e-7 * (1 + np.max(np.abs(target))))
    epigraph = float(result.x[-1])
    if not np.isfinite(epigraph) or epigraph < -tolerance:
        raise RuntimeError("Membership LP returned an invalid residual variable")
    residual = max(0.0, epigraph)
    witness_counts = np.asarray(mapping @ result.x[:-1]).reshape(baseline.shape)
    witness_residual = float(np.max(np.abs(objective_mapping @ result.x[:-1] - target)))
    if abs(witness_residual - residual) > 10 * tolerance:
        raise RuntimeError("Membership residual and witness disagree")
    return {"linf_residual": residual, "numerical_tolerance": tolerance,
            "baseline_feasible_within_tolerance": witness_residual <= tolerance,
            "witness_linf_residual": witness_residual,
            "projection_dimension": dimension,
            "distance_space": "full_real_cell_counts" if projection is None else "projected_contrasts",
            "witness_full_count_linf_residual": float(np.max(np.abs(witness_counts - baseline))),
            "primal_feasibility": primal_check,
            "witness_counts": witness_counts}


def diagnose_means(label_set, mean, baseline, max_bound_queries=10000):
    mean, baseline = np.asarray(mean, dtype=float), np.asarray(baseline, dtype=float)
    expected = (label_set.n_cells - 1, label_set.n_genes)
    if mean.shape != expected or baseline.shape != expected or not np.isfinite(mean).all() or not np.isfinite(baseline).all():
        raise ValueError("Positive finite mean and baseline must match the real-cell label matrix")
    if (mean <= 0).any() or (baseline <= 0).any():
        raise ValueError("Relative Poisson theory requires strictly positive candidate and baseline means")
    coefficients = np.vstack([np.log(baseline) - np.log(mean), np.zeros((1, label_set.n_genes))])
    constant = float((mean - baseline).sum())
    primal, adversarial = label_set.maximize(coefficients)
    dual = support_dual_certificate(label_set, coefficients)
    if not np.isclose(primal, dual["primal_value"], atol=dual["tolerance"], rtol=1e-7):
        raise RuntimeError("Training support and independent certificate LP disagree")
    membership = baseline_membership(label_set, baseline)
    lower, upper = label_set.count_bounds(exact=True, max_queries=max_bound_queries)
    box_support = float(np.sum(coefficients * np.where(coefficients >= 0, upper, lower)))
    count = mean.size
    kl = float(np.sum(kl_div(baseline, mean)) / count)
    risk_tolerance = (dual["tolerance"] + 1e-7 * (1 + abs(box_support))) / count
    witness_risk = float((constant + np.sum(coefficients[:-1] * membership["witness_counts"])) / count)
    # Unlike a rounded membership flag, this remains valid for an infeasible
    # baseline: evaluate the actual feasible closest-label witness, and bound
    # its difference from KL by Holder's inequality. Use actual witness error,
    # not just the LP's epigraph variable, to include its numerical residual.
    correction = membership["witness_linf_residual"] * float(np.abs(coefficients[:-1]).sum()) / count
    result = {"joint_relative_risk": (constant + primal) / count,
              "marginal_box_relative_risk": (constant + box_support) / count,
              "box_minus_joint": (box_support - primal) / count,
              "poisson_kl_baseline_to_candidate": kl,
              "baseline_witness_relative_lower_bound": witness_risk,
              "membership_distance_risk_correction": correction,
              "poisson_kl_minus_distance_correction": kl - correction,
              "risk_numerical_tolerance": risk_tolerance,
              "membership": {k: v for k, v in membership.items() if k != "witness_counts"},
              "support_dual": dual,
              "interpretation": "All risks are fixed-offset, per-real-cell-per-gene empirical averages conditional on the declared feasible set. No patient population or true-label coverage guarantee."}
    if any(not np.isfinite(value) for value in result.values() if isinstance(value, (int, float))):
        raise FloatingPointError("Risk arithmetic is non-finite; inspect numerical scale and block size")
    if result["box_minus_joint"] < -risk_tolerance:
        raise RuntimeError("Joint risk exceeds exact marginal-box risk")
    if result["joint_relative_risk"] + risk_tolerance < witness_risk:
        raise RuntimeError("Worst-case objective is below its feasible baseline-approximation witness")
    if witness_risk + risk_tolerance < kl - correction:
        raise RuntimeError("Feasible-witness risk violates the KL-minus-distance correction")
    return result, adversarial, membership["witness_counts"]


def diagnose_predictions(data_path, graphs_path, predictions_path, output, reference_library,
                         model_key="joint_relative", baseline_key="covariate_nb", max_bound_queries=10000):
    from .data import load_dataset
    from .weak_training import _validate_graph_alignment
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    if not np.isfinite(reference_library) or reference_library <= 0:
        raise ValueError("Use the positive training reference library recorded with these predictions")
    data, blocks = load_dataset(data_path), load_graphs(graphs_path)
    _validate_graph_alignment(blocks, data)
    with np.load(predictions_path, allow_pickle=False) as bundle:
        ids, genes = bundle["cell_ids"].astype(str), bundle["gene_ids"].astype(str)
        model, baseline = np.asarray(bundle[model_key], dtype=float), np.asarray(bundle[baseline_key], dtype=float)
    if ids.ndim != 1 or genes.ndim != 1:
        raise ValueError("Prediction cell/gene identifiers must be one-dimensional")
    if len(set(ids)) != len(ids) or len(set(genes)) != len(genes) or not len(ids) or not len(genes):
        raise ValueError("Prediction identifiers must be nonempty unique sets")
    if model.shape != (len(ids), len(genes)) or baseline.shape != model.shape:
        raise ValueError("Prediction arrays do not match their axes")
    if not set(ids) <= set(data.cells.cell_id) or not set(genes) <= set(data.genes):
        raise ValueError("Prediction cells/genes must exist in prepared data")
    index = {cell: i for i, cell in enumerate(ids)}
    data_index = {cell: i for i, cell in enumerate(data.cells.cell_id)}
    library = np.asarray(data.counts.sum(axis=1)).ravel()
    results, witnesses, covered = [], {}, set()
    for block in blocks:
        overlap = set(block.cell_ids) & set(ids)
        if not overlap:
            continue
        if overlap != set(block.cell_ids):
            raise ValueError("Diagnostics require complete graph blocks; do not drop coupled neighboring cells")
        selected = subset_genes(block, genes.tolist())
        positions = [index[cell] for cell in block.cell_ids]
        exposure = library[[data_index[cell] for cell in block.cell_ids]] / reference_library
        diagnostic, adversarial, witness = diagnose_means(selected.label_set,
            model[positions] * exposure[:, None], baseline[positions] * exposure[:, None], max_bound_queries)
        diagnostic.update(block_id=block.block_id, donor_id=block.donor_id, n_cells=len(block.cell_ids),
                          n_genes=len(genes),
                          gene_subset_policy=selected.metadata.get("gene_subset_policy"))
        results.append(diagnostic)
        prefix = f"block{len(results)-1:05d}"
        witnesses[f"{prefix}_cell_ids"] = np.asarray(block.cell_ids)
        witnesses[f"{prefix}_adversarial_labels"] = adversarial
        witnesses[f"{prefix}_closest_baseline_labels"] = witness
        covered.update(overlap)
    if covered != set(ids):
        raise ValueError("Predictions are not fully covered by the graphs")
    output.mkdir(parents=True, exist_ok=True)
    (output / "diagnostics.json").write_text(json.dumps({"model": model_key, "baseline": baseline_key,
        "reference_library": reference_library, "blocks": results,
        "note": "Numerical diagnostics only. Exact membership implies the collapse statement; tolerance-level membership is approximate, with an explicit witness-distance correction. Full infeasibility does not guarantee escape in a restricted model class, useful neural optimization or biological identifiability."}, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in r.items() if isinstance(v, (int, float, bool)) or k in {"block_id", "donor_id"}}
                  for r in results]).to_csv(output / "block_risks.csv", index=False)
    np.savez_compressed(output / "witnesses.npz", gene_ids=genes, **witnesses)
    return results
