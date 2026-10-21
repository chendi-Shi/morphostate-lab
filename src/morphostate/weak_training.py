"""Executable, budgeted ambiguity-supervised pilot experiments.

The inner optimizer acts on divisible molecular mass. Its maximizer is detached
before backpropagation (a Danskin subgradient); we do not differentiate through
HiGHS. All heads share frozen image inputs, a frozen covariate baseline, the same
validation criterion and a locked donor split. These are research baselines,
not implementations of third-party author systems or claims of novel theory.
"""

from __future__ import annotations

import hashlib
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .data import load_dataset
from .experiment import donor_weights, fit_scaler, predict_nb, select_genes, write_json
from .metrics import bootstrap_donor_gain, donor_metrics, poisson_deviance
from .models import ConditionalNBModel, model_from_config
from .splits import make_donor_split
from .training import fit_nb_baseline, set_seed

METHODS = (
    "fixed", "joint_relative", "joint_absolute", "interval_relative",
    "optimistic_relative", "core_only", "random_assignment", "ensemble",
    "aggregate", "no_geometry", "no_background_budget",
)


def poisson_assignment_loss(log_mu, baseline_log_mu, label_set, method, *,
                            fixed_labels=None, bounds=None, rng=None):
    """Return differentiable mean loss and optimizer timing for a REAL-cell block.

    The last graph column is background and has zero loss coefficient. Relative
    Poisson NLL cancels the count-factorial terms exactly. ``joint_absolute``
    instead optimizes the Poisson *quasi-loss* mu-y log(mu), not full worst-case
    NLL: its omitted factorial depends on the uncertain label. It is deliberately
    named and reported as that convention rather than claiming an exact NLL.
    """
    if log_mu.ndim != 2 or baseline_log_mu.shape != log_mu.shape:
        raise ValueError("Log means must be matching real-cell by gene matrices")
    if tuple(log_mu.shape) != (label_set.n_cells - 1, label_set.n_genes):
        raise ValueError("Graph must contain the real cells followed by one background column")
    if not torch.isfinite(log_mu).all() or not torch.isfinite(baseline_log_mu).all():
        raise ValueError("Log means must be finite")
    relative = method in {"joint_relative", "interval_relative", "optimistic_relative",
                          "no_geometry", "no_background_budget"}
    coefficient = baseline_log_mu - log_mu if relative else -log_mu
    constant = log_mu.exp() - baseline_log_mu.exp() if relative else log_mu.exp()
    coefficient_array = np.vstack([coefficient.detach().cpu().numpy(),
                                   np.zeros((1, label_set.n_genes))])
    started = time.perf_counter()
    if method in {"fixed", "core_only", "ensemble"}:
        if fixed_labels is None:
            raise ValueError(f"{method} requires explicit fixed labels")
        labels = np.asarray(fixed_labels, dtype=np.float64)
    elif method == "interval_relative":
        if bounds is None:
            raise ValueError("interval_relative requires precomputed exact marginal bounds")
        lower, upper = bounds
        labels = np.where(coefficient_array > 0, upper, lower)
    elif method == "optimistic_relative":
        labels = label_set.maximize(-coefficient_array)[1]
    elif method == "random_assignment":
        if rng is None:
            raise ValueError("random_assignment requires a seeded generator")
        # A random LP objective produces a feasible vertex, NOT a uniform draw
        # from the ownership polytope or independent molecule assignments.
        labels = label_set.maximize(rng.normal(size=coefficient_array.shape))[1]
    elif method in {"joint_relative", "joint_absolute", "no_geometry", "no_background_budget"}:
        labels = label_set.maximize(coefficient_array)[1]
    else:
        raise ValueError(f"Unknown assignment objective: {method}")
    solve_seconds = time.perf_counter() - started
    if labels.shape != coefficient_array.shape or not np.isfinite(labels).all() or (labels < -1e-7).any():
        raise ValueError("Invalid optimizer/fixed-label output")
    target = torch.as_tensor(labels[:-1], dtype=log_mu.dtype, device=log_mu.device)
    return (constant + coefficient * target).mean(), solve_seconds


def _core_labels(label_set):
    """Counts of structurally single-owner observations, not zero-filled full RNA."""
    owners_per_row = np.diff(label_set.candidate_owners.indptr)
    single = owners_per_row == 1
    owners = label_set.candidate_owners.indices[label_set.candidate_owners.indptr[:-1]]
    keep = single & (owners < label_set.n_cells - 1)
    labels = np.zeros((label_set.n_cells, label_set.n_genes), dtype=np.float64)
    np.add.at(labels, (owners[keep], label_set.gene_index[keep]), label_set.mass[keep])
    return labels


def _aggregate_groups(label_set, background_policy="fixed_observed"):
    """Real-owner support components, with explicit treatment of background.

    Under ``fixed_observed`` we aggregate the supplied hard segmentation counts.
    Such sums are NOT assignment invariant if the component can exchange mass
    with background. ``exclude`` retains only components that cannot do so.
    The background node is never used to join distant real-cell components.
    """
    if background_policy not in {"fixed_observed", "exclude"}:
        raise ValueError("aggregate_background must be fixed_observed or exclude")
    n = label_set.n_cells - 1
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    touched = set()
    for row in range(label_set.n_observations):
        owners = label_set.candidate_owners.indices[
            label_set.candidate_owners.indptr[row]:label_set.candidate_owners.indptr[row + 1]]
        real = owners[owners < n]
        for owner in real[1:]:
            parent[find(int(owner))] = find(int(real[0]))
        if n in owners:
            touched.update(int(owner) for owner in real)
    bg_roots = {find(i) for i in touched}
    groups = {}
    for i in range(n):
        root = find(i)
        if background_policy != "exclude" or root not in bg_roots:
            groups.setdefault(root, []).append(i)
    return [np.asarray(group, dtype=int) for group in groups.values()]


def _aggregate_loss(log_mu, fixed_labels, groups):
    """Component Poisson quasi-loss, cell-weighted and gene-normalized."""
    if not groups:
        raise ValueError("No aggregate component remains after excluding background-connected cells")
    result = log_mu.sum() * 0
    target = torch.as_tensor(fixed_labels[:-1], dtype=log_mu.dtype, device=log_mu.device)
    for indices in groups:
        indices = torch.as_tensor(indices, device=log_mu.device)
        log_total = torch.logsumexp(log_mu[indices], dim=0)
        total = target[indices].sum(0)
        result = result + (log_total.exp() - total * log_total).mean()
    return result / log_mu.shape[0]


def _validate_graph_alignment(blocks, data):
    """Reject missing/duplicate cells, cross-donor blocks and altered hard labels."""
    lookup = {str(cell): i for i, cell in enumerate(data.cells.cell_id)}
    seen, indices = set(), []
    for block in blocks:
        if list(block.gene_ids) != list(data.genes):
            raise ValueError("Graph gene panel/order differs from dataset genes")
        if not block.cell_ids or len(set(block.cell_ids)) != len(block.cell_ids):
            raise ValueError("Every graph needs nonempty unique real cell IDs")
        if any(cell not in lookup for cell in block.cell_ids):
            raise ValueError("Graph contains unknown cell IDs")
        if seen.intersection(block.cell_ids):
            raise ValueError("A real cell appears in more than one graph block")
        rows = np.asarray([lookup[cell] for cell in block.cell_ids], dtype=int)
        donors = set(data.cells.iloc[rows].donor_id.astype(str))
        if donors != {str(block.donor_id)}:
            raise ValueError("Graph crosses donor boundaries or has the wrong donor_id")
        if len(set(data.cells.iloc[rows].slide_id.astype(str))) != 1:
            raise ValueError("Graph block cannot cross physical slides")
        if block.label_set.n_cells != len(rows) + 1:
            raise ValueError("Graph must have exactly one final background column")
        if block.label_set.background_index != len(rows):
            raise ValueError("Background must be the last graph column")
        fixed = block.label_set.fixed_labels(block.fixed_owner)
        if not np.allclose(fixed[:-1], data.counts[rows].toarray(), rtol=0, atol=1e-7):
            raise ValueError("Graph fixed ownership counts differ from dataset counts")
        seen.update(block.cell_ids)
        indices.append(rows)
    if seen != set(lookup):
        raise ValueError("Graphs must cover every dataset cell exactly once; subset the dataset and graphs together")
    return indices


@torch.no_grad()
def _validation_deviance(model, x, c, y, exposure, weights):
    rate = predict_nb(model, x, c)
    values = poisson_deviance(y, rate * exposure[:, None])
    return float(np.average(values, weights=weights))


def _fit_weak(x, c, y, exposure, blocks, block_rows, val_indices, baseline, method, config, weak):
    """Fit one head. A block is indivisible; minibatch normalization is global."""
    from .geometry import without_budgets

    seed = int(config.get("seed", 7))
    set_seed(seed)
    name = config.get("device", "cpu")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    epochs = int(config.get("epochs", 100))
    patience = int(config.get("patience", 15))
    blocks_per_step = int(weak.get("blocks_per_step", 1))
    max_steps = int(config.get("max_steps", epochs * math.ceil(len(blocks) / max(1, blocks_per_step))))
    min_delta = float(config.get("min_delta", 1e-4))
    if min(epochs, patience, blocks_per_step, max_steps) < 1 or min_delta < 0:
        raise ValueError("epochs, patience, blocks_per_step and max_steps must be positive")
    if not blocks or len(blocks) != len(block_rows):
        raise ValueError("Training needs aligned nonempty graph blocks")
    model = ConditionalNBModel(baseline, x.shape[1], int(config["rank"]), int(config.get("hidden_dim", 0)))
    # Dispersion is irrelevant to the Poisson objective and must not be trained.
    model.raw_dispersion.requires_grad_(False)
    model.to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(config.get("lr", 0.01)),
                                  weight_decay=float(config.get("weight_decay", 1e-4)))
    rng = np.random.default_rng(seed)
    totals = {}
    for block, rows in zip(blocks, block_rows):
        totals[block.donor_id] = totals.get(block.donor_id, 0) + len(rows)
    weights = [len(rows) / (totals[block.donor_id] * len(totals))
               for block, rows in zip(blocks, block_rows)]
    caches, preprocessing_seconds = [], 0.0
    for block in blocks:
        started = time.perf_counter()
        if method == "no_geometry":
            block = without_budgets(block, geometry=True, background=False)
        elif method == "no_background_budget":
            block = without_budgets(block, geometry=False, background=True)
        label_set = block.label_set
        fixed = label_set.fixed_labels(block.fixed_owner)
        bounds = None
        groups = None
        if method == "interval_relative":
            bounds = label_set.count_bounds(exact=True, max_queries=int(weak.get("max_bound_queries", 10000)))
        elif method == "core_only":
            fixed = _core_labels(label_set)
        elif method == "ensemble":
            fixed = label_set.maximize(rng.normal(size=fixed.shape))[1]
        elif method == "aggregate":
            groups = _aggregate_groups(label_set, weak.get("aggregate_background", "fixed_observed"))
        caches.append((label_set, fixed, bounds, groups))
        preprocessing_seconds += time.perf_counter() - started
    if method == "core_only" and not any(cache[1][:-1].sum() > 0 for cache in caches):
        raise ValueError("core_only has no structurally single-owner real molecules; supply independent nuclear ownership evidence")
    if method == "aggregate" and not any(cache[3] for cache in caches):
        raise ValueError("No aggregate component remains after background exclusion")
    val_weights = np.asarray(config["val_sample_weights"], dtype=np.float64)
    history, best_state = [], None
    best, reference, bad_epochs, best_epoch = float("inf"), float("inf"), 0, 0
    steps, solves, solve_seconds = 0, 0, 0.0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, epochs + 1):
        model.to(device).train()
        order = rng.permutation(len(blocks))
        epoch_numerator, epoch_weight = 0.0, 0.0
        for start in range(0, len(order), blocks_per_step):
            batch = order[start:start + blocks_per_step]
            optimizer.zero_grad(set_to_none=True)
            for index in batch:
                rows = block_rows[index]
                label_set, fixed, bounds, groups = caches[index]
                if method == "aggregate" and not groups:
                    continue
                xx = torch.as_tensor(x[rows], device=device)
                cc = torch.as_tensor(c[rows], device=device)
                ss = torch.as_tensor(exposure[rows], device=device)
                log_mu = model.log_rate(xx, cc).clamp(-20, 20) + ss.log()[:, None]
                with torch.no_grad():
                    base_log_mu = model.baseline.log_rate(None, cc).clamp(-20, 20) + ss.log()[:, None]
                if method == "aggregate":
                    loss = _aggregate_loss(log_mu, fixed, groups)
                else:
                    loss, elapsed = poisson_assignment_loss(log_mu, base_log_mu, label_set,
                        method, fixed_labels=fixed, bounds=bounds, rng=rng)
                    solve_seconds += elapsed
                    solves += int(method in {"joint_relative", "joint_absolute", "optimistic_relative",
                                             "no_geometry", "no_background_budget", "random_assignment"})
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite ambiguity training loss")
                # Uniform block sampling, inverse inclusion correction and
                # globally normalized donor weights. Never normalize within a
                # block by its donor composition (each block has one donor).
                (loss * weights[index] * len(blocks) / len(batch)).backward()
                epoch_numerator += float(loss.detach().cpu()) * weights[index]
                epoch_weight += weights[index]
            torch.nn.utils.clip_grad_norm_(parameters, float(config.get("max_grad_norm", 5.0)),
                                           error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            if steps >= max_steps:
                break
        score = _validation_deviance(model, x[val_indices], c[val_indices], y[val_indices],
                                     exposure[val_indices], val_weights)
        if not np.isfinite(score):
            raise FloatingPointError("Non-finite validation deviance")
        improved = score < best
        history.append({"epoch": epoch, "optimizer_steps": steps,
                        "train_objective": epoch_numerator / max(epoch_weight, 1e-20),
                        "validation_fixed_poisson_deviance": score, "is_best": improved})
        if improved:
            best, best_epoch = score, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if score < reference - min_delta:
            reference, bad_epochs = score, 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience or steps >= max_steps:
            break
    model.load_state_dict(best_state)
    model.cpu().eval()
    model.training_summary = {
        "best_epoch": best_epoch, "best_val_deviance": best, "optimizer_steps": steps,
        "configured_step_budget": max_steps, "epochs_completed": len(history),
        "inner_solve_seconds": solve_seconds, "inner_solves": solves,
        "label_precomputation_seconds": preprocessing_seconds,
        "training_seconds": time.perf_counter() - started,
        "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "device": str(device), "restored_best_validation_checkpoint": True,
        "inner_optimizer": "scipy.optimize.linprog(method=highs) for budgeted graphs; exact row maximization otherwise",
        "random_sampling": "Random LP-objective vertices, not uniform polytope samples" if method in {"random_assignment", "ensemble"} else None,
        "gradient": "Detached exact inner solution / Danskin subgradient",
    }
    return model, history


def _checkpoint(model, shared):
    return {**shared, "model_config": model.model_config(), "state_dict": model.state_dict()}


def _mean_metrics(frame):
    keys = ("poisson_deviance", "log_rate_mse", "gene_pcc", "within_type_pcc",
            "known_cell_type_fraction", "within_type_evaluable_fraction")
    result = {}
    for key in keys:
        value = pd.to_numeric(frame[key], errors="coerce").mean()
        result[key] = None if pd.isna(value) else float(value)
    return result


def run_weak_experiment(config_path, output_override=None):
    """Run every requested objective with validation-only capacity selection."""
    from .geometry import load_graphs, subset_genes

    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_path = (config_path.parent / config["data"]).resolve()
    graph_path = (config_path.parent / config["graphs"]).resolve()
    output = Path(output_override).resolve() if output_override else (config_path.parent / config["output"]).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}")
    data, blocks = load_dataset(data_path), load_graphs(graph_path)
    block_rows = _validate_graph_alignment(blocks, data)
    source = data.provenance["covariate_source"]
    feature_source = data.provenance.get("feature_source", "unknown")
    if feature_source == "rna_oracle":
        raise ValueError("RNA-derived features cannot serve as histology image features")
    if source == "rna_oracle" and not config.get("allow_oracle", False):
        raise ValueError("RNA-derived covariates require allow_oracle: true")
    weak = dict(config.get("weak", {}))
    methods = list(dict.fromkeys(weak.get("methods", ["fixed", "joint_relative", "interval_relative"])))
    if not methods or any(method not in METHODS for method in methods):
        raise ValueError(f"weak.methods must be a nonempty subset of {METHODS}")
    prep = config.get("preprocess", {})
    if "max_cells_per_donor" in prep:
        raise ValueError("Weak experiments preserve entire graph blocks: remove max_cells_per_donor and explicitly subset dataset+graphs upstream")
    seed = int(config.get("seed", 7))
    cells, counts = data.cells, data.counts
    split = make_donor_split(cells, seed=seed, **config.get("split", {}))
    from .covariates import check_covariate_provenance
    check_covariate_provenance(data.provenance, {
        name: sorted(cells.iloc[indices].donor_id.astype(str).unique().tolist())
        for name, indices in split.items()})
    tr, va, te = split["train"], split["val"], split["test"]
    train_set = set(tr.tolist())
    train_blocks, train_rows = [], []
    for block, rows in zip(blocks, block_rows):
        intersects = train_set.intersection(rows.tolist())
        if intersects and len(intersects) != len(rows):
            raise ValueError("A graph block crosses the training split")
        if intersects:
            train_blocks.append(block)
            train_rows.append(rows)
    library = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    median_library = float(np.median(library[tr]))
    exposure = library / median_library
    if "max_genes" in prep or "min_train_cells" in prep:
        gene_indices = select_genes(counts[tr], exposure[tr], int(prep.get("max_genes", len(data.genes))),
                                    int(prep.get("min_train_cells", 1)))
    else:
        gene_indices = np.arange(len(data.genes))
    genes = [data.genes[i] for i in gene_indices]
    gene_conditioning = len(genes) != len(data.genes)
    if gene_conditioning:
        # Shared budgets cannot simply be reused after dropping observations:
        # subset_genes conditions removed genes on their fixed assignment.
        blocks = [subset_genes(block, genes) for block in blocks]
        chosen = {block.block_id: block for block in blocks}
        train_blocks = [chosen[block.block_id] for block in train_blocks]
    if len(cells) * len(genes) > int(prep.get("max_dense_elements", 40000000)):
        raise MemoryError("Pilot dense count limit exceeded; create smaller complete graph blocks/dataset upstream")
    y = counts[:, gene_indices].toarray().astype(np.float32)
    x_raw, c_raw = data.features, data.covariates
    remove_covariates = bool(weak.get("remove_covariates", False))
    if remove_covariates:
        c_raw = np.zeros_like(c_raw)
    x_mean, x_scale = fit_scaler(x_raw[tr])
    c_mean, c_scale = fit_scaler(c_raw[tr])
    x = ((x_raw - x_mean) / x_scale).astype(np.float32)
    x_standard = x.copy()
    c = ((c_raw - c_mean) / c_scale).astype(np.float32)
    cov_to_image = np.linalg.lstsq(np.column_stack([c[tr], np.ones(len(tr))]), x[tr], rcond=1e-6)[0].astype(np.float32)
    x = (x - np.column_stack([c, np.ones(len(c))]) @ cov_to_image).astype(np.float32)
    training = dict(config.get("training", {}))
    ranks = list(dict.fromkeys(int(value) for value in weak.get("ranks", training.get("ranks", [4]))))
    hidden_dims = list(dict.fromkeys(int(value) for value in weak.get("hidden_dims", [training.get("hidden_dim", 0)])))
    if not ranks or any(rank < 1 or rank > min(x.shape[1], y.shape[1]) for rank in ranks):
        raise ValueError("Candidate ranks must be positive and <= min(feature_dim, n_genes)")
    if not hidden_dims or any(hidden < 0 for hidden in hidden_dims):
        raise ValueError("hidden_dims must be nonempty nonnegative integers")
    training.update(seed=seed, sample_weights=donor_weights(cells.iloc[tr]),
                    val_sample_weights=donor_weights(cells.iloc[va]))
    ensemble_members = int(weak.get("ensemble_members", 3))
    if ensemble_members < 1:
        raise ValueError("ensemble_members must be positive")
    output.mkdir(parents=True, exist_ok=True)
    run_mode = "oracle" if source == "rna_oracle" else (
        "synthetic" if "synthetic" in (source, feature_source) else "image_covariates")
    write_json(output / "config.json", config)
    split_rows = cells[["cell_id", "donor_id", "slide_id", "cell_type"]].copy()
    split_rows["split"] = ""
    for name, indices in split.items():
        split_rows.loc[indices, "split"] = name
    split_rows.to_csv(output / "split.csv", index=False)
    write_json(output / "split_donors.json", {name: sorted(cells.iloc[idx].donor_id.astype(str).unique().tolist()) for name, idx in split.items()})
    provenance = {
        "mode": run_mode, "dataset": data.provenance, "dataset_path": str(data_path),
        "graph_path": str(graph_path), "seed": seed, "n_cells": len(cells), "n_genes": len(genes),
        "median_train_library": median_library, "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__, "remove_covariates": remove_covariates,
        "input_fingerprints": {name: hashlib.sha256((data_path / name).read_bytes()).hexdigest()
                               for name in ("cells.csv", "genes.csv", "provenance.json")},
        "graph_manifest_sha256": hashlib.sha256((graph_path / "manifest.json").read_bytes()).hexdigest(),
        "graph_blocks": [{"block_id": block.block_id, "donor_id": block.donor_id,
                          "real_cells": len(block.cell_ids), "observations": block.label_set.n_observations,
                          "edges": block.label_set.candidate_owners.nnz,
                          "geometry_budget": block.label_set.max_total_cost,
                          "background_budget": block.label_set.max_background_mass,
                          "metadata": block.metadata} for block in blocks],
        "gene_subset_conditions_removed_molecules_on_fixed_owners": gene_conditioning,
        "graph_note": "Graph/budget validity is an input assumption. No budget calibrated against validation/test prediction.",
        "selection": "Full-count ambiguity heads selected by fixed-label donor-macro validation Poisson deviance; core_only uses corresponding core-subset targets and is excluded from primary ranking. Never test metrics.",
        "baseline_selection": "The shared frozen covariate NB baseline retains the existing validation NB-NLL early stopping rule.",
        "exposure": "Observed library / train median used only for supervised training/evaluation; prediction needs no RNA. Offsets stay fixed while ownership varies, so this objective does not model uncertain cell library sizes.",
    }
    write_json(output / "provenance.json", provenance)
    print(f"Weak mode={run_mode}; train/val/test={len(tr)}/{len(va)}/{len(te)}; blocks={len(train_blocks)}", flush=True)
    baseline, base_history = fit_nb_baseline(c[tr], y[tr], exposure[tr], c[va], y[va], exposure[va], training)
    write_json(output / "history_covariate_nb.json", base_history)
    shared = {"x_mean": torch.from_numpy(x_mean), "x_scale": torch.from_numpy(x_scale),
              "c_mean": torch.from_numpy(c_mean), "c_scale": torch.from_numpy(c_scale),
              "cov_to_image": torch.from_numpy(cov_to_image), "genes": genes,
              "median_train_library": median_library, "mode": run_mode,
              "covariate_source": source, "feature_source": feature_source,
              "feature_encoder": data.provenance.get("feature_encoder"),
              "remove_covariates": remove_covariates, "weak_checkpoint_version": 1}
    torch.save(_checkpoint(baseline, {**shared, "method": "covariate_nb"}), output / "covariate_nb.pt")
    core_y = None
    if "core_only" in methods:
        core_y = np.zeros_like(y)
        for block, rows in zip(blocks, block_rows):
            core_y[rows] = _core_labels(block.label_set)[:-1]
    selected, selection, model_metadata = {}, {}, {}
    for method in methods:
        selection_y = core_y if method == "core_only" else y
        candidates, best_score, best_models, best_config = [], float("inf"), None, None
        for rank in ranks:
            for hidden in hidden_dims:
                candidate = dict(training, rank=rank, hidden_dim=hidden)
                members = ensemble_members if method == "ensemble" else 1
                total_budget = int(training.get("max_steps", int(training.get("epochs", 100)) * math.ceil(len(train_blocks) / int(weak.get("blocks_per_step", 1)))))
                if total_budget < members:
                    raise ValueError("Total step budget must allow at least one step per ensemble member")
                models, member_summaries = [], []
                for member in range(members):
                    params = dict(candidate, seed=seed + 1009 * member,
                                  max_steps=total_budget // members + int(member < total_budget % members))
                    model, history = _fit_weak(x, c, selection_y, exposure, train_blocks, train_rows, va,
                                               baseline, method, params, weak)
                    models.append(model)
                    member_summaries.append(model.training_summary)
                    write_json(output / f"history_{method}_r{rank}_h{hidden}_m{member}.json", history)
                validation_rate = np.mean([predict_nb(model, x[va], c[va]) for model in models], axis=0)
                score = float(np.average(poisson_deviance(selection_y[va], validation_rate * exposure[va, None]),
                                         weights=training["val_sample_weights"]))
                entry = {"rank": rank, "hidden_dim": hidden, "validation_fixed_poisson_deviance": score,
                         "members": member_summaries, "configured_total_step_budget": total_budget,
                         "actual_total_optimizer_steps": sum(item["optimizer_steps"] for item in member_summaries)}
                candidates.append(entry)
                if score < best_score:
                    best_score, best_models, best_config = score, models, {"rank": rank, "hidden_dim": hidden}
        selected[method] = best_models
        selection[method] = {"candidates": candidates, "selected": best_config,
                             "criterion": ("locked validation donor-macro core-subset Poisson deviance"
                                           if method == "core_only" else "locked validation donor-macro fixed-label Poisson deviance")}
        saved = {**shared, "method": method, "selected": best_config}
        if len(best_models) == 1:
            saved = _checkpoint(best_models[0], saved)
        else:
            saved["members"] = [{"model_config": model.model_config(), "state_dict": model.state_dict()} for model in best_models]
        torch.save(saved, output / f"{method}.pt")
        model_metadata[method] = {
            "eligible_for_primary_ranking": method != "core_only" and run_mode != "oracle",
            "target_scope": "structurally_certain_core_subset" if method == "core_only" else "fixed_observed_cell_counts",
            "objective": "absolute Poisson quasi-loss (factorial omitted)" if method == "joint_absolute" else method,
            "aggregate_background_policy": weak.get("aggregate_background", "fixed_observed") if method == "aggregate" else None,
            "note": ("Predicts only the core-molecule subset, not all cell RNA; full-count scores are sensitivity diagnostics."
                     if method == "core_only" else "Aggregate labels depend on fixed segmentation when background transfer is feasible."
                     if method == "aggregate" else "Pilot implementation, not a third-party author-model reproduction."),
        }
        print(f"{method}: validation deviance={best_score:.5f}, selected={best_config}", flush=True)
    write_json(output / "selection.json", selection)
    # No test prediction is requested until every model/architecture is locked.
    rates = {"covariate_nb": predict_nb(baseline, x[te], c[te])}
    for method, models in selected.items():
        rates[method] = np.mean([predict_nb(model, x[te], c[te]) for model in models], axis=0)
    model_metadata["covariate_nb"] = {"eligible_for_primary_ranking": run_mode != "oracle", "target_scope": "fixed_observed_cell_counts"}
    comparison_config = config.get("comparisons", {})
    if comparison_config.get("enabled", False):
        from .comparisons import fit_predict_comparisons
        from .models import RidgePredictor
        type_oracle = bool(comparison_config.get("include_type_oracle", False))
        extra_rates, extra_metadata = fit_predict_comparisons(
            x_standard[tr], c[tr], y[tr], exposure[tr], x_standard[va], c[va], y[va], exposure[va], x_standard[te], c[te],
            train_types=cells.iloc[tr].cell_type.to_numpy() if type_oracle else None,
            test_types=cells.iloc[te].cell_type.to_numpy() if type_oracle else None,
            train_donors=cells.iloc[tr].donor_id.to_numpy(), sample_weight=donor_weights(cells.iloc[tr]),
            val_sample_weight=donor_weights(cells.iloc[va]), seed=seed,
            ranks=comparison_config.get("ranks", ranks), epochs=int(comparison_config.get("epochs", 100)),
            patience=int(comparison_config.get("patience", 15)), hidden_dim=int(comparison_config.get("hidden_dim", 64)),
            alpha=float(comparison_config.get("alpha", 10)), learning_rate=float(comparison_config.get("lr", .001)),
            batch_size=int(comparison_config.get("batch_size", 256)))
        rates.update(extra_rates)
        write_json(output / "comparison_selection.json", extra_metadata)
        for name in extra_rates:
            model_metadata[name] = dict(extra_metadata["models"].get(name, {}),
                eligible_for_primary_ranking=run_mode != "oracle" and not name.endswith("_oracle"),
                target_scope="fixed_observed_cell_counts")
        for mode in ("covariate", "image", "residual"):
            ridge = RidgePredictor(mode=mode, alpha=float(comparison_config.get("alpha", 10)))
            ridge.fit(x_standard[tr], c[tr], np.log1p(y[tr] / exposure[tr, None]))
            name = f"{mode}_ridge"
            rates[name] = np.expm1(np.clip(ridge.predict(x_standard[te], c[te]), 0, 20))
            np.savez_compressed(output / f"{name}_parameters.npz",
                **{f"{part}_{key}": val for part, comp in ridge.components_.items() for key, val in comp.items()})
            model_metadata[name] = {"eligible_for_primary_ranking": run_mode != "oracle",
                                    "target_scope": "fixed_observed_cell_counts"}
    summary = {"mode": run_mode, "models": {}, "model_metadata": model_metadata,
               "evaluation_groups": {"primary": [name for name in rates if model_metadata[name]["eligible_for_primary_ranking"]],
                                     "target_mismatch_diagnostics": [name for name in rates
                                                                      if not model_metadata[name]["eligible_for_primary_ranking"]]},
               "note": "Cell counts use the original fixed segmentation, not independent true ownership. No biological validation implied."}
    frames = {}
    for name, rate in rates.items():
        frame = donor_metrics(y[te], rate, exposure[te], cells.iloc[te])
        frames[name] = frame
        frame.to_csv(output / f"{name}_per_donor.csv", index=False)
        summary["models"][name] = _mean_metrics(frame)
    repetitions = int(config.get("evaluation", {}).get("bootstrap_repetitions", 1000))
    summary["versus_covariate_nb"] = {name: bootstrap_donor_gain(frames["covariate_nb"], frame, seed, repetitions)
                                        for name, frame in frames.items() if name != "covariate_nb"
                                        and model_metadata[name]["eligible_for_primary_ranking"]}
    if "joint_relative" in frames and "interval_relative" in frames:
        summary["joint_vs_interval"] = bootstrap_donor_gain(frames["interval_relative"], frames["joint_relative"], seed, repetitions)
    if "core_only" in rates:
        core_frame = donor_metrics(core_y[te], rates["core_only"], exposure[te], cells.iloc[te])
        core_frame.to_csv(output / "core_subset_target_per_donor.csv", index=False)
        summary["core_subset_target_metrics"] = _mean_metrics(core_frame)
    write_json(output / "metrics.json", summary)
    np.savez_compressed(output / "test_predictions.npz", cell_ids=cells.iloc[te].cell_id.to_numpy(dtype=str),
                        gene_ids=np.asarray(genes), **rates)
    lines = ["# Ambiguity-supervised experiment", "", f"Mode: **{run_mode}**.", "",
             "Every architecture and checkpoint was selected using validation patients only.", "",
             "| Model | Donor-macro fixed-count deviance | Primary eligible |", "|---|---:|---|"]
    for name, values in summary["models"].items():
        lines.append(f"| {name} | {values['poisson_deviance']:.5f} | {model_metadata[name]['eligible_for_primary_ranking']} |")
    lines += ["", "Core-only predicts a different molecular subset and is excluded from primary ranking.",
              "Joint absolute uses Poisson quasi-loss, not count-factorial-inclusive worst-case NLL.",
              "Fixed-label test scores cannot establish true cellular ownership or biological state recovery.",
              "See selection.json for actual training/inner-solver budgets and timing."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def predict_weak(checkpoint, feature_path, covariate_path, output, allow_oracle=False):
    """Predict standardized rates without RNA for single-head or ensemble files."""
    from .importers import _strings, read_feature_metadata

    target = Path(output)
    if target.suffix != ".npz":
        raise ValueError("Prediction output must end in .npz")
    if target.exists() or target.with_suffix(".json").exists():
        raise FileExistsError(target)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["mode"] == "oracle" and not allow_oracle:
        raise ValueError("Oracle checkpoint requires allow_oracle=True")
    if saved.get("feature_source") == "image" and saved.get("feature_encoder") is not None:
        if read_feature_metadata(feature_path)["feature_encoder"] != saved["feature_encoder"]:
            raise ValueError("Prediction encoder differs from training")
    with np.load(feature_path, allow_pickle=False) as feat, np.load(covariate_path, allow_pickle=False) as cov:
        ids = np.asarray(_strings(feat["cell_ids"], "feature cell_ids"))
        other = np.asarray(_strings(cov["cell_ids"], "covariate cell_ids"))
        if not len(ids) or len(set(ids)) != len(ids) or len(set(other)) != len(other) or set(ids) != set(other):
            raise ValueError("Feature/covariate IDs must be nonempty unique identical sets")
        x = np.asarray(feat["features"], dtype=np.float32)
        c = np.asarray(cov["covariates"], dtype=np.float32)
        if x.ndim != 2 or c.ndim != 2 or len(x) != len(ids) or len(c) != len(other):
            raise ValueError("Invalid feature/covariate row dimensions")
        if x.shape[1] != len(saved["x_mean"]) or c.shape[1] != len(saved["c_mean"]):
            raise ValueError("Feature/covariate dimensions differ from checkpoint")
        if not np.isfinite(x).all() or not np.isfinite(c).all():
            raise ValueError("Prediction inputs must be finite")
        lookup = {cell: i for i, cell in enumerate(other)}
        c = c[[lookup[cell] for cell in ids]]
    if saved.get("remove_covariates", False):
        c = np.zeros_like(c)
    x = (x - saved["x_mean"].numpy()) / saved["x_scale"].numpy()
    c = (c - saved["c_mean"].numpy()) / saved["c_scale"].numpy()
    x = (x - np.column_stack([c, np.ones(len(c))]) @ saved["cov_to_image"].numpy()).astype(np.float32)
    predictions = []
    for member in saved.get("members", [saved]):
        model = model_from_config(member["model_config"])
        model.load_state_dict(member["state_dict"])
        predictions.append(predict_nb(model, x, c))
    rates = np.mean(predictions, axis=0)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, cell_ids=ids, gene_ids=np.asarray(saved["genes"]), rates=rates)
    write_json(target.with_suffix(".json"), {"mode": saved["mode"], "method": saved.get("method"),
        "checkpoint": str(checkpoint), "reference_library": saved["median_train_library"],
        "ensemble_members": len(predictions), "remove_covariates": saved.get("remove_covariates", False),
        "target_scope": "structurally_certain_core_subset" if saved.get("method") == "core_only" else "fixed_observed_cell_counts",
        "note": "Expected standardized counts at train reference exposure; no RNA consumed."})
