"""Predeclared donor folds, ablation jobs, resumable execution and paired summaries."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .data import load_dataset


def nested_update(original, updates):
    result = copy.deepcopy(original)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = nested_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def make_donor_folds(cells, folds=5, seed=7, val_fraction=0.2, group_column="donor_id"):
    """Make outer folds once; all training seeds share identical test patients.

    Optional grouping (e.g. TMA) must assign each donor to exactly one group.
    Validation also holds out entire groups. This is group randomization, not
    disease stratification; supply explicit folds when that design is needed.
    """
    if "donor_id" not in cells or group_column not in cells or not 0 < val_fraction < 1:
        raise ValueError("Missing grouping column or invalid validation fraction")
    columns = list(dict.fromkeys(["donor_id", group_column]))
    if cells[columns].isna().any().any() or cells.empty:
        raise ValueError("Grouping and donor IDs must be nonempty and complete")
    mappings = cells[columns].astype(str).drop_duplicates()
    if any((mappings[column].str.strip() != mappings[column]).any()
           or mappings[column].eq("").any() for column in columns):
        raise ValueError("Grouping and donor IDs must be nonempty trimmed strings")
    if group_column == "donor_id":
        donor_groups = {str(d): str(d) for d in cells.donor_id.unique()}
    else:
        if mappings.groupby("donor_id")[group_column].nunique().max() != 1:
            raise ValueError("A donor crosses grouping units; combine connected groups before splitting")
        donor_groups = dict(zip(mappings.donor_id, mappings[group_column], strict=True))
    groups = sorted(set(donor_groups.values()))
    if folds != "loo" and (not isinstance(folds, int) or isinstance(folds, bool)):
        raise ValueError("folds must be an integer or 'loo'")
    n_folds = len(groups) if folds == "loo" else folds
    if len(groups) < 3 or not 2 <= n_folds <= len(groups):
        raise ValueError("Need >=3 independent groups and 2 <= folds <= n_groups")
    rng = np.random.default_rng(seed)
    chunks = np.array_split(rng.permutation(groups), n_folds)
    result = []
    for held_out in chunks:
        remaining = np.array(sorted(set(groups) - set(held_out)))
        if len(remaining) < 2:
            raise ValueError("Each outer fold needs independent train and validation groups")
        remaining = rng.permutation(remaining)
        n_val = min(len(remaining) - 1, max(1, int(np.floor(len(remaining) * val_fraction))))
        assigned = {"test": set(held_out), "val": set(remaining[:n_val]), "train": set(remaining[n_val:])}
        result.append({name: sorted(d for d, group in donor_groups.items() if group in values)
                       for name, values in assigned.items()})
    return result


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _write_json(path, value):
    """Atomic receipts survive an interrupted process without truncated JSON."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _completion_receipt(job, runtime_config=None):
    directory = Path(job["config"]["output"])
    required = [directory / "config.json", directory / "metrics.json", directory / "split_donors.json"]
    if any(not path.is_file() for path in required):
        raise ValueError(f"Completed job lacks required output evidence: {job['job_id']}")
    actual_config = json.loads(required[0].read_text(encoding="utf-8"))
    expected_hash = _digest(runtime_config) if runtime_config is not None else job["config_sha256"]
    if _digest(actual_config) != expected_hash:
        raise ValueError(f"Completed job config differs from manifest: {job['job_id']}")
    split = json.loads(required[2].read_text(encoding="utf-8"))
    expected_split = job["config"]["split"]["explicit"]
    if (set(split) != set(expected_split)
            or any(sorted(split[name]) != sorted(expected_split[name]) for name in expected_split)):
        raise ValueError(f"Completed job donor split differs from manifest: {job['job_id']}")
    metrics = json.loads(required[1].read_text(encoding="utf-8"))
    if not isinstance(metrics.get("models"), dict) or not metrics["models"]:
        raise ValueError(f"Completed job has no declared model results: {job['job_id']}")
    for model in metrics["models"]:
        if not isinstance(model, str) or Path(model).name != model or "/" in model or "\\" in model:
            raise ValueError("Invalid result model name")
        if not (directory / f"{model}_per_donor.csv").is_file():
            raise ValueError(f"Completed job lacks per-donor result for {model}")
    files = sorted(set(required) | set(directory.glob("*_per_donor.csv")))
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def _load_runtime_config(job, status):
    if "runtime_config_path" not in status:
        return None
    runtime = yaml.safe_load(Path(status["runtime_config_path"]).read_text(encoding="utf-8"))
    if _digest(runtime) != status.get("runtime_config_sha256"):
        raise ValueError(f"Runtime config changed after execution: {job['job_id']}")
    expected = copy.deepcopy(job["config"])
    expected["data"] = status["prepared_data_path"]
    if runtime != expected:
        raise ValueError("Prepared runtime config may only replace the input data directory")
    return runtime


def _prepare_image_covariates(job, specification, output, status):
    settings = specification.get("image_covariates", {})
    if not isinstance(settings, dict) or set(settings) - {"enabled", "neighbors", "epochs", "patience"}:
        raise ValueError("image_covariates accepts enabled, neighbors, epochs and patience only")
    if not settings.get("enabled", False):
        return None
    from .covariates import build_image_covariates
    prepared = output / "data" / job["job_id"]
    split_path = output / "splits" / f"{job['job_id']}.json"
    split_path.parent.mkdir(exist_ok=True)
    _write_json(split_path, job["config"]["split"]["explicit"])
    options = {key: value for key, value in settings.items() if key != "enabled"}
    build_image_covariates(job["config"]["data"], split_path, prepared, seed=job["seed"], **options)
    runtime = copy.deepcopy(job["config"])
    runtime["data"] = str(prepared.resolve())
    config_path = output / "configs" / f"{job['job_id']}__prepared.yaml"
    config_path.write_text(yaml.safe_dump(runtime, sort_keys=False), encoding="utf-8")
    status.update(runtime_config_path=str(config_path.resolve()), runtime_config_sha256=_digest(runtime),
                  prepared_data_path=runtime["data"], covariate_split_path=str(split_path.resolve()))
    return runtime


def prepare_suite(config_path):
    path = Path(config_path).resolve()
    specification = yaml.safe_load(path.read_text(encoding="utf-8"))
    preparation = specification.get("image_covariates", {})
    if (not isinstance(preparation, dict)
            or set(preparation) - {"enabled", "neighbors", "epochs", "patience"}
            or not isinstance(preparation.get("enabled", False), bool)):
        raise ValueError("Invalid image_covariates settings")
    base_path = (path.parent / specification["base_config"]).resolve()
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    for key in ("data", "graphs"):
        if key in base:
            base[key] = str((base_path.parent / base[key]).resolve())
    data = load_dataset(base["data"])
    folds = specification.get("explicit_folds")
    if folds is None:
        folds = make_donor_folds(data.cells, specification.get("folds", 5),
                                 int(specification.get("split_seed", 7)),
                                 float(specification.get("val_fraction", 0.2)),
                                 specification.get("group_column", "donor_id"))
    from .splits import make_donor_split
    for split in folds:
        make_donor_split(data.cells, explicit=split)
    all_test = [donor for split in folds for donor in split["test"]]
    if len(all_test) != len(set(all_test)) or set(all_test) != set(data.cells.donor_id):
        raise ValueError("Outer folds must test each donor exactly once")
    seeds = specification.get("seeds", [7, 17, 29])
    if (not isinstance(seeds, list) or not seeds
            or not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in seeds)
            or len(seeds) != len(set(seeds))):
        raise ValueError("seeds must be a nonempty list of unique integers")
    variants = specification.get("variants", {"main": {}})
    if not isinstance(variants, dict) or not variants:
        raise ValueError("variants must be a nonempty mapping of names to overrides")
    output = (path.parent / specification["output"]).resolve()
    engine = specification.get("engine", "weak")
    if engine not in {"weak", "baseline"}:
        raise ValueError("engine must be weak or baseline")
    jobs = []
    for variant, updates in variants.items():
        if not isinstance(variant, str) or not variant or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in variant):
            raise ValueError("Variant names may contain only letters, digits, underscore and hyphen")
        if not isinstance(updates, dict) or "split" in updates or "seed" in updates or "output" in updates:
            raise ValueError("Variants cannot override splits, seed or output")
        for fold_number, split in enumerate(folds):
            for seed in seeds:
                job_id = f"{variant}__fold{fold_number:02d}__seed{seed}"
                config = nested_update(base, updates)
                # Variant data/graph replacements resolve relative to the suite.
                for key in ("data", "graphs"):
                    if key in updates:
                        config[key] = str((path.parent / updates[key]).resolve())
                config.update(seed=seed, split={"explicit": split}, output=str(output / "jobs" / job_id))
                jobs.append({"job_id": job_id, "variant": variant, "fold": fold_number,
                             "seed": seed, "engine": engine, "config": config, "config_sha256": _digest(config)})
    return specification, output, jobs


def run_suite(config_path, plan_only=False, resume=False):
    specification, output, jobs = prepare_suite(config_path)
    manifest = {"specification": specification, "jobs": jobs}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError("Existing suite differs; use a new output directory")
        if not resume:
            raise FileExistsError("Suite exists; pass --resume to skip verified completed jobs")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError("Suite output is not empty")
    output.mkdir(parents=True, exist_ok=True)
    (output / "configs").mkdir(exist_ok=True)
    _write_json(manifest_path, manifest)
    for job in jobs:
        (output / "configs" / f"{job['job_id']}.yaml").write_text(yaml.safe_dump(job["config"], sort_keys=False), encoding="utf-8")
    if plan_only:
        print(f"Prepared {len(jobs)} locked jobs at {output}")
        return manifest
    ledger_path = output / "status.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {}
    errors = []
    for job in jobs:
        job_id = job["job_id"]
        run_path = Path(job["config"]["output"])
        done = ledger.get(job_id, {})
        if done.get("status") == "completed":
            runtime_config = _load_runtime_config(job, done)
            if (done.get("config_sha256") != job["config_sha256"]
                    or done.get("artifacts_sha256") != _completion_receipt(job, runtime_config)):
                raise ValueError(f"Completed job is inconsistent: {job_id}")
            continue
        if run_path.exists() and any(run_path.iterdir()):
            # Do not erase partial experiments or overwrite their evidence.
            raise FileExistsError(f"Partial job exists: {run_path}; move it aside explicitly before resuming")
        start = time.perf_counter()
        ledger[job_id] = {"status": "running", "config_sha256": job["config_sha256"]}
        _write_json(ledger_path, ledger)
        print(f"[{job_id}] starting", flush=True)
        try:
            runtime_config = _prepare_image_covariates(job, specification, output, ledger[job_id])
            _write_json(ledger_path, ledger)
            run_config = (Path(ledger[job_id]["runtime_config_path"]) if runtime_config is not None
                          else output / "configs" / f"{job_id}.yaml")
            if job["engine"] == "weak":
                from .weak_training import run_weak_experiment
                run_weak_experiment(run_config)
            else:
                from .experiment import run_experiment
                run_experiment(run_config)
            ledger[job_id].update(status="completed", artifacts_sha256=_completion_receipt(job, runtime_config))
        except Exception as exc:
            ledger[job_id].update(status="failed", error=f"{type(exc).__name__}: {exc}")
            errors.append(job_id)
        finally:
            ledger[job_id]["seconds"] = time.perf_counter() - start
            _write_json(ledger_path, ledger)
        if errors and not specification.get("continue_on_error", False):
            raise RuntimeError(f"Suite job failed: {job_id}; see {ledger_path}")
    if errors and not any(ledger.get(job["job_id"], {}).get("status") == "completed" for job in jobs):
        raise RuntimeError(f"All suite jobs failed; see {ledger_path}: {errors}")
    summary = summarize_suite(output, reference=specification.get(
        "reference", "fixed" if specification.get("engine", "weak") == "weak" else "covariate_nb"))
    if errors:
        raise RuntimeError(f"Failed jobs retained in status.json: {errors}")
    return summary


def summarize_suite(output, reference="fixed", repetitions=2000):
    """Average matched seeds within donor; isolate oracle/different-target rows.

    A partially completed variant may have diagnostic means but does not produce
    a primary paired comparison. Declared model outputs, exact planned donors,
    identical complete seed sets and a common target are checked first.
    """
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    ledger = json.loads((output / "status.json").read_text(encoding="utf-8"))
    jobs = manifest["jobs"]
    if not jobs or len({job["job_id"] for job in jobs}) != len(jobs):
        raise ValueError("Manifest needs nonempty unique job IDs")
    job_keys = [(job["variant"], job["fold"], job["seed"]) for job in jobs]
    if len(set(job_keys)) != len(job_keys):
        raise ValueError("Manifest repeats a variant/fold/seed job")
    planned = {}
    for job in jobs:
        split = job["config"].get("split", {}).get("explicit")
        if not isinstance(split, dict) or not split.get("test"):
            raise ValueError("Summary requires each job's locked explicit donor split")
        for donor in split["test"]:
            key = (job["variant"], str(donor))
            record = planned.setdefault(key, {"fold": job["fold"], "seeds": set()})
            if record["fold"] != job["fold"] or job["seed"] in record["seeds"]:
                raise ValueError("A donor is planned in duplicate outer folds or seeds")
            record["seeds"].add(job["seed"])
    for variant in {job["variant"] for job in jobs}:
        seed_sets = {tuple(sorted(record["seeds"])) for (name, _), record in planned.items() if name == variant}
        if len(seed_sets) != 1:
            raise ValueError("Every outer fold within a variant must plan the same training seeds")
    tables = []
    for job in jobs:
        if ledger.get(job["job_id"], {}).get("status") != "completed":
            continue
        directory = Path(job["config"]["output"])
        report = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        declared = report.get("models")
        if not isinstance(declared, dict) or not declared:
            raise ValueError(f"Missing declared model metrics for {job['job_id']}")
        metadata = report.get("model_metadata", {})
        groups = report.get("evaluation_groups", {})
        for model in declared:
            if not (directory / f"{model}_per_donor.csv").is_file():
                raise ValueError(f"Missing declared per-donor output: {job['job_id']}/{model}")
        for file in sorted(directory.glob("*_per_donor.csv")):
            frame = pd.read_csv(file, dtype={"donor_id": str})
            model = file.name.removesuffix("_per_donor.csv")
            if ("donor_id" not in frame or "poisson_deviance" not in frame
                    or frame.empty or frame.donor_id.isna().any() or frame.donor_id.duplicated().any()):
                raise ValueError(f"Invalid per-donor results: {file}")
            expected = set(str(value) for value in job["config"]["split"]["explicit"]["test"])
            if set(frame.donor_id) != expected:
                raise ValueError(f"Per-donor output differs from locked test donors: {file}")
            frame["poisson_deviance"] = pd.to_numeric(frame.poisson_deviance, errors="raise")
            if not np.isfinite(frame.poisson_deviance).all() or (frame.poisson_deviance < 0).any():
                raise ValueError(f"Non-finite/negative per-donor deviance: {file}")
            information = metadata.get(model, {})
            scope = information.get("target_scope", "fixed_observed_cell_counts")
            # Existing baseline reports use this older name for the same
            # original-segmentation whole-cell molecular target.
            if scope == "all_molecules":
                scope = "fixed_observed_cell_counts"
            reasons = []
            if model not in declared:
                reasons.append("auxiliary_output_not_a_declared_model")
            if report.get("mode") not in {"synthetic", "image_covariates"}:
                reasons.append("oracle_or_unverified_input_mode")
            if information.get("eligible_for_primary_ranking", True) is not True:
                reasons.append("model_declared_ineligible")
            if "primary" in groups and model not in groups["primary"]:
                reasons.append("outside_primary_evaluation_group")
            if scope != "fixed_observed_cell_counts" or model == "core_only":
                reasons.append("different_target_scope")
            frame["model"], frame["target_scope"] = model, scope
            frame["primary_eligible"] = not reasons
            frame["diagnostic_reason"] = ";".join(reasons)
            frame["mode"] = report.get("mode", "unknown")
            frame["variant"], frame["seed"], frame["fold"] = job["variant"], job["seed"], job["fold"]
            tables.append(frame)
    if not tables:
        raise ValueError("No completed per-donor results found")
    rows = pd.concat(tables, ignore_index=True)
    keys = ["variant", "model", "seed", "donor_id"]
    if rows.duplicated(keys).any():
        raise ValueError("A donor appears in multiple outer test folds for the same model/seed")
    if "n_cells" in rows:
        supported = rows.loc[rows.primary_eligible]
        if (supported.groupby(["variant", "seed", "donor_id"])["n_cells"].nunique() > 1).any():
            raise ValueError("Primary models must use identical per-donor evaluation cell coverage")
    metrics = [name for name in ("poisson_deviance", "log_rate_mse", "gene_pcc", "within_type_pcc") if name in rows]
    grouped = rows.groupby(["variant", "model", "donor_id"], sort=True)
    mean_records = []
    for (variant, model, donor), frame in grouped:
        scopes, modes = set(frame.target_scope), set(frame["mode"])
        if len(scopes) != 1 or len(modes) != 1 or frame.primary_eligible.nunique() != 1:
            raise ValueError("Model input mode, eligibility or target changes across seeds")
        expected = planned[(variant, donor)]["seeds"]
        observed = set(frame.seed)
        if not observed <= expected:
            raise ValueError("An unplanned seed appears in a donor result")
        complete = observed == expected
        mean_records.append({"variant": variant, "model": model, "donor_id": donor,
            **{metric: pd.to_numeric(frame[metric], errors="raise").mean() for metric in metrics},
            "seeds_observed": len(observed), "seeds_expected": len(expected), "seeds_complete": complete,
            "target_scope": next(iter(scopes)), "mode": next(iter(modes)),
            "primary_eligible": bool(frame.primary_eligible.iloc[0]) and complete,
            "diagnostic_reason": str(frame.diagnostic_reason.iloc[0]) + (";incomplete_seeds" if not complete else "")})
    donor_means = pd.DataFrame(mean_records)
    rows.to_csv(output / "all_runs_per_donor.csv", index=False)
    donor_means.to_csv(output / "seed_averaged_per_donor.csv", index=False)
    primary = donor_means.loc[donor_means.primary_eligible].copy()
    diagnostics = donor_means.loc[~donor_means.primary_eligible].copy()
    diagnostics.to_csv(output / "diagnostic_seed_averaged_per_donor.csv", index=False)
    primary.to_csv(output / "primary_seed_averaged_per_donor.csv", index=False)
    overall = primary.groupby(["variant", "model"], as_index=False)[metrics].mean()
    overall["n_donors"] = primary.groupby(["variant", "model"])["donor_id"].nunique().to_numpy()
    overall.to_csv(output / "summary.csv", index=False)
    diagnostics.groupby(["variant", "model", "mode", "target_scope"], as_index=False)[metrics].mean().to_csv(
        output / "diagnostic_summary.csv", index=False)
    comparisons, skipped = [], []
    rng = np.random.default_rng(7)
    for variant in donor_means.variant.unique():
        subset = primary[primary.variant == variant]
        base = subset[subset.model == reference][["donor_id", "poisson_deviance"]]
        expected_donors = {donor for candidate_variant, donor in planned if candidate_variant == variant}
        for model in donor_means.loc[donor_means.variant == variant, "model"].unique():
            if model == reference:
                continue
            values = subset[subset.model == model]
            paired = values.merge(base, on="donor_id", suffixes=("", "_reference"), validate="one_to_one")
            if (set(paired.donor_id) != expected_donors or len(paired) != len(base)
                    or len(paired) != len(values) or not len(paired)):
                skipped.append({"variant": variant, "model": model, "reference": reference,
                                "reason": "ineligible input/target or incomplete planned donors/seeds"})
                continue
            gains = paired.poisson_deviance_reference.to_numpy() - paired.poisson_deviance.to_numpy()
            interval = None
            if len(gains) >= 2:
                boot = rng.choice(gains, (repetitions, len(gains)), replace=True).mean(1)
                interval = np.quantile(boot, [.025, .975]).tolist()
            comparisons.append({"variant": variant, "model": model, "reference": reference,
                                "n_donors": len(gains), "mean_gain": float(gains.mean()), "bootstrap_interval": interval})
    result = {"paired": comparisons, "skipped_comparisons": skipped,
              "completed_jobs": sum(ledger.get(job["job_id"], {}).get("status") == "completed" for job in jobs),
              "expected_jobs": len(jobs), "incomplete_seed_groups": int((~donor_means.seeds_complete).sum()),
              "diagnostic_donor_rows": len(diagnostics),
              "warning": "Only complete matched seeds and all planned test donors enter primary pairs. Oracle, core-subset and auxiliary outputs are separated into diagnostic CSVs. Small donor intervals are descriptive."}
    _write_json(output / "summary.json", result)
    lines = ["# Locked experiment suite", "", result["warning"], "", "| Variant | Model | Donors | Deviance |", "|---|---|---:|---:|"]
    for row in overall.itertuples():
        lines.append(f"| {row.variant} | {row.model} | {row.n_donors} | {row.poisson_deviance:.5f} |")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
