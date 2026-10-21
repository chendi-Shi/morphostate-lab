import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from morphostate.data import load_dataset, save_dataset
from morphostate.suite import make_donor_folds, run_suite, summarize_suite
from morphostate.synthetic import make_synthetic


def test_folds_keep_donors_and_grouping_units_together():
    cells = pd.DataFrame({"donor_id": [f"d{i}" for i in range(12) for _ in range(2)],
                          "tma": [f"t{i // 2}" for i in range(12) for _ in range(2)]})
    splits = make_donor_folds(cells, 3, group_column="tma")
    assert len({d for split in splits for d in split["test"]}) == 12
    for split in splits:
        assert not set(split["train"]) & set(split["test"])
        lookup = dict(zip(cells.donor_id, cells.tma, strict=True))
        assert not {lookup[d] for d in split["train"]} & {lookup[d] for d in split["val"]}
    assert splits == make_donor_folds(cells, 3, group_column="tma")


def test_donor_cannot_cross_group_units():
    cells = pd.DataFrame({"donor_id": ["a", "a", "b", "c"], "tma": ["x", "y", "z", "w"]})
    with pytest.raises(ValueError, match="crosses"):
        make_donor_folds(cells, 3, group_column="tma")


def test_suite_averages_seeds_before_donor_bootstrap(tmp_path):
    jobs, status = [], {}
    for fold, donor in enumerate(["a", "b"]):
        for seed in [7, 17]:
            name = f"f{fold}s{seed}"
            directory = tmp_path / name
            directory.mkdir()
            for model, score in [("fixed", 5.), ("joint_relative", 3. if donor == "a" else 4.)]:
                pd.DataFrame({"donor_id": [donor], "poisson_deviance": [score]}).to_csv(directory / f"{model}_per_donor.csv", index=False)
            (directory / "metrics.json").write_text(json.dumps({"mode": "synthetic", "models": {"fixed": {}, "joint_relative": {}}}))
            jobs.append({"job_id": name, "variant": "main", "fold": fold, "seed": seed,
                         "config": {"output": str(directory), "split": {"explicit": {"test": [donor]}}}})
            status[name] = {"status": "completed"}
    (tmp_path / "manifest.json").write_text(json.dumps({"jobs": jobs}))
    (tmp_path / "status.json").write_text(json.dumps(status))
    result = summarize_suite(tmp_path, repetitions=20)
    assert result["paired"][0]["n_donors"] == 2
    assert result["paired"][0]["mean_gain"] == 1.5
    assert result["incomplete_seed_groups"] == 0


def _result_suite(tmp_path, modes=("synthetic", "synthetic"), omit_seed=None):
    jobs, status = [], {}
    for fold, donor in enumerate(["a", "b"]):
        for seed in [7, 17]:
            name = f"f{fold}s{seed}"
            directory = tmp_path / name
            directory.mkdir()
            models = {"fixed": {}, "joint_relative": {}, "core_only": {}, "other_target": {}}
            report = {"mode": modes[fold], "models": models,
                      "model_metadata": {"core_only": {"eligible_for_primary_ranking": False,
                                             "target_scope": "structurally_certain_core_subset"},
                                         "other_target": {"target_scope": "protein_scores"}},
                      "evaluation_groups": {"primary": ["fixed", "joint_relative", "other_target"]}}
            (directory / "metrics.json").write_text(json.dumps(report))
            for model in [*models, "core_subset_target"]:
                pd.DataFrame({"donor_id": [donor], "poisson_deviance": [5 if model == "fixed" else 1]}).to_csv(
                    directory / f"{model}_per_donor.csv", index=False)
            jobs.append({"job_id": name, "variant": "main", "fold": fold, "seed": seed,
                         "config": {"output": str(directory), "split": {"explicit": {"test": [donor]}}}})
            status[name] = {"status": "failed" if omit_seed == (fold, seed) else "completed"}
    (tmp_path / "manifest.json").write_text(json.dumps({"jobs": jobs}))
    (tmp_path / "status.json").write_text(json.dumps(status))
    return jobs, status


def test_primary_pairs_exclude_core_auxiliary_and_different_targets(tmp_path):
    _result_suite(tmp_path)
    summary = summarize_suite(tmp_path, repetitions=10)
    assert [row["model"] for row in summary["paired"]] == ["joint_relative"]
    primary = pd.read_csv(tmp_path / "summary.csv")
    assert set(primary.model) == {"fixed", "joint_relative"}
    diagnostics = pd.read_csv(tmp_path / "diagnostic_summary.csv")
    assert set(diagnostics.model) == {"core_only", "core_subset_target", "other_target"}


def test_oracle_variants_do_not_enter_primary_table(tmp_path):
    _result_suite(tmp_path, modes=("oracle", "oracle"))
    summary = summarize_suite(tmp_path, repetitions=10)
    assert summary["paired"] == []
    assert pd.read_csv(tmp_path / "summary.csv").empty
    assert len(pd.read_csv(tmp_path / "diagnostic_summary.csv")) == 5


def test_partial_seed_results_never_produce_primary_pairs(tmp_path):
    _result_suite(tmp_path, omit_seed=(1, 17))
    summary = summarize_suite(tmp_path, repetitions=10)
    assert summary["paired"] == []
    assert summary["incomplete_seed_groups"] == 5
    assert summary["completed_jobs"] == 3


def test_changed_donor_rows_are_rejected_before_pairing(tmp_path):
    jobs, _ = _result_suite(tmp_path)
    target = tmp_path / jobs[0]["job_id"] / "joint_relative_per_donor.csv"
    pd.DataFrame({"donor_id": ["unexpected_patient"], "poisson_deviance": [1]}).to_csv(target, index=False)
    with pytest.raises(ValueError, match="locked test donors"):
        summarize_suite(tmp_path)


def test_mixed_input_modes_across_seeds_are_not_silently_averaged(tmp_path):
    jobs, _ = _result_suite(tmp_path)
    path = tmp_path / jobs[0]["job_id"] / "metrics.json"
    report = json.loads(path.read_text())
    report["mode"] = "oracle"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="changes across seeds"):
        summarize_suite(tmp_path)


def test_invalid_fold_and_group_values_fail_early():
    cells = pd.DataFrame({"donor_id": ["a", "b", "c", "d"]})
    with pytest.raises(ValueError, match="integer"):
        make_donor_folds(cells, folds=2.4)
    cells.loc[0, "donor_id"] = np.nan
    with pytest.raises(ValueError, match="complete"):
        make_donor_folds(cells, folds=2)


def _suite_config(tmp_path, prepare_covariates=False):
    make_synthetic(tmp_path / "data", donors=6, cells_per_donor=4, genes=4, features=4)
    data = load_dataset(tmp_path / "data")
    data.cells["cell_type"] = np.tile(["type_a", "type_b", "type_a", "type_b"], 6)
    data.cells["x_centroid"] = np.tile(np.arange(4), 6)
    data.cells["y_centroid"] = np.zeros(24)
    save_dataset(data, tmp_path / "data")
    base = {"data": "data", "output": "unused", "seed": 7,
            "training": {"epochs": 1, "patience": 1, "ranks": [1], "batch_size": 16},
            "preprocess": {"max_genes": 4, "min_train_cells": 1, "max_cells_per_donor": 4},
            "evaluation": {"bootstrap_repetitions": 10, "permutation_repeats": 0}}
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base))
    suite = {"base_config": "base.yaml", "engine": "baseline", "output": "suite",
             "folds": 3, "seeds": [7], "variants": {"main": {}}}
    if prepare_covariates:
        suite["image_covariates"] = {"enabled": True, "neighbors": 2, "epochs": 2, "patience": 1}
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(suite))
    return path


def test_suite_prepares_fold_local_covariates_and_resume_verifies_evidence(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    path = _suite_config(tmp_path, prepare_covariates=True)
    summary = run_suite(path)
    assert summary["completed_jobs"] == 3
    assert summary["paired"]
    output = tmp_path / "suite"
    manifest = json.loads((output / "manifest.json").read_text())
    status = json.loads((output / "status.json").read_text())
    for job in manifest["jobs"]:
        done = status[job["job_id"]]
        prepared = load_dataset(done["prepared_data_path"])
        assert set(prepared.provenance["covariate_fit_donors"]) == set(job["config"]["split"]["explicit"]["train"])
        assert not set(prepared.provenance["covariate_fit_donors"]) & set(job["config"]["split"]["explicit"]["test"])
        runtime = yaml.safe_load(open(done["runtime_config_path"], encoding="utf-8"))
        assert runtime["data"] != job["config"]["data"]
        assert done["config_sha256"] == job["config_sha256"]
        assert done["runtime_config_sha256"] != done["config_sha256"]
        assert done["artifacts_sha256"]
    import morphostate.covariates as covariates

    def must_not_fit(*args, **kwargs):
        raise AssertionError("Completed resume refitted image covariates")

    monkeypatch.setattr(covariates, "build_image_covariates", must_not_fit)
    resumed = run_suite(path, resume=True)
    assert resumed["paired"] == summary["paired"]
    target = Path(manifest["jobs"][0]["config"]["output"]) / "covariate_nb_per_donor.csv"
    target.write_text(target.read_text() + "\n")
    with pytest.raises(ValueError, match="inconsistent"):
        run_suite(path, resume=True)


def test_plan_only_and_failed_job_preserve_the_original_exception(tmp_path, monkeypatch):
    path = _suite_config(tmp_path)
    plan = run_suite(path, plan_only=True)
    assert len(plan["jobs"]) == 3
    assert not (tmp_path / "suite" / "status.json").exists()
    import morphostate.experiment as experiment

    def failure(*args, **kwargs):
        raise ValueError("deliberate data problem")

    monkeypatch.setattr(experiment, "run_experiment", failure)
    with pytest.raises(RuntimeError, match="Suite job failed"):
        run_suite(path, resume=True)
    ledger = json.loads((tmp_path / "suite" / "status.json").read_text())
    assert "ValueError: deliberate data problem" in next(iter(ledger.values()))["error"]
