import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from morphostate.sota import convert_ghist_csv, convert_pixel2gene_pickles, run_reproduction


def test_ghist_exact_author_csv_log1p_format(tmp_path):
    # inference.py builds DataFrame(all_expr, index=all_ids, columns=gene_names)
    # then to_csv; expr_scale has already been removed at this point.
    source = tmp_path / "epoch_3_expr.csv"
    pd.DataFrame(np.log1p([[2., 3.], [4., 7.]]), index=["001", "002"], columns=["G1", "G2"]).to_csv(source)
    mapped = tmp_path / "map.csv"
    mapped.write_text("source_id,cell_id\n001,slide::001\n002,slide::002\n", encoding="utf-8")
    output = tmp_path / "out.npz"
    meta = convert_ghist_csv(source, output, id_map=mapped)
    with np.load(output) as data:
        np.testing.assert_allclose(data["expected_counts"], [[2, 3], [4, 7]])
        assert data["cell_ids"].tolist() == ["slide::001", "slide::002"]
        assert "rates" not in data
    assert meta["value_key"] == "expected_counts"
    convert_ghist_csv(source, output, training_target_normalization="reference_library", reference_library=100)
    assert "rates" in np.load(output)


def test_ghist_no_silent_unit_conversion_or_duplicate_ids(tmp_path):
    source = tmp_path / "expr.csv"
    source.write_text(",G1\nc1,1\nc2,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="reference"):
        convert_ghist_csv(source, tmp_path / "out.npz", training_target_normalization="reference_library")
    source.write_text(",G1,G1\nc1,1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        convert_ghist_csv(source, tmp_path / "out.npz")


def test_pixel2gene_author_grid_fractional_masks_and_trust(tmp_path):
    (tmp_path / "genes.txt").write_text("G1\nG2\n", encoding="utf-8")
    for gene, grid in {"G1": [[2, 4], [6, 8]], "G2": [[3, 5], [7, 9]]}.items():
        with (tmp_path / f"{gene}.pickle").open("wb") as stream:
            pickle.dump(np.asarray(grid, dtype=np.float16), stream)
    mapping = tmp_path / "mask.npz"
    np.savez(mapping, cell_ids=["a", "b"], row=[0, 0, 1], col=[0, 0, 1], cell_index=[0, 1, 1], weight=[.25, .75, 1.])
    with pytest.raises(ValueError, match="trust"):
        convert_pixel2gene_pickles(tmp_path, mapping, tmp_path / "genes.txt", tmp_path / "out.npz")
    convert_pixel2gene_pickles(tmp_path, mapping, tmp_path / "genes.txt", tmp_path / "out.npz", trust_pickle=True)
    with np.load(tmp_path / "out.npz") as data:
        np.testing.assert_allclose(data["expected_counts"], [[.5, .75], [9.5, 11.25]])
    np.savez(mapping, cell_ids=["a", "b"], row=[0, 0], col=[0, 0], cell_index=[0, 1], weight=[1., 1.])
    with pytest.raises(ValueError, match="mass"):
        convert_pixel2gene_pickles(tmp_path, mapping, tmp_path / "genes.txt", tmp_path / "bad.npz", trust_pickle=True)


def _git_executable():
    git = shutil.which("git")
    bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe"
    if not git and bundled.exists():
        git = str(bundled)
    if not git:
        pytest.skip("git unavailable")
    return git


def reproduction_fixture(tmp_path):
    git = _git_executable()
    author = tmp_path / "author"
    author.mkdir()
    # Fixture uses the verified author CSV format, not the GHIST neural network.
    (author / "predict.py").write_text("from pathlib import Path\nimport sys\np=Path(sys.argv[1]);p.mkdir(exist_ok=True)\n(p/'prediction.csv').write_text(',G1\\nc3,0.6931471805599453\\n')\n(p/'model.bin').write_bytes(b'fixture fitted artifact')\n", encoding="utf-8")
    subprocess.run([git, "init", str(author)], check=True, capture_output=True)
    subprocess.run([git, "-C", str(author), "add", "predict.py"], check=True, capture_output=True)
    subprocess.run([git, "-C", str(author), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "fixture"], check=True, capture_output=True)
    revision = subprocess.check_output([git, "-C", str(author), "rev-parse", "HEAD"], text=True).strip()
    run = tmp_path / "locked"
    run.mkdir()
    pd.DataFrame({"cell_id": ["c1", "c2", "c3"], "donor_id": ["d1", "d2", "d3"], "split": ["train", "val", "test"]}).to_csv(run / "split.csv", index=False)
    config = {"name": "GHIST-format fixture", "checkout": str(author), "revision": revision, "git_executable": git,
        "locked_run": str(run), "output": str(tmp_path / "result"),
        "commands": [{"argv": [sys.executable, "predict.py", "{output}"]}],
        "checkpoints": ["{output}/model.bin"], "declaration": {"training_donors": ["d1"], "validation_donors": ["d2"], "encoder": "fixture", "input_mode": "image", "pretraining_overlap": "none"},
        "conversion": {"adapter": "ghist_csv", "source": "{output}/prediction.csv", "training_target_normalization": "raw_counts"}}
    path = tmp_path / "repro.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path, config


def test_runner_executes_pinned_code_and_preserves_native_units(tmp_path):
    path, config = reproduction_fixture(tmp_path)
    audit = run_reproduction(path)
    assert audit["status"] == "converted_native_counts"
    assert audit["commands"][0]["returncode"] == 0
    assert len(audit["checkpoint_sha256"]) == 1
    result = Path(config["output"])
    assert json.loads((result / "audit.json").read_text())["revision"] == config["revision"]
    assert (result / "manifests/test.csv").exists()
    assert not (result / "evaluation").exists()


def test_runner_normalized_target_protocol_evaluates_locked_test(tmp_path):
    from scipy import sparse

    from morphostate.data import Dataset, save_dataset

    path, config = reproduction_fixture(tmp_path)
    dataset_path = tmp_path / "dataset"
    cells = pd.DataFrame({"cell_id": ["c1", "c2", "c3"], "donor_id": ["d1", "d2", "d3"],
        "slide_id": ["s1", "s2", "s3"], "cell_type": ["T", "T", "T"]})
    dataset = Dataset(sparse.csr_matrix([[2], [3], [4]]), np.ones((3, 1)), np.ones((3, 1)), cells, ["G1"],
        {"dataset_id": "fixture", "feature_source": "synthetic", "covariate_source": "synthetic"})
    save_dataset(dataset, dataset_path)
    locked = Path(config["locked_run"])
    split = cells.copy()
    split["split"] = ["train", "val", "test"]
    split.to_csv(locked / "split.csv", index=False)
    (locked / "provenance.json").write_text(json.dumps({"dataset_path": str(dataset_path), "median_train_library": 2., "mode": "synthetic"}), encoding="utf-8")
    np.savez(locked / "test_predictions.npz", cell_ids=["c3"], gene_ids=["G1"], covariate_nb=[[1.]])
    config["conversion"].update(training_target_normalization="reference_library", reference_library=2.)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    audit = run_reproduction(path)
    assert audit["status"] == "evaluated"
    result = Path(config["output"])
    report = json.loads((result / "evaluation/metrics.json").read_text())
    assert report["versus_covariate_nb"]["n_donors"] == 1
    assert report["metrics"]["poisson_deviance"] >= 0


@pytest.mark.parametrize("problem", ["revision", "leak", "dirty", "failed_command", "cwd_escape"])
def test_runner_rejects_bad_execution(tmp_path, problem):
    path, config = reproduction_fixture(tmp_path)
    if problem == "revision":
        config["revision"] = "0" * 40
    elif problem == "leak":
        config["declaration"]["training_donors"] = ["d3"]
    elif problem == "dirty":
        (Path(config["checkout"]) / "predict.py").write_text("modified", encoding="utf-8")
    elif problem == "failed_command":
        config["commands"] = [{"argv": [sys.executable, "-c", "raise SystemExit(4)"]}]
    else:
        config["commands"][0]["cwd"] = ".."
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)):
        run_reproduction(path)
    if problem == "failed_command":
        audit = json.loads((Path(config["output"]) / "audit.json").read_text())
        assert audit["status"] == "failed"
        assert audit["commands"][0]["returncode"] == 4
