"""Auditable author-code execution and explicit, unit-aware output adapters.

The runner executes user-specified argv in a pinned local checkout. It does not
download code, install dependencies, select checkpoints using test labels, or claim
that writing split manifests proves an arbitrary external script obeyed them.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .external import evaluate_external


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids(values, name):
    result = np.asarray(values, dtype=str)
    if result.ndim != 1 or not len(result) or any(not x.strip() for x in result) or len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique nonempty IDs")
    return result


def _write_predictions(values, cells, genes, output, *, training_target_normalization, reference_library):
    """Refuse to turn native raw-count prediction into reference rates silently."""
    if training_target_normalization not in {"raw_counts", "reference_library"}:
        raise ValueError("training_target_normalization must be raw_counts or reference_library")
    cells, genes = _ids(cells, "cell_ids"), _ids(genes, "gene_ids")
    values = np.asarray(values, dtype=float)
    if values.shape != (len(cells), len(genes)) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("predictions must be finite nonnegative and match cell/gene IDs")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    key = "rates" if training_target_normalization == "reference_library" else "expected_counts"
    if key == "rates" and (reference_library is None or not np.isfinite(reference_library) or reference_library <= 0):
        raise ValueError("normalized targets require a positive training reference library")
    np.savez_compressed(output, cell_ids=cells, gene_ids=genes, **{key: values})
    return {"path": str(output.resolve()), "value_key": key, "reference_library": reference_library,
        "training_target_normalization": training_target_normalization, "sha256": file_sha256(output)}


def convert_ghist_csv(source, output, *, training_target_normalization="raw_counts", reference_library=None, id_map=None):
    """Read GHIST's actual ``epoch_*_expr.csv`` output and invert log1p.

    Verified against inference.py and dataio/dataset_input.py at the pinned revision.
    The CSV's first column is the author cell ID and the header contains gene names.
    GHIST has already divided by expr_scale; dividing again would be incorrect.
    An optional CSV maps ``source_id,cell_id``; partial/many-to-one maps are refused.
    """
    with Path(source).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        genes = _ids(header[1:], "GHIST genes")
        rows = list(reader)
    if not rows or any(len(row) != len(header) for row in rows):
        raise ValueError("GHIST CSV has missing or ragged rows")
    cells = _ids([row[0] for row in rows], "GHIST cell IDs")
    log_values = np.asarray([row[1:] for row in rows], dtype=float)
    if not np.isfinite(log_values).all() or (log_values < 0).any() or (log_values > 80).any():
        raise ValueError("GHIST log1p outputs must be finite, nonnegative and bounded")
    if id_map is not None:
        mapping = pd.read_csv(id_map, dtype=str, keep_default_na=False)
        if not {"source_id", "cell_id"} <= set(mapping):
            raise ValueError("ID map requires source_id and cell_id")
        source_ids = _ids(mapping.source_id, "map source_id")
        target_ids = _ids(mapping.cell_id, "map cell_id")
        if set(source_ids) != set(cells):
            raise ValueError("ID map must cover exactly the GHIST output IDs")
        lookup = dict(zip(source_ids, target_ids))
        cells = [lookup[c] for c in cells]
    metadata = _write_predictions(np.expm1(log_values), cells, genes, output,
        training_target_normalization=training_target_normalization, reference_library=reference_library)
    metadata.update({"adapter": "ghist_csv", "source_sha256": file_sha256(source), "transform": "expm1; expr_scale already removed by author inference"})
    return metadata


def convert_pixel2gene_pickles(source, mapping_path, genes_path, output, *, trust_pickle=False,
        training_target_normalization="raw_counts", reference_library=None):
    """Aggregate author ``cnts-super/{gene}.pickle`` grids on fixed common masks.

    ``mapping_path`` NPZ has cell_ids plus equal-length row,col,cell_index,weight.
    Weights specify the fraction of each prediction pixel allocated to a cell.
    Fraction sums per pixel cannot exceed one. Mapping must come from registered
    geometry, not test RNA. No guessed scale or automatic segmentation is applied.
    """
    if not trust_pickle:
        raise ValueError("Author pickle inputs execute Python when loaded; explicitly set trust_pickle for trusted local author outputs")
    genes = _ids(Path(genes_path).read_text(encoding="utf-8").splitlines(), "Pixel2Gene genes")
    if any(Path(g).name != g or "/" in g or "\\" in g or g in {".", ".."} for g in genes):
        raise ValueError("Gene names cannot contain path components")
    with np.load(mapping_path, allow_pickle=False) as mapping:
        cells = _ids(mapping["cell_ids"], "mapping cell_ids")
        row, col, owner = [np.asarray(mapping[k]) for k in ("row", "col", "cell_index")]
        weight = np.asarray(mapping["weight"], float)
    if not len(row) or any(v.ndim != 1 or v.shape != row.shape for v in (row, col, owner, weight)):
        raise ValueError("mapping arrays must be aligned nonempty vectors")
    if any(not np.issubdtype(v.dtype, np.integer) for v in (row, col, owner)) or min(row.min(), col.min(), owner.min()) < 0 or owner.max() >= len(cells):
        raise ValueError("mapping indices must be valid nonnegative integers")
    if not np.isfinite(weight).all() or (weight <= 0).any():
        raise ValueError("mapping weights must be finite and positive")
    sums = pd.DataFrame({"r": row, "c": col, "w": weight}).groupby(["r", "c"]).w.sum()
    if (sums > 1 + 1e-6).any() or len(np.unique(owner)) != len(cells):
        raise ValueError("mapping duplicates mass or omits cells")
    values = np.zeros((len(cells), len(genes)), float)
    hashes = {}
    for j, gene in enumerate(genes):
        path = Path(source) / f"{gene}.pickle"
        with path.open("rb") as stream:
            grid = np.asarray(pickle.load(stream), dtype=float)
        if grid.ndim != 2 or row.max() >= grid.shape[0] or col.max() >= grid.shape[1]:
            raise ValueError("Pixel2Gene grid does not match common-mask coordinate frame")
        selected = grid[row, col]
        if not np.isfinite(selected).all() or (selected < 0).any():
            raise ValueError("mapped pixels contain missing, negative or nonfinite predictions")
        np.add.at(values[:, j], owner, selected * weight)
        hashes[gene] = file_sha256(path)
    metadata = _write_predictions(values, cells, genes, output,
        training_target_normalization=training_target_normalization, reference_library=reference_library)
    metadata.update({"adapter": "pixel2gene_pickles", "grid_sha256": hashes, "mapping_sha256": file_sha256(mapping_path),
        "aggregation": "fractional area sum on supplied common masks; no Zarr read (author writer does not populate it)"})
    return metadata


def _git(checkout, args, executable):
    run = subprocess.run([executable, "-C", str(checkout), *args], check=True, capture_output=True, text=True)
    return run.stdout.strip()


def _validate_split(split):
    if not {"cell_id", "donor_id", "split"} <= set(split) or split.empty:
        raise ValueError("locked split requires cell_id, donor_id, split")
    _ids(split.cell_id, "locked cell_id")
    if set(split.split) != {"train", "val", "test"} or split.donor_id.str.strip().eq("").any():
        raise ValueError("locked split must include nonempty train/val/test donor IDs")
    if (split.groupby("donor_id").split.nunique() != 1).any():
        raise ValueError("donor leakage in locked split")


def run_reproduction(config_path, output=None):
    """Run pinned author code, convert its output, then score a locked test once.

    Config paths are relative to the YAML file, command working directories to the
    checkout. Commands are explicit argv lists (``shell=False``). Config must name
    its external environment interpreter, files/checkpoints and scientific units.
    """
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("reproduction config must be a mapping")
    base = config_path.parent
    def resolve(p):
        return (base / str(p)).resolve()
    checkout, run_dir = resolve(config["checkout"]), resolve(config["locked_run"])
    output = Path(output).resolve() if output is not None else resolve(config["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("reproduction output must be empty")
    revision = str(config["revision"])
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full immutable Git commit SHA")
    git_executable = str(config.get("git_executable", "git"))
    observed = _git(checkout, ["rev-parse", "HEAD"], git_executable)
    if observed != revision:
        raise ValueError(f"checkout revision differs: expected {revision}, observed {observed}")
    dirty = _git(checkout, ["status", "--porcelain", "--untracked-files=no"], git_executable)
    if dirty:
        raise ValueError("author checkout has modified tracked files; commit the adaptation and pin its revision")
    split = pd.read_csv(run_dir / "split.csv", dtype=str, keep_default_na=False)
    _validate_split(split)
    declaration = dict(config["declaration"])
    for label, key in (("train", "training_donors"), ("val", "validation_donors")):
        declared = _ids(declaration.get(key, []), key)
        locked = set(split.loc[split.split == label, "donor_id"])
        if set(declared) != locked:
            raise ValueError(f"{key} differs from locked split; additional data requires a separate declared protocol")
    if declaration.get("input_mode") not in {"image", "rna_oracle", "multimodal"} or not {"encoder", "pretraining_overlap"} <= set(declaration):
        raise ValueError("declaration requires input_mode, encoder and pretraining_overlap")
    declaration["code_revision"] = revision
    commands = config.get("commands")
    if not isinstance(commands, list) or not commands:
        raise ValueError("commands must be nonempty explicit argv command objects")
    for command in commands:
        if not isinstance(command, dict) or not isinstance(command.get("argv"), list) or not command["argv"] or any(not isinstance(v, str) or not v for v in command["argv"]):
            raise ValueError("each command requires a nonempty list of string argv")
        cwd = (checkout / command.get("cwd", ".")).resolve()
        if cwd != checkout and checkout not in cwd.parents:
            raise ValueError("command cwd must stay inside author checkout")
    output.mkdir(parents=True, exist_ok=True)
    manifests = output / "manifests"
    manifests.mkdir()
    replacements = {"checkout": str(checkout), "output": str(output), "locked_run": str(run_dir)}
    for label in ("train", "val", "test"):
        path = manifests / f"{label}.csv"
        split.loc[split.split == label].to_csv(path, index=False)
        replacements[f"{label}_manifest"] = str(path)
    def substitute(token):
        for key, value in replacements.items():
            token = token.replace("{" + key + "}", value)
        return token
    env = os.environ.copy()
    for label in ("train", "val", "test"):
        env[f"MORPHOSTATE_{label.upper()}_MANIFEST"] = replacements[f"{label}_manifest"]
    config_text = config_path.read_text(encoding="utf-8")
    (output / "reproduction.yaml").write_text(config_text, encoding="utf-8")
    audit = {"revision": revision, "checkout": str(checkout), "split_sha256": file_sha256(run_dir / "split.csv"),
        "commands": [], "status": "running", "declaration": declaration,
        "limitation": "Manifests and declarations are auditable records, not a sandbox or proof that author code used only declared donors."}
    def save():
        (output / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    save()
    try:
        for i, command in enumerate(commands):
            argv = [substitute(v) for v in command["argv"]]
            cwd = (checkout / command.get("cwd", ".")).resolve()
            started = time.monotonic()
            record = {"argv": argv, "cwd": str(cwd), "stdout": f"command_{i}.stdout.log", "stderr": f"command_{i}.stderr.log"}
            audit["commands"].append(record)
            save()
            with (output / record["stdout"]).open("w", encoding="utf-8") as stdout, (output / record["stderr"]).open("w", encoding="utf-8") as stderr:
                result = subprocess.run(argv, cwd=cwd, env=env, shell=False, stdout=stdout, stderr=stderr,
                    timeout=float(command.get("timeout_seconds", 86400)),
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            record.update({"returncode": result.returncode, "elapsed_seconds": time.monotonic() - started})
            save()
            if result.returncode:
                raise RuntimeError(f"Author command {i} failed with exit code {result.returncode}; see saved logs")
        if _git(checkout, ["rev-parse", "HEAD"], git_executable) != revision or _git(checkout, ["status", "--porcelain", "--untracked-files=no"], git_executable):
            raise ValueError("author tracked code changed during execution")
        audit["checkpoint_sha256"] = {str(resolve(substitute(p))): file_sha256(resolve(substitute(p))) for p in config.get("checkpoints", [])}
        if not audit["checkpoint_sha256"]:
            raise ValueError("checkpoints must identify at least one fitted model artifact")
        conversion = {k: substitute(v) if isinstance(v, str) else v for k, v in config["conversion"].items()}
        for key in ("source", "id_map", "mapping_path", "genes_path"):
            if key in conversion and conversion[key] is not None:
                conversion[key] = str(resolve(conversion[key]))
        adapter = conversion.pop("adapter")
        prediction_path = output / "converted_predictions.npz"
        if adapter == "ghist_csv":
            converted = convert_ghist_csv(output=prediction_path, **conversion)
        elif adapter == "pixel2gene_pickles":
            converted = convert_pixel2gene_pickles(output=prediction_path, **conversion)
        else:
            raise ValueError("Supported verified adapters: ghist_csv, pixel2gene_pickles")
        audit["conversion"] = converted
        (output / "declaration.json").write_text(json.dumps(declaration, indent=2), encoding="utf-8")
        if converted["value_key"] == "rates":
            evaluate_external(run_dir, prediction_path, converted["reference_library"], config["name"], output / "declaration.json", output / "evaluation")
            audit["status"] = "evaluated"
        else:
            audit["status"] = "converted_native_counts"
            audit["evaluation_note"] = "Raw-count predictions preserved. Rate-based evaluation skipped because units differ; do not divide by held-out observed RNA library."
        save()
    except Exception as error:
        audit["status"], audit["error"] = "failed", str(error)
        save()
        raise
    return audit
