"""Score externally reproduced models on a locked run without selecting them."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_dataset
from .importers import _strings
from .metrics import bootstrap_donor_gain, donor_metrics


def align_external_predictions(path, cell_ids, gene_ids, reference_library, training_library):
    """Require complete IDs and nonnegative rates at an explicit library reference.

    Log expression, arbitrary scores, and spot predictions require a scientifically
    justified external adapter. This function does not silently reinterpret units.
    """
    if not np.isfinite(reference_library) or reference_library <= 0:
        raise ValueError("reference_library must be finite and positive")
    if not np.isfinite(training_library) or training_library <= 0:
        raise ValueError("training_library must be finite and positive")
    with np.load(path, allow_pickle=False) as bundle:
        supplied_cells = _strings(bundle["cell_ids"], "external cell_ids")
        supplied_genes = _strings(bundle["gene_ids"], "external gene_ids")
        values = np.asarray(bundle["rates"], dtype=np.float64)
    if set(supplied_cells) != set(cell_ids) or set(supplied_genes) != set(gene_ids):
        raise ValueError("External predictions must cover exactly the locked test cells and genes")
    if values.shape != (len(supplied_cells), len(supplied_genes)):
        raise ValueError("External rates shape disagrees with ID arrays")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("External predictions must be finite nonnegative rates, not log values")
    row_lookup = {key: i for i, key in enumerate(supplied_cells)}
    col_lookup = {key: i for i, key in enumerate(supplied_genes)}
    rows = [row_lookup[key] for key in cell_ids]
    cols = [col_lookup[key] for key in gene_ids]
    return values[np.ix_(rows, cols)] * (training_library / reference_library)


def evaluate_external(run_dir, prediction_path, reference_library, model_name, metadata_path, output):
    """Audit a model's declared training overlap and evaluate its frozen predictions.

    The declaration is not independent proof about pretraining or hyperparameter
    selection. This evaluator cannot make incompatible biological units comparable.
    """
    run_dir, output = Path(run_dir), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("External evaluation output must be empty")
    if not model_name.strip():
        raise ValueError("model_name must be nonempty")
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    split = pd.read_csv(run_dir / "split.csv", dtype=str)
    cells = split.loc[split.split == "test"].reset_index(drop=True)
    declaration = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    required = {"training_donors", "validation_donors", "code_revision", "encoder", "input_mode", "pretraining_overlap"}
    if not required <= declaration.keys():
        raise ValueError(f"External metadata missing: {sorted(required - declaration.keys())}")
    if declaration["input_mode"] not in {"image", "rna_oracle", "multimodal"}:
        raise ValueError("input_mode must be image, rna_oracle or multimodal")
    training_donors = _strings(np.asarray(declaration["training_donors"]), "training_donors")
    validation_donors = _strings(np.asarray(declaration["validation_donors"]), "validation_donors")
    if set(training_donors) & set(validation_donors):
        raise ValueError("Declared training and validation donors overlap")
    if (set(training_donors) | set(validation_donors)) & set(cells.donor_id):
        raise ValueError("Declared external training/validation donors overlap the locked test")
    dataset = load_dataset(provenance["dataset_path"])
    with np.load(run_dir / "test_predictions.npz", allow_pickle=False) as original:
        genes = original["gene_ids"].astype(str).tolist()
        original_ids = original["cell_ids"].astype(str).tolist()
        baseline_rates = original["covariate_nb"]
    if original_ids != cells.cell_id.tolist():
        raise ValueError("Run split and saved prediction order differ")
    reference = float(provenance["median_train_library"])
    rates = align_external_predictions(prediction_path, original_ids, genes, reference_library, reference)
    row_lookup = {key: i for i, key in enumerate(dataset.cells.cell_id)}
    gene_lookup = {key: i for i, key in enumerate(dataset.genes)}
    rows = [row_lookup[key] for key in original_ids]
    columns = [gene_lookup[key] for key in genes]
    exposures = np.asarray(dataset.counts[rows].sum(axis=1)).ravel() / reference
    counts = dataset.counts[rows][:, columns].toarray()
    scored = donor_metrics(counts, rates, exposures, cells)
    baseline = donor_metrics(counts, baseline_rates, exposures, cells)
    metric_names = [name for name in scored if name not in {"donor_id", "n_cells"}]
    means = {name: pd.to_numeric(scored[name], errors="coerce").mean() for name in metric_names}
    summary = {"model": model_name, "run_mode": provenance["mode"], "declaration": declaration,
               "metrics": {key: None if pd.isna(value) else float(value) for key, value in means.items()},
               "versus_covariate_nb": bootstrap_donor_gain(baseline, scored),
               "reference_library": reference,
               "note": "External reproduction scored only; no hyperparameter selection. Declaration is not an independent provenance audit."}
    output.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output / "per_donor.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved external evaluation to {output}", flush=True)
    return summary
