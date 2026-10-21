"""Command-line interface for auditable pilot experiments."""

import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="morphostate", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Generate synthetic data for software verification")
    demo.add_argument("--output", required=True)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--donors", type=int, default=12)
    demo.add_argument("--cells-per-donor", type=int, default=80)
    demo.add_argument("--genes", type=int, default=32)
    demo.add_argument("--features", type=int, default=12)
    validate = sub.add_parser("validate", help="Validate counts, IDs, dimensions, and provenance")
    validate.add_argument("--data", required=True)
    run = sub.add_parser("run", help="Train baselines and select conditional rank on validation donors")
    run.add_argument("--config", required=True)
    run.add_argument("--output", help="Override run output; must be empty")
    imp = sub.add_parser("import-xenium", help="Import one single-donor Xenium sample with aligned embeddings")
    for flag in ("matrix", "cells", "features", "covariates", "donor-id", "slide-id", "output"):
        imp.add_argument(f"--{flag}", required=True)
    imp.add_argument("--covariate-source", choices=("image", "rna_oracle"), required=True)
    imp.add_argument("--dataset-id", default="custom")
    merge = sub.add_parser("merge", help="Merge identical panels and feature schemas; never silently intersect genes")
    merge.add_argument("--inputs", nargs="+", required=True)
    merge.add_argument("--output", required=True)
    predict = sub.add_parser("predict", help="Predict standardized rates without reading any test RNA")
    for flag in ("checkpoint", "features", "covariates", "output"):
        predict.add_argument(f"--{flag}", required=True)
    predict.add_argument("--allow-oracle", action="store_true")
    extract = sub.add_parser("extract-features", help="Encode already registered image patches with timm")
    extract.add_argument("--manifest", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--model", default="resnet18")
    weights = extract.add_mutually_exclusive_group(required=True)
    weights.add_argument("--pretrained", action="store_true")
    weights.add_argument("--checkpoint")
    extract.add_argument("--batch-size", type=int, default=32)
    extract.add_argument("--device", default="auto")
    external = sub.add_parser("evaluate-external", help="Score a reproduced model on locked test IDs and units")
    for flag in ("run", "predictions", "model-name", "metadata", "output"):
        external.add_argument(f"--{flag}", required=True)
    external.add_argument("--reference-library", type=float, required=True)
    args = parser.parse_args(argv)
    if args.command == "demo":
        from .synthetic import make_synthetic
        make_synthetic(args.output, args.seed, args.donors, args.cells_per_donor, args.genes, args.features)
        print(f"Synthetic dataset saved to {args.output}; not real biological evidence.")
    elif args.command == "validate":
        from .data import load_dataset
        data = load_dataset(args.data)
        print(json.dumps({"cells": len(data.cells), "genes": len(data.genes),
                          "donors": data.cells.donor_id.nunique(), "provenance": data.provenance}, indent=2))
    elif args.command == "run":
        from .experiment import run_experiment
        run_experiment(args.config, args.output)
    elif args.command == "import-xenium":
        from .importers import import_xenium
        import_xenium(args.matrix, args.cells, args.features, args.covariates,
                      args.donor_id, args.slide_id, args.output, args.covariate_source, args.dataset_id)
        print(f"Prepared sample saved to {args.output}")
    elif args.command == "merge":
        from .importers import merge_datasets
        merge_datasets(args.inputs, args.output)
        print(f"Merged dataset saved to {args.output}")
    elif args.command == "predict":
        from .experiment import predict_from_checkpoint
        predict_from_checkpoint(args.checkpoint, args.features, args.covariates, args.output, args.allow_oracle)
    elif args.command == "extract-features":
        from .features import extract_features
        extract_features(args.manifest, args.output, args.model, args.pretrained, args.checkpoint,
                         args.batch_size, args.device)
    elif args.command == "evaluate-external":
        from .external import evaluate_external
        evaluate_external(args.run, args.predictions, args.reference_library, args.model_name, args.metadata, args.output)


if __name__ == "__main__":
    main()

