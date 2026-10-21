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
    weak = sub.add_parser("weak-run", help="Train coupled ambiguous-label methods and ablations")
    weak.add_argument("--config", required=True)
    weak.add_argument("--output")
    weak_predict = sub.add_parser("weak-predict", help="Predict from a weak single-model or ensemble checkpoint")
    for flag in ("checkpoint", "features", "covariates", "output"):
        weak_predict.add_argument(f"--{flag}", required=True)
    weak_predict.add_argument("--allow-oracle", action="store_true")
    synthetic_graphs = sub.add_parser("synthetic-graphs", help="Create conserved synthetic label uncertainty")
    for flag in ("data", "output"):
        synthetic_graphs.add_argument(f"--{flag}", required=True)
    synthetic_graphs.add_argument("--seed", type=int, default=7)
    synthetic_graphs.add_argument("--ambiguous-fraction", type=float, default=0.6)
    synthetic_graphs.add_argument("--background-fraction", type=float, default=0.1)
    synthetic_graphs.add_argument("--geometry-fraction", type=float, default=0.25)
    graph = sub.add_parser("build-graphs", help="Build patient-pure graphs from registered molecule coordinates")
    for flag in ("data", "transcripts", "output", "coordinate-units"):
        graph.add_argument(f"--{flag}", required=True)
    graph.add_argument("--radius", type=float, required=True)
    graph.add_argument("--geometry-budget", type=float)
    graph.add_argument("--background-fraction", type=float)
    graph.add_argument("--max-observations", type=int, default=2000000)
    graph.add_argument("--max-edges", type=int, default=2000000)
    graph.add_argument("--max-cells-per-block", type=int, default=5000)
    suite = sub.add_parser("suite", help="Run locked patient folds, seeds and config variants")
    suite.add_argument("--config", required=True)
    suite.add_argument("--plan-only", action="store_true")
    suite.add_argument("--resume", action="store_true")
    summarize = sub.add_parser("summarize", help="Summarize a suite with donor-level paired bootstrap")
    summarize.add_argument("--suite", required=True)
    summarize.add_argument("--reference", default="fixed")
    summarize.add_argument("--repetitions", type=int, default=2000)
    reproduce = sub.add_parser("reproduce", help="Execute a pinned author checkout and convert its predictions")
    reproduce.add_argument("--config", required=True)
    reproduce.add_argument("--output")
    affine = sub.add_parser("fit-affine", help="Fit a coordinate transform from training landmarks")
    affine.add_argument("--landmarks", required=True)
    affine.add_argument("--output", required=True)
    affine.add_argument("--residual-limit", type=float)
    patches = sub.add_parser("extract-patches", help="Crop registered H&E patches at level-0 coordinates")
    for flag in ("image", "cells", "transform", "output"):
        patches.add_argument(f"--{flag}", required=True)
    patches.add_argument("--patch-pixels", type=int, default=224)
    patches.add_argument("--backend", choices=("pillow", "openslide"), default="pillow")
    patches.add_argument("--edge-policy", choices=("error", "white"), default="error")
    covariates = sub.add_parser("image-covariates", help="Fit train-only image type probabilities and neighborhood composition")
    for flag in ("data", "split", "output"):
        covariates.add_argument(f"--{flag}", required=True)
    covariates.add_argument("--neighbors", type=int, default=12)
    covariates.add_argument("--epochs", type=int, default=100)
    covariates.add_argument("--patience", type=int, default=15)
    covariates.add_argument("--seed", type=int, default=7)
    states = sub.add_parser("evaluate-states", help="Evaluate prespecified programs against independent state labels")
    for flag in ("predictions", "labels", "programs", "output"):
        states.add_argument(f"--{flag}", required=True)
    states.add_argument("--label-source", choices=("protein", "if", "expert", "synthetic"), required=True)
    states.add_argument("--rate-key", default="rates")
    contrasts = sub.add_parser("contrast-bounds", help="Compute exact feasible linear-contrast bounds")
    for flag in ("graphs", "coefficients", "output"):
        contrasts.add_argument(f"--{flag}", required=True)
    risk = sub.add_parser("bound-risk", help="Report all contrast coverage/risk at prespecified width thresholds")
    for flag in ("bounds", "truth", "output"):
        risk.add_argument(f"--{flag}", required=True)
    risk.add_argument("--thresholds", type=float, nargs="+", required=True)
    risk.add_argument("--truth-source", choices=("synthetic", "independent_assay"), required=True)
    stress = sub.add_parser("stress-case", help="Construct a conserved state-erasure counterexample with feasible witnesses")
    stress.add_argument("--output", required=True)
    stress.add_argument("--seed", type=int, default=7)
    stress.add_argument("--donors", type=int, default=6)
    stress.add_argument("--pairs-per-donor", type=int, default=4)
    stress.add_argument("--state-strength", type=float, default=0.5)
    stress.add_argument("--state-prevalence", type=float, default=0.25)
    stress.add_argument("--ambiguity", type=float, default=0.6)
    stress.add_argument("--exclude-null", action="store_true")
    theory = sub.add_parser("theory-diagnostics", help="Audit LP duality, baseline feasibility and joint/box relative risks")
    for flag in ("data", "graphs", "predictions", "output"):
        theory.add_argument(f"--{flag}", required=True)
    theory.add_argument("--reference-library", type=float, required=True)
    theory.add_argument("--model-key", default="joint_relative")
    theory.add_argument("--baseline-key", default="covariate_nb")
    theory.add_argument("--max-bound-queries", type=int, default=10000)
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
    elif args.command == "weak-run":
        from .weak_training import run_weak_experiment
        run_weak_experiment(args.config, args.output)
    elif args.command == "weak-predict":
        from .weak_training import predict_weak
        predict_weak(args.checkpoint, args.features, args.covariates, args.output, args.allow_oracle)
    elif args.command == "synthetic-graphs":
        from .geometry import make_synthetic_graphs
        make_synthetic_graphs(args.data, args.output, seed=args.seed, ambiguous_fraction=args.ambiguous_fraction,
                              background_fraction=args.background_fraction, geometry_fraction=args.geometry_fraction)
    elif args.command == "build-graphs":
        from .geometry import construct_geometry_graphs
        construct_geometry_graphs(args.data, args.transcripts, args.output, args.radius,
                                   args.geometry_budget, args.background_fraction,
                                   coordinate_units=args.coordinate_units, max_observations=args.max_observations,
                                   max_edges=args.max_edges, max_cells_per_block=args.max_cells_per_block)
    elif args.command == "suite":
        from .suite import run_suite
        run_suite(args.config, args.plan_only, args.resume)
    elif args.command == "summarize":
        from .suite import summarize_suite
        summarize_suite(args.suite, args.reference, args.repetitions)
    elif args.command == "reproduce":
        from .sota import run_reproduction
        run_reproduction(args.config, args.output)
    elif args.command == "fit-affine":
        from .imaging import fit_affine_landmarks
        fit_affine_landmarks(args.landmarks, args.output, args.residual_limit)
    elif args.command == "extract-patches":
        from .imaging import extract_registered_patches
        extract_registered_patches(args.image, args.cells, args.transform, args.output,
                                   args.patch_pixels, args.backend, args.edge_policy)
    elif args.command == "image-covariates":
        from .covariates import build_image_covariates
        build_image_covariates(args.data, args.split, args.output, args.neighbors, args.epochs, args.patience, args.seed)
    elif args.command == "evaluate-states":
        from .state_evaluation import evaluate_states
        evaluate_states(args.predictions, args.labels, args.programs, args.output, args.label_source, args.rate_key)
    elif args.command == "contrast-bounds":
        from .state_evaluation import evaluate_contrasts
        evaluate_contrasts(args.graphs, args.coefficients, args.output)
    elif args.command == "bound-risk":
        from .selective import evaluate_bound_risk
        evaluate_bound_risk(args.bounds, args.truth, args.output, args.thresholds, args.truth_source)
    elif args.command == "stress-case":
        from .stress import make_stress_case
        make_stress_case(args.output, args.seed, args.donors, args.pairs_per_donor,
                          args.state_strength, args.state_prevalence, args.ambiguity, not args.exclude_null)
    elif args.command == "theory-diagnostics":
        from .theory import diagnose_predictions
        diagnose_predictions(args.data, args.graphs, args.predictions, args.output, args.reference_library,
                              args.model_key, args.baseline_key, args.max_bound_queries)


if __name__ == "__main__":
    main()

