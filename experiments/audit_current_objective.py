"""Replay the 2026-10-22 synthetic development diagnostic, not a SOTA test.

Run from an installed repository with:
    python experiments/audit_current_objective.py --output runs/objective_audit_replay
This data seed was already inspected during method development.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from morphostate.geometry import make_synthetic_graphs
from morphostate.synthetic import make_synthetic
from morphostate.weak_training import run_weak_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/objective_audit_replay"))
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    make_synthetic(out / "data", seed=23, donors=8, cells_per_donor=20, genes=8, features=8)
    make_synthetic_graphs(out / "data", out / "graphs", seed=23, ambiguous_fraction=0.6,
                          background_fraction=0.1, geometry_fraction=0.25)
    config = yaml.safe_load(Path(__file__).with_suffix(".yaml").read_text(encoding="utf-8"))
    config.update(data=str(out / "data"), graphs=str(out / "graphs"))
    report = {"scope": "Previously inspected synthetic development data; not SOTA or an independent test",
              "seeds": [7, 17, 29], "runs": []}
    started = time.perf_counter()
    for seed in report["seeds"]:
        config.update(seed=seed, output=str(out / f"seed_{seed}"))
        path = out / f"seed_{seed}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        result = run_weak_experiment(path)
        report["runs"].append({"seed": seed, "metrics": result["models"]})
        report["elapsed_seconds"] = time.perf_counter() - started
        (out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({"seed": seed, "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
