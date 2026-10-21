import json
from pathlib import Path

import pytest
import torch
import yaml

from morphostate.cli import main


def test_documented_weak_cli_runs_all_methods_and_comparisons(tmp_path):
    torch.set_num_threads(1)
    main(["demo", "--output", str(tmp_path / "data"), "--donors", "6", "--cells-per-donor", "4",
          "--genes", "4", "--features", "4"])
    main(["synthetic-graphs", "--data", str(tmp_path / "data"), "--output", str(tmp_path / "graphs")])
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/weak_demo.yaml").read_text())
    config.update(data="data", graphs="graphs", output="run")
    config["training"].update(epochs=1, max_steps=2)
    config["comparisons"].update(epochs=1, include_type_oracle=True)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    main(["weak-run", "--config", str(path)])
    result = json.loads((tmp_path / "run/metrics.json").read_text())
    assert len(result["models"]) == 22
    assert {"core_only", "type_mean_oracle"} <= set(result["evaluation_groups"]["target_mismatch_diagnostics"])
    assert "regularized_partial_cca" in result["evaluation_groups"]["primary"]
    assert (tmp_path / "run/comparison_selection.json").exists()


@pytest.mark.parametrize("command", ["build-graphs", "weak-predict", "suite", "summarize", "reproduce",
                                    "fit-affine", "extract-patches", "image-covariates", "evaluate-states", "contrast-bounds",
                                    "stress-case", "bound-risk", "theory-diagnostics"])
def test_new_cli_help(command):
    with pytest.raises(SystemExit) as error:
        main([command, "--help"])
    assert error.value.code == 0
