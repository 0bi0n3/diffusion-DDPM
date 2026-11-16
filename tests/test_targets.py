"""The target interface (#9): --target on train/sample/score/run, recorded and enforced. Through the CLI only."""
import json

import numpy as np
import pytest
import torch

from changemap.cli import main as cli_main
from grid import main
from test_end_to_end import run_pipeline, tier  # noqa: F401  (fixture)


def test_target_recorded_in_config_and_report(tier, tmp_path):
    _, rep = run_pipeline(tier, tmp_path)
    assert json.loads((tmp_path / "run" / "config.json").read_text())["target"] == "grid_map"
    assert rep["target"] == "grid_map"


def _exits_nonzero(argv):
    with pytest.raises(SystemExit) as e:
        main(argv)
    assert e.value.code not in (0, None)


def test_sample_refuses_a_run_made_for_another_target(tier, tmp_path):
    run_pipeline(tier, tmp_path)
    cfg_path = tmp_path / "run" / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg_path.write_text(json.dumps(dict(cfg, target="masked_frame")))
    _exits_nonzero(["sample", "--run", str(tmp_path / "run"), "--tier", str(tier), "--split", "val", "--k", "1",
                    "--steps", "2", "--out", str(tmp_path / "s2")])


def test_score_refuses_samples_made_for_another_target(tier, tmp_path):
    samples, _ = run_pipeline(tier, tmp_path)
    path = next(samples.glob("*.npz"))
    z = dict(np.load(path))
    assert str(z["target"]) == "grid_map"
    np.savez(path, **dict(z, target="masked_frame"))
    _exits_nonzero(["score", "--tier", str(tier), "--split", "val", "--lpips-dir", str(tmp_path),
                    "--samples", str(samples), "--out", str(tmp_path / "r2")])


def test_resume_refuses_a_changed_target(tier, tmp_path):
    out = tmp_path / "o"
    argv = ["train", "--tier", str(tier), "--steps", "4", "--batch", "4", "--hidden", "32", "--depth", "2",
            "--heads", "2", "--out", str(out), "--val-n", "2", "--val-sample-steps", "2"]
    assert main(argv + ["--stop-after", "2"]) == 0
    st = torch.load(out / "last.pt", map_location="cpu")
    st["config"]["target"] = "masked_frame"
    torch.save(st, out / "last.pt")
    _exits_nonzero(argv)


def test_unknown_target_refused(tier, tmp_path):
    _exits_nonzero(["train", "--tier", str(tier), "--target", "bogus", "--steps", "1", "--out", str(tmp_path / "o")])
