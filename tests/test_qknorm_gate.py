"""Ticket #19: QK-normalisation option and stability gate, tested through the CLI seam."""
import json

import numpy as np
import pytest
import torch

from changemap.cli import main as cli_main
from grid import main
from test_arms import write_context

ROWS = [
    {"id": f"w{i}", "target_status": "target", "regions": [[100.0 + 40 * i, 100.0, 500.0 + 40 * i, 400.0]],
     "box_ok_50_20": [True]}
    for i in range(6)
] + [{"id": "nc", "target_status": "no_change", "regions": [], "box_ok_50_20": []}]


@pytest.fixture
def tier(tmp_path):
    d = tmp_path / "tier"
    d.mkdir()
    for split in ("train", "val"):
        rows = [dict(r, id=f"{split}_{r['id']}") for r in ROWS]
        (d / f"{split}_targets.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return d


def train(tier, out, steps, *extra):
    return main(["train", "--tier", str(tier), "--arm", "unconditional", "--steps", str(steps), "--batch", "4",
                 "--hidden", "32", "--depth", "2", "--heads", "2", "--out", str(out), "--val-n", "4",
                 "--val-sample-steps", "2", *extra])


def test_qk_norm_recorded_and_refused_on_resume(tier, tmp_path):
    out_default = tmp_path / "default"
    assert train(tier, out_default, 2, "--stop-after", "1") == 0
    assert json.loads((out_default / "config.json").read_text())["qk_norm"] is False
    with pytest.raises(SystemExit) as e:
        train(tier, out_default, 2, "--qk-norm", "on")
    assert e.value.code not in (0, None)

    out_on = tmp_path / "on"
    assert train(tier, out_on, 2, "--stop-after", "1", "--qk-norm", "on") == 0
    assert json.loads((out_on / "config.json").read_text())["qk_norm"] is True
    with pytest.raises(SystemExit) as e:
        train(tier, out_on, 2, "--qk-norm", "off")
    assert e.value.code not in (0, None)


def test_qk_norm_on_trains_samples_and_differs(tier, tmp_path):
    for split in ("train", "val"):
        write_context(tier, split, [{"id": f"{split}_{r['id']}"} for r in ROWS], 3)
    off_run, on_run = tmp_path / "off", tmp_path / "on"
    off_s, on_s = tmp_path / "s_off", tmp_path / "s_on"
    common = ["--tier", str(tier), "--arm", "frame_caption", "--steps", "3", "--batch", "4",
              "--hidden", "32", "--depth", "2", "--heads", "2", "--val-n", "3", "--val-sample-steps", "2"]
    assert main(["train", *common, "--out", str(off_run), "--qk-norm", "off"]) == 0
    assert main(["train", *common, "--out", str(on_run), "--qk-norm", "on"]) == 0
    assert main(["sample", "--run", str(off_run), "--tier", str(tier), "--split", "val", "--k", "1",
                 "--steps", "2", "--out", str(off_s)]) == 0
    assert main(["sample", "--run", str(on_run), "--tier", str(tier), "--split", "val", "--k", "1",
                 "--steps", "2", "--out", str(on_s)]) == 0

    sd_off = torch.load(off_run / "model.pt", map_location="cpu")
    sd_on = torch.load(on_run / "model.pt", map_location="cpu")
    assert not any("q_norm" in k or "k_norm" in k for k in sd_off)
    for i in range(2):
        for prefix in (f"blocks.{i}.attn", f"cross.{i}"):
            for norm in ("q_norm.weight", "k_norm.weight"):
                key = f"{prefix}.{norm}"
                assert key in sd_on and sd_on[key].shape == (16,)

    maps_off = np.load(off_s / "frame_caption.npz")["maps"]
    maps_on = np.load(on_s / "frame_caption.npz")["maps"]
    assert not np.array_equal(maps_off, maps_on)

    # Resume with qk_norm on is bit-exact
    part = tmp_path / "on_part"
    assert main(["train", *common, "--out", str(part), "--qk-norm", "on", "--stop-after", "1"]) == 0
    assert main(["train", *common, "--out", str(part), "--qk-norm", "on"]) == 0
    sd_part = torch.load(part / "model.pt", map_location="cpu")
    assert sd_on.keys() == sd_part.keys() and all(torch.equal(sd_on[k], sd_part[k]) for k in sd_on)


def test_qk_norm_off_is_byte_identical(tier, tmp_path):
    default_run, off_run = tmp_path / "default", tmp_path / "off"
    default_s, off_s = tmp_path / "s_default", tmp_path / "s_off"
    assert train(tier, default_run, 3) == 0
    assert train(tier, off_run, 3, "--qk-norm", "off") == 0
    assert main(["sample", "--run", str(default_run), "--tier", str(tier), "--split", "val", "--k", "1",
                 "--steps", "2", "--out", str(default_s)]) == 0
    assert main(["sample", "--run", str(off_run), "--tier", str(tier), "--split", "val", "--k", "1",
                 "--steps", "2", "--out", str(off_s)]) == 0

    sd_default = torch.load(default_run / "model.pt", map_location="cpu")
    sd_off = torch.load(off_run / "model.pt", map_location="cpu")
    assert list(sd_default.keys()) == list(sd_off.keys())
    assert all(torch.equal(sd_default[k], sd_off[k]) for k in sd_default)
    assert not any("q_norm" in k or "k_norm" in k for k in sd_default)

    from changemap.model import build_model
    baseline_keys = list(build_model("unconditional", 32, 2, 2, 1).state_dict().keys())
    assert list(sd_off.keys()) == baseline_keys

    maps_default = np.load(default_s / "unconditional.npz")["maps"]
    maps_off = np.load(off_s / "unconditional.npz")["maps"]
    assert maps_default.tobytes() == maps_off.tobytes()
    assert (default_run / "log.json").read_bytes() == (off_run / "log.json").read_bytes()


def _write_log(run_dir, train_entries):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "log.json").write_text(json.dumps({"train": train_entries, "validation": []}) + "\n")


def test_gate_fails_loss_spike_and_names_first_bad_step(tmp_path, capsys):
    run = tmp_path / "loss_spike"
    _write_log(run, [
        {"step": 1, "loss": 1.0, "lr": 1e-4, "grad_norm": 1.0},
        {"step": 2, "loss": 0.5, "lr": 1e-4, "grad_norm": 1.0},
        {"step": 3, "loss": 0.2, "lr": 1e-4, "grad_norm": 1.0},
        {"step": 4, "loss": 2.0, "lr": 1e-4, "grad_norm": 1.0},    # exact 10x min (0.2): still ok
        {"step": 5, "loss": 2.01, "lr": 1e-4, "grad_norm": 1.0},   # > 10x min: first bad step
        {"step": 6, "loss": 9.0, "lr": 1e-4, "grad_norm": 1.0},
    ])
    rc = cli_main(["gate", "--run", str(run)])
    captured = capsys.readouterr()
    msg = captured.out + captured.err
    assert rc != 0
    assert "step 5" in msg and "step 6" not in msg


def test_gate_fails_grad_norm_spike_and_names_first_bad_step(tmp_path, capsys):
    run = tmp_path / "grad_spike"
    _write_log(run, [
        {"step": 1, "loss": 1.0, "lr": 1e-4, "grad_norm": 1.0},
        {"step": 2, "loss": 0.9, "lr": 1e-4, "grad_norm": 50.0},
        {"step": 3, "loss": 0.8, "lr": 1e-4, "grad_norm": 100.0},   # exact 100: still ok
        {"step": 4, "loss": 0.7, "lr": 1e-4, "grad_norm": 100.5},   # > 100: first bad step
        {"step": 5, "loss": 0.7, "lr": 1e-4, "grad_norm": 400.0},
    ])
    rc = cli_main(["gate", "--run", str(run)])
    captured = capsys.readouterr()
    msg = captured.out + captured.err
    assert rc != 0
    assert "step 4" in msg and "step 5" not in msg


def test_gate_passes_clean_cpu_run(tier, tmp_path, capsys):
    for name, extra in (("off", ()), ("on", ("--qk-norm", "on"))):
        out = tmp_path / name
        assert train(tier, out, 4, *extra) == 0
        rc = cli_main(["gate", "--run", str(out)])
        captured = capsys.readouterr()
        assert rc == 0
        assert "PASS" in captured.out
    # Direct path to log.json is also accepted
    assert cli_main(["gate", "--run", str(tmp_path / "off" / "log.json")]) == 0


def test_gate_errors_on_missing_or_empty_log(tmp_path):
    with pytest.raises(SystemExit) as e1:
        cli_main(["gate", "--run", str(tmp_path / "missing")])
    assert e1.value.code not in (0, None)

    empty = tmp_path / "empty"
    _write_log(empty, [])
    with pytest.raises(SystemExit) as e2:
        cli_main(["gate", "--run", str(empty)])
    assert e2.value.code not in (0, None)





