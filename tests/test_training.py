"""Ticket #6: training robustness, through the CLI seam; assertions on files."""
import hashlib
import json
import subprocess
import sys

import pytest
import torch

from changemap.cli import main
from fake_wandb import FakeWandb

REGION = [[100.0, 100.0, 500.0, 400.0]]
ROWS = [{"id": f"w{i}", "target_status": "target", "regions": [[100.0 + 40 * i, 100.0, 500.0 + 40 * i, 400.0]],
         "box_ok_50_20": [True]} for i in range(6)] + [{"id": "nc", "target_status": "no_change", "regions": [],
                                                       "box_ok_50_20": []}]


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


def weights(path, key=None):
    sd = torch.load(path, map_location="cpu")
    if key == "ema":
        return sd["ema"]["module"]
    return sd[key] if key else sd


def same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def record(out):
    return json.loads((out / "run_record.json").read_text())


def test_stopped_and_resumed_equals_uninterrupted(tier, tmp_path):
    full, part = tmp_path / "full", tmp_path / "part"
    assert train(tier, full, 6, "--val-every", "3") == 0
    assert train(tier, part, 6, "--val-every", "3", "--stop-after", "2") == 0
    assert record(part)["status"] == "interrupted" and record(part)["step"] == 2
    assert (part / "last.pt").exists()
    assert train(tier, part, 6, "--val-every", "3") == 0          # resumes from last.pt
    assert record(part)["status"] == "finished" and record(part)["step"] == 6
    for key in ("model", "ema"):
        assert same(weights(full / "last.pt", key), weights(part / "last.pt", key))
    assert same(weights(full / "model.pt"), weights(part / "model.pt"))
    log_f, log_p = (json.loads((d / "log.json").read_text()) for d in (full, part))
    assert log_f == log_p                                         # losses, lrs and validation entries alike
    assert [e["step"] for e in log_p["train"]] == list(range(1, 7))   # no step twice, none missing


def test_resume_of_a_finished_run_changes_nothing(tier, tmp_path):
    out = tmp_path / "o"
    train(tier, out, 3)
    before = weights(out / "model.pt")
    assert train(tier, out, 3) == 0
    assert same(before, weights(out / "model.pt"))


def test_resume_with_different_config_refused(tier, tmp_path):
    out = tmp_path / "o"
    train(tier, out, 4, "--stop-after", "2")
    with pytest.raises(SystemExit) as e:
        train(tier, out, 4, "--lr", "0.5")
    assert e.value.code not in (0, None)


def test_ema_is_the_sampling_weights(tier, tmp_path):
    out = tmp_path / "o"
    train(tier, out, 4)
    ema, raw = weights(out / "last.pt", "ema"), weights(out / "last.pt", "model")
    assert same(weights(out / "model.pt"), ema) and not same(ema, raw)
    assert main(["sample", "--run", str(out), "--tier", str(tier), "--split", "val", "--k", "1", "--steps", "2",
                 "--out", str(tmp_path / "s")]) == 0


def test_validation_entries_at_the_interval(tier, tmp_path):
    out = tmp_path / "o"
    train(tier, out, 7, "--val-every", "3")
    val = json.loads((out / "log.json").read_text())["validation"]
    assert [e["step"] for e in val] == [3, 6, 7]                  # every 3 steps, and the last step
    for e in val:
        assert e["val_loss"] > 0 and 0 <= e["val_grid_mse"] <= 1
        assert not any("iou" in k.lower() for k in e)             # no matched mIoU: the scoring metric moved on


def test_validation_is_repeatable(tier, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    train(tier, a, 2, "--val-every", "1")
    train(tier, b, 2, "--val-every", "1")
    assert json.loads((a / "log.json").read_text())["validation"] == json.loads((b / "log.json").read_text())["validation"]


def test_cosine_lr_with_warmup(tier, tmp_path):
    out = tmp_path / "o"
    train(tier, out, 20, "--lr", "0.001", "--warmup", "4")
    lrs = [e["lr"] for e in json.loads((out / "log.json").read_text())["train"]]
    assert lrs[0] < lrs[3] == pytest.approx(0.001)                # linear warm-up to the peak
    assert all(a >= b for a, b in zip(lrs[3:], lrs[4:]))          # then cosine decay
    assert lrs[-1] < 0.05 * lrs[3]


def test_run_record(tier, tmp_path):
    (tier / "test_targets.jsonl").write_text(json.dumps(ROWS[0]) + "\n")
    out = tmp_path / "o"
    train(tier, out, 2)
    rec = record(out)
    head = subprocess.run(["git", "-C", str(__import__("pathlib").Path(__file__).parents[1]), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert rec["code_commit"] == head
    sha = {n: hashlib.sha256((tier / n).read_bytes()).hexdigest() for n in ("train_targets.jsonl", "val_targets.jsonl")}
    assert rec["data_sha256"] == sha                              # test split never read, so never hashed
    assert rec["config"]["arm"] == "unconditional" and rec["config"]["steps"] == 2
    assert rec["device"] and rec["wall_time_seconds"] > 0
    assert rec["peak_memory"]["host_rss_mb"] > 0
    assert rec["status"] == "finished"


def test_run_record_commit_from_environment(tier, tmp_path, monkeypatch):
    monkeypatch.setenv("CHANGEMAP_COMMIT", "f" * 40)
    train(tier, tmp_path / "o", 1)
    assert record(tmp_path / "o")["code_commit"] == "f" * 40


def test_wandb_logging_with_fake(tier, tmp_path, monkeypatch):
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    out = tmp_path / "o"
    args = ("--val-every", "2", "--log-every", "1", "--wandb-mode", "online")
    train(tier, out, 4, *args, "--stop-after", "2")
    train(tier, out, 4, *args)
    assert fake.inits[0]["project"] and fake.inits[0]["id"] == fake.inits[1]["id"]   # same W&B run after resume
    steps = [d["step"] for d in fake.logged if "train/loss" in d]
    assert steps == [1, 2, 3, 4]                                  # never twice
    val = [d for d in fake.logged if "val/loss" in d]
    assert [d["step"] for d in val] == [2, 4] and all("val/grid_mse" in d for d in val)
    assert fake.finished == 2


def test_wandb_failure_does_not_stop_training(tier, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", FakeWandb(fail=("init", "log")))
    out = tmp_path / "o"
    assert train(tier, out, 2, "--wandb-mode", "online") == 0
    assert record(out)["status"] == "finished"


def test_wandb_off_by_default(tier, tmp_path, monkeypatch):
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    train(tier, tmp_path / "o", 2)
    assert fake.inits == []


# ---- job script (dry run only: nothing is submitted) -----------------------------------------------------------

def render(tmp_path, *extra):
    from changemap import cluster
    script = tmp_path / "train.sbatch"
    rc = cluster.main(["render", "--arm", "unconditional", "--sha", "abc1234", "--tier", "/data/tier_1k",
                       "--out", "/runs/unconditional", "--steps", "20000", "--script", str(script), *extra])
    return rc, script


def test_job_script_renders_with_resume_and_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("nothing may be run or submitted"))
    rc, script = render(tmp_path)
    assert rc == 0
    text = script.read_text()
    assert "#SBATCH --time=24:00:00" in text and "#SBATCH --gpus=1" in text
    assert "#SBATCH --partition=a100" in text and "#SBATCH --mem=" in text
    assert "--signal=B:USR1@" in text
    assert "--resume auto" in text and "--arm unconditional" in text and "--steps 20000" in text
    assert "--device cuda" in text and "--wandb-mode online" in text
    assert "/runs/unconditional" in text and "CHANGEMAP_COMMIT" in text
    limit = int(text.split("--time-limit ")[1].split()[0])
    assert 0 < limit < 24 * 3600                                  # the program stops itself before Slurm does


def test_job_script_is_valid_bash(tmp_path):
    _, script = render(tmp_path)
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0


def test_job_script_refuses_over_24h(tmp_path):
    with pytest.raises(SystemExit) as e:
        render(tmp_path, "--time", "24:00:01")
    assert e.value.code not in (0, None)


def test_submit_dry_run_prints_command_and_submits_nothing(tmp_path, monkeypatch, capsys):
    from changemap import cluster
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("nothing may be run or submitted"))
    rc = cluster.main(["--home", str(tmp_path), "submit", "--dry-run", "--arm", "frame_caption", "--sha", "abc1234",
                       "--tier", "/data/tier_1k", "--out", "/runs/fc", "--steps", "20000",
                       "--dependency", "afterany:123"])
    out = capsys.readouterr().out
    assert rc == 0 and "sbatch" in out and "--dependency=afterany:123" in out and "--time=24:00:00" in out
    assert not (tmp_path / "ledger.tsv").exists()


def test_context_arm_resume_and_record_hashes_context(tier, tmp_path):
    from test_arms import write_context
    for split in ("train", "val"):
        write_context(tier, split, [{"id": f"{split}_{r['id']}"} for r in ROWS], 3)
    argv = ["--tier", str(tier), "--arm", "frame_caption", "--steps", "4", "--batch", "4", "--hidden", "32",
            "--depth", "2", "--heads", "2", "--val-n", "3", "--val-every", "2", "--val-sample-steps", "2"]
    full, part = tmp_path / "full", tmp_path / "part"
    assert main(["train", *argv, "--out", str(full)]) == 0
    assert main(["train", *argv, "--out", str(part), "--stop-after", "1"]) == 0
    assert main(["train", *argv, "--out", str(part)]) == 0
    assert same(weights(full / "model.pt"), weights(part / "model.pt"))
    assert {"train_context.npz", "val_context.npz"} <= set(record(part)["data_sha256"])
    assert [e["step"] for e in json.loads((part / "log.json").read_text())["validation"]] == [2, 4]
