"""Conditioning Arms end to end through the CLI: context swaps, parameter counts, shared noise, `run` with arms."""
import json
import shutil

import numpy as np
import pytest

from changemap.cli import main
from test_end_to_end import TRAIN, VAL, tier  # noqa: F401

ARMS = ["unconditional", "caption_only", "frame_only", "frame_caption"]
SMALL = ["--hidden", "32", "--depth", "2", "--heads", "2"]


def write_context(tier, split, rows, seed):
    """<split>_context.npz: ids (n,), f8 (n, 28, 50, 768), caption (n, 77, 768); one row per Window incl. Dropped."""
    g = np.random.default_rng(seed)
    n = len(rows)
    np.savez(tier / f"{split}_context.npz", ids=np.array([r["id"] for r in rows]),
             f8=g.normal(size=(n, 28, 50, 768)).astype(np.float16),
             caption=g.normal(size=(n, 77, 768)).astype(np.float16))


@pytest.fixture
def ctier(tier):
    write_context(tier, "train", TRAIN, 1)
    write_context(tier, "val", VAL, 2)
    return tier


def swapped(ctier, tmp_path, field):
    """A copy of the tier whose val context has `field` rolled by one Window (same ids, other content)."""
    d = tmp_path / f"swap_{field}"
    shutil.copytree(ctier, d)
    z = dict(np.load(ctier / "val_context.npz"))
    z[field] = np.roll(z[field], 1, axis=0)
    np.savez(d / "val_context.npz", **z)
    return d


def train(tier, arm, out, steps=4):
    assert main(["train", "--tier", str(tier), "--arm", arm, "--steps", str(steps), "--batch", "4", "--lr", "1e-3",
                 *SMALL, "--out", str(out)]) == 0
    return out


def sample(run, tier, out, seed=35, steps=2):
    assert main(["sample", "--run", str(run), "--tier", str(tier), "--split", "val", "--k", "2",
                 "--steps", str(steps), "--seed", str(seed), "--out", str(out)]) == 0
    return np.load(next(out.glob("*.npz")))["maps"]


@pytest.fixture
def runs(ctier, tmp_path):
    return {a: train(ctier, a, tmp_path / "runs" / a) for a in ARMS}


def test_context_swaps_change_only_the_arms_that_use_them(ctier, runs, tmp_path):
    f8_swap, cap_swap = swapped(ctier, tmp_path, "f8"), swapped(ctier, tmp_path, "caption")
    for arm, uses_f8, uses_cap in (("unconditional", 0, 0), ("caption_only", 0, 1), ("frame_only", 1, 0),
                                   ("frame_caption", 1, 1)):
        base = sample(runs[arm], ctier, tmp_path / f"b_{arm}")
        with_f8 = sample(runs[arm], f8_swap, tmp_path / f"f_{arm}")
        with_cap = sample(runs[arm], cap_swap, tmp_path / f"c_{arm}")
        assert np.array_equal(base, with_f8) == (not uses_f8), arm
        assert np.array_equal(base, with_cap) == (not uses_cap), arm


def test_all_arms_have_the_same_parameter_count(runs):
    counts = {a: json.loads((r / "config.json").read_text())["n_params"] for a, r in runs.items()}
    assert len(set(counts.values())) == 1 and counts["unconditional"] > 0


def test_unconditional_needs_no_context_files(tier, tmp_path):    # plain tier: no *_context.npz at all
    run = train(tier, "unconditional", tmp_path / "u")
    assert sample(run, tier, tmp_path / "s").shape == (5, 2, 28, 50)


def test_context_arm_without_context_file_refused(tier, tmp_path):
    with pytest.raises(SystemExit) as e:
        main(["train", "--tier", str(tier), "--arm", "frame_caption", "--steps", "1", "--out", str(tmp_path / "r")])
    assert e.value.code not in (0, None)


def test_arms_share_per_window_noise(runs, ctier, tmp_path):
    # zero Euler steps returns the starting noise itself
    start = {a: sample(runs[a], ctier, tmp_path / f"n_{a}", seed=35, steps=0) for a in ARMS}
    other = sample(runs["frame_caption"], ctier, tmp_path / "n_other", seed=36, steps=0)
    for a in ARMS[1:]:
        assert np.array_equal(start[a], start["unconditional"])
    assert not np.array_equal(other, start["frame_caption"])


def test_run_accepts_list_of_arms_and_scores_them_together(ctier, tmp_path):
    out = tmp_path / "all"
    assert main(["run", "--tier", str(ctier), "--arms", *ARMS, "--steps", "2", "--batch", "2", *SMALL,
                 "--k", "1", "--sample-steps", "2", "--out", str(out)]) == 0
    rep = json.loads((out / "report" / "report.json").read_text())
    assert set(ARMS) <= set(rep["rows"])
