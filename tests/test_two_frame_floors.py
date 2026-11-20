"""Two-frame floors (Mean-Map and Previous-Target) for both `change_map_pixels` and `map_latent` (#18)."""
import json

import numpy as np
import pytest

from changemap.cli import main as cli
from test_change_map_pixels import H, W, pair_row, true_map, write_two_frame_tier

BOX_A = [192, 108, 576, 324]        # 80x45 = 3,600 px at 224x400 (total 89,600 px)
BOX_B = [960, 540, 1344, 756]       # 80x45 = 3,600 px at 224x400, disjoint from BOX_A


def _write_pixel_arm(tmp_path, name, ids, maps):
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    np.savez(d / f"{name}.npz", ids=np.array(ids), maps=maps.astype(np.float32), arm=name, target="change_map_pixels")
    return d


def _score(tier, tmp_path, target, sample_dirs=(), out="rep"):
    argv = ["score", "--tier", str(tier), "--split", "val", "--target", target,
            "--lpips-dir", str(tmp_path), "--out", str(tmp_path / out)]
    for d in sample_dirs:
        argv += ["--samples", str(d)]
    assert cli(argv) == 0
    return json.loads((tmp_path / out / "report.json").read_text())


def test_mean_map_floor_pixel_matches_independent_truth(tmp_path):
    """3 non-dropped train pairs (2 with BOX_A, 1 no_change) + 1 dropped full-screen pair -> independent mean map is
    2/3 inside BOX_A (3,600 px) and 0 outside (86,000 px)."""
    train_rows = [
        pair_row("tr_0-1", "tr0", [0, 1], "target", [BOX_A]),
        pair_row("tr_1-2", "tr0", [1, 2], "target", [BOX_A]),
        pair_row("tr_2-3", "tr0", [2, 3], "no_change"),
        pair_row("tr_3-4", "tr0", [3, 4], "dropped", [[0, 0, 1920, 1080]]),
    ]
    val_rows = [
        pair_row("va_0-1", "va0", [0, 1], "target", [BOX_A]),
        pair_row("va_1-2", "va0", [1, 2], "no_change"),
    ]
    tier = write_two_frame_tier(tmp_path / "tier", train_rows=train_rows, val_rows=val_rows)
    arm = _write_pixel_arm(tmp_path, "frame_caption", ["va_0-1", "va_1-2"],
                           np.stack([true_map([BOX_A]), np.zeros((H, W), np.float32)])[:, None])

    rep = _score(tier, tmp_path, "change_map_pixels", [arm], out="rep_mean")
    assert "mean_map_floor" in rep["rows"]
    mm = rep["rows"]["mean_map_floor"]
    assert mm["per_window"]["va_0-1"]["mse"] == [pytest.approx(1 / 224, rel=1e-6)]
    assert mm["per_window"]["va_0-1"]["spill"] == [pytest.approx(0.0, abs=1e-12)]
    assert mm["per_window"]["va_1-2"]["mse"] == [pytest.approx(1 / 56, rel=1e-6)]
    assert mm["per_window"]["va_1-2"]["spill"] == [pytest.approx(1.0, abs=1e-12)]
    assert mm["summary"]["all"]["single"]["mse"] == pytest.approx(5 / 448, rel=1e-6)
    cmp_pairs = {(c["a"], c["b"], c["metric"]) for c in rep["comparisons"]}
    assert ("frame_caption:single", "mean_map_floor:single", "mse") in cmp_pairs
    assert ("frame_caption:single", "mean_map_floor:single", "lpips") in cmp_pairs


def test_previous_target_floor_same_window_fallback_and_missing_count(tmp_path):
    """Previous-Target takes Ft-1 -> Ft from the same Window when present (including dropped -> missing and
    no_change -> usable empty), falls back to `<split>_previous.jsonl` for the first pair of a Window, and counts
    Step Pairs with no usable previous pair in report.json and report.md."""
    train_rows = [
        pair_row("tr_0-1", "tr0", [0, 1], "target", [BOX_A]),
        pair_row("tr_1-2", "tr0", [1, 2], "no_change"),
    ]
    val_rows = [
        pair_row("wA_10-11", "wA", [10, 11], "target", [BOX_A]),
        pair_row("wA_11-12", "wA", [11, 12], "target", [BOX_A]),
        pair_row("wA_12-13", "wA", [12, 13], "dropped", [[0, 0, 1920, 1080]]),
        pair_row("wA_13-14", "wA", [13, 14], "target", [BOX_A]),
        pair_row("wB_20-21", "wB", [20, 21], "target", [BOX_B]),
        pair_row("wB_21-22", "wB", [21, 22], "no_change"),
        pair_row("wB_22-23", "wB", [22, 23], "no_change"),
    ]
    tier = write_two_frame_tier(tmp_path / "tier", train_rows=train_rows, val_rows=val_rows)
    prev_records = [
        {"id": "wA_10-11", "previous": {"regions": [BOX_A], "box_ok_50_20": [True]}},
        # Conflicting record for a non-first pair: same-Window previous pair (BOX_A) must win over tier record (BOX_B)
        {"id": "wA_11-12", "previous": {"regions": [BOX_B], "box_ok_50_20": [True]}},
    ]
    (tier / "val_previous.jsonl").write_text("\n".join(json.dumps(r) for r in prev_records) + "\n")

    scored_ids = [r["id"] for r in val_rows if r["target_status"] != "dropped"]
    arm = _write_pixel_arm(tmp_path, "frame_caption", scored_ids, np.zeros((len(scored_ids), 1, H, W), np.float32))
    rep = _score(tier, tmp_path, "change_map_pixels", [arm], out="rep_prev")

    assert list(rep["rows"]) == ["frame_caption", "all_black_floor", "mean_map_floor", "previous_target_floor"]
    assert rep["n_no_previous"] == 2
    pt = rep["rows"]["previous_target_floor"]
    assert pt["n_no_previous"] == 2
    pw = pt["per_window"]
    assert pw["wA_10-11"]["mse"] == [0.0] and pw["wA_10-11"]["lpips"] == [0.0]         # first pair -> tier record
    assert pw["wA_11-12"]["mse"] == [0.0] and pw["wA_11-12"]["lpips"] == [0.0]         # same-Window previous wins
    assert pw["wA_13-14"]["mse"] == [pytest.approx(9 / 224, rel=1e-6)]                 # dropped previous -> all black
    assert pw["wB_20-21"]["mse"] == [pytest.approx(9 / 224, rel=1e-6)]                 # absent previous -> all black
    assert pw["wB_21-22"]["mse"] == [pytest.approx(9 / 224, rel=1e-6)]                 # same-Window BOX_B vs no_change
    assert pw["wB_21-22"]["spill"] == [pytest.approx(1.0, abs=1e-12)]
    assert pw["wB_22-23"]["mse"] == [0.0]                                              # same-Window no_change -> usable

    md = (tmp_path / "rep_prev" / "report.md").read_text()
    assert "Previous-Target Floor:" in md and "2 of 6" in md
    cmp_pairs = {(c["a"], c["b"], c["metric"]) for c in rep["comparisons"]}
    for floor in ("all_black_floor", "mean_map_floor", "previous_target_floor"):
        assert ("frame_caption:single", f"{floor}:single", "mse") in cmp_pairs
        assert ("frame_caption:single", f"{floor}:single", "region_lpips") in cmp_pairs

