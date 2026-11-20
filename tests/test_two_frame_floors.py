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
