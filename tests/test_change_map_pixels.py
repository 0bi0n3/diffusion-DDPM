"""Tracer bullet for the `change_map_pixels` target (#16) end to end through the CLI on a synthetic two-frame tier."""
import json

import numpy as np
import pytest
from PIL import Image

from changemap.cli import main as cli

H, W = 224, 400
SMALL = ["--hidden", "32", "--depth", "2", "--heads", "2"]


def px(v, scale):
    return int(np.floor(v * scale + 0.5))


def pixel_box(b):
    """Exact scaling: x * 400/1920, y * 224/1080, nearest-pixel edges, at least one pixel wide and high."""
    x0, y0, x1, y1 = px(b[0], W / 1920), px(b[1], H / 1080), px(b[2], W / 1920), px(b[3], H / 1080)
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


def true_map(regions):
    m = np.zeros((H, W), np.float32)
    for b in regions:
        x0, y0, x1, y1 = pixel_box(b)
        m[y0:y1, x0:x1] = 1.0
    return m


def pair_row(pid, source_win, frames_ab, status, regions=(), ok=None):
    ok = [True] * len(regions) if ok is None else list(ok)
    return {"id": pid, "source_window_id": source_win, "frames": list(frames_ab),
            "target_status": status, "regions": list(regions), "box_ok_50_20": ok}


def index_row(r):
    regs = [b for b, ok in zip(r["regions"], r["box_ok_50_20"]) if ok] if r["target_status"] == "target" else []
    win = f"{r['id']}@{r['source_window_id']}"
    return {"id": f"pixel_difference/last_step/goal_subgoal_action/{win}",
            "window": win, "file": f"map_{win}.png", "n_regions": len(regs),
            "regions_px_1920x1080": [[float(v) for v in b] for b in regs]}


TRAIN = [
    pair_row("t2_10-11", "t2_10", [10, 11], "target", [[192, 108, 576, 324]]),
    pair_row("t2_11-12", "t2_10", [11, 12], "no_change"),
    pair_row("t2_20-21", "t2_20", [20, 21], "target", [[960, 540, 1344, 756]]),
    pair_row("t2_21-22", "t2_20", [21, 22], "dropped", [[0, 0, 1920, 1080]]),
    pair_row("t2_30-31", "t2_30", [30, 31], "target", [[100, 100, 300, 200], [50, 50, 60, 60]], [True, False]),
]

VAL = [
    pair_row("t1_10-11", "t1_10", [10, 11], "target", [[192, 108, 576, 324]]),
    pair_row("t1_11-12", "t1_10", [11, 12], "target", [[960, 540, 1344, 756]]),
    pair_row("t1_20-21", "t1_20", [20, 21], "no_change"),
    pair_row("t1_21-22", "t1_20", [21, 22], "dropped", [[0, 0, 1920, 1080]]),
    pair_row("t1_30-31", "t1_30", [30, 31], "target", [[642, 442, 1028, 582], [498, 298, 514, 312]], [True, False]),
]

SCORED_VAL = [r for r in VAL if r["target_status"] != "dropped"]
VAL_IDS = [r["id"] for r in SCORED_VAL]
NO_CHANGE_IDS = {"t1_20-21"}


def write_context(tier, split, rows, seed):
    g = np.random.default_rng(seed)
    ids = np.array([r["id"] for r in rows])
    f8 = g.standard_normal((len(rows), 28, 50, 768)).astype(np.float16)
    cap = g.standard_normal((len(rows), 77, 768)).astype(np.float16)
    np.savez(tier / f"{split}_context.npz", ids=ids, f8=f8, caption=cap)


def write_two_frame_tier(d, train_rows=TRAIN, val_rows=VAL):
    d.mkdir(parents=True, exist_ok=True)
    for split, rows, seed in (("train", train_rows, 11), ("val", val_rows, 12)):
        (d / f"{split}_targets.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        idx_rows = [index_row(r) for r in rows if r["target_status"] != "dropped"]
        (d / f"{split}_index.jsonl").write_text("\n".join(json.dumps(r) for r in idx_rows) + "\n")
        write_context(d, split, rows, seed)
    return d


@pytest.fixture
def tftier(tmp_path):
    return write_two_frame_tier(tmp_path / "tier")


def exits_nonzero(argv):
    with pytest.raises(SystemExit) as e:
        cli(argv)
    assert e.value.code not in (0, None)


def test_change_map_pixels_train_sample_score_end_to_end(tftier, tmp_path):
    run, samples, report_dir = tmp_path / "run", tmp_path / "samples", tmp_path / "report"
    assert cli(["train", "--tier", str(tftier), "--target", "change_map_pixels", "--arm", "frame_caption",
                "--steps", "3", "--batch", "2", *SMALL, "--out", str(run)]) == 0
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["target"] == "change_map_pixels" and cfg["arm"] == "frame_caption"
    rec = json.loads((run / "run_record.json").read_text())
    assert {"train_index.jsonl", "val_index.jsonl"} <= set(rec["data_sha256"])

    assert cli(["sample", "--run", str(run), "--tier", str(tftier), "--split", "val",
                "--target", "change_map_pixels", "--k", "2", "--steps", "2", "--out", str(samples)]) == 0
    z = np.load(samples / "frame_caption.npz")
    assert str(z["target"]) == "change_map_pixels"
    assert list(z["ids"]) == VAL_IDS
    assert "t1_21-22" not in list(z["ids"])                        # Dropped Pair is absent
    assert z["maps"].shape == (len(VAL_IDS), 2, H, W)
    assert z["maps"].min() >= 0 and z["maps"].max() <= 1

    # sample and score refuse another target
    exits_nonzero(["sample", "--run", str(run), "--tier", str(tftier), "--split", "val",
                   "--target", "masked_frame", "--k", "1", "--steps", "2", "--out", str(tmp_path / "bad_s")])
    exits_nonzero(["score", "--tier", str(tftier), "--split", "val", "--target", "grid_map",
                   "--lpips-dir", str(tmp_path), "--samples", str(samples), "--out", str(tmp_path / "bad_r")])

    assert cli(["score", "--tier", str(tftier), "--split", "val", "--target", "change_map_pixels",
                "--lpips-dir", str(tmp_path), "--samples", str(samples), "--out", str(report_dir)]) == 0
    rep = json.loads((report_dir / "report.json").read_text())
    assert rep["target"] == "change_map_pixels"
    assert rep["window_ids"] == VAL_IDS
    assert list(rep["rows"]) == ["frame_caption", "all_black_floor"]
    fc = rep["rows"]["frame_caption"]
    assert fc["k"] == 2
    assert set(fc["per_window"]["t1_10-11"]) == {"lpips", "region_lpips", "mse", "spill"}
    assert set(fc["summary"]) == {"all", "change", "no_change"}
    assert fc["summary"]["all"]["n"] == 4
    assert fc["summary"]["change"]["n"] == 3
    assert fc["summary"]["no_change"]["n"] == 1
    assert fc["summary"]["all"]["single"]["psnr"] is not None or fc["summary"]["all"]["single"]["mse"] == 0
    # No Change Step Pair has all-black truth (All-Black Floor scores 0 MSE, 0 LPIPS, None region_lpips, None spill)
    ab_nc = rep["rows"]["all_black_floor"]["per_window"]["t1_20-21"]
    assert ab_nc["mse"] == [0.0] and ab_nc["lpips"] == [0.0]
    assert ab_nc["region_lpips"] == [None] and ab_nc["spill"] == [None]
    assert (report_dir / "report.md").exists() and list((report_dir / "panels").glob("*.png"))
    md = (report_dir / "report.md").read_text()
    assert "# Change Map (pixels) scoring" in md and "Step Pairs:" in md and "Last Step" not in md


def _write_arm(tmp_path, name, ids, maps):
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    np.savez(d / f"{name}.npz", ids=np.array(ids), maps=maps.astype(np.float32), arm=name, target="change_map_pixels")
    return d


def _score(tier, tmp_path, sample_dirs=(), out="rep"):
    argv = ["score", "--tier", str(tier), "--split", "val", "--target", "change_map_pixels",
            "--lpips-dir", str(tmp_path), "--out", str(tmp_path / out)]
    for d in sample_dirs:
        argv += ["--samples", str(d)]
    assert cli(argv) == 0
    return json.loads((tmp_path / out / "report.json").read_text())


def _fixture_png_downscaled(regions):
    """Draw a 1600x896 white-on-black PNG independently (matching the dataset generator) and downscale nearest-neighbour."""
    canvas = np.zeros((896, 1600), np.uint8)
    for b in regions:
        x0, y0 = px(b[0], 1600 / 1920), px(b[1], 896 / 1080)
        x1, y1 = px(b[2], 1600 / 1920), px(b[3], 896 / 1080)
        canvas[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)] = 255
    img = Image.fromarray(canvas).resize((W, H), Image.NEAREST)
    return (np.asarray(img, np.float32) > 127).astype(np.float32)


def test_true_map_scores_zero_and_matches_fixture_png_downscaled_nearest(tftier, tmp_path):
    truth_maps = []
    png_maps = []
    for r in SCORED_VAL:
        regs = [b for b, ok in zip(r["regions"], r["box_ok_50_20"]) if ok] if r["target_status"] == "target" else []
        truth_maps.append(true_map(regs))
        png_maps.append(_fixture_png_downscaled(regs))
    truth_arr = np.stack(truth_maps)[:, None]
    png_arr = np.stack(png_maps)[:, None]
    assert np.array_equal(truth_arr, png_arr)

    rep = _score(tftier, tmp_path, [_write_arm(tmp_path, "perfect", VAL_IDS, png_arr)], out="rep_perfect")
    pw = rep["rows"]["perfect"]["per_window"]
    for wid in VAL_IDS:
        assert pw[wid]["lpips"] == [0.0]
        assert pw[wid]["mse"] == [0.0]
        if wid in NO_CHANGE_IDS:
            assert pw[wid]["region_lpips"] == [None]
            assert pw[wid]["spill"] == [None]
        else:
            assert pw[wid]["region_lpips"] == [0.0]
            assert pw[wid]["spill"] == [0.0]


def test_mismatched_or_missing_index_row_exits_nonzero(tftier, tmp_path):
    idx_path = tftier / "val_index.jsonl"
    rows = [json.loads(line) for line in idx_path.read_text().splitlines() if line.strip()]
    # Corrupt the regions of the first val row (refused on score) and first train row (refused on train)
    bad = [dict(r) for r in rows]
    bad[0]["regions_px_1920x1080"] = [[0.0, 0.0, 100.0, 100.0]]
    idx_path.write_text("\n".join(json.dumps(r) for r in bad) + "\n")
    exits_nonzero(["score", "--tier", str(tftier), "--split", "val", "--target", "change_map_pixels",
                   "--lpips-dir", str(tmp_path), "--out", str(tmp_path / "r_mismatch")])

    # Missing index row
    idx_path.write_text("\n".join(json.dumps(r) for r in rows[1:]) + "\n")
    exits_nonzero(["score", "--tier", str(tftier), "--split", "val", "--target", "change_map_pixels",
                   "--lpips-dir", str(tmp_path), "--out", str(tmp_path / "r_missing")])

    # Corrupt train index row -> train refuses
    train_idx = tftier / "train_index.jsonl"
    trows = [json.loads(line) for line in train_idx.read_text().splitlines() if line.strip()]
    trows[0]["regions_px_1920x1080"] = [[0.0, 0.0, 100.0, 100.0]]
    train_idx.write_text("\n".join(json.dumps(r) for r in trows) + "\n")
    exits_nonzero(["train", "--tier", str(tftier), "--target", "change_map_pixels", "--arm", "frame_caption",
                   "--steps", "1", "--batch", "2", *SMALL, "--out", str(tmp_path / "r_train_bad")])


def test_bootstrap_resamples_whole_windows_so_step_pairs_move_together(tmp_path):
    """Two Windows each have two Step Pairs whose (arm - all_black_floor) MSE diffs are (-a, +a).
    If Step Pairs of a Window always move together in every bootstrap resample, every resample sums to 0
    (ci_lo == ci_hi == 0.0); independent Step Pair resampling would yield a wide [-a, +a] interval."""
    box = [192, 108, 576, 324]
    val_rows = [
        pair_row("wA_0-1", "wA", [0, 1], "target", [box]),
        pair_row("wA_1-2", "wA", [1, 2], "no_change"),
        pair_row("wB_0-1", "wB", [0, 1], "target", [box]),
        pair_row("wB_1-2", "wB", [1, 2], "no_change"),
    ]
    tier = write_two_frame_tier(tmp_path / "bt_tier", train_rows=TRAIN, val_rows=val_rows)
    box_map = true_map([box])
    # Arm predicts box_map on every Step Pair: exact on target pairs (diff -a), wrong on no_change pairs (diff +a)
    maps = np.stack([box_map] * 4)[:, None]
    arm_dir = _write_arm(tmp_path, "frame_caption", [r["id"] for r in val_rows], maps)
    rep = _score(tier, tmp_path, [arm_dir], out="rep_boot")
    cmp_mse = next(c for c in rep["comparisons"]
                   if c["a"] == "frame_caption:single" and c["b"] == "all_black_floor:single" and c["metric"] == "mse")
    assert cmp_mse["mean_diff"] == pytest.approx(0.0, abs=1e-12)
    assert cmp_mse["ci_lo"] == pytest.approx(0.0, abs=1e-12)
    assert cmp_mse["ci_hi"] == pytest.approx(0.0, abs=1e-12)
