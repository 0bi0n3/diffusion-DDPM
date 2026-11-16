"""The Masked Frame target (#10) end to end through the CLI: frame cache -> train -> sample -> score.

The synthetic tier gets a fake frame cache `<split>_frames.npz` (ids, f8, f9: uint8 (n, 224, 400, 3)). Truth is built
here independently of the package: mask from the exact-pixel scaling rule, RGB = F9 where the mask is on, 0 elsewhere."""
import json

import numpy as np
import pytest

from changemap.cli import main as cli
from test_end_to_end import TRAIN, VAL, IDS, NO_CHANGE, row, sq  # noqa: F401

H, W = 224, 400


def px(v, scale):
    return int(np.floor(v * scale + 0.5))


def pixel_box(b):
    """Exact scaling: x * 400/1920, y * 224/1080, nearest-pixel edges, at least one pixel row/column."""
    x0, y0, x1, y1 = px(b[0], W / 1920), px(b[1], H / 1080), px(b[2], W / 1920), px(b[3], H / 1080)
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


def true_mask(regions):
    m = np.zeros((H, W), np.float32)
    for b in regions:
        x0, y0, x1, y1 = pixel_box(b)
        m[y0:y1, x0:x1] = 1
    return m


def fake_frames(rows, seed):
    """F9 random (a pure black patch in the first region of every Window), F8 = F9 changed outside the first region."""
    g = np.random.default_rng(seed)
    f8, f9 = [], []
    for r in rows:
        a = g.integers(1, 256, (H, W, 3), dtype=np.uint8)
        if r["regions"]:
            x0, y0, x1, y1 = pixel_box(r["regions"][0])
            a[y0:y1, x0:x1] = 0
        b = a.copy()
        b[:8, :8] = 77
        f9.append(a)
        f8.append(b)
    return np.stack(f8), np.stack(f9)


def write_frames(tier, split, rows, seed):
    f8, f9 = fake_frames(rows, seed)
    np.savez(tier / f"{split}_frames.npz", ids=np.array([r["id"] for r in rows]), f8=f8, f9=f9)


@pytest.fixture
def mtier(tmp_path):
    d = tmp_path / "tier"
    d.mkdir()
    for split, rows, seed in (("train", TRAIN, 1), ("val", VAL, 2)):
        (d / f"{split}_targets.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        write_frames(d, split, rows, seed)
    return d


SMALL = ["--hidden", "32", "--depth", "2", "--heads", "2"]
FLOORS = ["all_black_floor", "mean_map_floor", "previous_target_floor", "copy_f8_floor"]
SCORED = [r for r in VAL if r["target_status"] != "dropped"]


def truth_frames():
    _, f9 = fake_frames(VAL, 2)
    by = {r["id"]: f9[i] for i, r in enumerate(VAL)}
    out = []
    for r in SCORED:
        regs = r["regions"][:1] if r["id"] == "t1_50" else r["regions"]    # t1_50's second box is not significant
        m = true_mask(regs)
        out.append((by[r["id"]] * m[..., None].astype(np.uint8), m))
    return np.stack([o[0] for o in out]), np.stack([o[1] for o in out])


def score(tier, tmp_path, dirs=(), out="report", extra=()):
    argv = ["score", "--tier", str(tier), "--split", "val", "--lpips-dir", str(tmp_path), "--out", str(tmp_path / out)]
    for d in dirs:
        argv += ["--samples", str(d)]
    assert cli(argv + list(extra)) == 0
    return json.loads((tmp_path / out / "report.json").read_text())


def test_masked_frame_train_sample_score_end_to_end(mtier, tmp_path):
    run, samples = tmp_path / "run", tmp_path / "samples"
    assert cli(["train", "--tier", str(mtier), "--steps", "4", "--batch", "4", *SMALL, "--out", str(run)]) == 0
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["target"] == "masked_frame"                         # the new default
    assert cli(["sample", "--run", str(run), "--tier", str(mtier), "--split", "val", "--k", "2", "--steps", "2",
                "--out", str(samples)]) == 0
    z = np.load(next(samples.glob("*.npz")))
    assert str(z["target"]) == "masked_frame"
    assert z["rgb"].shape == (5, 2, H, W, 3) and z["rgb"].dtype == np.uint8
    assert z["mask"].shape == (5, 2, H, W) and z["mask"].min() >= 0 and z["mask"].max() <= 1
    rep = score(mtier, tmp_path, [samples])
    assert rep["target"] == "masked_frame" and rep["window_ids"] == IDS
    assert list(rep["rows"]) == ["unconditional", *FLOORS]
    row_ = rep["rows"]["unconditional"]
    assert row_["k"] == 2 and set(row_["per_window"]["t1_10"]) >= {"lpips", "region_lpips", "mse", "region_mse",
                                                                    "mask_mse", "spill", "grid_lpips", "grid_mse"}
    assert (tmp_path / "report" / "report.md").exists() and list((tmp_path / "report" / "panels").glob("*.png"))


def write_arm(tmp_path, name, rgb, mask, k=1):
    d = tmp_path / name
    d.mkdir()
    np.savez(d / f"{name}.npz", ids=np.array(IDS), rgb=np.repeat(rgb[:, None], k, 1), mask=np.repeat(mask[:, None], k, 1),
             arm=name, target="masked_frame")
    return d


def test_true_masked_frame_scores_perfectly_and_black_content_is_not_masked(mtier, tmp_path):
    rgb, mask = truth_frames()
    rep = score(mtier, tmp_path, [write_arm(tmp_path, "perfect", rgb, mask)])
    for wid, w in rep["rows"]["perfect"]["per_window"].items():
        for m in ("lpips", "mse", "mask_mse", "grid_lpips", "grid_mse"):
            assert w[m] == [0.0], (wid, m)
        if wid in NO_CHANGE:
            assert w["region_lpips"] == [None] and w["region_mse"] == [None]
        else:
            assert w["region_lpips"] == [0.0] and w["region_mse"] == [0.0]
        assert w["spill"] in ([0.0], [None])
    # the same RGB with the mask off: black F9 content inside a region (RGB 0, mask 1 in truth) is told apart from masked
    rep2 = score(mtier, tmp_path, [write_arm(tmp_path, "nomask", rgb, np.zeros_like(mask))], out="r2")
    w = rep2["rows"]["nomask"]["per_window"]["t1_10"]
    assert w["mask_mse"][0] == pytest.approx(true_mask(VAL[0]["regions"]).mean())
    assert w["region_mse"] == [0.0] and w["mse"] == [0.0]    # t1_10's region is pure black: pixels identical to masked ...
    w = rep2["rows"]["nomask"]["per_window"]["t1_20"]        # ... t1_20 has a second, textured region
    assert w["region_mse"] == [0.0] and w["region_lpips"] == [0.0]     # RGB inside the regions is right ...
    assert w["mse"][0] > 0                                   # ... but composited with no mask it is dropped
    assert w["spill"] == [None]


def test_all_black_floor_exact_pixels(mtier, tmp_path):
    rep = score(mtier, tmp_path)
    assert list(rep["rows"]) == FLOORS
    _, mask = truth_frames()
    pw = rep["rows"]["all_black_floor"]["per_window"]
    for wid, m in zip(IDS, mask):
        assert pw[wid]["mask_mse"][0] == pytest.approx(float(m.astype(np.float64).mean()), abs=1e-12)    # exact on-pixel share
        assert pw[wid]["spill"] == [None]                                             # nothing on: undefined
    assert pw["t1_30"]["mse"] == [0.0] and pw["t1_30"]["region_lpips"] == [None]
    assert pw["t1_20"]["region_lpips"][0] > 0 and pw["t1_10"]["region_lpips"] == [0.0]   # t1_10 is pure black
    assert rep["rows"]["all_black_floor"]["summary"]["no_change"]["single"]["region_lpips"] is None


def test_region_scaling_is_exact_pixels(tmp_path):
    """Hand-computed: [960, 540, 1344, 756] -> x 200..280, y 112..157 (156.8 rounds up) = 80 x 45 pixels;
    [100, 100, 104, 102] is under one pixel high: x 21..22, y 21..21 -> keeps one row = 1 pixel."""
    rows = [row("t3_1", "target", [[960, 540, 1344, 756]]), row("t3_2", "target", [[100, 100, 104, 102]]),
            row("t3_3", "target", [[0, 0, 1920, 1080]])]
    d = tmp_path / "t"
    d.mkdir()
    for split, seed in (("train", 1), ("val", 2)):
        rr = [dict(r, id=r["id"].replace("t3", "t4" if split == "train" else "t3")) for r in rows]
        (d / f"{split}_targets.jsonl").write_text("\n".join(json.dumps(r) for r in rr) + "\n")
        write_frames(d, split, rr, seed)
    pw = score(d, tmp_path)["rows"]["all_black_floor"]["per_window"]
    assert pw["t3_1"]["mask_mse"][0] == pytest.approx(3600 / (H * W), abs=1e-12)
    assert pw["t3_2"]["mask_mse"][0] == pytest.approx(1 / (H * W), abs=1e-12)
    assert pw["t3_3"]["mask_mse"][0] == 1.0


def exits_nonzero(argv):
    with pytest.raises(SystemExit) as e:
        cli(argv)
    assert e.value.code not in (0, None)


def test_default_is_masked_frame_and_grid_map_still_selectable(mtier, tmp_path):
    base = ["train", "--tier", str(mtier), "--steps", "1", "--batch", "2", *SMALL]
    assert cli(base + ["--out", str(tmp_path / "a")]) == 0
    assert cli(base + ["--target", "grid_map", "--out", str(tmp_path / "b")]) == 0
    assert json.loads((tmp_path / "a" / "config.json").read_text())["target"] == "masked_frame"
    assert json.loads((tmp_path / "b" / "config.json").read_text())["target"] == "grid_map"


def test_score_and_sample_refuse_a_grid_map_run_as_masked_frame(mtier, tmp_path):
    run, samples = tmp_path / "run", tmp_path / "s"
    assert cli(["train", "--tier", str(mtier), "--target", "grid_map", "--steps", "1", "--batch", "2", *SMALL,
                "--out", str(run)]) == 0
    exits_nonzero(["sample", "--run", str(run), "--tier", str(mtier), "--split", "val", "--k", "1", "--steps", "2",
                   "--out", str(samples)])                              # default target is masked_frame
    assert cli(["sample", "--run", str(run), "--tier", str(mtier), "--split", "val", "--k", "1", "--steps", "2",
                "--target", "grid_map", "--out", str(samples)]) == 0
    exits_nonzero(["score", "--tier", str(mtier), "--split", "val", "--lpips-dir", str(tmp_path),
                   "--samples", str(samples), "--out", str(tmp_path / "r")])


def test_missing_frame_cache_is_refused(mtier, tmp_path):
    (mtier / "train_frames.npz").unlink()
    exits_nonzero(["train", "--tier", str(mtier), "--steps", "1", "--out", str(tmp_path / "o")])


def test_run_chains_masked_frame_and_records_frame_cache_hash(mtier, tmp_path):
    out = tmp_path / "all"
    assert cli(["run", "--tier", str(mtier), "--arms", "unconditional", "--steps", "2", "--batch", "2", *SMALL,
                "--k", "1", "--sample-steps", "2", "--lpips-dir", str(tmp_path), "--out", str(out)]) == 0
    rep = json.loads((out / "report" / "report.json").read_text())
    assert rep["target"] == "masked_frame" and set(rep["rows"]) == {"unconditional", *FLOORS}
    rec = json.loads((out / "runs" / "unconditional" / "run_record.json").read_text())
    assert {"train_frames.npz", "val_frames.npz"} <= set(rec["data_sha256"])
    log = json.loads((out / "runs" / "unconditional" / "log.json").read_text())
    assert 0 <= log["validation"][-1]["val_mse"] <= 1


def test_test_split_frames_need_the_verdict_flag(mtier, tmp_path):
    (mtier / "test_targets.jsonl").write_text(json.dumps(VAL[0]) + "\n")
    write_frames(mtier, "test", VAL[:1], 3)
    exits_nonzero(["score", "--tier", str(mtier), "--split", "test", "--lpips-dir", str(tmp_path),
                   "--out", str(tmp_path / "r")])
    assert cli(["score", "--tier", str(mtier), "--split", "test", "--verdict", "--lpips-dir", str(tmp_path),
                "--out", str(tmp_path / "r")]) == 0


def test_copy_f8_floor_has_perfect_location_and_copies_f8_pixels(mtier, tmp_path):
    rep = score(mtier, tmp_path)
    assert rep["rows"]["copy_f8_floor"]["k"] == 1
    pw = rep["rows"]["copy_f8_floor"]["per_window"]
    for wid in IDS:
        assert pw[wid]["mask_mse"] == [0.0] and pw[wid]["spill"] in ([0.0], [None])
        assert pw[wid]["grid_mse"] == [0.0]
    # F8 == F9 inside the regions (F8 differs only in the top-left 8x8) -> equals the truth ...
    for wid in ("t1_10", "t1_20", "t1_30", "t1_60"):
        assert pw[wid]["lpips"] == [0.0] and pw[wid]["mse"] == [0.0], wid
    for wid in ("t1_10", "t1_20", "t1_60"):
        assert pw[wid]["region_lpips"] == [0.0] and pw[wid]["region_mse"] == [0.0], wid
    # ... t1_50's region covers the top-left corner where F8 != F9 -> differs
    assert pw["t1_50"]["region_mse"][0] > 0 and pw["t1_50"]["mse"][0] > 0 and pw["t1_50"]["region_lpips"][0] > 0


def test_previous_target_floor_is_f7_f8_regions_with_f8_pixels_and_counts_missing(mtier, tmp_path):
    rep = score(mtier, tmp_path)
    row_ = rep["rows"]["previous_target_floor"]
    pw, black = row_["per_window"], rep["rows"]["all_black_floor"]["per_window"]
    assert pw["t1_10"]["mask_mse"] == [0.0] and pw["t1_10"]["region_mse"] == [0.0]    # F7 -> F8 repeats the target
    assert pw["t1_10"]["mse"] == [0.0] and pw["t1_10"]["spill"] == [0.0]
    _, mask = truth_frames()
    by = dict(zip(IDS, mask))
    prev = {"t1_20": VAL[1]["regions"][:1], "t1_60": [VAL[5]["previous"]["regions"][0]]}
    for wid, regs in prev.items():
        assert pw[wid]["mask_mse"][0] == pytest.approx(float(np.mean((true_mask(regs).astype(np.float64) - by[wid]) ** 2)), abs=1e-12)
    assert pw["t1_60"]["spill"] == [1.0]                          # previous regions entirely elsewhere
    for wid in ("t1_30", "t1_50"):                                # no previous pair: all-zero mask, as the All-Black Floor
        assert pw[wid] == black[wid]
    assert row_["n_no_previous"] == 2 and rep["n_no_previous"] == 2 and "all-zero mask" in rep["no_previous_rule"]


def test_mean_map_floor_is_the_mean_training_mask_and_rgb(mtier, tmp_path):
    _, f9 = fake_frames(TRAIN, 1)
    tm = [true_mask(r["regions"][:1] if r["id"] == "t2_50" else r["regions"]) for r in TRAIN if r["target_status"] != "dropped"]
    # (fake_frames is indexed over all rows of TRAIN, including the Dropped Pair at index 3)
    keep = [i for i, r in enumerate(TRAIN) if r["target_status"] != "dropped"]
    tf = [f9[i] * m[..., None].astype(np.uint8) for i, m in zip(keep, tm)]
    mean_mask, mean_rgb = np.mean(tm, axis=0), np.rint(np.mean(tf, axis=0))
    rep = score(mtier, tmp_path)
    assert rep["rows"]["mean_map_floor"]["k"] == 1
    pw = rep["rows"]["mean_map_floor"]["per_window"]
    true_rgb, true_m = truth_frames()
    for i, wid in enumerate(IDS):
        assert pw[wid]["mask_mse"][0] == pytest.approx(float(np.mean((mean_mask.astype(np.float64) - true_m[i]) ** 2)), abs=1e-6)
        if wid in NO_CHANGE:
            assert pw[wid]["region_mse"] == [None]
        else:
            tmk = true_m[i] > 0.5
            want = float(np.mean(((mean_rgb[tmk] - true_rgb[i][tmk]) / 255) ** 2))
            assert pw[wid]["region_mse"][0] == pytest.approx(want, abs=1e-6), wid


def noisy_arm(tmp_path, name, amount, k=2):
    """An arm whose k samples are the truth with `amount` of random RGB noise inside the regions (mask exact)."""
    rgb, mask = truth_frames()
    g = np.random.default_rng(abs(hash(name)) % 1000)
    r = np.stack([[np.clip(rgb[i].astype(int) + (g.integers(-amount, amount + 1, rgb[i].shape) * (mask[i][..., None] > 0.5)),
                           0, 255).astype(np.uint8) for _ in range(k)] for i in range(len(rgb))])
    d = tmp_path / name
    d.mkdir()
    np.savez(d / f"{name}.npz", ids=np.array(IDS), rgb=r, mask=np.repeat(mask[:, None], k, 1), arm=name, target="masked_frame")
    return d


def test_bootstrap_comparisons_for_h1_to_h5_and_report_labels(mtier, tmp_path):
    arms = {"unconditional": 90, "caption_only": 60, "frame_caption": 30}
    rep = score(mtier, tmp_path, [noisy_arm(tmp_path, a, v) for a, v in arms.items()])
    assert list(rep["rows"]) == [*arms, *FLOORS]
    got = {(c["a"], c["b"], c["metric"]) for c in rep["comparisons"]}
    metrics_ = ("lpips", "region_lpips", "mse")
    want = {(f"{a}:single", f"{f}:single", m) for a in arms for f in FLOORS for m in metrics_}
    want |= {("frame_caption:single", "caption_only:single", m) for m in metrics_}
    want |= {("caption_only:single", "unconditional:single", m) for m in metrics_}
    want |= {(f"{a}:best_of_k", f"{a}:single", m) for a in arms for m in metrics_}
    assert want <= got
    for c in rep["comparisons"]:
        assert (c["resamples"], c["seed"]) == (1000, 35)
    h = {c["hypothesis"]: (c["a"], c["b"]) for c in rep["comparisons"] if c["metric"] == "region_lpips" and "hypothesis" in c}
    assert h == {"H1": ("frame_caption:single", "previous_target_floor:single"),
                 "H2": ("frame_caption:single", "caption_only:single"),
                 "H3": ("caption_only:single", "unconditional:single"),
                 "H4": ("frame_caption:best_of_k", "frame_caption:single"),
                 "H5": ("frame_caption:single", "copy_f8_floor:single")}
    # best-of-k never worse than single
    r = rep["rows"]["frame_caption"]
    for part in r["summary"].values():
        for m in ("lpips", "mse", "region_lpips"):
            if part["single"][m] is not None:
                assert part["best_of_k"][m] <= part["single"][m] + 1e-12
    md = (tmp_path / "report" / "report.md").read_text()
    assert "oracle location" in md and "Copy-F8 Floor" in md and "PSNR" in md
    assert "H5" in md and "caption leak" in md.lower() and "LPIPS caveat" in md
    assert rep["rows"]["copy_f8_floor"]["summary"]["all"]["single"]["psnr"] is not None or \
        rep["rows"]["copy_f8_floor"]["summary"]["all"]["single"]["mse"] == 0
