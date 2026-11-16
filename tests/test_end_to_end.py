"""End to end through the CLI only: train -> sample -> score on a synthetic tier; assert on files."""
import json
import math

import numpy as np
import pytest

from changemap.cli import main

SQ_W, SQ_H = 1920 / 50, 1080 / 28


def sq(r0, c0, r1, c1):
    """Pixel box covering Token Grid squares rows r0..r1, cols c0..c1 (inclusive)."""
    return [c0 * SQ_W, r0 * SQ_H, (c1 + 1) * SQ_W, (r1 + 1) * SQ_H]


def row(wid, status, regions=(), ok=None, previous=None):
    regions = [list(r) for r in regions]
    r = {"id": wid, "target_status": status, "regions": regions,
         "box_ok_50_20": [True] * len(regions) if ok is None else ok}
    if previous is not None:
        r["previous"] = previous
    return r


def prev(*boxes):
    return {"regions": [list(b) for b in boxes], "box_ok_50_20": [True] * len(boxes)}


# val: 5 scored Windows (one No Change) + 1 Dropped Pair; one Window has a non-significant extra box.
# Previous pair (F7 -> F8): t1_10 repeats its target exactly, t1_20 differs, t1_30 and t1_50 have none (absent, dropped).
VAL = [
    row("t1_10", "target", [sq(2, 3, 5, 9)], previous=prev(sq(2, 3, 5, 9))),
    row("t1_20", "target", [sq(10, 10, 12, 20), sq(20, 30, 24, 40)], previous=prev(sq(10, 10, 12, 20))),
    row("t1_30", "no_change"),
    row("t1_40", "dropped"),
    row("t1_50", "target", [sq(0, 0, 3, 5), [100, 100, 110, 105]], ok=[True, False], previous={"dropped": True}),
    row("t1_60", "target", [sq(15, 40, 20, 49)], previous=prev(sq(1, 1, 2, 2))),
]
TRAIN = [dict(r, id=r["id"].replace("t1", "t2")) for r in VAL]
IDS = ["t1_10", "t1_20", "t1_30", "t1_50", "t1_60"]
NO_CHANGE = {"t1_30"}


@pytest.fixture(autouse=True)
def fake_lpips(monkeypatch):
    monkeypatch.setenv("CHANGEMAP_LPIPS_MODULE", "fake_lpips")
    monkeypatch.delenv("CHANGEMAP_LPIPS_DIR", raising=False)


@pytest.fixture
def tier(tmp_path):
    d = tmp_path / "tier"
    d.mkdir()
    for split, rows in (("train", TRAIN), ("val", VAL)):
        (d / f"{split}_targets.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return d


def true_maps(tier):
    from changemap.data import load_windows
    return {w.id: w.grid_map for w in load_windows(tier, "val")}


def write_arm(tmp_path, tier, arm, fn, k=1, name=None):
    """A samples dir holding one arm whose maps are fn(true_map, j) for sample j."""
    ws = true_maps(tier)
    maps = np.stack([np.stack([fn(m, j) for j in range(k)]) for m in ws.values()]).astype(np.float32)
    d = tmp_path / (name or f"samples_{arm}")
    d.mkdir(exist_ok=True)
    np.savez(d / f"{arm}.npz", ids=np.array(list(ws)), maps=maps, arm=arm)
    return d


def score(tier, tmp_path, dirs, out="report", extra=()):
    argv = ["score", "--tier", str(tier), "--split", "val", "--lpips-dir", str(tmp_path), "--out", str(tmp_path / out)]
    for d in dirs:
        argv += ["--samples", str(d)]
    assert main(argv + list(extra)) == 0
    return json.loads((tmp_path / out / "report.json").read_text())


def run_pipeline(tier, tmp_path, seed=35):
    run, samples = tmp_path / "run", tmp_path / f"samples{seed}"
    if not (run / "config.json").exists():
        assert main(["train", "--tier", str(tier), "--arm", "unconditional", "--steps", "4", "--batch", "4",
                     "--hidden", "32", "--depth", "2", "--heads", "2", "--out", str(run)]) == 0
    assert main(["sample", "--run", str(run), "--tier", str(tier), "--split", "val", "--k", "2",
                 "--steps", "3", "--seed", str(seed), "--out", str(samples)]) == 0
    return samples, score(tier, tmp_path, [samples], out=f"report{seed}")


def load(samples):
    z = np.load(next(samples.glob("*.npz")))
    return list(z["ids"]), z["maps"]


FLOORS = ("all_black_floor", "mean_map_floor", "previous_target_floor")


def test_every_floor_and_arm_on_identical_windows_with_split_means(tier, tmp_path):
    _, rep = run_pipeline(tier, tmp_path)
    assert rep["window_ids"] == IDS                      # Dropped Pair absent, No Change Window present
    assert rep["n_windows"] == 5 and rep["n_no_change"] == 1
    assert {"unconditional", *FLOORS} <= set(rep["rows"])
    for r in rep["rows"].values():
        assert sorted(r["per_window"]) == sorted(IDS)    # identical Window ids on every row
        for part, n in (("all", 5), ("change", 4), ("no_change", 1)):
            s = r["summary"][part]
            assert s["n"] == n
            for v in ("single", "best_of_k"):
                assert s[v]["lpips"] is not None and s[v]["mse"] is not None
    assert rep["rows"]["unconditional"]["k"] == 2 and len(rep["rows"]["unconditional"]["per_window"]["t1_10"]["lpips"]) == 2
    assert all(rep["rows"][f]["k"] == 1 for f in FLOORS)
    assert "iou" not in json.dumps(rep).lower().replace("previous", "")     # "previous" contains the letters


def test_psnr_per_row_from_mean_mse_and_none_per_window(tier, tmp_path):
    _, rep = run_pipeline(tier, tmp_path)
    for r in rep["rows"].values():
        for w in r["per_window"].values():
            assert set(w) == {"lpips", "mse"}            # no per-Window PSNR
        for s in r["summary"].values():
            for v in ("single", "best_of_k"):
                mse, psnr = s[v]["mse"], s[v]["psnr"]
                if mse is None:
                    assert psnr is None
                elif mse > 0:
                    assert math.isfinite(psnr) and psnr == pytest.approx(10 * math.log10(1 / mse))
    # the single-sample row mean is the mean over Windows of each Window's mean over k
    r = rep["rows"]["unconditional"]
    mean_mse = np.mean([np.mean(w["mse"]) for w in r["per_window"].values()])
    assert r["summary"]["all"]["single"]["mse"] == pytest.approx(mean_mse)


def test_all_black_floor_exact_zero_on_no_change_and_positive_on_change(tier, tmp_path):
    rep = score(tier, tmp_path, [])
    black = rep["rows"]["all_black_floor"]
    assert black["per_window"]["t1_30"] == {"lpips": [0.0], "mse": [0.0]}
    assert black["summary"]["no_change"]["single"]["lpips"] == 0.0
    assert black["summary"]["no_change"]["single"]["mse"] == 0.0
    assert black["summary"]["no_change"]["single"]["psnr"] is None      # exact match: no finite PSNR
    for wid in IDS:
        if wid not in NO_CHANGE:
            assert black["per_window"][wid]["lpips"][0] > 0 and black["per_window"][wid]["mse"][0] > 0
    assert math.isfinite(black["summary"]["change"]["single"]["psnr"])


def test_map_against_itself_gives_zero(tier, tmp_path):
    d = write_arm(tmp_path, tier, "perfect", lambda m, j: m)
    rep = score(tier, tmp_path, [d])
    for w in rep["rows"]["perfect"]["per_window"].values():
        assert w == {"lpips": [0.0], "mse": [0.0]}


def test_previous_target_floor_uses_previous_pair_and_counts_missing(tier, tmp_path):
    rep = score(tier, tmp_path, [])
    pw = rep["rows"]["previous_target_floor"]["per_window"]
    assert pw["t1_10"] == {"lpips": [0.0], "mse": [0.0]}          # F7 -> F8 equals the target
    assert pw["t1_20"]["mse"][0] > 0
    # no previous pair (absent / dropped) is scored as all black: same numbers as the All-Black Floor
    black = rep["rows"]["all_black_floor"]["per_window"]
    for wid in ("t1_30", "t1_50"):
        assert pw[wid] == black[wid]
    assert rep["n_no_previous"] == 2 and rep["no_previous_rule"].lower().count("all black") == 1
    assert rep["rows"]["previous_target_floor"]["n_no_previous"] == 2


def test_previous_regions_from_explicit_split_file(tier, tmp_path):
    (tier / "val_previous.jsonl").write_text(
        "\n".join(json.dumps({"id": i, "previous": None if i != "t1_30" else prev(sq(4, 4, 6, 6))}) for i in IDS) + "\n")
    rep = score(tier, tmp_path, [])
    assert rep["n_no_previous"] == 4
    assert rep["rows"]["previous_target_floor"]["per_window"]["t1_30"]["mse"][0] > 0
    assert rep["rows"]["previous_target_floor"]["per_window"]["t1_10"]["mse"][0] > 0   # the file wins over inline


def test_mean_map_floor_is_the_average_training_map(tier, tmp_path):
    from changemap.data import load_windows
    train = np.mean([w.grid_map for w in load_windows(tier, "train")], axis=0)
    rep = score(tier, tmp_path, [])
    for w in load_windows(tier, "val"):
        assert rep["rows"]["mean_map_floor"]["per_window"][w.id]["mse"][0] == pytest.approx(
            float(np.mean((train - w.grid_map) ** 2)), rel=1e-5)


def test_best_of_k_never_worse_than_single(tier, tmp_path):
    _, rep = run_pipeline(tier, tmp_path)
    r = rep["rows"]["unconditional"]
    for s in r["summary"].values():
        for m in ("lpips", "mse"):
            assert s["best_of_k"][m] <= s["single"][m] + 1e-12
    for w in r["per_window"].values():
        assert min(w["lpips"]) <= np.mean(w["lpips"]) + 1e-12
    cmp = {(c["a"], c["b"], c["metric"]): c for c in rep["comparisons"]}
    c = cmp[("unconditional:best_of_k", "unconditional:single", "lpips")]
    assert c["mean_diff"] <= 0 and c["ci_hi"] <= 1e-12


def test_bootstrap_floor_against_itself_is_exactly_zero(tier, tmp_path):
    d = write_arm(tmp_path, tier, "black_copy", lambda m, j: np.zeros_like(m))
    rep = score(tier, tmp_path, [d])
    c = {(c["a"], c["b"], c["metric"]): c for c in rep["comparisons"]}[("black_copy:single", "all_black_floor:single", "lpips")]
    assert c["mean_diff"] == 0.0 and c["ci_lo"] == 0.0 and c["ci_hi"] == 0.0
    assert c["resamples"] == 1000 and c["seed"] == 35 and c["n"] == 5


def test_bootstrap_pairs_present_and_deterministic(tier, tmp_path):
    dirs = [write_arm(tmp_path, tier, a, lambda m, j, s=s: np.clip(m * s, 0, 1), k=2)
            for a, s in (("unconditional", 0.2), ("caption_only", 0.5), ("frame_caption", 0.9))]
    rep = score(tier, tmp_path, dirs, out="r1")
    pairs = {(c["a"], c["b"]) for c in rep["comparisons"]}
    for arm in ("unconditional", "caption_only", "frame_caption"):
        for fl in FLOORS:
            assert (f"{arm}:single", f"{fl}:single") in pairs
        assert (f"{arm}:best_of_k", f"{arm}:single") in pairs
    assert ("frame_caption:single", "caption_only:single") in pairs
    assert ("caption_only:single", "unconditional:single") in pairs
    assert not any(a.startswith("frame_only") for a, _ in pairs)             # arm absent: no comparison
    rep2 = score(tier, tmp_path, dirs, out="r2")
    assert rep2["comparisons"] == rep["comparisons"]
    c = {(c["a"], c["b"], c["metric"]): c for c in rep["comparisons"]}[("frame_caption:single", "caption_only:single", "mse")]
    assert c["ci_lo"] <= c["mean_diff"] <= c["ci_hi"] and c["mean_diff"] < 0       # 0.9 x true beats 0.5 x true


def test_report_md_and_panels(tier, tmp_path):
    _, rep = run_pipeline(tier, tmp_path)
    md = (tmp_path / "report35" / "report.md").read_text()
    assert "caption" in md.lower() and "Gemma" in md                         # caption-leak caveat
    assert "natural images" in md and "not comparable" in md and "DiT2VLM" in md   # LPIPS caveat
    assert "frame 1" in md.lower() and "frame 10" in md.lower() and "F9" not in md
    for name in ("unconditional", *FLOORS):
        assert name in md
    panels = sorted((tmp_path / "report35" / "panels").glob("*.png"))
    assert len(panels) >= 1 and all(p.stat().st_size > 1000 for p in panels)
    assert rep["panel_windows"] == IDS[:len(rep["panel_windows"])]


def test_lpips_dir_from_environment(tier, tmp_path, monkeypatch):
    monkeypatch.setenv("CHANGEMAP_LPIPS_DIR", str(tmp_path))
    assert main(["score", "--tier", str(tier), "--split", "val", "--out", str(tmp_path / "e")]) == 0


def test_missing_real_lpips_weights_exit_nonzero(tier, tmp_path, monkeypatch):
    monkeypatch.delenv("CHANGEMAP_LPIPS_MODULE")
    with pytest.raises(SystemExit) as e:
        main(["score", "--tier", str(tier), "--split", "val", "--lpips-dir", str(tmp_path / "nope"),
              "--out", str(tmp_path / "x")])
    assert e.value.code not in (0, None)


def test_batched_lpips_matches_one_at_a_time():
    import torch
    import fake_lpips
    from changemap import metrics
    m = fake_lpips.load()
    g = torch.Generator().manual_seed(1)
    a, b = torch.rand(5, 28, 50, generator=g), torch.rand(5, 28, 50, generator=g)
    batched = metrics.lpips_per_sample(m, a.numpy(), b.numpy(), batch=2)
    one = [float(m(metrics.to_lpips_input(x[None].numpy()), metrics.to_lpips_input(y[None].numpy()))) for x, y in zip(a, b)]
    assert np.allclose(batched, one, atol=1e-6)


def test_same_seed_same_samples_different_seed_differs(tier, tmp_path):
    s1, _ = run_pipeline(tier, tmp_path, seed=35)
    again = tmp_path / "again"
    assert main(["sample", "--run", str(tmp_path / "run"), "--tier", str(tier), "--split", "val", "--k", "2",
                 "--steps", "3", "--seed", "35", "--out", str(again)]) == 0
    s2, _ = run_pipeline(tier, tmp_path, seed=36)
    ids1, m1 = load(s1)
    ids2, m2 = load(again)
    ids3, m3 = load(s2)
    assert ids1 == ids2 == ids3 and m1.shape == (5, 2, 28, 50)
    assert m1.min() >= 0 and m1.max() <= 1
    assert np.array_equal(m1, m2)
    assert not np.array_equal(m1, m3)
    assert not np.array_equal(m1[:, 0], m1[:, 1])       # k samples differ


def test_test_split_refused_without_verdict(tier, tmp_path):
    run_pipeline(tier, tmp_path)
    (tier / "test_targets.jsonl").write_text(json.dumps(VAL[0]) + "\n")
    base = ["sample", "--run", str(tmp_path / "run"), "--tier", str(tier), "--split", "test", "--k", "1",
            "--steps", "2", "--out", str(tmp_path / "t")]
    with pytest.raises(SystemExit) as e:
        main(base)
    assert e.value.code not in (0, None)
    assert main(base + ["--verdict"]) == 0


def test_unknown_arm_refused(tier, tmp_path):
    with pytest.raises(SystemExit) as e:
        main(["train", "--tier", str(tier), "--arm", "bogus_arm", "--steps", "1", "--out", str(tmp_path / "r")])
    assert e.value.code not in (0, None)


def test_run_chains_all_three(tier, tmp_path):
    out = tmp_path / "all"
    assert main(["run", "--tier", str(tier), "--arms", "unconditional", "--steps", "2", "--batch", "2",
                 "--hidden", "32", "--depth", "2", "--heads", "2", "--k", "1", "--sample-steps", "2",
                 "--lpips-dir", str(tmp_path), "--out", str(out)]) == 0
    assert "previous_target_floor" in json.loads((out / "report" / "report.json").read_text())["rows"]
