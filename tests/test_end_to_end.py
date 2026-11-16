"""End to end through the CLI only: train -> sample -> score on a synthetic tier; assert on files."""
import json

import numpy as np
import pytest

from changemap.cli import main

SQ_W, SQ_H = 1920 / 50, 1080 / 28


def sq(r0, c0, r1, c1):
    """Pixel box covering Token Grid squares rows r0..r1, cols c0..c1 (inclusive)."""
    return [c0 * SQ_W, r0 * SQ_H, (c1 + 1) * SQ_W, (r1 + 1) * SQ_H]


def row(wid, status, regions=(), ok=None):
    regions = [list(r) for r in regions]
    return {"id": wid, "target_status": status, "regions": regions,
            "box_ok_50_20": [True] * len(regions) if ok is None else ok}


# val: 5 scored Windows (one No Change) + 1 Dropped Pair; one Window has a non-significant extra box
VAL = [
    row("t1_10", "target", [sq(2, 3, 5, 9)]),
    row("t1_20", "target", [sq(10, 10, 12, 20), sq(20, 30, 24, 40)]),
    row("t1_30", "no_change"),
    row("t1_40", "dropped"),
    row("t1_50", "target", [sq(0, 0, 3, 5), [100, 100, 110, 105]], ok=[True, False]),
    row("t1_60", "target", [sq(15, 40, 20, 49)]),
]
TRAIN = [dict(r, id=r["id"].replace("t1", "t2")) for r in VAL]


@pytest.fixture
def tier(tmp_path):
    d = tmp_path / "tier"
    d.mkdir()
    for split, rows in (("train", TRAIN), ("val", VAL)):
        (d / f"{split}_targets.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return d


def run_pipeline(tier, tmp_path, seed=35):
    run, samples, report = tmp_path / "run", tmp_path / f"samples{seed}", tmp_path / f"report{seed}"
    if not (run / "config.json").exists():
        assert main(["train", "--tier", str(tier), "--arm", "unconditional", "--steps", "4", "--batch", "4",
                     "--hidden", "32", "--depth", "2", "--heads", "2", "--out", str(run)]) == 0
    assert main(["sample", "--run", str(run), "--tier", str(tier), "--split", "val", "--k", "2",
                 "--steps", "3", "--seed", str(seed), "--out", str(samples)]) == 0
    assert main(["score", "--samples", str(samples), "--tier", str(tier), "--split", "val",
                 "--out", str(report)]) == 0
    return samples, json.loads((report / "report.json").read_text())


def load(samples):
    z = np.load(next(samples.glob("*.npz")))
    return list(z["ids"]), z["maps"]


def test_report_rows_and_window_ids(tier, tmp_path):
    _, rep = run_pipeline(tier, tmp_path)
    ids = ["t1_10", "t1_20", "t1_30", "t1_50", "t1_60"]
    assert rep["window_ids"] == ids                     # Dropped Pair absent, No Change Window present
    assert rep["n_windows"] == 5 and rep["n_no_change"] == 1
    assert {"unconditional", "all_black_floor", "oracle"} <= set(rep["rows"])
    for r in rep["rows"].values():
        assert sorted(r["terms"]) == sorted(ids)        # identical Window ids on every row
    black = rep["rows"]["all_black_floor"]
    assert black["matched_miou"] == 0.0 and black["no_change_accuracy"] == 1.0 and black["boxes_per_window"] == 0.0
    assert "No Change" in rep["scoring_rule"] and "pooled" in rep["scoring_rule"]
    # oracle: grid-aligned regions are recovered exactly by map_boxes at "half"
    assert rep["rows"]["oracle"]["matched_miou"] == pytest.approx(1.0)
    assert rep["rows"]["oracle"]["no_change_accuracy"] == 1.0
    u = rep["rows"]["unconditional"]
    assert len(u["per_sample_miou"]) == 2 and u["matched_miou"] == pytest.approx(np.mean(u["per_sample_miou"]))
    assert all(len(v) == 2 for v in u["terms"].values())      # per-Window terms kept per sample
    assert 0 <= u["matched_miou"] <= 1 and 0 <= u["no_change_accuracy"] <= 1


def test_pooled_formula_by_hand():
    from changemap.multibox import pooled, window_terms
    exact = window_terms([(0, 0, 2, 1)], [(0, 0, 2, 1)])                     # (1, 1)
    half = window_terms([(0, 0, 2, 1)], [(1, 0, 3, 1)])                      # IoU 1/3: (1/3, 1)
    false_alarm = window_terms([(0, 0, 1, 1)], [])                           # No Change Window with a box: (0, 1)
    quiet = window_terms([], [])                                             # No Change Window, no box: (0, 0)
    miss = window_terms([], [(0, 0, 1, 1), (2, 2, 3, 3)])                    # (0, 2)
    assert exact == (1.0, 1) and quiet == (0.0, 0) and false_alarm == (0.0, 1) and miss == (0.0, 2)
    assert half[0] == pytest.approx(1 / 3) and half[1] == 1
    assert pooled([exact, half, false_alarm, quiet]) == pytest.approx((1 + 1 / 3) / 3)
    assert pooled([quiet]) is None
    # 2 predicted vs 1 true: one pair matched, denominator 2 + 1 - 1 = 2
    assert window_terms([(0, 0, 2, 1), (5, 5, 6, 6)], [(0, 0, 2, 1)]) == (1.0, 2)


def test_oracle_matches_map_boxes_directly(tier, tmp_path):
    import changemap.siblings  # noqa: F401
    from dit2vlm import ceiling, hints
    _, rep = run_pipeline(tier, tmp_path)
    for wid, regions in (("t1_10", VAL[0]["regions"]), ("t1_50", VAL[4]["regions"][:1])):
        boxes = ceiling.map_boxes(hints.change_map(regions, (28, 50)), "half")
        assert np.allclose(boxes, regions, atol=1e-6)
        assert rep["rows"]["oracle"]["terms"][wid] == [[1.0, 1]]


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
                 "--out", str(out)]) == 0
    assert "oracle" in json.loads((out / "report" / "report.json").read_text())["rows"]
