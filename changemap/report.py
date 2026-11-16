"""report.md of a scoring run."""
from __future__ import annotations

CAPTION_LEAK = ("Caption leak: Gemma wrote the Caption (goal, subgoal, action) while looking at frames 1 to 10 of the "
                "Window, so the Caption can describe the change that the model is asked to draw. A strong "
                "Caption-only result is therefore not evidence that a Caption written before the fact would help.")
LPIPS_CAVEAT = ("LPIPS caveat: LPIPS was trained on natural images. On white rectangles on black it mostly measures "
                "where edges and blobs are. These scores are image-similarity scores of Grid Maps and are not "
                "comparable with the box-based scores of DiT2VLM.")
MASKED_LPIPS_CAVEAT = ("LPIPS caveat: LPIPS was trained on natural images. On Godot screenshots with most of the frame "
                       "masked it mostly measures where content is and how it looks locally. Region LPIPS compares the "
                       "uncomposited RGB inside the true regions (crop of at least 32x32 pixels), so it judges content "
                       "separately from whether the generated Change Mask found the region. These scores are not "
                       "comparable with the Grid Map scores or with the box-based scores of DiT2VLM.")
MASKED_FRAMES = ("Frames are numbered 1 to 10: a Window spans frame 1 to frame 10, the Previous Frame is frame 9, and "
                 "the Last Step goes from frame 9 to frame 10. The target is frame 10 at 224x400 where the Change Mask "
                 "is on and black elsewhere. Where: Mask MSE, Spill (share of mask-on pixels outside the true regions; "
                 "n/a when nothing is on) and the generated mask pooled to a Grid Map (Grid LPIPS, Grid MSE). What: "
                 "Region LPIPS and Region MSE (RGB inside the true regions; No Change Windows have none, shown as n/a). "
                 "LPIPS and MSE are over the composited full frame.")
HEADERS = {"lpips": "LPIPS", "mse": "MSE", "region_lpips": "Region LPIPS", "region_mse": "Region MSE",
           "mask_mse": "Mask MSE", "spill": "Spill", "grid_lpips": "Grid LPIPS", "grid_mse": "Grid MSE"}
FRAMES = ("Frames are numbered 1 to 10: a Window spans frame 1 to frame 10, the Previous Frame is frame 9, and the "
          "Last Step goes from frame 9 to frame 10. The Previous-Target Floor repeats the Grid Map of the step before "
          "it (frame 8 to frame 9).")


def _f(v, nd=4):
    return "n/a" if v is None else f"{v:.{nd}f}"


def render(rep: dict) -> str:
    masked = rep["target"] == "masked_frame"
    names = [m for m in next(iter(rep["rows"].values()))["per_window"][rep["window_ids"][0]]] if rep["rows"] else []
    cols = names + ["psnr"] if "mse" in names else names
    head = [HEADERS.get(m, m) if m != "psnr" else "PSNR (dB)" for m in cols]
    caveat = MASKED_LPIPS_CAVEAT if masked else LPIPS_CAVEAT
    L = ["# Masked Frame scoring" if masked else "# Change Map scoring", "",
         f"Split: {rep['split']}. Windows: {rep['n_windows']} ({rep['n_no_change']} No Change). "
         f"Samples per Window: arms k={rep['k']}, floors 1. "
         + ("Lower is better for every metric." if masked else "Lower LPIPS and MSE are better."), "", MASKED_FRAMES if masked else FRAMES, "",
         "## Caveats", "", f"- {CAPTION_LEAK}", f"- {caveat}", ""]
    if "previous_target_floor" in rep["rows"]:
        L += [f"Previous-Target Floor: {rep['no_previous_rule']} Windows without a usable previous pair: "
              f"{rep['n_no_previous']} of {rep['n_windows']}.", ""]
    for variant, title in (("single", "Single sample (mean over the k samples)"),
                           ("best_of_k", "Best of k (oracle selection: picks the sample closest to the target)")):
        L += [f"## {title}", "", f"| row | part | n | {' | '.join(head)} |", "|---|---|---|" + "---|" * len(cols)]
        for name, r in rep["rows"].items():
            for part in ("all", "change", "no_change"):
                s = r["summary"][part]
                v = s[variant]
                cells = " | ".join(_f(v[m], 2 if m == "psnr" else 5 if m.endswith("mse") else 4) for m in cols)
                L.append(f"| {name} | {part} | {s['n']} | {cells} |")
        L.append("")
    L += ["PSNR is computed from each row's mean MSE, never per Window; it is n/a when the mean MSE is 0.", "",
          f"## Paired bootstrap ({rep['bootstrap']['resamples']} resamples, seed {rep['bootstrap']['seed']}, 95% interval)",
          "", "Difference a - b of per-Window values; negative means a is better (lower).", "",
          "| a | b | metric | mean diff | 95% low | 95% high |", "|---|---|---|---|---|---|"]
    for c in rep["comparisons"]:
        L.append(f"| {c['a']} | {c['b']} | {c['metric']} | {_f(c['mean_diff'], 5)} | {_f(c['ci_lo'], 5)} | {_f(c['ci_hi'], 5)} |")
    L += ["", f"Panels: panels/ ({', '.join(rep['panel_windows'])}).", ""]
    return "\n".join(L)
