"""report.md of a scoring run."""
from __future__ import annotations

CAPTION_LEAK = ("Caption leak: Gemma wrote the Caption (goal, subgoal, action) while looking at frames 1 to 10 of the "
                "Window, so the Caption can describe the change that the model is asked to draw. A strong "
                "Caption-only result is therefore not evidence that a Caption written before the fact would help.")
LPIPS_CAVEAT = ("LPIPS caveat: LPIPS was trained on natural images. On white rectangles on black it mostly measures "
                "where edges and blobs are. These scores are image-similarity scores of Grid Maps and are not "
                "comparable with the box-based scores of DiT2VLM.")
FRAMES = ("Frames are numbered 1 to 10: a Window spans frame 1 to frame 10, the Previous Frame is frame 9, and the "
          "Last Step goes from frame 9 to frame 10. The Previous-Target Floor repeats the Grid Map of the step before "
          "it (frame 8 to frame 9).")


def _f(v, nd=4):
    return "n/a" if v is None else f"{v:.{nd}f}"


def render(rep: dict) -> str:
    L = ["# Change Map scoring", "",
         f"Split: {rep['split']}. Windows: {rep['n_windows']} ({rep['n_no_change']} No Change). "
         f"Samples per Window: arms k={rep['k']}, floors 1. Lower LPIPS and MSE are better.", "", FRAMES, "",
         "## Caveats", "", f"- {CAPTION_LEAK}", f"- {LPIPS_CAVEAT}", "",
         f"Previous-Target Floor: {rep['no_previous_rule']} Windows without a usable previous pair: "
         f"{rep['n_no_previous']} of {rep['n_windows']}.", ""]
    for variant, title in (("single", "Single sample (mean over the k samples)"),
                           ("best_of_k", "Best of k (oracle selection: picks the sample closest to the target)")):
        L += [f"## {title}", "", "| row | part | n | LPIPS | MSE | PSNR (dB) |", "|---|---|---|---|---|---|"]
        for name, r in rep["rows"].items():
            for part in ("all", "change", "no_change"):
                s = r["summary"][part]
                v = s[variant]
                L.append(f"| {name} | {part} | {s['n']} | {_f(v['lpips'])} | {_f(v['mse'], 5)} | {_f(v['psnr'], 2)} |")
        L.append("")
    L += ["PSNR is computed from each row's mean MSE, never per Window; it is n/a when the mean MSE is 0.", "",
          f"## Paired bootstrap ({rep['bootstrap']['resamples']} resamples, seed {rep['bootstrap']['seed']}, 95% interval)",
          "", "Difference a - b of per-Window values; negative means a is better (lower).", "",
          "| a | b | metric | mean diff | 95% low | 95% high |", "|---|---|---|---|---|---|"]
    for c in rep["comparisons"]:
        L.append(f"| {c['a']} | {c['b']} | {c['metric']} | {_f(c['mean_diff'], 5)} | {_f(c['ci_lo'], 5)} | {_f(c['ci_hi'], 5)} |")
    L += ["", f"Panels: panels/ ({', '.join(rep['panel_windows'])}).", ""]
    return "\n".join(L)
