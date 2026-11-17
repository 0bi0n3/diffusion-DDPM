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
FLOOR_NOTES = ("Floors are scored exactly like arms (k = 1, same Window ids). Each supplies a mask and RGB. All-Black Floor: "
               "all-zero mask, fill RGB. Mean-Map Floor: the mean training mask and mean training RGB. Previous-Target "
               "Floor: the frame 8 to frame 9 regions as the mask, with the Previous Frame's RGB inside it. Copy-F8 Floor: "
               "the true mask with the Previous Frame's RGB inside it. Copy-F8 Floor is an oracle location (it is given "
               "where the change is), so it separates predicting new content from copying the Previous Frame.")
HYPOTHESES = ("Hypotheses (Region LPIPS; a hypothesis is supported when the upper bound of the interval is below 0): "
              "H1 Frame+Caption vs the Previous-Target Floor; H2 Frame+Caption vs Caption-only; H3 Caption-only vs "
              "Unconditional (read with the caption leak); H4 best-of-k vs single sample for Frame+Caption; H5 "
              "Frame+Caption vs the Copy-F8 Floor (does the model predict new content beyond copying the Previous Frame).")
ROW_LABELS = {"copy_f8_floor": "copy_f8_floor (oracle location)"}
TOKEN_METHOD = ("Token-space scoring: generated and true Masked Frames are composited (RGB where the mask is on, 0 elsewhere), "
                "upscaled bicubically x4 to 896x1600 and encoded by the same frozen encoder ({enc}), giving 28x50 tokens. "
                "Token Cosine is the mean cosine over all 1,400 squares; Region Token Cosine is the same mean over the "
                "squares that overlap a true region (n/a for No Change Windows); higher is better for both. Token MSE is "
                "{mse}. Best of k takes the highest cosine and the lowest Token MSE. True-frame encodings are cached "
                "({cache}).")
TOKEN_MSE_NORMALISED = "computed after normalising tokens with the DiT study's latent stats ({stats})"
TOKEN_MSE_RAW = "computed on unnormalised tokens (no --latent-stats given), so it is not comparable with the DiT study's normalised MSE"
TOKEN_HYPOTHESES = ("Token-space reading of H1 to H5 (secondary): each is also compared on Region Token Cosine, where higher is "
                    "better, so a hypothesis is supported when the lower bound of a - b is above 0.")
HEADERS = {"token_cosine": "Token Cosine", "region_token_cosine": "Region Token Cosine", "token_mse": "Token MSE",
           "lpips": "LPIPS", "mse": "MSE", "region_lpips": "Region LPIPS", "region_mse": "Region MSE",
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
         + (("Lower is better for every metric except Token Cosine and Region Token Cosine (higher is better)."
             if "tokens" in rep else "Lower is better for every metric.") if masked else "Lower LPIPS and MSE are better."), "", MASKED_FRAMES if masked else FRAMES, "",
         "## Caveats", "", f"- {CAPTION_LEAK}", f"- {caveat}", ""]
    if masked:
        L += [FLOOR_NOTES, "", HYPOTHESES, ""]
    if "tokens" in rep:
        t = rep["tokens"]
        enc = f"{t['encoder']['id']} revision {t['encoder']['revision']}"
        mse = (TOKEN_MSE_NORMALISED.format(stats=t["latent_stats"]) if t["normalised"] else TOKEN_MSE_RAW)
        L += [TOKEN_METHOD.format(enc=enc, mse=mse, cache=t["cache"]), "", TOKEN_HYPOTHESES, ""]
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
                L.append(f"| {ROW_LABELS.get(name, name) if masked else name} | {part} | {s['n']} | {cells} |")
        L.append("")
    L += ["PSNR is computed from each row's mean MSE, never per Window; it is n/a when the mean MSE is 0.", "",
          f"## Paired bootstrap ({rep['bootstrap']['resamples']} resamples, seed {rep['bootstrap']['seed']}, 95% interval)",
          "", "Difference a - b of per-Window values; negative means a is better (lower)"
          + ("; for Token Cosine and Region Token Cosine higher is better, so positive means a is better." if "tokens" in rep else "."), "",
          "| a | b | metric | mean diff | 95% low | 95% high |" + (" hypothesis |" if masked else ""),
          "|---|---|---|---|---|---|" + ("---|" if masked else "")]
    for c in rep["comparisons"]:
        L.append(f"| {c['a']} | {c['b']} | {c['metric']} | {_f(c['mean_diff'], 5)} | {_f(c['ci_lo'], 5)} | {_f(c['ci_hi'], 5)} |"
                 + (f" {c.get('hypothesis', '')} |" if masked else ""))
    if rep.get("hypothesis_readings"):
        L += ["", "## Pixel and token conclusions", "",
              "Region LPIPS supports a hypothesis when its upper bound is below 0; Region Token Cosine when its lower bound is above 0. "
              "A disagreement is flagged here and not resolved.", "",
              "| hypothesis | a | b | Region LPIPS upper bound | pixel | Region Token Cosine lower bound | token | |",
              "|---|---|---|---|---|---|---|---|"]
        for r in rep["hypothesis_readings"]:
            L.append(f"| {r['hypothesis']} | {r['a']} | {r['b']} | {_f(r['pixel_ci_hi'], 5)} | "
                     f"{'supported' if r['pixel_supported'] else 'not supported'} | {_f(r['token_ci_lo'], 5)} | "
                     f"{'supported' if r['token_supported'] else 'not supported'} | {'DISAGREE' if r['disagree'] else 'agree'} |")
        dis = [r["hypothesis"] for r in rep["hypothesis_readings"] if r["disagree"]]
        L += ["", f"Disagreement between pixel and token conclusions: {', '.join(dis)}." if dis
              else "No hypothesis has disagreeing pixel and token conclusions."]
    L += ["", f"Panels: panels/ ({', '.join(rep['panel_windows'])}).", ""]
    return "\n".join(L)
