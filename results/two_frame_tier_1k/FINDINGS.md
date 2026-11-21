# Findings: Two-Frame `tier_1k` Change Map (pixels) and Map Latent (Spec #14, Ticket #21)

Pre-registration: [`PREREG.md`](PREREG.md) (committed at `c03ecff` before any `#21` GPU job).  
Paired scoring tables: [`report.md`](report.md) and [`report.json`](report.json).

---

## 1. Run Provenance, Stability, and Overfitting Checks

All GPU runs executed on AISurrey (`a100` partition, container `dit2vlm_v2.sif`, code commit `376e32df85326d58ae2100f6bcda0bdbe68946f2`, `arm=frame_caption`, `qk_norm=on`, `grad_clip=1.0`, `seed=35`). Validation split only (`893` non-dropped Step Pairs: `604` change across `97` source Windows, `289` No Change, `8` without a usable previous pair); the test split and BornCG were never read.

| Job ID | Name | Target | Steps | Wall Time | Status | Gate / Overfit |
| --- | --- | --- | --- | --- | --- | --- |
| `220255` | `tf-gate-pixels` | `change_map_pixels` | 1,000 | 00:08:50 | COMPLETED | `changemap gate` **PASS** |
| `220328` | `tf-gate-latent` | `map_latent` | 1,000 | 00:15:39 | COMPLETED | `changemap gate` **PASS** |
| `220361` | `tf-full-pixels` | `change_map_pixels` | 20,000 | 03:00:17 | COMPLETED | No divergence (`max grad_norm 3.46`); no overfit (`val_loss` `0.1633 -> 0.0227`, monotonic) |
| `220362` | `tf-full-latent` | `map_latent` | 20,000 | 05:20:36 | COMPLETED | `changemap gate` **PASS** (`max grad_norm 3.71`); no overfit (`val_loss` `0.9799 -> 0.6327`, monotonic) |
| `220363` | `changemap-tf-score` | both (`k=8`, 50 Euler) | — | 01:27:10 | OUT_OF_MEMORY (64G) | Completed both sample `.npz` files (`2.4G` pixel, `29G` latent); OOM when `score` loaded 29G latent array |
| `221040` | `changemap-tf-score` | paired score (128G) | — | 00:50:03 | COMPLETED | Reused existing sample `.npz` files; peak RAM `63.8G`; wrote `report.json` and `report.md` |

- **Note on `full_pixels` `changemap gate` check**: `changemap gate --run results/two_frame_tier_1k/full_pixels` exited 1 at step `16068` (`loss 0.0812 > 10x running minimum 0.008021`, `grad_norm 0.282`). Across all 20,000 steps, loss remained finite and well-behaved (`last 10 losses: 0.0135–0.0451`), `max grad_norm` was `3.456` (vs limit `100`), and `val_loss` decreased monotonically at every logged validation step (`1000` through `20000`). This is benign flow-matching minibatch variance once the running minimum dropped below `0.008`, not training instability.
- **Checkpoint selection**: Both runs were scored with their final EMA checkpoint (`model.pt`, step `20,000`) per `PREREG.md`. Neither run triggered the pre-registered overfitting rule.

---

## 2. Pre-Registered Hypothesis Verdicts (H1–H4)

All paired comparisons use **Window-resampled** bootstrap (`1,000` resamples, `seed=35`, `95%` CI) over the `604` validation change Step Pairs (`97` source Windows). Differences are `a - b`.

| Hypothesis | Comparison (`a - b`) | Metric (direction) | `a` | `b` | Mean diff | 95% CI | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **H1** | `change_map_pixels:single` vs `previous_target_floor` | Region LPIPS (lower better) | `0.5694` | `0.3129` | `+0.25646` | `[+0.24312, +0.26855]` | **Refuted** (upper bound > 0) |
| **H2** | `map_latent:single` vs `previous_target_floor` | Region Token Cosine (higher better) | `0.5826` | `0.7132` | `-0.13055` | `[-0.14601, -0.11325]` | **Refuted** (lower bound < 0) |
| **H3** | `map_latent:single` vs `change_map_pixels:single` | Region Token Cosine (two-sided) | `0.5826` | `0.5915` | `-0.00887` | `[-0.01679, -0.00077]` | **`change_map_pixels` better** (upper bound < 0) |
| **H4 (pixels)** | `change_map_pixels:best_of_k` vs `single` | Region LPIPS (lower better) | `0.5021` | `0.5694` | `-0.06725` | `[-0.07363, -0.06163]` | **Supported** (upper bound < 0) |
| **H4 (pixels)** | `change_map_pixels:best_of_k` vs `single` | Region Token Cosine (higher better) | `0.6942` | `0.5915` | `+0.10268` | `[+0.09617, +0.10900]` | **Supported** (lower bound > 0) |
| **H4 (latent)** | `map_latent:best_of_k` vs `single` | Region Token Cosine (higher better) | `0.6855` | `0.5826` | `+0.10285` | `[+0.09689, +0.10908]` | **Supported** (lower bound > 0) |

### Summary Answers
- **Does either run beat the Previous-Target Floor?** **No.** Neither `change_map_pixels` nor `map_latent` beats `previous_target_floor` in pixel space (`Region LPIPS`: `0.5694` vs `0.3129`) or in DINOv3 latent space (`Region Token Cosine`: `0.5915` / `0.5826` vs `0.7132`; `Token MSE`: `1.4458` / `1.5050` vs `0.8708`). Even oracle **best-of-8** selection (`Region LPIPS 0.5021`; `Region Token Cosine 0.6942` / `0.6855`) remains below single-sample `previous_target_floor`.
- **Both runs do beat `all_black_floor` on Region Token Cosine** (`0.5915` and `0.5826` vs `0.5499`), and `change_map_pixels` beats `mean_map_floor` on Region LPIPS (`0.5694` vs `0.5865`, CI `[-0.02756, -0.00701]`), showing genuine spatial localization signal above trivial static baselines, though far short of temporal persistence (`previous_target_floor`).
- **Pixel vs Latent target (H3)**: Contrary to the pre-registered guess, `change_map_pixels` encoded to DINOv3 tokens slightly but consistently outperformed direct `map_latent` prediction on `Region Token Cosine` (`0.5915` vs `0.5826`), full-frame `Token Cosine` (`0.7633` vs `0.7360`), and `Token MSE` (`1.4458` vs `1.5050`, diff `+0.05924`, CI `[+0.02663, +0.09319]`).

---

## 3. Caveats

- **Map Latent not decoded**: `map_latent` (`frame_caption_map_latent`) predictions are evaluated strictly in DINOv3 token space (`28x50x768`) and are **not** decoded to pixels; pixel-space columns are `n/a` and no image panels are generated for `map_latent`.
- **Caption leak**: Gemma wrote each Window Caption from all 10 frames (`F0`–`F9`), so the Caption can describe changes occurring in the target Step Pair or neighboring Step Pairs. Even with this leak, `frame_caption` does not beat `previous_target_floor`.
- **LPIPS on binary Change Maps**: VGG LPIPS was trained on natural RGB images, not white-on-black UI masks.
- **Validation split only**: Both checkpoint monitoring and scoring used the `val` split (`893` Step Pairs); this is not a held-out test verdict.

---

## 4. Visual Inspection of `change_map_pixels` Panels

Four panels were rendered locally on CPU via `changemap panels` under [`panels/`](panels/):
- [`frame_caption_t78_1187-1188@t78_1187.png`](panels/frame_caption_t78_1187-1188@t78_1187.png)
- [`frame_caption_t78_1190-1191@t78_1187.png`](panels/frame_caption_t78_1190-1191@t78_1187.png)
- [`frame_caption_t78_1192-1193@t78_1187.png`](panels/frame_caption_t78_1192-1193@t78_1187.png)
- [`frame_caption_t78_1193-1194@t78_1187.png`](panels/frame_caption_t78_1193-1194@t78_1187.png)

Qualitative observations:
- The DiT produces sharp, coherent rectangular white UI change masks localized to the active central code-editor pane of the IDE — no mode collapse or salt-and-pepper noise.
- Across the `k=8` samples for a given Step Pair, the model samples diverse plausible sub-regions within that editor pane (thin line edits, larger editor-pane blocks, or blank maps), explaining why best-of-8 improves substantially over single-sample while single-sample lags `previous_target_floor` (which copies the exact crisp mask from `t-1`).

---

## 5. Post-Hoc Diagnostics (Explicitly Post-Hoc)

- **Why blank `change_map_pixels` samples still get `LPIPS ~ 0.39`**: In panel `t78_1187-1188@t78_1187`, `sample 1` is visually all black (`0.0%` pixels `> 0.5`) yet scores `LPIPS = 0.387`, and `no_change` Step Pairs average `LPIPS = 0.3987` despite very low pixel MSE (`0.02655` single; `0.00077` / `31.11 dB` best-of-8). Inspecting `samples/pixel/frame_caption.npz` shows that background pixels (`< 0.5`, `95.7%` of all pixels) have small non-zero continuous diffusion residuals (`mean = 0.0054`, `p90 = 0.0165`, `p99 = 0.0326`, i.e. `1`–`8` in uint8). Because `ChangeMapPixelsTarget.score_samples` passes continuous `[0, 1]` maps directly to VGG LPIPS without binarizing at `0.5`, this invisible background texture incurs a `~0.38` LPIPS penalty relative to exact-zero `all_black_floor` (`0.0779`) and `previous_target_floor` (`0.1022`).
- Note that this post-hoc observation does **not** rescue H1 or H2: in latent space (`Region Token Cosine`) and on `Spill` (`0.6854` vs `0.4741`), `change_map_pixels` and `map_latent` still trail `previous_target_floor` by a wide margin.
