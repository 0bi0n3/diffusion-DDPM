# PREREG: two-frame tier_1k, Change Map (pixels) and Map Latent (spec #14, ticket #21)

Written and committed before any AISurrey GPU job for #21 is submitted.

## What is run

- **Data**: two-frame `tier_1k` Step Pairs staged in #20 (`results/two_frame_tier_1k_data/staging_manifest.json`):
  - `train`: 7,103 non-dropped Step Pairs (2,140 No Change; 97 Dropped Pairs excluded).
  - `val`: 893 non-dropped Step Pairs (289 No Change; 7 Dropped Pairs excluded).
  - The test split is never read and BornCG stays sealed.
- **Conditioning Arm**: `frame_caption` only, for both targets.
  - Input Frame `Ft`: frozen DINOv3 ViT-B/16 tokens `(28, 50, 768)`, normalised once with the DiT study's frame stats.
  - Caption: `goal. subgoal. action` (never `evidence`), encoded with CLIP ViT-L/14 (`openai/clip-vit-large-patch14`).
- **Targets** (two separate runs trained identically, changing only the target):
  1. `change_map_pixels`: exact-pixel Change Map at `224x400`, 1 channel (white Significant Regions on black, drawn from the verified Change Map index), folded `8x8` onto `(28, 50, 64)` in `[-1, 1]`. Decoded to `(224, 400)` in `[0, 1]`; pixels are on where `> 0.5`.
  2. `map_latent`: frozen DINOv3 ViT-B/16 tokens `(28, 50, 768)` of that same Change Map image (`224x400`, repeated to RGB, bicubic x4 to `896x1600`, 2x2 pooled), normalised per channel with the training Map Latents' mean and std. **Map Latent predictions are not decoded to pixels in this experiment**, so no pixel-space metrics or image panels are reported for the `map_latent` run and no number depends on an untrained decoder.
- **Model and training**: `ChangeMapDiT` with `hidden=384`, `depth=12`, `heads=6`, channels from target (`64` or `768`), **`--qk-norm on`** (RMSNorm on queries and keys per head in self-attention and Caption cross-attention), `20,000` steps, batch size `64` (~180 passes over the 7,103 training Step Pairs), `seed=35`, AdamW `lr=1e-4` with `500` warm-up steps then cosine decay, EMA decay `0.9999`, global gradient-norm clipping `1.0`, bf16 autocast, validation every `1,000` steps.
- **Sampling**: `k = 8` samples per validation Step Pair, `50` Euler steps, `seed = 35`, from the EMA weights.
- **Scoring**: paired scoring of both runs in one AISurrey GPU job against the three two-frame floors (All-Black, Mean-Map, Previous-Target), split into `all`, `change`, and `no_change` Step Pairs:
  - Pixel space (`change_map_pixels` and pixel floors): full-frame LPIPS at `224x400`, Region LPIPS (padded crop of at least `32x32` around the true Significant Regions, change Step Pairs only), pixel MSE, PSNR per row, Spill.
  - Latent space (`map_latent`, encoded `change_map_pixels`, and encoded floors): Token Cosine (all 1,400 squares), Region Token Cosine (squares where the true pooled map > 0, change Step Pairs only), Token MSE (in training Map Latent normalised space).
  - Paired bootstrap: `1,000` resamples, `seed = 35`, `95%` confidence intervals, **Window-resampled** (whole Windows are resampled together so all Step Pairs sharing a Window move together).
  - Panels: rendered locally on CPU for `change_map_pixels` (Input Frame, true Change Map, `k` generated maps, per-sample scores).

## Stability-gate rule (before full runs)

Because both Caption arms collapsed in #7 without QK-normalisation, each target (`change_map_pixels` and `map_latent`) first runs a short stability-gate job on AISurrey (`1,000` steps, `--arm frame_caption`, `--qk-norm on`, `--grad-clip 1.0`, batch `64`, `seed 35`, `lr 1e-4`, `hidden 384`, `depth 12`, `heads 6`). Both logs are checked with `changemap gate --run <dir>`:

- **Fail** if at any logged training step: training loss exceeds `10x` its running minimum (or is non-finite), OR pre-clip gradient norm exceeds `100` (or is non-finite).
- **Stop rule**: if either gate fails, stop immediately, commit the gate findings, and do **not** launch the 20k full runs.

## Checkpoint selection rule (on validation)

Only the last EMA weights (`model.pt`, step 20,000) are used for scoring. Overfitting is defined as:

1. `val_loss` at step 20,000 exceeds the minimum `val_loss` over the 20 logged validation steps by more than `5%` (relative), **AND**
2. `val_loss` rose at each of the last three logged validation steps.

If overfitting is triggered for a run, that run is still scored and reported with its last EMA checkpoint, overfitting is flagged in the report, and checkpoints are never swapped after seeing scores. Any shorter or regularised re-run would be a separate, separately declared deviation, not a replacement. Both checkpoint checking and scoring use the validation split (893 Step Pairs); this is stated as a limitation and is not a test verdict.

## Hypotheses (Window-resampled paired 95% bootstrap intervals)

All comparisons are paired over the same validation change Step Pairs (604 Step Pairs with a Significant Region; Region LPIPS and Region Token Cosine are absent for the 289 No Change Step Pairs). Differences are written `a - b`:

- **H1** (pixel space, Region LPIPS, lower is better): `change_map_pixels` (single sample) beats the Previous-Target Floor. Supported when the **upper bound** of the 95% CI of `change_map_pixels - previous_target_floor` is `< 0`.
- **H2** (latent space, Region Token Cosine, higher is better): `map_latent` (single sample) beats the encoded Previous-Target Floor. Supported when the **lower bound** of the 95% CI of `map_latent - previous_target_floor` is `> 0`.
- **H3** (latent space, Region Token Cosine, two-sided): `map_latent` (single sample) vs `change_map_pixels` encoded to Map Latents (single sample). Reported either way: `map_latent` better if the lower bound of `map_latent - change_map_pixels` is `> 0`; `change_map_pixels` better if the upper bound is `< 0`; inconclusive if the 95% CI contains `0`.
- **H4** (best-of-8 vs single-sample, for each run):
  - `change_map_pixels`: best-of-8 beats single-sample on Region LPIPS (upper bound `< 0`) and on Region Token Cosine (lower bound `> 0`).
  - `map_latent`: best-of-8 beats single-sample on Region Token Cosine (lower bound `> 0`).

Full-frame LPIPS, pixel MSE, PSNR, Spill, Token Cosine, Token MSE, and No Change breakdowns are descriptive and carry no pre-registered hypothesis. Any additional analysis after inspecting results will be explicitly labelled **post-hoc**.

## Caveats

- **Caption leak**: Gemma wrote each Caption from all ten frames (F0–F9) of a Window, so the Caption can describe changes in the current Step Pair or other Step Pairs of the same Window. Caption contributions cannot be read as evidence that a pre-hoc Caption would help.
- **LPIPS on Change Maps**: LPIPS was trained on natural images, not white-on-black UI Change Maps; it primarily reflects edge and blob placement.
- **Map Latent not decoded**: `map_latent` outputs are evaluated only in DINOv3 token space and are not decoded to pixels.

## Honest guesses (written before any GPU run)

Context: #7 trained on 789 Last-Step Masked Frames (256 channels of RGB + mask) and failed to beat the floors while Caption arms collapsed without QK-norm. Here the target is a much simpler binary Change Map (or its DINOv3 Map Latent), training has ~9x more Step Pairs (7,103, ~180 passes at 20k steps), and QK-norm stabilises attention. On the other hand, 1-second UI changes are often local and stochastic, and consecutive Step Pairs frequently edit the same UI region, making the Previous-Target Floor strong.

- **Stability gates**: ~85% both pass with QK-norm on and gradient clipping 1.0.
- **H1 (`change_map_pixels` beats Previous-Target on Region LPIPS)**: ~45% supported. Nine times more data and a simpler 1-channel target should let the DiT learn coarse change locations from Input Frame + Caption, but fuzzy/noisy pixel edges are penalised by LPIPS against crisp binary rectangles.
- **H2 (`map_latent` beats encoded Previous-Target on Region Token Cosine)**: ~55% supported. Predicting directly in DINOv3 space avoids pixel-edge sharpness penalties and lets the model hedge smoothly toward the conditional mean latent, which often improves cosine similarity.
- **H3 (`map_latent` vs encoded `change_map_pixels` on Region Token Cosine)**: ~60% `map_latent` higher, ~15% `change_map_pixels` higher, ~25% CI contains 0. `map_latent` optimises directly in the evaluation space, whereas `change_map_pixels` must produce sharp binary maps that survive DINOv3 encoding.
- **H4 (best-of-8 beats single-sample for each run)**: ~95% supported for both runs. Selecting the best of 8 samples is an oracle min/max over sample variance and beats the mean almost by construction; it is reported as a variance check rather than a claim of useful diversity.
