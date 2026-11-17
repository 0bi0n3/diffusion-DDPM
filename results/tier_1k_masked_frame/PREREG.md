# PREREG: tier_1k, Masked Frame, four Conditioning Arms (ticket #7)

Written and committed before any job is submitted. Code at master d37bf36 plus the job-script changes of this branch; the
model, loss, sampler and scorer are those of d37bf36.

## What is run

- Target: `masked_frame` (F9 at 224x400, RGB plus Change Mask, folded 8x8 into 28x50x256).
- Conditioning Arms: Unconditional, Caption-only, Frame-only, Frame+Caption. Four separate runs, one GPU each.
- Data: tier_1k, train split for training (789 Windows) and validation split for everything else (100 Windows).
  The test split is never read and BornCG stays sealed.
- Training: 20,000 steps, batch 64, seed 35, lr 1e-4 with 500 warm-up steps then cosine decay, EMA decay 0.9999,
  hidden 384, depth 12, heads 6, bf16 autocast. Validation every 1,000 steps on the 100 validation Windows
  (flow-matching loss, plus MSE of 4 EMA samples drawn with 10 Euler steps).
- Sampling: k = 8 samples per validation Window, 50 Euler steps, seed 35, from the EMA weights.
- Scoring: `changemap score` with the real DINOv3 encoder (token metrics on), the DiT study's latent stats, LPIPS VGG16
  weights offline, paired bootstrap of 1,000 resamples with seed 35, 95% intervals. Floors: All-Black, Mean-Map,
  Previous-Target, Copy-F8 (oracle location).

## Checkpoint selection (on validation)

Only the last EMA weights (`model.pt`, step 20,000) are kept; intermediate checkpoints are overwritten. The rule is:
use the last EMA checkpoint, unless the validation log (`log.json`, entry "validation") of that arm shows overfitting.
Exactly: overfitting means that `val_loss` at step 20,000 exceeds the minimum `val_loss` over the 20 logged steps by more
than 5% (relative) AND `val_loss` rose at each of the last three logged steps. If that happens for an arm, the arm is
still scored and reported with its last EMA checkpoint, the overfit is flagged in the report, and any shorter or
regularised re-run is a new, separately declared run (a deviation from this PREREG), not a replacement. If the
condition is not met, no selection is made and nothing is chosen after seeing the scores. Selection and scoring both use
the validation split (only 100 Windows); this is stated as a limitation, the verdict is not on test.

## Hypotheses (Region LPIPS, lower is better)

A hypothesis is supported when the upper bound of the paired 95% interval of the difference (first minus second) is
below 0. All comparisons are paired over the same validation change Windows (Region LPIPS is absent for No Change
Windows).

- H1: Frame+Caption minus the Previous-Target Floor (single sample).
- H2: Frame+Caption minus Caption-only (single sample).
- H3: Caption-only minus Unconditional (single sample). Read with the caption leak (Gemma wrote the Caption from F0-F9).
- H4: Frame+Caption best-of-8 minus Frame+Caption single sample.
- H5: Frame+Caption minus the Copy-F8 Floor (single sample): does the model predict new content beyond the Previous Frame.

Note on the floors: each floor's RGB is composited with its own mask, so Copy-F8 carries F8's pixels inside the true
regions and Previous-Target carries them only inside the F7-to-F8 regions. Copy-F8 is therefore the stronger floor and
H5 is the higher bar.

## Secondary token-space reading (Region Token Cosine, higher is better)

Each of H1-H5 is also read on Region Token Cosine (same comparisons, DINOv3 tokens of the composited, 4x upscaled
frames). The difference is (first minus second) and the hypothesis is supported in token space when the LOWER bound of
the 95% interval is above 0. Where the pixel and token readings disagree, that is reported as a disagreement and not
resolved after the fact. The pixel reading is primary. Token MSE, Token Cosine, full-frame LPIPS, MSE, Mask MSE and
Spill are descriptive and carry no hypothesis.

Caveats repeated in the report: the caption leak, and LPIPS trained on natural images, not UI screens.

## Honest guesses (written before seeing any result)

Context: in a CPU smoke run on 20 validation Windows Copy-F8 Region LPIPS was .096 against All-Black .416; the DiT latent
study found nothing beat copying the previous frame; only 789 training Windows for a 44.5M-parameter pixel model.

- H1 (beats Previous-Target): 40% supported. The floor shows F8 only where the previous step changed, so it is partly
  black over the true region; but a pixel model trained on 789 Windows will likely give blurry or noisy content, which
  LPIPS punishes hard.
- H2 (Frame beats Caption-only): 75% supported. F8's tokens carry the layout and colours of the region about to change,
  which a caption cannot; this is the most likely real effect.
- H3 (Caption beats Unconditional): 25% supported. The caption mainly says which action happens, which helps where more
  than what, and Region LPIPS uses the true crop; the leak could help, but with 789 Windows I expect the interval to
  contain 0.
- H4 (best-of-8 beats single): 95% supported. The minimum of eight noisy scores is lower almost by construction; this
  says little about useful diversity and is read that way.
- H5 (beats Copy-F8): 5% supported. Copy-F8 is an oracle on location and the screen mostly keeps its pixels; the model
  would need to beat copying with a small training set.
- Token space: I expect the same direction as pixel space for H2, H4 and H5; the most likely disagreement is H1 or H3,
  where an interval can straddle 0 in one space and not the other.
