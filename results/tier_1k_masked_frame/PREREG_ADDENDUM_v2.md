# PREREG addendum: run v2 (declared deviation)

Written and committed after the v1 results (below) and before any v2 job is submitted. v1 stays the PREREG run and
is reported as such; v2 is a separate, declared run, not a replacement.

## Why

In v1 the Frame+Caption arm diverged. Its training loss fell to about 0.0105 by step 15k, then jumped
between steps 15k and 17k to about 0.93 (the level at the start of training) and stayed there. Its validation loss
went from 0.067 to 17.5. No loss was non-finite. The training loop had no gradient clipping. With EMA decay 0.9999
(about the last 10k steps), the step-20k EMA weights that the PREREG scores include about 4k collapsed steps, so H1,
H2, H4 and H5 (all of which use Frame+Caption) are compromised by a training failure rather than answered.

v1's PREREG checkpoint rule also flags Caption-only (final validation loss +47% over its minimum at step 5k, rising
for the last three validations) and Frame-only (+15% over its minimum at 9k). Unconditional did not overfit.

## What changes in v2

One change only: global gradient-norm clipping at 1.0 (the DiT/RAE default), for all four arms, so the arms stay
trained alike. Everything else is as in the PREREG: target, arms, tier_1k, 20k steps, batch 64, seed 35, lr, warm-up,
EMA, model size, k = 8, 50 Euler steps, scoring, floors, H1-H5 with the same support rules, the secondary
token reading, and the checkpoint rule (last EMA checkpoint; overfitting flagged, never swapped). The training log now
also records the pre-clip gradient norm at every step.

## Guesses for v2 (before any v2 result)

v1 already showed that the three stable arms are worse than the All-Black and Previous-Target floors on Region
LPIPS (change Windows: Caption-only .536, Unconditional .537, Frame-only .565; All-Black .445; Previous-Target .371;
Copy-F8 .126). Clipping should stop the divergence but is not expected to fix that.

- H1 (Frame+Caption beats Previous-Target): 10%.
- H2 (Frame+Caption beats Caption-only): 40%.
- H3 (Caption-only beats Unconditional): 20% on pixels. In v1 the pixel reading was not supported (−.001 [−.010, .008])
  while the token reading was (+.026 [.014, .038]): a disagreement, recorded, not resolved.
- H4 (best-of-8 beats single): 95%.
- H5 (Frame+Caption beats Copy-F8): 2%.

## v1 results (PREREG run, as scored)

`v1/report.json`, `v1/report.md`, and each arm's `log.json`, `run_record.json` and `config.json` sit beside this
file. Frame+Caption v1 is reported as diverged.
