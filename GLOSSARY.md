# Change Map Diffusion

A diffusion model that generates the Change Map of a Window's last Step Pair, conditioned on what the screen looks like and what the user intends.

## Data

**Window**:
A captioned 10 s stretch of a Godot tutorial video, sampled as ten frames F0–F9 at 1 fps.
_Avoid_: clip, sample, segment

**Step Pair**:
Two consecutive frames of a Window (Ft → Ft+1). The Last Step is F8 → F9.
_Avoid_: frame pair, transition

**Previous Frame**:
F8, the screen just before the Last Step. The model's image context.
_Avoid_: previous image, source frame, input image

**Significant Region**:
A box where pixels change across a Step Pair and that passes the 50/20 size rule (long side ≥ 50 px, short side ≥ 20 px at 1920×1080); at most 10 per Step Pair.
_Avoid_: change region (unfiltered), block, box

**Change Map**:
A black image with every Significant Region of the Last Step drawn as a filled white rectangle. The thing the model generates.
_Avoid_: mask, target image, change-maps

**No Change Window**:
A Window whose Last Step has no Significant Region; its Change Map is all black.
_Avoid_: empty map, negative

**Dropped Pair**:
A Step Pair whose change regions all cover more than half the screen (e.g. a scene cut); excluded, not a No Change Window.

**Caption**:
The goal, subgoal and action text of a Window. Never includes `evidence`.
_Avoid_: prompt, description, context

## Experiment

**Conditioning Arm**:
One of four models trained alike that differ only in context: Unconditional, Caption-only, Frame-only (Previous Frame), Frame+Caption.
_Avoid_: ablation, variant

**Grid Map**:
The 28×50 coverage form of a Change Map: each Token Grid square holds the fraction of it covered by Significant Regions, in [0, 1] (scaled to [−1, 1] for training). The model's target and output.
_Avoid_: heatmap, mask

**All-Black Floor**:
The score of always predicting No Change (an empty Grid Map). What a model must beat to have learned anything about where change happens.
_Avoid_: empty baseline, null model

**Mean-Map Floor**:
The score of predicting the average training Grid Map for every Window. A location prior with no context.
_Avoid_: average baseline

**Previous-Target Floor**:
The score of predicting the Grid Map of the Step Pair before the Last Step (F7 → F8). "The same place changes again". Borrowed from DiT2VLM.
_Avoid_: repeat baseline, copy-previous
