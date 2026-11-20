# Change Map Diffusion

A diffusion model that generates the Masked Frame (default) or the Grid Map of a Window's last Step Pair, conditioned on what the screen looks like and what the user intends.

## Data

**Window**:
A captioned 10 s stretch of a Godot tutorial video, sampled as ten frames F0–F9 at 1 fps.
_Avoid_: clip, sample, segment

**Step Pair**:
Two consecutive frames of a Window (Ft → Ft+1). The Last Step is F8 → F9.
_Avoid_: frame pair, transition

**Input Frame**:
Ft, the first frame of a Step Pair (the screen just before the change). The Previous Frame (F8) is the Input Frame of the Last Step.
_Avoid_: source frame, input image

**Previous Frame**:
F8, the screen just before the Last Step. The model's image context on Last-Step tiers.
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

**Target**:
What the model generates for a Window or Step Pair, chosen by name (`masked_frame`, the default, `grid_map`, `change_map_pixels`, or `map_latent`) and recorded with every run, samples file and report, so a run is never scored as another target.
_Avoid_: output type, mode

**Change Map (pixels)**:
The `change_map_pixels` target: a Step Pair's Change Map drawn from its verified index regions at exact pixels at 224×400 (1 channel, x·400/1920, y·224/1080, nearest-pixel edges, at least 1 px) and folded 8×8 into (28, 50, 64) in [−1, 1]; decoded to (224, 400) in [0, 1], on where > 0.5.
_Avoid_: pixel mask, binary image

**Map Latent**:
The `map_latent` target: the frozen DINOv3 tokens (28×50×768) of a Step Pair's Change Map image (drawn at 224×400, white on black, repeated to RGB, bicubic ×4 to 896×1600, 2×2 pooled), normalised per channel with the training Map Latents' mean and std. Scored in latent space only; not decoded to pixels.
_Avoid_: change-map embedding, pixel latent

**Masked Frame**:
F9 at 224×400 as RGB plus a Change Mask, with RGB held at the fill value (black, 0) outside the mask. Folded 8×8 into one 256-channel Token Grid square per block (28×50×256; channel order per square: pixel row-major, then R, G, B, mask). The default target.
_Avoid_: masked image, crop

**Change Mask**:
The pixel-exact mask of the Last Step's Significant Regions on the 224×400 frame (1 inside, 0 outside; all zero for a No Change Window). A generated mask is on where it exceeds 0.5. Regions scale from 1920×1080 at x·400/1920, y·224/1080 with edges rounded to the nearest pixel, at least one pixel wide and high.
_Avoid_: Change Map (that is the white-on-black image), segmentation

**Frame Cache**:
`<split>_frames.npz` beside a tier's targets: Window ids with F8 and F9 as uint8 (n, 224, 400, 3), unmasked. Its sha256 is in the run record.

**Region LPIPS**:
LPIPS between generated and true RGB, uncomposited, on the bounding box of the true Significant Regions padded to at least 32×32 pixels. Absent (not zero) for a No Change Window.

**Mask MSE**:
Mean squared error between the generated and the true Change Mask over the full frame.

**Spill**:
The share of mask-on generated pixels that lie outside the true regions; undefined when nothing is on.

**Grid Map**:
The 28×50 coverage form of a Change Map: each Token Grid square holds the fraction of it covered by Significant Regions, in [0, 1] (scaled to [−1, 1] for training). The model's target and output.
_Avoid_: heatmap, mask

**All-Black Floor**:
The score of always predicting No Change (an empty Grid Map). What a model must beat to have learned anything about where change happens.
_Avoid_: empty baseline, null model

**Mean-Map Floor**:
The score of predicting the average training Grid Map for every Window. A location prior with no context. For the Masked Frame: the mean training mask and mean training RGB.
_Avoid_: average baseline

**Previous-Target Floor**:
The score of predicting the Grid Map of the Step Pair before the Last Step (F7 → F8). "The same place changes again". Borrowed from DiT2VLM. For the Masked Frame: the F7 → F8 regions as the mask, with F8's RGB inside it; a Window with no previous pair gets an all-zero mask and is counted.
_Avoid_: repeat baseline, copy-previous

**Copy-F8 Floor**:
Masked Frame floor: the true Change Mask with F8's RGB inside it. Location is given (an oracle), so it scores perfectly on Mask MSE and Spill; what it shows is whether a model predicts new content beyond copying the Previous Frame (H5).
_Avoid_: copy baseline, oracle model
