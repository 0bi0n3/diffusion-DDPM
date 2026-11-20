# Change Map (pixels) scoring

Split: val. Step Pairs: 10 (3 No Change). Samples per Step Pair: arms k=2, floors 1. Lower is better for every metric except Token Cosine and Region Token Cosine (higher is better).

Each row is one Step Pair (Input Frame Ft -> Ft+1). The target is the exact-pixel Change Map at 224x400 (white Significant Regions on black; all black for No Change Step Pairs). Paired bootstrap resamples whole Windows so the Step Pairs of a Window move together.

## Caveats

- Caption leak: Gemma wrote the Caption (goal, subgoal, action) while looking at frames 1 to 10 of the Window, so the Caption can describe the change that the model is asked to draw. A strong Caption-only result is therefore not evidence that a Caption written before the fact would help.
- LPIPS caveat: LPIPS was trained on natural images. On white-on-black Change Maps at 224x400 it mostly measures where edges and blobs are. Region LPIPS compares the padded crop around the true Significant Regions (at least 32x32 pixels) on Step Pairs with a change.

Latent-space scoring: Change Maps (224x400, white on black, repeated to RGB) are upscaled bicubically x4 to 896x1600 and encoded by the frozen encoder (standin revision 1), giving 28x50 tokens. Token Cosine is the mean cosine over all 1,400 squares; Region Token Cosine is the same mean over squares where the true pooled map > 0 (n/a for No Change Step Pairs); higher is better for both. Token MSE is computed after normalising tokens with the training Map Latent stats (/mnt/hitchcock/scratch/oberon/changemap_data/two_frame_tier_1k_smoke10/token_cache/train_stats__standin__1.npz). Best of k takes the highest cosine and the lowest Token MSE. True Map Latents are cached (/mnt/hitchcock/scratch/oberon/changemap_data/two_frame_tier_1k_smoke10/token_cache/val__standin__1.npz).

## Single sample (mean over the k samples)

| row | part | n | LPIPS | Region LPIPS | MSE | Spill | Token Cosine | Region Token Cosine | Token MSE | PSNR (dB) |
|---|---|---|---|---|---|---|---|---|---|---|
| frame_caption | all | 10 | 0.8455 | 0.8654 | 0.37918 | 0.9629 | 0.0487 | 1.0000 | 3.54649 | 4.21 |
| frame_caption | change | 7 | 0.8498 | 0.8654 | 0.37922 | 0.9470 | 0.0696 | 1.0000 | 3.52915 | 4.21 |
| frame_caption | no_change | 3 | 0.8355 | n/a | 0.37909 | 1.0000 | 0.0000 | n/a | 3.58696 | 4.21 |
| all_black_floor | all | 10 | 0.0637 | 0.4032 | 0.03706 | n/a | 0.9513 | 0.0000 | 0.48681 | 14.31 |
| all_black_floor | change | 7 | 0.0909 | 0.4032 | 0.05294 | n/a | 0.9304 | 0.0000 | 0.69545 | 12.76 |
| all_black_floor | no_change | 3 | 0.0000 | n/a | 0.00000 | n/a | 1.0000 | n/a | 0.00000 | n/a |

## Best of k (oracle selection: picks the sample closest to the target)

| row | part | n | LPIPS | Region LPIPS | MSE | Spill | Token Cosine | Region Token Cosine | Token MSE | PSNR (dB) |
|---|---|---|---|---|---|---|---|---|---|---|
| frame_caption | all | 10 | 0.8431 | 0.8613 | 0.37849 | 0.9626 | 0.0487 | 1.0000 | 3.53725 | 4.22 |
| frame_caption | change | 7 | 0.8475 | 0.8613 | 0.37864 | 0.9466 | 0.0696 | 1.0000 | 3.52127 | 4.22 |
| frame_caption | no_change | 3 | 0.8329 | n/a | 0.37812 | 1.0000 | 0.0000 | n/a | 3.57455 | 4.22 |
| all_black_floor | all | 10 | 0.0637 | 0.4032 | 0.03706 | n/a | 0.9513 | 0.0000 | 0.48681 | 14.31 |
| all_black_floor | change | 7 | 0.0909 | 0.4032 | 0.05294 | n/a | 0.9304 | 0.0000 | 0.69545 | 12.76 |
| all_black_floor | no_change | 3 | 0.0000 | n/a | 0.00000 | n/a | 1.0000 | n/a | 0.00000 | n/a |

PSNR is computed from each row's mean MSE, never per Window; it is n/a when the mean MSE is 0.

## Paired bootstrap (1000 resamples, seed 35, 95% interval)

Difference a - b of per-Window values; negative means a is better (lower); for Token Cosine and Region Token Cosine higher is better, so positive means a is better.

| a | b | metric | mean diff | 95% low | 95% high |
|---|---|---|---|---|---|
| frame_caption:single | all_black_floor:single | lpips | 0.78182 | 0.77870 | 0.80992 |
| frame_caption:single | all_black_floor:single | region_lpips | 0.46220 | 0.45404 | 0.51115 |
| frame_caption:single | all_black_floor:single | mse | 0.34212 | 0.33839 | 0.37572 |
| frame_caption:single | all_black_floor:single | region_token_cosine | 1.00000 | 1.00000 | 1.00000 |
| frame_caption:best_of_k | frame_caption:single | lpips | -0.00234 | -0.00253 | -0.00061 |
| frame_caption:best_of_k | frame_caption:single | region_lpips | -0.00409 | -0.01154 | -0.00284 |
| frame_caption:best_of_k | frame_caption:single | mse | -0.00070 | -0.00075 | -0.00024 |
| frame_caption:best_of_k | frame_caption:single | region_token_cosine | 0.00000 | 0.00000 | 0.00000 |

Panels: panels/ (t78_1187-1188, t78_1188-1189, t78_1189-1190, t78_1190-1191).
