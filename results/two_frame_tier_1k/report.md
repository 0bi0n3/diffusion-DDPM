# Change Map (pixels) scoring

Split: val. Step Pairs: 893 (289 No Change). Samples per Step Pair: arms k=8, floors 1. Lower is better for every metric except Token Cosine and Region Token Cosine (higher is better).

Each row is one Step Pair (Input Frame Ft -> Ft+1). The target is the exact-pixel Change Map at 224x400 (white Significant Regions on black; all black for No Change Step Pairs). Paired bootstrap resamples whole Windows so the Step Pairs of a Window move together.

## Caveats

- Caption leak: Gemma wrote the Caption (goal, subgoal, action) while looking at frames 1 to 10 of the Window, so the Caption can describe the change that the model is asked to draw. A strong Caption-only result is therefore not evidence that a Caption written before the fact would help.
- LPIPS caveat: LPIPS was trained on natural images. On white-on-black Change Maps at 224x400 it mostly measures where edges and blobs are. Region LPIPS compares the padded crop around the true Significant Regions (at least 32x32 pixels) on Step Pairs with a change.

Latent-space scoring: Change Maps (224x400, white on black, repeated to RGB) are upscaled bicubically x4 to 896x1600 and encoded by the frozen encoder (facebook/dinov3-vitb16-pretrain-lvd1689m revision 5931719e67bbdb9737e363e781fb0c67687896bc), giving 28x50 tokens. Token Cosine is the mean cosine over all 1,400 squares; Region Token Cosine is the same mean over squares where the true pooled map > 0 (n/a for No Change Step Pairs); higher is better for both. Token MSE is computed after normalising tokens with the training Map Latent stats (/mnt/fast/nobackup/scratch4weeks/od00383/change_map/two_frame_tier_1k/token_cache/train_stats__facebook-dinov3-vitb16-pretrain-lvd1689m__5931719e67bbdb9737e363e781fb0c67687896bc.npz). Best of k takes the highest cosine and the lowest Token MSE. True Map Latents are cached (/mnt/fast/nobackup/scratch4weeks/od00383/change_map/two_frame_tier_1k/token_cache/val__facebook-dinov3-vitb16-pretrain-lvd1689m__5931719e67bbdb9737e363e781fb0c67687896bc.npz).

Map Latent rows (`*_map_latent`) are scored in latent space only; their predictions are not decoded to pixels (pixel-space columns are n/a and no image panels are generated for them).

Previous-Target Floor: A Step Pair with no previous Step Pair (absent or dropped) is scored as all black. Step Pairs without a usable previous pair: 8 of 893.

## Single sample (mean over the k samples)

| row | part | n | LPIPS | Region LPIPS | MSE | Spill | Token Cosine | Region Token Cosine | Token MSE | PSNR (dB) |
|---|---|---|---|---|---|---|---|---|---|---|
| frame_caption | all | 893 | 0.4298 | 0.5694 | 0.07594 | 0.7837 | 0.7633 | 0.5915 | 1.44580 | 11.20 |
| frame_caption | change | 604 | 0.4447 | 0.5694 | 0.09958 | 0.6854 | 0.7406 | 0.5915 | 1.56224 | 10.02 |
| frame_caption | no_change | 289 | 0.3987 | n/a | 0.02655 | 1.0000 | 0.8107 | n/a | 1.20244 | 15.76 |
| all_black_floor | all | 893 | 0.0779 | 0.3754 | 0.05733 | n/a | 0.8344 | 0.5499 | 0.95604 | 12.42 |
| all_black_floor | change | 604 | 0.1152 | 0.3754 | 0.08476 | n/a | 0.7552 | 0.5499 | 1.41348 | 10.72 |
| all_black_floor | no_change | 289 | 0.0000 | n/a | 0.00000 | n/a | 1.0000 | n/a | 0.00000 | n/a |
| mean_map_floor | all | 893 | 0.4107 | 0.5865 | 0.05226 | n/a | 0.7148 | 0.5890 | 1.73515 | 12.82 |
| mean_map_floor | change | 604 | 0.4261 | 0.5865 | 0.07382 | n/a | 0.7015 | 0.5890 | 1.80762 | 11.32 |
| mean_map_floor | no_change | 289 | 0.3784 | n/a | 0.00721 | n/a | 0.7426 | n/a | 1.58369 | 21.42 |
| previous_target_floor | all | 893 | 0.1022 | 0.3129 | 0.08018 | 0.5910 | 0.8487 | 0.7132 | 0.87077 | 10.96 |
| previous_target_floor | change | 604 | 0.1275 | 0.3129 | 0.10259 | 0.4741 | 0.8290 | 0.7132 | 0.98231 | 9.89 |
| previous_target_floor | no_change | 289 | 0.0493 | n/a | 0.03332 | 1.0000 | 0.8898 | n/a | 0.63765 | 14.77 |
| frame_caption_map_latent | all | 893 | n/a | n/a | n/a | n/a | 0.7360 | 0.5826 | 1.50504 | n/a |
| frame_caption_map_latent | change | 604 | n/a | n/a | n/a | n/a | 0.7189 | 0.5826 | 1.60441 | n/a |
| frame_caption_map_latent | no_change | 289 | n/a | n/a | n/a | n/a | 0.7718 | n/a | 1.29735 | n/a |

## Best of k (oracle selection: picks the sample closest to the target)

| row | part | n | LPIPS | Region LPIPS | MSE | Spill | Token Cosine | Region Token Cosine | Token MSE | PSNR (dB) |
|---|---|---|---|---|---|---|---|---|---|---|
| frame_caption | all | 893 | 0.3775 | 0.5021 | 0.04156 | 0.6555 | 0.8430 | 0.6942 | 0.98993 | 13.81 |
| frame_caption | change | 604 | 0.3946 | 0.5021 | 0.06107 | 0.4988 | 0.8187 | 0.6942 | 1.10382 | 12.14 |
| frame_caption | no_change | 289 | 0.3417 | n/a | 0.00077 | 1.0000 | 0.8938 | n/a | 0.75190 | 31.11 |
| all_black_floor | all | 893 | 0.0779 | 0.3754 | 0.05733 | n/a | 0.8344 | 0.5499 | 0.95604 | 12.42 |
| all_black_floor | change | 604 | 0.1152 | 0.3754 | 0.08476 | n/a | 0.7552 | 0.5499 | 1.41348 | 10.72 |
| all_black_floor | no_change | 289 | 0.0000 | n/a | 0.00000 | n/a | 1.0000 | n/a | 0.00000 | n/a |
| mean_map_floor | all | 893 | 0.4107 | 0.5865 | 0.05226 | n/a | 0.7148 | 0.5890 | 1.73515 | 12.82 |
| mean_map_floor | change | 604 | 0.4261 | 0.5865 | 0.07382 | n/a | 0.7015 | 0.5890 | 1.80762 | 11.32 |
| mean_map_floor | no_change | 289 | 0.3784 | n/a | 0.00721 | n/a | 0.7426 | n/a | 1.58369 | 21.42 |
| previous_target_floor | all | 893 | 0.1022 | 0.3129 | 0.08018 | 0.5910 | 0.8487 | 0.7132 | 0.87077 | 10.96 |
| previous_target_floor | change | 604 | 0.1275 | 0.3129 | 0.10259 | 0.4741 | 0.8290 | 0.7132 | 0.98231 | 9.89 |
| previous_target_floor | no_change | 289 | 0.0493 | n/a | 0.03332 | 1.0000 | 0.8898 | n/a | 0.63765 | 14.77 |
| frame_caption_map_latent | all | 893 | n/a | n/a | n/a | n/a | 0.8123 | 0.6855 | 1.02449 | n/a |
| frame_caption_map_latent | change | 604 | n/a | n/a | n/a | n/a | 0.7871 | 0.6855 | 1.16521 | n/a |
| frame_caption_map_latent | no_change | 289 | n/a | n/a | n/a | n/a | 0.8651 | n/a | 0.73037 | n/a |

PSNR is computed from each row's mean MSE, never per Window; it is n/a when the mean MSE is 0.

## Paired bootstrap (1000 resamples, seed 35, 95% interval)

Difference a - b of per-Window values; negative means a is better (lower); for Token Cosine and Region Token Cosine higher is better, so positive means a is better.

| a | b | metric | mean diff | 95% low | 95% high | hypothesis |
|---|---|---|---|---|---|---|
| frame_caption:single | all_black_floor:single | lpips | 0.35187 | 0.34543 | 0.35787 |  |
| frame_caption:single | all_black_floor:single | region_lpips | 0.19394 | 0.18598 | 0.20108 |  |
| frame_caption:single | all_black_floor:single | mse | 0.01862 | 0.01337 | 0.02452 |  |
| frame_caption:single | all_black_floor:single | region_token_cosine | 0.04162 | 0.03163 | 0.05276 |  |
| frame_caption:single | mean_map_floor:single | lpips | 0.01912 | 0.01479 | 0.02367 |  |
| frame_caption:single | mean_map_floor:single | region_lpips | -0.01712 | -0.02756 | -0.00701 |  |
| frame_caption:single | mean_map_floor:single | mse | 0.02368 | 0.01809 | 0.02947 |  |
| frame_caption:single | mean_map_floor:single | region_token_cosine | 0.00249 | -0.00624 | 0.01203 |  |
| frame_caption:single | previous_target_floor:single | lpips | 0.32763 | 0.31823 | 0.33674 |  |
| frame_caption:single | previous_target_floor:single | region_lpips | 0.25646 | 0.24312 | 0.26855 | H1 |
| frame_caption:single | previous_target_floor:single | mse | -0.00423 | -0.01313 | 0.00510 |  |
| frame_caption:single | previous_target_floor:single | region_token_cosine | -0.12168 | -0.13672 | -0.10642 |  |
| frame_caption:best_of_k | frame_caption:single | lpips | -0.05233 | -0.05442 | -0.05044 |  |
| frame_caption:best_of_k | frame_caption:single | region_lpips | -0.06725 | -0.07363 | -0.06163 | H4 |
| frame_caption:best_of_k | frame_caption:single | mse | -0.03439 | -0.04134 | -0.02811 |  |
| frame_caption:best_of_k | frame_caption:single | region_token_cosine | 0.10268 | 0.09617 | 0.10900 | H4 |
| frame_caption_map_latent:single | all_black_floor:single | region_token_cosine | 0.03275 | 0.02180 | 0.04573 |  |
| frame_caption_map_latent:single | all_black_floor:single | token_mse | 0.54900 | 0.48869 | 0.61742 |  |
| frame_caption_map_latent:single | mean_map_floor:single | region_token_cosine | -0.00638 | -0.01680 | 0.00419 |  |
| frame_caption_map_latent:single | mean_map_floor:single | token_mse | -0.23011 | -0.27510 | -0.18706 |  |
| frame_caption_map_latent:single | previous_target_floor:single | region_token_cosine | -0.13055 | -0.14601 | -0.11325 | H2 |
| frame_caption_map_latent:single | previous_target_floor:single | token_mse | 0.63427 | 0.57851 | 0.68860 |  |
| frame_caption_map_latent:single | frame_caption:single | region_token_cosine | -0.00887 | -0.01679 | -0.00077 | H3 |
| frame_caption_map_latent:single | frame_caption:single | token_mse | 0.05924 | 0.02663 | 0.09319 |  |
| frame_caption_map_latent:best_of_k | frame_caption_map_latent:single | region_token_cosine | 0.10285 | 0.09689 | 0.10908 | H4 |
| frame_caption_map_latent:best_of_k | frame_caption_map_latent:single | token_mse | -0.48055 | -0.51008 | -0.45009 |  |

## Pixel and token conclusions

Region LPIPS supports a hypothesis when its upper bound is below 0; Region Token Cosine when its lower bound is above 0. A disagreement is flagged here and not resolved.

| hypothesis | a | b | Region LPIPS upper bound | pixel | Region Token Cosine lower bound | token | |
|---|---|---|---|---|---|---|---|
| H4 | frame_caption:best_of_k | frame_caption:single | -0.06163 | supported | 0.09689 | supported | agree |

No hypothesis has disagreeing pixel and token conclusions.

Panels: panels/ (t78_1187-1188@t78_1187, t78_1188-1189@t78_1187, t78_1189-1190@t78_1187, t78_1190-1191@t78_1187).
