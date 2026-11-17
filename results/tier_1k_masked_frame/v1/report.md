# Masked Frame scoring

Split: val. Windows: 100 (30 No Change). Samples per Window: arms k=8, floors 1. Lower is better for every metric except Token Cosine and Region Token Cosine (higher is better).

Frames are numbered 1 to 10: a Window spans frame 1 to frame 10, the Previous Frame is frame 9, and the Last Step goes from frame 9 to frame 10. The target is frame 10 at 224x400 where the Change Mask is on and black elsewhere. Where: Mask MSE, Spill (share of mask-on pixels outside the true regions; n/a when nothing is on) and the generated mask pooled to a Grid Map (Grid LPIPS, Grid MSE). What: Region LPIPS and Region MSE (RGB inside the true regions; No Change Windows have none, shown as n/a). LPIPS and MSE are over the composited full frame.

## Caveats

- Caption leak: Gemma wrote the Caption (goal, subgoal, action) while looking at frames 1 to 10 of the Window, so the Caption can describe the change that the model is asked to draw. A strong Caption-only result is therefore not evidence that a Caption written before the fact would help.
- LPIPS caveat: LPIPS was trained on natural images. On Godot screenshots with most of the frame masked it mostly measures where content is and how it looks locally. Region LPIPS compares the uncomposited RGB inside the true regions (crop of at least 32x32 pixels), so it judges content separately from whether the generated Change Mask found the region. These scores are not comparable with the Grid Map scores or with the box-based scores of DiT2VLM.

Floors are scored exactly like arms (k = 1, same Window ids). Each supplies a mask and RGB. All-Black Floor: all-zero mask, fill RGB. Mean-Map Floor: the mean training mask and mean training RGB. Previous-Target Floor: the frame 8 to frame 9 regions as the mask, with the Previous Frame's RGB inside it. Copy-F8 Floor: the true mask with the Previous Frame's RGB inside it. Copy-F8 Floor is an oracle location (it is given where the change is), so it separates predicting new content from copying the Previous Frame.

Hypotheses (Region LPIPS; a hypothesis is supported when the upper bound of the interval is below 0): H1 Frame+Caption vs the Previous-Target Floor; H2 Frame+Caption vs Caption-only; H3 Caption-only vs Unconditional (read with the caption leak); H4 best-of-k vs single sample for Frame+Caption; H5 Frame+Caption vs the Copy-F8 Floor (does the model predict new content beyond copying the Previous Frame).

Token-space scoring: generated and true Masked Frames are composited (RGB where the mask is on, 0 elsewhere), upscaled bicubically x4 to 896x1600 and encoded by the same frozen encoder (facebook/dinov3-vitb16-pretrain-lvd1689m revision 5931719e67bbdb9737e363e781fb0c67687896bc), giving 28x50 tokens. Token Cosine is the mean cosine over all 1,400 squares; Region Token Cosine is the same mean over the squares that overlap a true region (n/a for No Change Windows); higher is better for both. Token MSE is computed after normalising tokens with the DiT study's latent stats (/mnt/fast/nobackup/scratch4weeks/od00383/change_map/stats/stats.npz). Best of k takes the highest cosine and the lowest Token MSE. True-frame encodings are cached (/mnt/fast/nobackup/scratch4weeks/od00383/change_map/tier_1k/token_cache/val__facebook-dinov3-vitb16-pretrain-lvd1689m__5931719e67bbdb9737e363e781fb0c67687896bc.npz).

Token-space reading of H1 to H5 (secondary): each is also compared on Region Token Cosine, where higher is better, so a hypothesis is supported when the lower bound of a - b is above 0.

Previous-Target Floor: A Window with no previous Step Pair (absent or dropped) gets an all-zero mask for the Previous-Target Floor. Windows without a usable previous pair: 2 of 100.

## Single sample (mean over the k samples)

| row | part | n | LPIPS | Region LPIPS | MSE | Region MSE | Mask MSE | Spill | Grid LPIPS | Grid MSE | Token Cosine | Region Token Cosine | Token MSE | PSNR (dB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unconditional | all | 100 | 0.1493 | 0.5368 | 0.00748 | 0.07725 | 0.10122 | 0.9123 | 0.0837 | 0.09534 | 0.7566 | 0.5405 | 1.59517 | 21.26 |
| unconditional | change | 70 | 0.1791 | 0.5368 | 0.00910 | 0.07725 | 0.12543 | 0.8747 | 0.1023 | 0.11765 | 0.7258 | 0.5405 | 1.77268 | 20.41 |
| unconditional | no_change | 30 | 0.0799 | n/a | 0.00370 | n/a | 0.04471 | 1.0000 | 0.0403 | 0.04327 | 0.8286 | n/a | 1.18097 | 24.32 |
| caption_only | all | 100 | 0.1217 | 0.5357 | 0.00496 | 0.08016 | 0.06697 | 0.8276 | 0.0662 | 0.06108 | 0.7890 | 0.5661 | 1.40373 | 23.05 |
| caption_only | change | 70 | 0.1469 | 0.5357 | 0.00669 | 0.08016 | 0.08972 | 0.7563 | 0.0832 | 0.08189 | 0.7634 | 0.5661 | 1.55376 | 21.75 |
| caption_only | no_change | 30 | 0.0628 | n/a | 0.00093 | n/a | 0.01390 | 1.0000 | 0.0266 | 0.01253 | 0.8489 | n/a | 1.05367 | 30.30 |
| frame_only | all | 100 | 0.1095 | 0.5650 | 0.00511 | 0.06997 | 0.06750 | 0.7904 | 0.0610 | 0.06235 | 0.8109 | 0.5870 | 1.25853 | 22.91 |
| frame_only | change | 70 | 0.1374 | 0.5650 | 0.00676 | 0.06997 | 0.08855 | 0.7066 | 0.0778 | 0.08165 | 0.7774 | 0.5870 | 1.46691 | 21.70 |
| frame_only | no_change | 30 | 0.0445 | n/a | 0.00128 | n/a | 0.01837 | 1.0000 | 0.0219 | 0.01730 | 0.8891 | n/a | 0.77230 | 28.92 |
| frame_caption | all | 100 | 0.7695 | 0.8057 | 0.36373 | 0.30965 | 0.68673 | 0.9443 | 0.6111 | 0.56229 | 0.6336 | 0.4968 | 2.46045 | 4.39 |
| frame_caption | change | 70 | 0.7738 | 0.8057 | 0.36187 | 0.30965 | 0.67502 | 0.9204 | 0.6064 | 0.54880 | 0.5969 | 0.4968 | 2.66702 | 4.41 |
| frame_caption | no_change | 30 | 0.7595 | n/a | 0.36807 | n/a | 0.71405 | 1.0000 | 0.6221 | 0.59376 | 0.7193 | n/a | 1.97845 | 4.34 |
| all_black_floor | all | 100 | 0.0748 | 0.4448 | 0.00405 | 0.08192 | 0.05549 | n/a | 0.0428 | 0.05110 | 0.8432 | 0.5477 | 1.08179 | 23.93 |
| all_black_floor | change | 70 | 0.1068 | 0.4448 | 0.00578 | 0.08192 | 0.07927 | n/a | 0.0611 | 0.07300 | 0.7760 | 0.5477 | 1.54541 | 22.38 |
| all_black_floor | no_change | 30 | 0.0000 | n/a | 0.00000 | n/a | 0.00000 | n/a | 0.0000 | 0.00000 | 1.0000 | n/a | 0.00000 | n/a |
| mean_map_floor | all | 100 | 0.0748 | 0.4931 | 0.00405 | 0.07027 | 0.05136 | n/a | 0.0428 | 0.05110 | 0.8432 | 0.5477 | 1.08179 | 23.93 |
| mean_map_floor | change | 70 | 0.1068 | 0.4931 | 0.00578 | 0.07027 | 0.07027 | n/a | 0.0611 | 0.07300 | 0.7760 | 0.5477 | 1.54541 | 22.38 |
| mean_map_floor | no_change | 30 | 0.0000 | n/a | 0.00000 | n/a | 0.00726 | n/a | 0.0000 | 0.00000 | 1.0000 | n/a | 0.00000 | n/a |
| previous_target_floor | all | 100 | 0.0968 | 0.3712 | 0.00531 | 0.05608 | 0.06661 | 0.5536 | 0.0550 | 0.06003 | 0.8664 | 0.7109 | 0.88473 | 22.75 |
| previous_target_floor | change | 70 | 0.1225 | 0.3712 | 0.00697 | 0.05608 | 0.08502 | 0.4499 | 0.0706 | 0.07638 | 0.8470 | 0.7109 | 0.99811 | 21.57 |
| previous_target_floor | no_change | 30 | 0.0368 | n/a | 0.00143 | n/a | 0.02363 | 1.0000 | 0.0186 | 0.02189 | 0.9118 | n/a | 0.62017 | 28.43 |
| copy_f8_floor (oracle location) | all | 100 | 0.0183 | 0.1261 | 0.00085 | 0.01472 | 0.00000 | 0.0000 | 0.0000 | 0.00000 | 0.9905 | 0.9583 | 0.06257 | 30.69 |
| copy_f8_floor (oracle location) | change | 70 | 0.0262 | 0.1261 | 0.00122 | 0.01472 | 0.00000 | 0.0000 | 0.0000 | 0.00000 | 0.9864 | 0.9583 | 0.08938 | 29.14 |
| copy_f8_floor (oracle location) | no_change | 30 | 0.0000 | n/a | 0.00000 | n/a | 0.00000 | n/a | 0.0000 | 0.00000 | 1.0000 | n/a | 0.00000 | n/a |

## Best of k (oracle selection: picks the sample closest to the target)

| row | part | n | LPIPS | Region LPIPS | MSE | Region MSE | Mask MSE | Spill | Grid LPIPS | Grid MSE | Token Cosine | Region Token Cosine | Token MSE | PSNR (dB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unconditional | all | 100 | 0.0750 | 0.4848 | 0.00363 | 0.05720 | 0.04604 | 0.8173 | 0.0424 | 0.04134 | 0.8590 | 0.6029 | 0.92409 | 24.41 |
| unconditional | change | 70 | 0.1071 | 0.4848 | 0.00518 | 0.05720 | 0.06577 | 0.7389 | 0.0606 | 0.05905 | 0.7986 | 0.6029 | 1.32012 | 22.86 |
| unconditional | no_change | 30 | 0.0000 | n/a | 0.00000 | n/a | 0.00000 | 1.0000 | 0.0000 | 0.00000 | 1.0000 | n/a | 0.00000 | n/a |
| caption_only | all | 100 | 0.1116 | 0.5129 | 0.00476 | 0.07866 | 0.06453 | 0.7935 | 0.0614 | 0.05869 | 0.8132 | 0.5932 | 1.23897 | 23.23 |
| caption_only | change | 70 | 0.1376 | 0.5129 | 0.00647 | 0.07866 | 0.08715 | 0.7080 | 0.0788 | 0.07933 | 0.7848 | 0.5932 | 1.40914 | 21.89 |
| caption_only | no_change | 30 | 0.0510 | n/a | 0.00076 | n/a | 0.01174 | 1.0000 | 0.0208 | 0.01051 | 0.8795 | n/a | 0.84192 | 31.22 |
| frame_only | all | 100 | 0.0932 | 0.5319 | 0.00449 | 0.06497 | 0.05795 | 0.7375 | 0.0517 | 0.05302 | 0.8449 | 0.6231 | 1.02888 | 23.48 |
| frame_only | change | 70 | 0.1219 | 0.5319 | 0.00615 | 0.06497 | 0.07856 | 0.6324 | 0.0683 | 0.07181 | 0.8053 | 0.6231 | 1.28294 | 22.11 |
| frame_only | no_change | 30 | 0.0262 | n/a | 0.00061 | n/a | 0.00985 | 1.0000 | 0.0129 | 0.00918 | 0.9375 | n/a | 0.43607 | 32.14 |
| frame_caption | all | 100 | 0.7616 | 0.7938 | 0.35608 | 0.30294 | 0.67855 | 0.9441 | 0.6085 | 0.54964 | 0.6397 | 0.5023 | 2.40851 | 4.48 |
| frame_caption | change | 70 | 0.7659 | 0.7938 | 0.35420 | 0.30294 | 0.66729 | 0.9201 | 0.6037 | 0.53659 | 0.6025 | 0.5023 | 2.61724 | 4.51 |
| frame_caption | no_change | 30 | 0.7515 | n/a | 0.36047 | n/a | 0.70483 | 1.0000 | 0.6197 | 0.58011 | 0.7264 | n/a | 1.92146 | 4.43 |
| all_black_floor | all | 100 | 0.0748 | 0.4448 | 0.00405 | 0.08192 | 0.05549 | n/a | 0.0428 | 0.05110 | 0.8432 | 0.5477 | 1.08179 | 23.93 |
| all_black_floor | change | 70 | 0.1068 | 0.4448 | 0.00578 | 0.08192 | 0.07927 | n/a | 0.0611 | 0.07300 | 0.7760 | 0.5477 | 1.54541 | 22.38 |
| all_black_floor | no_change | 30 | 0.0000 | n/a | 0.00000 | n/a | 0.00000 | n/a | 0.0000 | 0.00000 | 1.0000 | n/a | 0.00000 | n/a |
| mean_map_floor | all | 100 | 0.0748 | 0.4931 | 0.00405 | 0.07027 | 0.05136 | n/a | 0.0428 | 0.05110 | 0.8432 | 0.5477 | 1.08179 | 23.93 |
| mean_map_floor | change | 70 | 0.1068 | 0.4931 | 0.00578 | 0.07027 | 0.07027 | n/a | 0.0611 | 0.07300 | 0.7760 | 0.5477 | 1.54541 | 22.38 |
| mean_map_floor | no_change | 30 | 0.0000 | n/a | 0.00000 | n/a | 0.00726 | n/a | 0.0000 | 0.00000 | 1.0000 | n/a | 0.00000 | n/a |
| previous_target_floor | all | 100 | 0.0968 | 0.3712 | 0.00531 | 0.05608 | 0.06661 | 0.5536 | 0.0550 | 0.06003 | 0.8664 | 0.7109 | 0.88473 | 22.75 |
| previous_target_floor | change | 70 | 0.1225 | 0.3712 | 0.00697 | 0.05608 | 0.08502 | 0.4499 | 0.0706 | 0.07638 | 0.8470 | 0.7109 | 0.99811 | 21.57 |
| previous_target_floor | no_change | 30 | 0.0368 | n/a | 0.00143 | n/a | 0.02363 | 1.0000 | 0.0186 | 0.02189 | 0.9118 | n/a | 0.62017 | 28.43 |
| copy_f8_floor (oracle location) | all | 100 | 0.0183 | 0.1261 | 0.00085 | 0.01472 | 0.00000 | 0.0000 | 0.0000 | 0.00000 | 0.9905 | 0.9583 | 0.06257 | 30.69 |
| copy_f8_floor (oracle location) | change | 70 | 0.0262 | 0.1261 | 0.00122 | 0.01472 | 0.00000 | 0.0000 | 0.0000 | 0.00000 | 0.9864 | 0.9583 | 0.08938 | 29.14 |
| copy_f8_floor (oracle location) | no_change | 30 | 0.0000 | n/a | 0.00000 | n/a | 0.00000 | n/a | 0.0000 | 0.00000 | 1.0000 | n/a | 0.00000 | n/a |

PSNR is computed from each row's mean MSE, never per Window; it is n/a when the mean MSE is 0.

## Paired bootstrap (1000 resamples, seed 35, 95% interval)

Difference a - b of per-Window values; negative means a is better (lower); for Token Cosine and Region Token Cosine higher is better, so positive means a is better.

| a | b | metric | mean diff | 95% low | 95% high | hypothesis |
|---|---|---|---|---|---|---|
| unconditional:single | all_black_floor:single | lpips | 0.07457 | 0.06657 | 0.08249 |  |
| unconditional:single | all_black_floor:single | region_lpips | 0.09198 | 0.08042 | 0.10473 |  |
| unconditional:single | all_black_floor:single | mse | 0.00343 | 0.00269 | 0.00424 |  |
| unconditional:single | all_black_floor:single | region_token_cosine | -0.00713 | -0.01404 | -0.00064 |  |
| unconditional:single | mean_map_floor:single | lpips | 0.07457 | 0.06657 | 0.08249 |  |
| unconditional:single | mean_map_floor:single | region_lpips | 0.04368 | 0.03717 | 0.05034 |  |
| unconditional:single | mean_map_floor:single | mse | 0.00343 | 0.00269 | 0.00424 |  |
| unconditional:single | mean_map_floor:single | region_token_cosine | -0.00713 | -0.01404 | -0.00064 |  |
| unconditional:single | previous_target_floor:single | lpips | 0.05257 | 0.03124 | 0.07108 |  |
| unconditional:single | previous_target_floor:single | region_lpips | 0.16556 | 0.13864 | 0.19523 |  |
| unconditional:single | previous_target_floor:single | mse | 0.00217 | -0.00056 | 0.00437 |  |
| unconditional:single | previous_target_floor:single | region_token_cosine | -0.17038 | -0.20669 | -0.13630 |  |
| unconditional:single | copy_f8_floor:single | lpips | 0.13100 | 0.11825 | 0.14397 |  |
| unconditional:single | copy_f8_floor:single | region_lpips | 0.41066 | 0.38582 | 0.43837 |  |
| unconditional:single | copy_f8_floor:single | mse | 0.00663 | 0.00531 | 0.00810 |  |
| unconditional:single | copy_f8_floor:single | region_token_cosine | -0.41774 | -0.42977 | -0.40598 |  |
| caption_only:single | all_black_floor:single | lpips | 0.04694 | 0.03681 | 0.05787 |  |
| caption_only:single | all_black_floor:single | region_lpips | 0.09093 | 0.07860 | 0.10408 |  |
| caption_only:single | all_black_floor:single | mse | 0.00091 | 0.00052 | 0.00132 |  |
| caption_only:single | all_black_floor:single | region_token_cosine | 0.01839 | 0.00800 | 0.02948 |  |
| caption_only:single | mean_map_floor:single | lpips | 0.04694 | 0.03681 | 0.05787 |  |
| caption_only:single | mean_map_floor:single | region_lpips | 0.04263 | 0.03196 | 0.05312 |  |
| caption_only:single | mean_map_floor:single | mse | 0.00091 | 0.00052 | 0.00132 |  |
| caption_only:single | mean_map_floor:single | region_token_cosine | 0.01839 | 0.00800 | 0.02948 |  |
| caption_only:single | previous_target_floor:single | lpips | 0.02494 | 0.00552 | 0.04253 |  |
| caption_only:single | previous_target_floor:single | region_lpips | 0.16451 | 0.13702 | 0.19433 |  |
| caption_only:single | previous_target_floor:single | mse | -0.00035 | -0.00313 | 0.00173 |  |
| caption_only:single | previous_target_floor:single | region_token_cosine | -0.14487 | -0.18525 | -0.10403 |  |
| caption_only:single | copy_f8_floor:single | lpips | 0.10337 | 0.08854 | 0.11894 |  |
| caption_only:single | copy_f8_floor:single | region_lpips | 0.40961 | 0.38423 | 0.43658 |  |
| caption_only:single | copy_f8_floor:single | mse | 0.00411 | 0.00291 | 0.00563 |  |
| caption_only:single | copy_f8_floor:single | region_token_cosine | -0.39222 | -0.40887 | -0.37583 |  |
| frame_only:single | all_black_floor:single | lpips | 0.03472 | 0.02453 | 0.04512 |  |
| frame_only:single | all_black_floor:single | region_lpips | 0.12022 | 0.09941 | 0.14034 |  |
| frame_only:single | all_black_floor:single | mse | 0.00107 | 0.00057 | 0.00167 |  |
| frame_only:single | all_black_floor:single | region_token_cosine | 0.03938 | 0.02243 | 0.05686 |  |
| frame_only:single | mean_map_floor:single | lpips | 0.03472 | 0.02453 | 0.04512 |  |
| frame_only:single | mean_map_floor:single | region_lpips | 0.07192 | 0.05593 | 0.08939 |  |
| frame_only:single | mean_map_floor:single | mse | 0.00107 | 0.00057 | 0.00167 |  |
| frame_only:single | mean_map_floor:single | region_token_cosine | 0.03938 | 0.02243 | 0.05686 |  |
| frame_only:single | previous_target_floor:single | lpips | 0.01272 | -0.01056 | 0.03191 |  |
| frame_only:single | previous_target_floor:single | region_lpips | 0.19380 | 0.16254 | 0.22525 |  |
| frame_only:single | previous_target_floor:single | mse | -0.00020 | -0.00305 | 0.00197 |  |
| frame_only:single | previous_target_floor:single | region_token_cosine | -0.12388 | -0.16119 | -0.08723 |  |
| frame_only:single | copy_f8_floor:single | lpips | 0.09115 | 0.07659 | 0.10585 |  |
| frame_only:single | copy_f8_floor:single | region_lpips | 0.43890 | 0.41368 | 0.46598 |  |
| frame_only:single | copy_f8_floor:single | mse | 0.00426 | 0.00297 | 0.00589 |  |
| frame_only:single | copy_f8_floor:single | region_token_cosine | -0.37124 | -0.38984 | -0.35129 |  |
| frame_caption:single | all_black_floor:single | lpips | 0.69473 | 0.67760 | 0.71010 |  |
| frame_caption:single | all_black_floor:single | region_lpips | 0.36089 | 0.32440 | 0.39977 |  |
| frame_caption:single | all_black_floor:single | mse | 0.35968 | 0.35520 | 0.36322 |  |
| frame_caption:single | all_black_floor:single | region_token_cosine | -0.05085 | -0.05823 | -0.04378 |  |
| frame_caption:single | mean_map_floor:single | lpips | 0.69473 | 0.67760 | 0.71010 |  |
| frame_caption:single | mean_map_floor:single | region_lpips | 0.31259 | 0.28467 | 0.34304 |  |
| frame_caption:single | mean_map_floor:single | mse | 0.35968 | 0.35520 | 0.36322 |  |
| frame_caption:single | mean_map_floor:single | region_token_cosine | -0.05085 | -0.05823 | -0.04378 |  |
| frame_caption:single | previous_target_floor:single | lpips | 0.67273 | 0.65022 | 0.69096 | H1 |
| frame_caption:single | previous_target_floor:single | region_lpips | 0.43447 | 0.39620 | 0.47326 | H1 |
| frame_caption:single | previous_target_floor:single | mse | 0.35842 | 0.35409 | 0.36215 | H1 |
| frame_caption:single | previous_target_floor:single | region_token_cosine | -0.21411 | -0.25015 | -0.17980 | H1 |
| frame_caption:single | copy_f8_floor:single | lpips | 0.75116 | 0.74479 | 0.75611 | H5 |
| frame_caption:single | copy_f8_floor:single | region_lpips | 0.67957 | 0.65208 | 0.70501 | H5 |
| frame_caption:single | copy_f8_floor:single | mse | 0.36288 | 0.35980 | 0.36533 | H5 |
| frame_caption:single | copy_f8_floor:single | region_token_cosine | -0.46146 | -0.47411 | -0.44936 | H5 |
| frame_caption:single | caption_only:single | lpips | 0.64779 | 0.62896 | 0.66482 | H2 |
| frame_caption:single | caption_only:single | region_lpips | 0.26995 | 0.23877 | 0.30209 | H2 |
| frame_caption:single | caption_only:single | mse | 0.35877 | 0.35444 | 0.36230 | H2 |
| frame_caption:single | caption_only:single | region_token_cosine | -0.06924 | -0.08355 | -0.05578 | H2 |
| caption_only:single | unconditional:single | lpips | -0.02763 | -0.03946 | -0.01603 | H3 |
| caption_only:single | unconditional:single | region_lpips | -0.00105 | -0.01033 | 0.00811 | H3 |
| caption_only:single | unconditional:single | mse | -0.00252 | -0.00332 | -0.00172 | H3 |
| caption_only:single | unconditional:single | region_token_cosine | 0.02552 | 0.01416 | 0.03799 | H3 |
| unconditional:best_of_k | unconditional:single | lpips | -0.07438 | -0.08219 | -0.06646 |  |
| unconditional:best_of_k | unconditional:single | region_lpips | -0.05198 | -0.05986 | -0.04449 |  |
| unconditional:best_of_k | unconditional:single | mse | -0.00385 | -0.00466 | -0.00316 |  |
| unconditional:best_of_k | unconditional:single | region_token_cosine | 0.06236 | 0.05424 | 0.07120 |  |
| caption_only:best_of_k | caption_only:single | lpips | -0.01008 | -0.01180 | -0.00847 |  |
| caption_only:best_of_k | caption_only:single | region_lpips | -0.02286 | -0.02693 | -0.01925 |  |
| caption_only:best_of_k | caption_only:single | mse | -0.00020 | -0.00025 | -0.00015 |  |
| caption_only:best_of_k | caption_only:single | region_token_cosine | 0.02717 | 0.02144 | 0.03366 |  |
| frame_only:best_of_k | frame_only:single | lpips | -0.01632 | -0.02208 | -0.01166 |  |
| frame_only:best_of_k | frame_only:single | region_lpips | -0.03312 | -0.04272 | -0.02390 |  |
| frame_only:best_of_k | frame_only:single | mse | -0.00062 | -0.00086 | -0.00043 |  |
| frame_only:best_of_k | frame_only:single | region_token_cosine | 0.03603 | 0.02638 | 0.04623 |  |
| frame_caption:best_of_k | frame_caption:single | lpips | -0.00790 | -0.00842 | -0.00738 | H4 |
| frame_caption:best_of_k | frame_caption:single | region_lpips | -0.01193 | -0.01336 | -0.01059 | H4 |
| frame_caption:best_of_k | frame_caption:single | mse | -0.00765 | -0.00813 | -0.00720 | H4 |
| frame_caption:best_of_k | frame_caption:single | region_token_cosine | 0.00548 | 0.00498 | 0.00595 | H4 |

## Pixel and token conclusions

Region LPIPS supports a hypothesis when its upper bound is below 0; Region Token Cosine when its lower bound is above 0. A disagreement is flagged here and not resolved.

| hypothesis | a | b | Region LPIPS upper bound | pixel | Region Token Cosine lower bound | token | |
|---|---|---|---|---|---|---|---|
| H1 | frame_caption:single | previous_target_floor:single | 0.47326 | not supported | -0.25015 | not supported | agree |
| H2 | frame_caption:single | caption_only:single | 0.30209 | not supported | -0.08355 | not supported | agree |
| H3 | caption_only:single | unconditional:single | 0.00811 | not supported | 0.01416 | supported | DISAGREE |
| H4 | frame_caption:best_of_k | frame_caption:single | -0.01059 | supported | 0.00498 | supported | agree |
| H5 | frame_caption:single | copy_f8_floor:single | 0.70501 | not supported | -0.47411 | not supported | agree |

Disagreement between pixel and token conclusions: H3.

Panels: panels/ (t78_1187, t69_218, t117_1539, t75_1525).
