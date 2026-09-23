# Calibrated 16×16 geometry sweep (100 steps)

All four losses used the same F0v2 256-pair training manifest, lambda=0.1, teacher temperature=1.25, and seeds 17/29/41. Each model was evaluated on the same image-disjoint 64-row held-out set at 16×16; teacher/student map normalization, top-k, soft-IoU and q=.95 selection were frozen from the independent 20-image validation calibration. Values below are mean ± sample SD across training seeds.

| Loss | q95 box IoU | soft-IoU | top-10 IoU | top-20 IoU | top-30 IoU | top-40 IoU | mass-in-box | pointing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| kl | 0.2754 ± 0.0297 | 0.0683 ± 0.0187 | 0.2805 ± 0.0111 | 0.2726 ± 0.0042 | 0.2692 ± 0.0004 | 0.2687 ± 0.0001 | 0.3247 ± 0.0067 | 0.3073 ± 0.0888 |
| kl_rank | 0.2853 ± 0.0049 | 0.0879 ± 0.0016 | 0.2845 ± 0.0036 | 0.2719 ± 0.0044 | 0.2692 ± 0.0033 | 0.2687 ± 0.0009 | 0.3357 ± 0.0057 | 0.3854 ± 0.0477 |
| kl_moment | 0.2789 ± 0.0034 | 0.0747 ± 0.0042 | 0.2799 ± 0.0015 | 0.2720 ± 0.0027 | 0.2687 ± 0.0013 | 0.2686 ± 0.0004 | 0.3199 ± 0.0070 | 0.2812 ± 0.0000 |
| js | 0.2913 ± 0.0132 | 0.0706 ± 0.0032 | 0.2754 ± 0.0003 | 0.2712 ± 0.0023 | 0.2698 ± 0.0008 | 0.2693 ± 0.0006 | 0.3242 ± 0.0043 | 0.3177 ± 0.0786 |

## Paired comparison against V1

For each seed, each loss was compared to a V1 caption-SFT checkpoint trained with the same seed (seed 17 uses the existing V1 checkpoint; seeds 29/41 were newly trained for 100 steps). The 10,000-resample percentile bootstrap pairs the same 64 held-out IDs within seed, then averages the three fixed seed-level deltas. These intervals quantify held-out-example uncertainty conditional on these three seeds; they are not a training-seed population CI.

| Loss | q95 IoU Δ [95% CI] | soft-IoU Δ [95% CI] | top-10 IoU Δ [95% CI] | mass-in-box Δ [95% CI] | pointing Δ [95% CI] |
|---|---:|---:|---:|---:|---:|
| kl | -0.0080 [-0.0244, +0.0081] | +0.0008 [-0.0053, +0.0070] | +0.0080 [-0.0027, +0.0188] | +0.0037 [-0.0060, +0.0139] | +0.0000 [-0.0781, +0.0781] |
| kl_rank | +0.0018 [-0.0144, +0.0174] | +0.0204 [+0.0149, +0.0262] | +0.0120 [+0.0027, +0.0217] | +0.0147 [+0.0047, +0.0246] | +0.0781 [+0.0000, +0.1562] |
| kl_moment | -0.0046 [-0.0197, +0.0104] | +0.0073 [+0.0007, +0.0138] | +0.0073 [-0.0013, +0.0165] | -0.0011 [-0.0126, +0.0103] | -0.0260 [-0.0990, +0.0469] |
| js | +0.0078 [-0.0062, +0.0222] | +0.0031 [-0.0021, +0.0084] | +0.0029 [-0.0033, +0.0090] | +0.0032 [-0.0071, +0.0137] | +0.0104 [-0.0625, +0.0833] |

## Reading the result

- **KL+rank is the best next capability-pilot candidate**, with positive sample-bootstrap intervals for soft-IoU and top-10 IoU, and positive deltas for mass-in-box/pointing. Its q=.95 calibrated box-IoU interval crosses zero, so localization-box superiority is not established.
- KL+moment consistently lowers pointing and q=.95 box IoU versus its matched V1 in this 100-step run, despite improving soft-IoU. It should not be selected by soft-IoU alone.
- JS improves the q=.95 mean numerically but its interval crosses zero and its soft-IoU gain is not decisive.
- KL has no clear gain over V1; q=.95 deltas vary in sign across seeds.
- These comparisons do not establish statistical superiority across training seeds (only three, all short 100-step runs). Treat the bootstrap as paired held-out-sample uncertainty, and confirm the selected candidate on a larger independent test / downstream benchmark.

## Reproducibility artifacts

Raw 64-row report JSONs and per-seed training config/loss logs are stored alongside this file. The paired bootstrap JSONs use 10,000 resamples and fixed RNG seed 20260923. Teacher/student calibration evidence and the previous 32×32-only comparison remain in the parent directory; the 32×32 run is not mixed into this table.
