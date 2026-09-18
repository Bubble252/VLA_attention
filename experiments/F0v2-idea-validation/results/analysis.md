# F0 idea validation analysis

## Scope

64 image-disjoint Flickr30k Entities test records; phrase-score gradient×activation; same processor and evaluation code. 20-step is an engineering smoke; 100-step is the directionality check.

## Results

| Run | pointing | mass-in-box | top20 box IoU |
|---|---:|---:|---:|
| V0_20 | 0.2812 | 0.3095 | 0.2693 |
| V1_20 | 0.3594 | 0.3183 | 0.2692 |
| V2_20 | 0.3594 | 0.3313 | 0.2696 |
| V3_20 | 0.2969 | 0.3197 | 0.2683 |
| V4_20 | 0.3594 | 0.3210 | 0.2682 |
| V1_100 | 0.3594 | 0.3378 | 0.2680 |
| V2_100 | 0.3438 | 0.3284 | 0.2681 |
| V3_100 | 0.3906 | 0.3365 | 0.2679 |
| V4_100 | 0.3438 | 0.3333 | 0.2696 |

## Paired 100-step deltas

V3 − V1: pointing improves, while mass-in-box and IoU do not show a positive mean delta. V4 − V2: mass-in-box and IoU improve, while pointing is not improved.

Bootstrap intervals use 10,000 paired resamples with seed 23. These are directional intervals, not a final large-scale significance claim.

## Teacher controls

Correct SD phrase maps outperform wrong-image and random controls on pointing and mass-in-box. The shifted control remains close to correct on IoU, so the current IoU negative control is not clean enough to support a strong causal claim.

## Decision

The 100-step run provides partial evidence: semantic attribution alignment can improve some spatial metrics, especially when combined with retention, but the full V3/V4 success criterion is not met. Do not launch F1-10k unchanged. First revise the loss strength/schedule or teacher-map evaluation, add a clean negative control, and rerun a small controlled sweep.
