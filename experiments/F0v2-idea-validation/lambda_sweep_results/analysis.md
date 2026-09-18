# Semantic-loss lambda sweep

Fixed 100-step training on the same F0v2 manifest and seed; evaluation uses the same 64 image-disjoint held-out records.

| variant | lambda_sem | pointing | mass-in-box | top20 box IoU |
|---|---:|---:|---:|---:|
| V3 | 0.00 | 0.3438 | 0.3255 | 0.2722 |
| V4 | 0.00 | 0.3750 | 0.3328 | 0.2701 |
| V3 | 0.01 | 0.3281 | 0.3254 | 0.2698 |
| V4 | 0.01 | 0.2969 | 0.3246 | 0.2675 |
| V3 | 0.03 | 0.2812 | 0.3273 | 0.2700 |
| V4 | 0.03 | 0.2500 | 0.3243 | 0.2688 |
| V3 | 0.10 | 0.3906 | 0.3365 | 0.2679 |
| V4 | 0.10 | 0.3438 | 0.3333 | 0.2696 |
| V3 | 0.30 | 0.3125 | 0.3137 | 0.2684 |
| V4 | 0.30 | 0.3125 | 0.3273 | 0.2662 |

Best directional V3 setting by mass then pointing: lambda=0.10. This is a selection result, not a final significance claim.

Best directional V4 setting by mass then pointing: lambda=0.10. This is a selection result, not a final significance claim.

## Interpretation

Use the sweep to decide whether a stable intermediate lambda improves both spatial metrics relative to lambda=0. If no lambda is consistently better, change map normalization/temperature before increasing data. Do not select a lambda using only one metric or one variant.

## Decision

For V3, lambda=0.10 is the best directional point on mass-in-box (+0.0110 over lambda=0) and pointing (+0.0469), but its IoU is lower by 0.0043. For V4, lambda=0.10 improves mass only marginally (+0.0005) while pointing and IoU decrease. Lambda=0.30 degrades V3 mass substantially and does not rescue V4. Thus the sweep does not show a lambda that improves both spatial metrics in both variants. It supports a narrow V3 lambda=0.10 follow-up, but does not justify F1-10k or claim a validated method. The next experiment should change map conditioning (temperature/normalization) while keeping lambda fixed at 0.10 for V3, with V4 retained as an interaction check.
