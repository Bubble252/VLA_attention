# F0 500-step confirmation

The 500-step run uses the same 256 training rows, seed 17, loss weights, and 64 image-disjoint held-out records. All four training loss traces are finite and each checkpoint was restored for evaluation.

| model | pointing | mass-in-box | top20 box IoU |
|---|---:|---:|---:|
| V1 | 0.2969 | 0.3284 | 0.2698 |
| V2 | 0.3906 | 0.3282 | 0.2690 |
| V3 | 0.3750 | 0.3312 | 0.2685 |
| V4 | 0.2344 | 0.2678 | 0.2692 |

## Decision

V3 is below V1 on pointing and IoU but slightly above on mass-in-box. V4 is below V2 on pointing and mass-in-box and nearly unchanged on IoU. The 500-step confirmation therefore does not support the required `V3 > V1` and `V4 > V2` claim. The semantic loss is not yet a validated improvement under this configuration. The correct SD teacher map still beats same-image wrong-word, wrong-image, and random controls on pointing and mass, so the teacher signal itself is meaningful; the failure is in transferring that signal through the current training objective. Do not start F1-10k unchanged.

Next research action: audit semantic-loss gradient scale and schedule, verify map normalization/temperature, then run a small lambda sweep with the same held-out protocol. Keep all current results as negative/partial evidence; do not delete them.
