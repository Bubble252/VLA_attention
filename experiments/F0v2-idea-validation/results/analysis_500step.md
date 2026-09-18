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

## Why this is the next experiment

The SD teacher map passes the spatial controls, but the student-side semantic loss does not produce a stable V3/V4 gain. This leaves two separable hypotheses:

1. **Signal transfer:** `lambda_sem=0.1` may be too weak, too strong, or introduced too early. In the 500-step logs V3 semantic loss remains around 1; in V4 it competes with the retention term. Gradient clipping also means the raw loss value does not directly reveal the parameter update.
2. **Map conditioning:** the current KL is applied to normalized raw `abs(gradient × activation)` against a 16×16 SD map. Its optimization behavior depends on epsilon, map sharpness, interpolation, and the second-order attribution path. A teacher can be meaningful for evaluation while this representation is poorly conditioned for training.

Therefore first hold map construction fixed and compare `lambda_sem ∈ {0, 0.01, 0.03, 0.1, 0.3}`. Record caption/semantic loss and gradient norms before and after clipping, then evaluate the same held-out split. Only a promising lambda range should proceed to temperature or normalization variants. This separates optimization scale from map formulation instead of treating any single metric fluctuation as a method gain.
