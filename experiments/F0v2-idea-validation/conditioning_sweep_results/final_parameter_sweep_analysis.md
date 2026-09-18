# F0 semantic-loss parameter sweep: final audit

## Controlled scope

All runs use the same 256-row F0v2 training manifest, 64 image-disjoint held-out records, Qwen LoRA configuration, SD1.5 phrase cache, and attribution evaluator. The sweep examined the parameters that directly control semantic-loss transfer:

1. coarse `lambda_sem`: `0, 0.01, 0.03, 0.10, 0.30` for V3 and V4 at 100 steps;
2. V3 fine lambda near `0.10`: `0.06, 0.08, 0.12, 0.15`;
3. V3 teacher map temperature at `lambda=0.10`: `0.50, 0.75, 1.00, 1.25, 1.50`;
4. V3 semantic warm-up at `lambda=0.10, T=1.25`, independently trained with seed 29: `0, 20, 50` steps.

## Best observed configurations

| Setting | Seed | Pointing | Mass-in-box | IoU | What it shows |
|---|---:|---:|---:|---:|---|
| V3 lambda=0, T=1 | 17 | 0.3438 | 0.3255 | 0.2722 | semantic-off matched baseline |
| V3 lambda=0.10, T=1 | 17 | 0.3906 | 0.3365 | 0.2679 | best coarse lambda for pointing/mass, IoU lower |
| V3 lambda=0.10, T=1.25 | 17 | **0.4219** | 0.3324 | **0.2723** | only seed-17 candidate above semantic-off baseline on all three means; IoU gain is +0.00006 |
| V3 lambda=0.10, T=1.50 | 17 | 0.3750 | **0.3408** | 0.2692 | strongest mass, weaker IoU |
| V3 lambda=0.10, T=1.25, no warm-up | 29 | 0.3594 | 0.3295 | 0.2689 | repeats pointing/mass gain, IoU lower |
| V3 lambda=0.10, T=1.25, warm-up 50 | 29 | 0.3750 | 0.3371 | 0.2681 | warm-up improves pointing/mass versus same-seed no-warm-up, IoU lower |

## What the sweep establishes

- `lambda=0.30` harms V3 mass and does not help V4, so the semantic term cannot simply be made stronger.
- `lambda=0.10` is the useful scale range; smaller lambda values do not provide consistent gains.
- Softening the teacher to `T=1.25` improves V3 pointing and reaches the strongest seed-17 joint result. A sharper teacher (`T<1`) does not improve the joint result.
- The independent seed reproduces pointing/mass improvements but not IoU. The paired bootstrap intervals for candidate versus semantic-off baseline include zero for all metrics on both seeds.
- Warm-up affects the optimization path, so the semantic term is not inert; nevertheless it does not solve the IoU instability.

## Decision

The completed parameter sweep identifies a scientifically useful **partial candidate**: V3 with `lambda_sem=0.10`, teacher temperature `1.25`, and possibly 50-step warm-up. It does **not** establish a robust method configuration because all three spatial metrics do not improve consistently across independent seeds. Do not run F1-10k under the current claim.

The next method change should not be more scalar search. It should change the geometry of the teacher/student match: compare the current raw-mass KL to a rank/quantile-normalized or thresholded-region objective, then repeat only the selected candidate and its semantic-off control over at least three seeds. This directly tests the remaining failure mode: a spatially meaningful teacher map whose distributional KL is poorly aligned with the desired compact region geometry.
