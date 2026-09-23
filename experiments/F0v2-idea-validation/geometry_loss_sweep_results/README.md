# F0v2 geometry-aware semantic loss sweep

## Scope and evidence

Four semantic-map objectives were trained on the same F0v2 256-pair manifest for 100 steps, using seeds 17, 29, and 41, `lambda_sem=0.1`, teacher temperature `1.25`, rank quantile `0.25`, and one held-out image-disjoint set of 64 examples. The training/evaluation reports and checkpoints are on VEPFS; the aggregated remote summary was generated at `results/F0v2_geometry_heldout/summary.{json,md}`.

The current local code supports fixed-grid map matching, KL, KL+rank, KL+moment, and JS. Unit tests passed on the 101 environment (`8 passed`). The 20-image independent validation calibration compared 16×16, 32×32, and 64×64 and provisionally selected 16×16 and threshold quantile 0.95. It measured student V3 soft-IoU of 0.0755/0.0604/0.0544 respectively, and top-20 box IoU of 0.3177/0.3170/0.3164. A portable audit copy is stored in [`calibration_v1.json`](calibration_v1.json); the source artifact is on VEPFS at `results/F0v2_geometry_calibration_V3_val/calibration.json`.

## Important resolution caveat

The 12 completed loss-sweep jobs were launched before the runner was changed to use the selected 16×16 grid. They trained with a common **32×32** grid. Their held-out maps were evaluated at both native top-k and calibrated **16×16** resolution. Therefore the four-way comparison is controlled among loss modes, but it is not yet the final “calibrated-grid training” comparison. A second sweep using the current `server/run_f0_geometry_loss_sweep.sh` default (`F0_GEOM_RESOLUTION=16`) is required before selecting a final loss.

The validation split is only 20 images and the held-out threshold evaluation is sensitive to the chosen support quantile. Keep 16×16/q=.95 as a frozen provisional rule for the next run; do not retune it on held-out test. Report top-10/20/30/40 support IoU, soft-IoU, mass-in-box and pointing together. Do not elevate q=.95 IoU alone as the objective.

## 32×32 training sweep: held-out results

All values below are mean ± sample standard deviation across 3 training seeds on the same 64 held-out image-phrase records. “Calibrated IoU” means the 16×16/q=.95 readout; it is an evaluation readout of models trained at 32×32, not the final calibrated-grid training result.

| Loss | Seeds | Calibrated IoU | Soft-IoU | Top-20 IoU | Mass-in-box | Pointing |
|---|---|---:|---:|---:|---:|---:|
| KL | 17, 29, 41 | 0.2838 ± 0.0064 | 0.0748 ± 0.0040 | 0.2709 ± 0.0015 | 0.3266 ± 0.0089 | 0.3438 ± 0.0563 |
| KL + rank | 17, 29, 41 | 0.2758 ± 0.0051 | **0.0799 ± 0.0062** | 0.2732 ± 0.0009 | 0.3306 ± 0.0061 | 0.3385 ± 0.0861 |
| KL + moment | 17, 29, 41 | **0.2893 ± 0.0073** | 0.0790 ± 0.0027 | **0.2748 ± 0.0011** | 0.3304 ± 0.0045 | 0.3333 ± 0.0180 |
| JS | 17, 29, 41 | 0.2812 ± 0.0103 | 0.0711 ± 0.0053 | 0.2717 ± 0.0013 | 0.3208 ± 0.0096 | 0.3229 ± 0.0771 |

## Interpretation and next gate

- KL+moment is the best provisional candidate on calibrated IoU, top-20 IoU and seed consistency of mass-in-box, but its calibrated-IoU gain over KL is only 0.0055 and the four loss modes have not yet been compared against V1/V2 under the exact same calibrated readout.
- KL+rank gives the strongest soft-IoU and mass-in-box means, but not the best calibrated q=.95 box IoU. This divergence confirms that one scalar metric cannot select the method.
- JS is not better than KL on this run.
- These are three-seed directional results, not statistical proof; the same 64 images are shared across seeds, and the table does not yet include paired bootstrap intervals.

Required before selecting a final loss:

1. Re-run the four loss modes at 16×16 training resolution, preserving all other settings and seeds.
2. Evaluate V1 and V2 on the same 64 records, with the frozen 16×16/q=.95 protocol; compare each method to its matched baseline.
3. Compute paired bootstrap intervals by sample for calibrated IoU, soft-IoU, top-k IoU, mass-in-box, and pointing.
4. Only then begin Lavender-style downstream capability evaluation using paired V0–V4 checkpoints. Capability scores cannot substitute for missing spatial gains.

No claim of IoU improvement or idea validation should be made from the provisional 32×32 sweep alone.

## Matched V1/V2 baseline status

V1 and V2 were re-evaluated on the same 64 held-out rows with the frozen 16×16/q=.95 rule, and reports exist on VEPFS at `results/F0v2_geometry_heldout/baseline_V1_gpu1/report.json` and `baseline_V2_gpu1/report.json`. The remote approval service began returning 502 before their summary fields could be captured into this workspace. Treat the baseline comparison as **not yet auditable here**; do not infer the result from partial console excerpts. Re-read the full JSON summaries and compute paired sample-level intervals before selecting a loss.

## Downstream capability evaluation status

The Lavender-style benchmark protocol is defined in `paper_draft/05_experiments_and_expected_conclusions.md` and `doc/03_execution_plan.md`, but no capability benchmark has been run yet. The next stage remains pending until (a) the 16×16 training-grid sweep is complete, (b) V1/V2 matched baselines and paired intervals are captured, and (c) one candidate is selected without using test results for calibration. Then run the six-task paired V0–V4 pilot: COCO Captions, VQAv2, TextVQA, POPE, MME, and WorldMedQA-V. Report task-native metrics; do not collapse incomplete benchmark groups into one average.

## 2026-09-23 calibrated-grid result

The actual 16×16 retraining sweep and matched V1-by-seed baseline were completed after remote access recovered. Unlike the earlier `geom16_*`-named held-out readouts, these models were trained with `common_resolution=[16,16]`. All four modes ran 100 steps for seeds 17/29/41, with λ=.1, teacher temperature 1.25, same 256-pair training manifest (SHA256 `2a37c49e…`), and same 64-row image-disjoint held-out manifest (SHA256 `7b16778e…`). V1 seed 17 uses the existing 100-step V1 trained at seed 17; V1 seeds 29 and 41 were newly trained as matched baselines. Each method was evaluated at the frozen 16×16 grid / q=.95 support rule.

| Loss | q95 box IoU | soft-IoU | top-10 IoU | top-20 IoU | mass-in-box | pointing |
|---|---:|---:|---:|---:|---:|---:|
| KL | 0.2754 ± 0.0297 | 0.0683 ± 0.0187 | 0.2805 ± 0.0111 | 0.2726 ± 0.0042 | 0.3247 ± 0.0067 | 0.3073 ± 0.0888 |
| KL+rank | 0.2853 ± 0.0049 | **0.0879 ± 0.0016** | **0.2845 ± 0.0036** | 0.2719 ± 0.0044 | **0.3357 ± 0.0057** | **0.3854 ± 0.0477** |
| KL+moment | 0.2789 ± 0.0034 | 0.0747 ± 0.0042 | 0.2799 ± 0.0015 | 0.2720 ± 0.0027 | 0.3199 ± 0.0070 | 0.2812 ± 0.0000 |
| JS | **0.2913 ± 0.0132** | 0.0706 ± 0.0032 | 0.2754 ± 0.0003 | 0.2712 ± 0.0023 | 0.3242 ± 0.0043 | 0.3177 ± 0.0786 |

Numbers are mean ± sample SD across three training seeds; full top-k/q curves, individual 64-row reports, training loss traces, and checksums are in `calibrated16_20260923/`. Paired bootstrap results compare each loss to same-seed V1 and resample the same 64 held-out IDs within each of the three fixed seeds (10,000 draws, RNG seed 20260923). These CIs reflect held-out sample uncertainty conditional on the selected seeds, not uncertainty over training seeds.

KL+rank is the leading next candidate because it improves soft-IoU and top-10 IoU over matched V1 with paired 95% sample-bootstrap intervals above zero; mass-in-box and pointing also trend upward. Its primary q=.95 box-IoU difference is only +0.0018 with CI [−0.0144,+0.0174], so box-localization superiority is **not** established. JS has the highest mean q=.95 IoU but its paired interval crosses zero; KL+moment improves soft-IoU but decreases q=.95 IoU and pointing. The data support carrying KL+rank to the Lavender-style capability pilot as a candidate, not claiming a confirmed win. The 100-step budget and only three seeds remain an important limitation.

The original 32×32-trained sweep remains separately archived in the parent README table and is not pooled with these results. The raw remote reports in `remote_artifacts_20260923/` are calibration evidence plus the original 32×32 held-out reports; the actual calibrated-grid training/evaluation artifacts are in `calibrated16_20260923/`.
