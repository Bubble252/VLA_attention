# Lavender-style downstream capability pilot (2026-09-23)

## Scope and protocol

Five Qwen2.5-VL-7B checkpoints were evaluated with deterministic decoding (`do_sample=false`, max_new_tokens=64) and a common image budget of 1,003,520 pixels. Each task uses the same fixed pilot IDs across checkpoints. The comparison is a capability pilot, not a full official benchmark run. V0 is base; V1 caption SFT; V2 caption SFT + DINOv2 retention; V3_KLrank caption SFT + calibrated semantic map KL+rank; V4_KLrank V2 + KL+rank. V3/V4 use one seed-17 100-step checkpoint each. Their results are therefore single-checkpoint diagnostics; uncertainty intervals do not include training-seed variation.

Pilot sizes: COCO Captions 128 images; VQAv2 sample 128; TextVQA 128; POPE 384 (128 each adversarial/popular/random); MME 64 complete question pairs (128 questions); WorldMedQA-V 256 examples.

VQA/TextVQA use an internal leave-one-annotator-out consensus implementation with generic normalization, not the official evaluator. WorldMedQA is a deterministic multilingual/country-stratified pilot sample. Caption METEOR was unavailable because Java was not installed. Run official benchmark scoring and full official splits before paper claims.

## Descriptive metrics

| Checkpoint | COCO CIDEr | VQAv2 consensus | TextVQA consensus | POPE accuracy | MME pair accuracy / score | WorldMedQA-V accuracy |
|---|---:|---:|---:|---:|---:|---:|
| V0 | 0.095 | 0.283 | 0.319 | 0.849 | 0.703 / 109 | 0.418 |
| V1 | 0.778 | 0.702 | 0.660 | 0.841 | 0.703 / 108 | 0.469 |
| V2 | 0.744 | 0.705 | 0.689 | 0.846 | 0.703 / 109 | 0.461 |
| V3_KLrank | 0.891 | 0.159 | 0.367 | 0.852 | 0.734 / 109 | 0.406 |
| V4_KLrank | 0.831 | 0.734 | 0.687 | 0.846 | 0.719 / 109 | 0.480 |

## Paired change from V1

Paired bootstrap resamples the same example IDs (10,000 resamples; seed 20260923). MME resamples 64 complete two-question pairs. Intervals quantify sampling uncertainty for these fixed checkpoints only, not variation across training seeds. V0 is shown as contextual baseline; central scientific contrasts are V3 vs V1 and V4 vs V2.

| Candidate | VQAv2 Δ [95% CI] | TextVQA Δ [95% CI] | POPE Δ [95% CI] | MME pair acc Δ [95% CI] | WorldMedQA-V Δ [95% CI] |
|---|---:|---:|---:|---:|---:|
| V0 | -0.419 [-0.508, -0.333] | -0.341 [-0.425, -0.259] | +0.008 [+0.000, +0.018] | +0.000 [-0.062, +0.062] | -0.051 [-0.098, -0.004] |
| V2 | +0.003 [-0.059, +0.065] | +0.029 [-0.007, +0.070] | +0.005 [-0.005, +0.016] | +0.000 [-0.062, +0.062] | -0.008 [-0.055, +0.035] |
| V3_KLrank | -0.543 [-0.625, -0.461] | -0.293 [-0.381, -0.204] | +0.010 [-0.003, +0.026] | +0.031 [-0.031, +0.094] | -0.062 [-0.121, -0.004] |
| V4_KLrank | +0.033 [-0.027, +0.093] | +0.027 [-0.033, +0.090] | +0.005 [-0.005, +0.016] | +0.016 [-0.047, +0.078] | +0.012 [-0.035, +0.055] |

## Interpretation

- Caption SFT (V1) materially improves the pilot language-and-vision metrics over the unadapted V0 checkpoint. This is a sanity check that the caption task/training path is active, not a claim of official benchmark performance.
- V2 retention is close to V1 on VQAv2, TextVQA, POPE and MME; its WorldMedQA point estimate is slightly lower, with paired interval crossing zero. On this pilot it does not show a clear broad capability gain or cost.
- V3_KLrank has the strongest caption score and a small POPE/MME point-estimate gain, but VQAv2 and TextVQA consensus scores drop sharply versus V1, and WorldMedQA is lower. The intervals exclude zero for the two VQA-style declines and WorldMedQA in this sample. This is a material capability trade-off and prevents claiming that the semantic map loss is capability-neutral.
- V4_KLrank is more balanced: its direct paired changes against V2 are VQAv2 +0.030 [-0.026, +0.088], TextVQA -0.002 [-0.053, +0.051], POPE 0.000 [0.000, 0.000], MME pair accuracy +0.016 [-0.047, +0.094], and WorldMedQA-V +0.020 [-0.027, +0.063]. These intervals all include zero (POPE predictions were identical on this pilot), so the semantic term has no established capability increment over DINO retention.
- Raw TextVQA outputs show a concrete V3 failure mode: repeated punctuation/continuation on several answers (e.g., the answer “atomic” followed by a long run of periods); a simple diagnostic flagged 28/128 V3 TextVQA outputs with at least 18 tokens or four identical consecutive tokens, compared with 3/128 V1, 1/128 V2, and 0/128 V4. The heuristic is only a degeneration indicator, not a benchmark metric. Inspect generation, prompt formatting, and training dynamics before calling the V3 VQA decline forgetting. This does not appear to be only an answer normalizer effect.

## Corrected MME provenance

The first main-pilot V0 and V2 MME reports used a binary-question schema while the others used paired-question scoring, so those initial files must not be compared. All five were rerun under `P4_capability_pilot_mme_pairedfix_20260923`; fixed manifest SHA256 is `618989cdf66949f1b462df3e3f91c6cc7d945ee6865109e84d28ad38f0360ab` (128 rows / 64 complete pairs). Use only the corrected MME reports in the table above.

## Reproducibility artifacts

- Raw five-by-six pilot reports: `experiment_workspace/results/P4_capability_pilot_20260923_v2/`.
- Corrected all-checkpoint MME reports: `experiment_workspace/results/P4_capability_pilot_mme_pairedfix_20260923/`.
- Paired-bootstrap results versus V1: `paired_bootstrap_vs_V1.json`; direct V4-vs-V2: `paired_bootstrap_v4_vs_v2.json`.
- All five-by-six main reports record manifest SHA, ordered ID/prompt SHA, image budget, deterministic decoding, per-sample latency and CUDA peak memory.

## Next evidence needed

1. Review V3 VQA/TextVQA raw predictions and confirm the drop reflects real answer degradation rather than prompt/parser normalization artifacts.
2. The direct V4-vs-V2 paired bootstrap is complete for the five non-caption tasks; run a paired corpus-caption bootstrap if caption is used as an inferential endpoint.
3. Extend downstream evaluation to at least three training seeds for V3/V4, then full official validation/test splits with official scorers; add blinded/random/correct teacher-map controls as planned.
4. Only promote a claim if downstream capability gains (or an explicitly quantified trade-off) align with geometry results and replicate across seeds.
