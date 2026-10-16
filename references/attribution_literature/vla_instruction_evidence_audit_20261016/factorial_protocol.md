# Factorial protocol

## Unit and factors

The statistical unit is an episode/init-state cluster, never a frame. For a scene with legal goals A/B:

- Instruction: A, B, paraphrases A/B; invalid/no-target prompts are diagnostic-only.
- Evidence: clean, intervene A, intervene B, area/position-matched non-target, background/light control.
- State: exact simulator state where possible; otherwise predeclared matched-state strata. Report phase separately.

For policy action `a(I,E,s)`, define `D` as a dimension-normalized distance between clean and intervened H-step native action distributions. For deterministic regression, report first-action and full-chunk weighted distance; for stochastic/flow policies, use fixed noise seeds and distributional distances.

```text
C_A = D(intervene A | instruction A) - D(intervene B | instruction A)
C_B = D(intervene B | instruction B) - D(intervene A | instruction B)
I_bind = (C_A + C_B) / 2
```

The sign is chosen so larger means the corresponding target evidence matters more. Report episode-cluster bootstrap CI and seed consistency. Do not average incompatible action units without frozen normalization.

## Intervention validity

Use at least two operators (feature-token replacement and image-space inpaint/blur) plus area/position-matched non-target controls. Require the direction of `I_bind` to agree. An intervention that changes object geometry, contact feasibility or camera calibration invalidates the pair.

## Behavioral endpoints

Primary: correct-target selection, official rollout success, phase success (approach/grasp/place), paired action response, and `I_bind`. Offline MAE, attribution maps, patch interventions and probes are secondary.

## Interpretation

- Instruction effect without positive `I_bind`: likely language prior or template response.
- Evidence effect without instruction reversal: likely visual saliency/location shortcut.
- Positive `I_bind` without target/rollout gain: mechanism diagnostic only.
- Positive `I_bind` plus target/rollout gain over matched strong baselines: supports binding benefit, still not a formal causal estimate.
