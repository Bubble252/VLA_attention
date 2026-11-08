# Goal17 E1 release/settle amendment

**Date**: 2026-11-08<br>
**Status**: `AMENDMENT_DRAFTED_BEFORE_CALIBRATION`<br>
**Scope**: execution wrapper only. Grounding Gate G-A and its threshold are unchanged.

## 1. What the first Goal17 run showed

Grounding was repaired:

| arm | white_right destination <=0.15 m | object in mask | destination margin | object margin |
|---|---:|---:|---:|---:|
| `G3-BASE` | 0/15 | 0/15 | 0/15 | 0/15 |
| `G4-NO-MARGIN` | 15/15 | 15/15 | 15/15 | 12/15 |
| `G4-MARGIN` | 15/15 | 15/15 | 14/15 | 12/15 |
| `G3-MARGIN` (Track C) | 0/15 | 5/15 | 0/15 | 4/15 |

`G4-MARGIN` passes G-A. Track C still fails, so the correct wording is **deployable grounding repaired**, not **compositional generalization solved**.

The first E1 execution run did not pass G-B:

| arm | success | released success | hold-only | target destination correct | wrong destination |
|---|---:|---:|---:|---:|---:|
| `B0` | 0/15 | 0/15 | 0/15 | 0/15 | 14/15 |
| `E1-ORACLE` | 4/15 | 1/15 | 3/15 | 10/15 | 3/15 |
| `E1-PRED` | 0/15 | 0/15 | 0/15 | 10/15 | 3/15 |

## 2. Failure localisation

`E1-PRED` did not fail at grounding: 10/15 reached the correct destination and 9/15 reached released placement. It failed after release.

The current wrapper uses `release.settle_steps = 4`. In `src/vla_attention/reguide_rebinding.py`, `settle_steps` is the total number of release-action emissions, including the first one. The rollout then terminates as soon as `settle_complete_step` is set. In the observed traces, predicted-destination rollouts therefore stop after four release actions. Several oracle rollouts succeed in the same window, but predicted-destination rollouts release at a slightly different point and have no time to register a stable placement.

This yields two competing hypotheses.

### H-SETTLE

The object is already at `plate_2`, but four release actions are too short for the simulator success predicate to fire. Extending the post-release hold should convert released placements into official successes.

- If true: longer settle increases `released_success` without changing destination error or wrong-destination count.
- If false: objects remain above the plate or fall away, and longer settle does not increase success.

### H-EXEC-GEOM

The predicted destination is accurate enough for the scoring threshold but not for the execution/release geometry. The release point is near the plate boundary, so extending settle cannot recover the rollout.

- If true: success remains zero even with long settle, and failures correlate with larger destination error / release XY error.
- If false: longer settle converts the existing released placements into success.

## 3. Calibration protocol

Only the three pre-registered calibration inits are used to choose `settle_steps`:

```text
init_0e94a4e6b9622c6b
init_2edf49999eb35566
init_3500df19a4d3b5a1
```

Candidates:

```text
4    existing full-run baseline
8
16
32
```

For each candidate, run `E1-PRED` only with:

```bash
SETTLE_STEPS=<N> bash server/run_goal17_grounding.sh calib-settle
```

Selection rule, frozen before seeing the calibration results:

1. reject any candidate with errors, NaN, or object fly-out;
2. among clean candidates, choose the smallest `settle_steps` with at least one official success and at least one released success on the three calibration inits;
3. if no candidate gets a success, stop tuning settle and record H-SETTLE as falsified; move to release-geometry / predicted-point analysis rather than extending the hold indefinitely.

## 4. Confirmation protocol

After the calibration choice is fixed, run the full 15-init E1 comparison with that single value:

```bash
SETTLE_STEPS=<N> bash server/run_goal17_grounding.sh eval-e1-settle
```

Report both:

- all 15 inits for the pre-registered Goal17 Gate;
- the 12 confirmation inits separately, because the three calibration inits were used to select the settle length.

G-B is unchanged: `E1-PRED success >=1/15`, `E1-PRED - B0 >= +1`, `released success >=1/15`, wrong-destination not above B0, and 0 error / NaN / fly-out.

## 5. What this amendment does not allow

- no change to G-A thresholds;
- no change to the frozen grounding checkpoint;
- no oracle fallback in `E1-PRED`;
- no tuning on the 12 confirmation inits;
- no claim that Track C supports compositional generalization if it remains 0/15.
