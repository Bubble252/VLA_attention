# Falsification gate

## Required controls

Every trainable arm shares backbone, action head, initialization, demonstrations, valid action targets, optimizer updates, effective batch, augmentation count, evaluator, seeds and teacher/cache calls. If the candidate needs new paired expert rollouts, the same rollouts enter every baseline arm.

## Success gate for a future method claim

All must hold on held-out episode clusters:

1. Correct-target selection and official rollout improve over B1, static semantic, RS-CL and inverse-dynamics baselines.
2. `I_bind` is positive with clustered CI and consistent across seeds/intervention operators.
3. Correct maps exceed wrong-map, shuffled-instruction and uniform/full-image controls.
4. ID success does not exceed the preregistered acceptable degradation.
5. Improvements remain after matching demonstrations, augmentation and compute.

## Immediate failure conditions

- Only attribution or `I_bind` improves; actions/rollouts do not.
- Wrong and correct maps are equivalent, or full-image pooling matches language selection.
- RS-CL/inverse dynamics matches the gain under equal budget.
- Instruction changes action but target-evidence intervention does not reverse accordingly.
- No legal target-retargeting action labels exist for training.
- Results depend on one OOD mask operator.

Current result: training-method gate is failed by data provenance and close prior work. The diagnostic protocol remains testable.
