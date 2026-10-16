# Causal graph and alternative explanations

The planned quantity is an **instruction–evidence action-sensitivity interaction**, not an identified causal effect.

```text
instruction ─┬─> language state ─┬─> predicted action
             └─> target binding ─┤
image objects ─> visual evidence ┤
proprio/state ───────────────────┤
phase/history ──────────────────>┤
layout/task template ───────────>┤
training data/augmentation ─────> learned policy
intervention operator ──────────> OOD artifacts ─> predicted action
```

Alternative explanations to rule out:

1. Language-only shortcut: action changes with words while target-specific evidence has no interaction.
2. Saliency-only shortcut: the same visual target dominates under both instructions.
3. Object-location/template shortcut: instruction and location remain confounded.
4. State/time shortcut: proprioception or phase predicts the action without target evidence.
5. Budget/data explanation: added demonstrations or augmentation, not the method, cause gains.
6. Explanation-only gain: attribution/interaction improves but target selection and rollout do not.
7. Intervention artifact: mask/inpainting changes the input distribution or geometry beyond evidence removal.

The factorial design blocks the first three only when instruction, scene, state and intervention budget are paired. State/time probes, matched training budgets and multiple intervention operators address the remaining alternatives.
