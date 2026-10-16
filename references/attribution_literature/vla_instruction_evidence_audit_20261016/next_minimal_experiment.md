# Next minimal experiment

Run at most three experiments, in this order:

1. **Frozen task0 spatial-reference sanity**: exploit the two black bowls; compare the official successful OpenVLA/OFT checkpoint under the legal task0 instruction with target-bowl, other-bowl and area-matched non-target interventions. This checks evidence specificity but is not a target-retargeting pair.
2. **Same-scene Goal diagnostic**: instantiate `put bowl on plate` and `put bowl on stove` from an audited shared simulator state; run both legal task predicates/instructions with plate, stove and non-target interventions. Measure `I_bind`, first/chunk action response, correct target approach and rollout success. No training labels are transferred.
3. **Multi-object/plate pair**: use LIBERO-90 Scene5 red mug left/right plate or source-mug swap after validating cross-task state loading. Repeat the factorial audit and primitive/grounding decomposition. If exact state loading fails, use a predeclared matched-state diagnostic and label the limitation.

Stop after these experiments and apply the gate in `falsification_gate.md`. Do not start an ECL-like training arm until a new method claim and legal paired-action data pass separate audits.
