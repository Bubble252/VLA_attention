# Final topic card

## Working title

**Does Your VLA Look at What You Asked For? A Factorial Audit of Instruction–Evidence Binding in Robot Actions**

## One-sentence story

VLA task success confounds understanding the instruction, using the corresponding visual target, and executing a motor primitive; a crossed instruction×target-evidence intervention audit separates these mechanisms at action and rollout level.

## Inputs and outputs

- Inputs: a frozen VLA, scene-compatible legal task instructions A/B, exact or audited matched simulator state, target A/B evidence regions, native action distribution/chunk.
- Outputs: `I_bind`, correct-target selection, approach/grasp/place success, official rollout success, and shortcut classification.

## Method

Run each state under instructions A/B and clean, A-evidence, B-evidence, non-target, background/light interventions. Compare native action responses with frozen normalization and clustered uncertainty. Decompose success into correct referent selection and primitive execution. Use multiple intervention operators to expose OOD artifacts.

## Closest work and delta

- LIT: separate visual and goal interventions; no crossed target-specific interaction statistic.
- Robust Skills: target-selection versus primitive metrics; no evidence intervention.
- ECL: fixed-instruction evidence masking for long-tail training; no legal instruction swap factorial.
- Anchor-Align: language-action alignment and semantic target tests; no evidence-specific action interaction.

## Data status

LIBERO Goal/90 contains scene-compatible target/source/role swap families. Current workspace lacks exact-state paired legal actions, so the frozen topic is diagnostic. Task0 contains two bowls and supports spatial-reference diagnosis, but lacks an alternative legal instruction/action pair for the second bowl.

## Kill criteria

No paper claim if `I_bind` cannot distinguish correct from wrong/shuffled controls, if its sign depends on mask operator, or if it does not predict correct-target/phase/rollout behavior beyond existing metrics.
