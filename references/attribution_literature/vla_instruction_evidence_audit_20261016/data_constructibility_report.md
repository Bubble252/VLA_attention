# LIBERO data constructibility report

## Verdict

**Diagnostic-only intervention with current data.** None of the 10 audited task pairs has both exact shared init state and currently inventoried legal expert actions. `Trainable paired supervision` is therefore not established.

## What is available

- Official source commit and all 130 BDDL tasks were parsed; scene-compatible goal/source/position/role swap families exist.
- LIBERO-Goal offers same objects/layout and different goals. LIBERO-90 contains especially useful pairs such as red mug→left/right plate, red/white mug→left plate, front/left caddy compartment, and left/right bowl roles.
- Current remote RLDS: 432 episodes, ten Spatial tasks, 35–47 episodes/task, 256×256 agent/wrist RGB, 7D delta EEF. All ten change the selected black bowl's start relation and keep plate as target.
- Task0 contains **two** black bowls; the instruction selects the bowl between plate and ramekin, so it supports spatial-reference diagnosis. Current data has no alternative instruction/action for the second bowl in the exact same state.

## Why strict training pairs fail now

- All ten shortlisted pair families share scene structure, but exact state matching across task-specific init arrays is false; minimum cross-task L2 is nonzero for every pair.
- Current server demonstrations cover Spatial only; Goal/90 expert actions are not inventoried.
- A shared scene template and similar state vector do not establish that an action from one goal is correct for another.

## Permissible downgrade

Instantiate scene-compatible BDDL task pairs with matched or deliberately shared simulator state and evaluate the same frozen policy under both official goals. This supports action-response, correct-target and rollout diagnostics. Training requires newly collected successful paired rollouts from an official expert/planner, with the extra data budget matched across baselines.

LIBERO-PRO can provide language rephrase/object/position stress tests, but its implementation/data is not present in the audited workspace; paper descriptions cannot substitute for a runnable dataset audit.
