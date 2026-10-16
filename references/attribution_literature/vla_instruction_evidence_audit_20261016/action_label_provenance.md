# Action-label provenance

## Allowed

- The official demonstration belonging to the exact task instruction and goal.
- A successful official/expert policy rollout in the exact instantiated task and audited initial state.
- A validated planner/controller whose rollout succeeds under the official task predicate.
- Instruction/evidence perturbations explicitly marked diagnostic-only, with no claim that the original action is correct for the changed target.

## Forbidden

- Original image/state + changed target instruction + original action as a new-target label.
- Hand-authored delta EEF labels, failed rollouts, or trajectories transferred across tasks without success verification.
- Treating nearest init-state vectors as identical states or as proof of action equivalence.

## Current evidence

The server inventory contains 432 legal 7D-delta-EEF episodes for ten LIBERO-Spatial tasks only. It contains no goal/object/90 pair audited here. Official LIBERO provides 50 human teleoperation demonstrations per task in the downloadable raw datasets, but those raw Goal/90 demonstrations are not currently present in the audited workspace. Even if downloaded, the task-specific init files contain no exact shared state among the audited pairs, so they do not automatically create strict paired counterfactual actions.
