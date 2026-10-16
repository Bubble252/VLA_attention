# Final topic verdict

## Selected gate: B — Freeze as diagnostic/mechanistic paper

Freeze the research question as: **Does a VLA's action retargeting actually depend on the visual evidence of the referent selected by language?** The contribution is a controlled instruction×evidence interaction audit with decomposed target-selection and primitive-execution metrics.

Do not freeze a new training loss. ECL and LIT already cover evidence-guided counterfactual training and goal/visual interventions; Anchor-Align covers semantic target retargeting; Robust Skills covers grounding-versus-execution decomposition. The bounded novelty verdict is Level 2 for the broad candidate, while the exact crossed interaction estimator is the remaining diagnostic delta.

Current LIBERO data supports diagnostic construction, not strict paired training: ten scene-compatible goal/source/role pairs were found, but none has an exact shared init state plus currently inventoried legal actions for both targets. Current 432 RLDS episodes cover Spatial only.

## Idea-evaluator verdict

**Accept with Revisions**, implemented as gate B rather than a method paper.

- Paper type: Novel Problem / controlled diagnostic setting.
- Fatal flaw F1 (method novelty): MAJOR for a method paper, defended by removing the training-method claim.
- Fatal flaw F6 (unverifiable paired-action claim): MAJOR for training, defended by diagnostic-only evaluation and official task predicates.
- Higher 6/10: clearer diagnosis may improve model selection, but no policy gain is promised.
- Faster 5/10: frozen-policy audit is cheaper than retraining, though full rollouts are required.
- Stronger 8/10: directly tests grounding under target/layout changes and matched nuisance interventions.
- Cheaper 7/10: reuses BDDL task families and frozen policies; exact paired expert data is avoided for the diagnostic.
- Broader 7/10: architecture-agnostic action-response measurement can apply to OpenVLA/OFT and flow policies with native distances.
- Paradigm probe: First Principles=yes (success does not identify language–visual binding); Elephant=yes (skill/grounding conflation is documented); Technology Cycle=partial; Hamming=partial. Disruptive potential: possible, with a deliberately narrow first paper.
- Feasibility: compute low-to-medium; data medium; engineering medium; timeline compatible with a focused diagnostic project. User weekly hours/advisor cadence were not provided, so lifecycle fit remains an attestation rather than measured fact.
- Integrity gate: passed for evidence, novelty qualification and falsification; no guaranteed effect claim.

## Reviewer delta

> Unlike LIT, which separately tests task-preserving visual perturbations and goal-changing instructions, and Robust Skills, which separates target selection from grasp execution through spatial task ladders, this work crosses legal target instructions with target-specific and matched non-target evidence interventions in the same physical state to quantify whether action retargeting is visually bound to the instructed referent.

## Paper claim boundary

The metric is an intervention sensitivity interaction, not a causal effect. The paper may claim that it distinguishes language-only, saliency-only and location shortcuts under controlled assumptions. It may not claim to improve policy behavior unless a later, separately audited method passes the full falsification gate.
