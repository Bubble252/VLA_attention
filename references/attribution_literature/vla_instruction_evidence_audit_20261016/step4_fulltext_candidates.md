# Step 4 — Full-text candidates and Step 5 verification

Seven candidates were selected by mechanism overlap, recency, and mandatory-anchor status. Each was read in introduction, method, experiments and limitations/discussion to the cited extent.

## ECL — Evidence-Guided Counterfactual Learning for Long-Tailed Robotic Policies

- Date/source: 2026-10-09; https://arxiv.org/abs/2610.12861
- Venue: PDF states ICLR 2027; independently verified only as arXiv v1
- Problem framing (verified): Long-tailed VLA policies fall back to head-task actions when task-specific visual evidence is weak.
- Core mechanism (verified): Instruction- and action-related attention product selects top-K visual tokens; a masked branch and factual-minus-masked prediction receive action supervision with frequency-aware weighting.
- Key insight (verified): Action prediction must be associated with task-specific visual evidence rather than frequent action priors.
- Application domain: MiniVLA/π0.5, LIBERO-Core-LT, two real robot platforms.
- Closest-passage evidence: §3.1 same-state two-task diagnostic; Eq.(3)–(9); §4.3 selection/masking ablations; §5 limitation on doubled training cost.
- Assumptions/scope: Instruction is held fixed in each evidence-removal pair and the main problem is long-tail bias. It does not estimate the two-factor instruction×object-evidence interaction across legal target instructions.
- Refined overlap (problem/core/insight/domain): partial / partial / match / match
- Verdict: 2/4 exact matches; Level 3 — Medium Overlap

## LIT — Breaking the Vision–Action Shortcut: Latent Interface Training for Generalizable Robotics Foundation Models

- Date/source: 2026-09-11; /home/bubble/类脑计算/参考/Breaking the Vision-Action Shortcut.pdf
- Venue: arXiv preprint
- Problem framing (verified): Visual shortcuts make action generation brittle under task-preserving visual shifts and goal changes.
- Core mechanism (verified): Two-stage spatial-goal action prior plus pose-supervised exclusive latent visual interface; counterfactual analysis separately perturbs visual input and instruction goal.
- Key insight (verified): Control how visual information reaches the action expert while retaining spatial goal information.
- Application domain: π0.5, MolmoAct2, FAST-WAM, ImageWAM; LIBERO/Plus and real robot.
- Closest-passage evidence: §III Eq.(2)–(8); §IV-E Fig.5 keeps scene/state fixed while swapping goal, and separately applies distractor/blur; Table III ablations.
- Assumptions/scope: Interventions are separate qualitative trajectory probes; no crossed instruction×target-specific evidence interaction estimator or decomposed primitive/grounding metric.
- Refined overlap (problem/core/insight/domain): match / partial / match / match
- Verdict: 3/4 exact matches; Level 2 — High Overlap

## Robust Skills — Robust Skills, Brittle Grounding: Diagnosing Restricted Generalization in Vision-Language Action Policies via Multi-Object Picking

- Date/source: 2026-02-27; https://arxiv.org/abs/2602.24143
- Venue: arXiv preprint
- Problem framing (verified): High VLA success can hide object-location shortcuts and brittle instruction-to-object binding.
- Core mechanism (verified): Multi-object task ladder and held-out object-region compositions; success, grasp-anything and reach decompose instruction grounding from primitive execution.
- Key insight (verified): Manipulation skill can remain robust while instruction-conditioned target selection collapses.
- Application domain: SmolVLA/π0.5, ManiSkill multi-object picking.
- Closest-passage evidence: §3.1–3.3; Table 2–5; §6.1 explicitly diagnostic, one primitive, no new method.
- Assumptions/scope: No target-specific visual evidence intervention or action sensitivity interaction; it is the strongest framing/metric prior.
- Refined overlap (problem/core/insight/domain): match / partial / match / match
- Verdict: 3/4 exact matches; Level 2 — High Overlap

## Shortcut Learning — Shortcut Learning in Generalist Robot Policies: The Role of Dataset Diversity and Fragmentation

- Date/source: 2025-08-08; /home/bubble/类脑计算/VLM终局/references/papers/Shortcut Learning in Generalist Robot Policies.pdf
- Venue: CoRL 2025 (PDF)
- Problem framing (verified): Robot policies ignore instructions/targets because visual factors correlate with actions across fragmented datasets.
- Core mechanism (verified): Controlled viewpoint×instruction/object confounding, factor-diversity analysis, and object augmentation that breaks correlations.
- Key insight (verified): Factor independence and interleaving are necessary to prevent visual shortcuts.
- Application domain: LIBERO-Spatial, SIMPLER, π0 and real robot.
- Closest-passage evidence: Fig.1 and §3.3; App.E uses two objects present with instruction/viewpoint swaps; App.F object removal/inpainting augmentation.
- Assumptions/scope: Studies data confounding and augmentation, not a target-evidence action-response interaction statistic.
- Refined overlap (problem/core/insight/domain): match / partial / match / match
- Verdict: 3/4 exact matches; Level 2 — High Overlap

## Anchor-Align — Generalizable VLA Finetuning via Representation Anchoring and Language-Action Alignment

- Date/source: 2026-07-15; /home/bubble/类脑计算/VLM终局/references/papers/Generalizable VLA Finetuning.pdf
- Venue: arXiv preprint
- Problem framing (verified): BC forgets semantic grounding and language/action outputs become inconsistent.
- Core mechanism (verified): Layer-wise frozen-VLM anchoring plus action-to-direction language labels; semantic target/layout perturbation evaluation.
- Key insight (verified): Preserved semantics and same-observation language/action alignment improve target re-grounding.
- Application domain: VLA-Adapter/StarVLA, LIBERO-PRO/Plus/CALVIN and xArm7.
- Closest-passage evidence: §3.2–3.3 Eq.(1)–(5); §4.3 Shuffle/Scatter; §4.4 pink-mug target change and failure decomposition.
- Assumptions/scope: Tests correct-object selection but does not cross instruction with interventions on each candidate object's evidence.
- Refined overlap (problem/core/insight/domain): partial / differ / match / match
- Verdict: 2/4 exact matches; Level 3 — Medium Overlap

## Pointing-VLA — Pointing-VLA: Typed Spatial Grounding Interfaces for Vision-Language-Action Manipulation

- Date/source: 2026-08; https://arxiv.org/abs/2608.23138
- Venue: arXiv preprint
- Problem framing (verified): Opaque/text geometry interfaces are brittle for source selection and placement.
- Core mechanism (verified): Typed point, object-functional heatmap and visual trajectory heads with fixed PICK/PLACE contracts.
- Key insight (verified): Different action stages require different explicit geometry readouts.
- Application domain: Bridge/WidowX, NORA, π0.5 and real robot.
- Closest-passage evidence: Structured PICK/PLACE contract; Table 2 composition ablation; Table 6 π0.5 guidance and failure-stage counts.
- Assumptions/scope: Explicit spatial interface and external controller; no counterfactual instruction-evidence interaction audit.
- Refined overlap (problem/core/insight/domain): partial / differ / partial / match
- Verdict: 1/4 exact matches; Level 4 — Low Overlap

## RoboIRG-Bench — RoboIRGBench: Benchmarking Implicit Referential Grounding in Vision-Language-Action Models

- Date/source: 2026-09; https://arxiv.org/abs/2609.34384
- Venue: arXiv preprint
- Problem framing (verified): Explicit task success does not imply robust referential grounding from context.
- Core mechanism (verified): Paired explicit/implicit instruction variants across direct, reasoning, spatial and contextual references.
- Key insight (verified): Language, perception, memory and execution errors should be separated.
- Application domain: RoboMME-based closed-loop VLA benchmark and FR3 real robot.
- Closest-passage evidence: §III preserves manipulation objectives while reformulating references; §IV evaluates 50 episodes×3 seeds.
- Assumptions/scope: Does not intervene on visual evidence or estimate action sensitivity to the instruction-evidence pairing.
- Refined overlap (problem/core/insight/domain): partial / partial / match / match
- Verdict: 2/4 exact matches; Level 3 — Medium Overlap
