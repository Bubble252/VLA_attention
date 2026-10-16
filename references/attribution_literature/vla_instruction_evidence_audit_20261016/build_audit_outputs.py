"""Build the three audits and final topic gate from frozen evidence."""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent

OVERLAP = {
    "Evidence-Guided Counterfactual Learning for Long-Tailed Robotic Policies": 3,
    "Inference-Time Attention Steering for Vision-Language-Action Driving Models": 1,
    "Mini-BEHAVIOR-Gran: Revealing U-Shaped Effects of Instruction Granularity on Language-Guided Embodied Agents": 1,
    "InstructVLA: Vision-Language-Action Instruction Tuning from Understanding to Manipulation": 2,
    "Grounding Hierarchical Vision-Language-Action Models Through Explicit Language-Action Alignment": 2,
    "Visually Grounding Language Instruction for History-Dependent Manipulation": 1,
    "Instrucrobo: Object-Centric Multi-Instruction Decoupling Model for Explainable Robotic Manipulation": 2,
}

ANCHORS = [
    {
        "short": "ECL",
        "title": "Evidence-Guided Counterfactual Learning for Long-Tailed Robotic Policies",
        "date": "2026-10-09",
        "source": "https://arxiv.org/abs/2610.12861",
        "venue": "PDF states ICLR 2027; independently verified only as arXiv v1",
        "problem": "Long-tailed VLA policies fall back to head-task actions when task-specific visual evidence is weak.",
        "mechanism": "Instruction- and action-related attention product selects top-K visual tokens; a masked branch and factual-minus-masked prediction receive action supervision with frequency-aware weighting.",
        "insight": "Action prediction must be associated with task-specific visual evidence rather than frequent action priors.",
        "domain": "MiniVLA/π0.5, LIBERO-Core-LT, two real robot platforms.",
        "axes": ["partial", "partial", "match", "match"],
        "evidence": "§3.1 same-state two-task diagnostic; Eq.(3)–(9); §4.3 selection/masking ablations; §5 limitation on doubled training cost.",
        "scope": "Instruction is held fixed in each evidence-removal pair and the main problem is long-tail bias. It does not estimate the two-factor instruction×object-evidence interaction across legal target instructions.",
    },
    {
        "short": "LIT",
        "title": "Breaking the Vision–Action Shortcut: Latent Interface Training for Generalizable Robotics Foundation Models",
        "date": "2026-09-11",
        "source": "/home/bubble/类脑计算/参考/Breaking the Vision-Action Shortcut.pdf",
        "venue": "arXiv preprint",
        "problem": "Visual shortcuts make action generation brittle under task-preserving visual shifts and goal changes.",
        "mechanism": "Two-stage spatial-goal action prior plus pose-supervised exclusive latent visual interface; counterfactual analysis separately perturbs visual input and instruction goal.",
        "insight": "Control how visual information reaches the action expert while retaining spatial goal information.",
        "domain": "π0.5, MolmoAct2, FAST-WAM, ImageWAM; LIBERO/Plus and real robot.",
        "axes": ["match", "partial", "match", "match"],
        "evidence": "§III Eq.(2)–(8); §IV-E Fig.5 keeps scene/state fixed while swapping goal, and separately applies distractor/blur; Table III ablations.",
        "scope": "Interventions are separate qualitative trajectory probes; no crossed instruction×target-specific evidence interaction estimator or decomposed primitive/grounding metric.",
    },
    {
        "short": "Robust Skills",
        "title": "Robust Skills, Brittle Grounding: Diagnosing Restricted Generalization in Vision-Language Action Policies via Multi-Object Picking",
        "date": "2026-02-27",
        "source": "https://arxiv.org/abs/2602.24143",
        "venue": "arXiv preprint",
        "problem": "High VLA success can hide object-location shortcuts and brittle instruction-to-object binding.",
        "mechanism": "Multi-object task ladder and held-out object-region compositions; success, grasp-anything and reach decompose instruction grounding from primitive execution.",
        "insight": "Manipulation skill can remain robust while instruction-conditioned target selection collapses.",
        "domain": "SmolVLA/π0.5, ManiSkill multi-object picking.",
        "axes": ["match", "partial", "match", "match"],
        "evidence": "§3.1–3.3; Table 2–5; §6.1 explicitly diagnostic, one primitive, no new method.",
        "scope": "No target-specific visual evidence intervention or action sensitivity interaction; it is the strongest framing/metric prior.",
    },
    {
        "short": "Shortcut Learning",
        "title": "Shortcut Learning in Generalist Robot Policies: The Role of Dataset Diversity and Fragmentation",
        "date": "2025-08-08",
        "source": "/home/bubble/类脑计算/VLM终局/references/papers/Shortcut Learning in Generalist Robot Policies.pdf",
        "venue": "CoRL 2025 (PDF)",
        "problem": "Robot policies ignore instructions/targets because visual factors correlate with actions across fragmented datasets.",
        "mechanism": "Controlled viewpoint×instruction/object confounding, factor-diversity analysis, and object augmentation that breaks correlations.",
        "insight": "Factor independence and interleaving are necessary to prevent visual shortcuts.",
        "domain": "LIBERO-Spatial, SIMPLER, π0 and real robot.",
        "axes": ["match", "partial", "match", "match"],
        "evidence": "Fig.1 and §3.3; App.E uses two objects present with instruction/viewpoint swaps; App.F object removal/inpainting augmentation.",
        "scope": "Studies data confounding and augmentation, not a target-evidence action-response interaction statistic.",
    },
    {
        "short": "Anchor-Align",
        "title": "Generalizable VLA Finetuning via Representation Anchoring and Language-Action Alignment",
        "date": "2026-07-15",
        "source": "/home/bubble/类脑计算/VLM终局/references/papers/Generalizable VLA Finetuning.pdf",
        "venue": "arXiv preprint",
        "problem": "BC forgets semantic grounding and language/action outputs become inconsistent.",
        "mechanism": "Layer-wise frozen-VLM anchoring plus action-to-direction language labels; semantic target/layout perturbation evaluation.",
        "insight": "Preserved semantics and same-observation language/action alignment improve target re-grounding.",
        "domain": "VLA-Adapter/StarVLA, LIBERO-PRO/Plus/CALVIN and xArm7.",
        "axes": ["partial", "differ", "match", "match"],
        "evidence": "§3.2–3.3 Eq.(1)–(5); §4.3 Shuffle/Scatter; §4.4 pink-mug target change and failure decomposition.",
        "scope": "Tests correct-object selection but does not cross instruction with interventions on each candidate object's evidence.",
    },
    {
        "short": "Pointing-VLA",
        "title": "Pointing-VLA: Typed Spatial Grounding Interfaces for Vision-Language-Action Manipulation",
        "date": "2026-08",
        "source": "https://arxiv.org/abs/2608.23138",
        "venue": "arXiv preprint",
        "problem": "Opaque/text geometry interfaces are brittle for source selection and placement.",
        "mechanism": "Typed point, object-functional heatmap and visual trajectory heads with fixed PICK/PLACE contracts.",
        "insight": "Different action stages require different explicit geometry readouts.",
        "domain": "Bridge/WidowX, NORA, π0.5 and real robot.",
        "axes": ["partial", "differ", "partial", "match"],
        "evidence": "Structured PICK/PLACE contract; Table 2 composition ablation; Table 6 π0.5 guidance and failure-stage counts.",
        "scope": "Explicit spatial interface and external controller; no counterfactual instruction-evidence interaction audit.",
    },
    {
        "short": "RoboIRG-Bench",
        "title": "RoboIRGBench: Benchmarking Implicit Referential Grounding in Vision-Language-Action Models",
        "date": "2026-09",
        "source": "https://arxiv.org/abs/2609.34384",
        "venue": "arXiv preprint",
        "problem": "Explicit task success does not imply robust referential grounding from context.",
        "mechanism": "Paired explicit/implicit instruction variants across direct, reasoning, spatial and contextual references.",
        "insight": "Language, perception, memory and execution errors should be separated.",
        "domain": "RoboMME-based closed-loop VLA benchmark and FR3 real robot.",
        "axes": ["partial", "partial", "match", "match"],
        "evidence": "§III preserves manipulation objectives while reformulating references; §IV evaluates 50 episodes×3 seeds.",
        "scope": "Does not intervene on visual evidence or estimate action sensitivity to the instruction-evidence pairing.",
    },
]


def normalize_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def first_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return "Abstract unavailable; no method inference from title alone."
    match = re.search(r"(?<=[.!?])\s", text)
    return text[: match.start() + 1] if match else text


def method_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if re.search(r"\b(propose|introduce|present|develop|construct|design)\b", sentence, re.I):
            return sentence
    return "No abstract-level mechanism extracted."


def overlap_label(count: int) -> str:
    return {0: "Level 5 — No Overlap", 1: "Level 4 — Low Overlap", 2: "Level 3 — Medium Overlap", 3: "Level 2 — High Overlap", 4: "Level 1 — Full Overlap"}[count]


def main() -> None:
    search = json.loads((ROOT / "step2_search_records.json").read_text())
    records = {}
    for source, papers in search["results"].items():
        for paper in papers:
            key = normalize_title(paper["title"])
            if key not in records:
                records[key] = dict(paper, sources=[source])
            elif source not in records[key]["sources"]:
                records[key]["sources"].append(source)
    for anchor in ANCHORS:
        key = normalize_title(anchor["title"])
        if key not in records:
            records[key] = {
                "title": anchor["title"], "publication_date": anchor["date"], "url": anchor["source"],
                "abstract": "Added as mandatory anchor or verified recall; see full-text comparison.", "sources": ["mandatory-anchor"],
            }
    score_map = {
        normalize_title(a["title"]): sum(x == "match" for x in a["axes"]) for a in ANCHORS
    }
    score_map.update({normalize_title(k): v for k, v in OVERLAP.items()})
    triage = []
    for record in records.values():
        title = record["title"]
        score = score_map.get(normalize_title(title), 0)
        triage.append(
            {
                "title": title,
                "date": record.get("publication_date") or str(record.get("year") or "unknown"),
                "problem_framing": first_sentence(record.get("abstract", "")),
                "core_mechanism": method_sentence(record.get("abstract", "")),
                "key_insight": "Potentially relevant to instruction/evidence/action" if score else "No concrete overlap established from abstract.",
                "application_domain": "VLA/robotics" if re.search(r"robot|vision.language.action|VLA", title + " " + record.get("abstract", ""), re.I) else "outside narrow VLA scope",
                "overlap_score": score,
                "source": record.get("url"),
                "provenance": record.get("sources", []),
            }
        )
    triage.sort(key=lambda x: (-x["overlap_score"], x["title"]))
    (ROOT / "step3_abstract_triage.json").write_text(json.dumps(triage, ensure_ascii=False, indent=2))
    md = ["# Step 3 — Abstract-level triage", "", "All deduplicated live-search records plus mandatory anchors are retained. Missing abstracts remain uncertain.", ""]
    for i, item in enumerate(triage, 1):
        md += [f"## {i}. {item['title']}", "", f"- Date: {item['date']}", f"- Problem framing: {item['problem_framing']}", f"- Core mechanism: {item['core_mechanism']}", f"- Key insight: {item['key_insight']}", f"- Application domain: {item['application_domain']}", f"- Overlap score: {item['overlap_score']}/4 (abstract triage only)", f"- Source: {item['source']}; {', '.join(item['provenance'])}", ""]
    (ROOT / "step3_abstract_triage.md").write_text("\n".join(md))
    comparisons = []
    for anchor in ANCHORS:
        matches = sum(x == "match" for x in anchor["axes"])
        comparisons.append(dict(anchor, matching_axes=matches, level=5 - matches, label=overlap_label(matches)))
    (ROOT / "step4_fulltext_candidates.json").write_text(json.dumps(comparisons, ensure_ascii=False, indent=2))
    full = ["# Step 4 — Full-text candidates and Step 5 verification", "", "Seven candidates were selected by mechanism overlap, recency, and mandatory-anchor status. Each was read in introduction, method, experiments and limitations/discussion to the cited extent.", ""]
    for item in comparisons:
        full += [f"## {item['short']} — {item['title']}", "", f"- Date/source: {item['date']}; {item['source']}", f"- Venue: {item['venue']}", f"- Problem framing (verified): {item['problem']}", f"- Core mechanism (verified): {item['mechanism']}", f"- Key insight (verified): {item['insight']}", f"- Application domain: {item['domain']}", f"- Closest-passage evidence: {item['evidence']}", f"- Assumptions/scope: {item['scope']}", f"- Refined overlap (problem/core/insight/domain): {' / '.join(item['axes'])}", f"- Verdict: {item['matching_axes']}/4 exact matches; {item['label']}", ""]
    (ROOT / "step4_fulltext_candidates.md").write_text("\n".join(full))
    minimum = min(x["level"] for x in comparisons)
    delta = (
        "Unlike ECL, which removes jointly instruction/action-attended tokens under a fixed instruction to counter long-tail action bias, "
        "and LIT, which separately tests task-preserving visual changes and goal changes, the candidate crosses legal target instructions "
        "with target-specific and matched non-target evidence interventions in the same physical state, quantifying whether action retargeting "
        "depends on the corresponding visual referent; this is an evaluation/diagnostic delta, not a distinct training-method claim."
    )
    comparison_md = ["# Step 5 — Novelty comparison", "", "## Proposed candidate", "", "- Problem: instruction changes and evidence removal separately cannot establish language-to-visual binding in action.", "- Mechanism: crossed instruction A/B × evidence A/B/non-target perturbations with action-response interaction and decomposed rollout metrics.", "- Insight: only the interaction removes language-only and vision-only shortcut explanations.", "- Domain: VLA multi-object/target manipulation; diagnostic first.", ""] + full[4:]
    (ROOT / "step5_novelty_comparison.md").write_text("\n".join(comparison_md))
    verdict = f"""# Novelty verdict

**{overlap_label(5-minimum)}** under the scoop-check mapping; the closest papers match three of four axes, so the overall level is {minimum}. No verified paper in this bounded audit implements the exact crossed interaction estimator, but ECL, LIT, Robust Skills and Shortcut Learning collectively cover the training mechanism, separate interventions, grounding/execution decomposition, and controlled factor-confounding story.

## Reviewer delta

> {delta}

## Decision boundary

The diagnostic interaction is defensible as a measurement contribution. The optional training claim is not: ECL already uses instruction/action evidence selection, counterfactual masking and action supervision; LIT already constrains visual conditioning and tests goal changes. Any future training method needs a separate novelty audit.

## Search provenance

- Attempt 1 through the required port 7897 returned TLS EOF for every live source; retained as `step2_search_records_attempt1_proxy_ssl_failure.json` and logs.
- Direct retry returned arXiv=15 and Crossref=15; DBLP/OpenAlex failed parsing/empty responses, OpenReview dependency was missing, and Semantic Scholar returned HTTP 429. Zero results from failed sources are not absence evidence.
- Live results are in `step2_search_records.json`; all deduplicated triage is in `step3_abstract_triage.*`.
"""
    (ROOT / "novelty_verdict.md").write_text(verdict)

    pairs = list(csv.DictReader((ROOT / "candidate_task_pairs.csv").open()))
    action_provenance = """# Action-label provenance

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
"""
    (ROOT / "action_label_provenance.md").write_text(action_provenance)
    data_report = f"""# LIBERO data constructibility report

## Verdict

**Diagnostic-only intervention with current data.** None of the {len(pairs)} audited task pairs has both exact shared init state and currently inventoried legal expert actions. `Trainable paired supervision` is therefore not established.

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
"""
    (ROOT / "data_constructibility_report.md").write_text(data_report)

    causal = """# Causal graph and alternative explanations

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
"""
    (ROOT / "causal_graph_and_alternatives.md").write_text(causal)
    factorial = """# Factorial protocol

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
"""
    (ROOT / "factorial_protocol.md").write_text(factorial)
    matrix = [
        ["M0", "native BC/B0", "none", "required", "action baseline"],
        ["M1", "B1 BlindVLA-style retention", "same", "required", "visual retention"],
        ["M2", "B1 + static semantic alignment", "same", "required", "existing semantic regularizer"],
        ["M3", "B1 + RS-CL/equivalent state contrastive", "same", "required", "robot-state relation baseline"],
        ["M4", "B1 + inverse dynamics", "same", "required", "action-aware visual representation baseline"],
        ["M5", "instruction-evidence binding candidate", "same", "diagnostic first", "candidate"],
        ["M6", "M5 wrong-map", "same", "diagnostic", "semantic specificity control"],
        ["M7", "M5 instruction shuffle", "same", "diagnostic", "language pairing control"],
        ["M8", "M5 target swap without legal action label", "same", "evaluation only", "forbidden as training label"],
    ]
    with (ROOT / "matched_budget_matrix.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "configuration", "demonstrations_augmentation_budget", "training_role", "explanation_tested"])
        writer.writerows(matrix)
    falsification = """# Falsification gate

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
"""
    (ROOT / "falsification_gate.md").write_text(falsification)

    final = f"""# Final topic verdict

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
"""
    (ROOT / "final_topic_verdict.md").write_text(final)
    card = """# Final topic card

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
"""
    (ROOT / "final_topic_card.md").write_text(card)
    minimal = """# Next minimal experiment

Run at most three experiments, in this order:

1. **Frozen task0 spatial-reference sanity**: exploit the two black bowls; compare the official successful OpenVLA/OFT checkpoint under the legal task0 instruction with target-bowl, other-bowl and area-matched non-target interventions. This checks evidence specificity but is not a target-retargeting pair.
2. **Same-scene Goal diagnostic**: instantiate `put bowl on plate` and `put bowl on stove` from an audited shared simulator state; run both legal task predicates/instructions with plate, stove and non-target interventions. Measure `I_bind`, first/chunk action response, correct target approach and rollout success. No training labels are transferred.
3. **Multi-object/plate pair**: use LIBERO-90 Scene5 red mug left/right plate or source-mug swap after validating cross-task state loading. Repeat the factorial audit and primitive/grounding decomposition. If exact state loading fails, use a predeclared matched-state diagnostic and label the limitation.

Stop after these experiments and apply the gate in `falsification_gate.md`. Do not start an ECL-like training arm until a new method claim and legal paired-action data pass separate audits.
"""
    (ROOT / "next_minimal_experiment.md").write_text(minimal)
    print(json.dumps({"deduplicated_records": len(triage), "fulltext_candidates": len(comparisons), "novelty_level": minimum, "final_gate": "B"}, indent=2))


if __name__ == "__main__":
    main()
