# Manipulation phase / skill segmentation methods for D1 `L_phase`

**Purpose in this project**: support the later D1 extension of Structure-Native Action Attribution Consistency. D1 needs a weak or learned phase signal to construct

```text
A_phase(t) = α_t · A_lang(source) + (1 - α_t) · A_lang(target)
L_phase = D(A_act(t), A_phase(t))
```

This file records related methods and how they map to our plan. It does not replace D0. First-run VLA remains `L_sem + L_contain`; phase methods are for later D1 experiments.

## 1. What is directly useful for us

| Class | Representative methods | What they provide | How we use it |
|---|---|---|---|
| Event / state rules | CHAMP, Robo2VLM-style proprioceptive phase derivation | changepoints from EEF pose, object relation, gripper aperture, force/contact | D1a / D1b phase labels in LIBERO |
| Demonstration skill segmentation | BP-AR-HMM + DMP, BUDS | segment unstructured demonstrations into repeated skills or subgoals | post-hoc phase boundary discovery; sanity check against our hand rules |
| Weak temporal alignment | TACO | align a known subtask sketch to unsegmented trajectories | if we know sketch `approach → grasp → move → place`, infer soft phase without dense labels |
| Visual + kinematic boundary prediction | RoboSegNet | predicts transition boundaries from RGB + proprioception | future learned phase detector; not first-run dependency |
| VLM / video progress reasoning | Robo2VLM, ProcVLM, ROVER | phase/progress labels or progress reasoning from trajectories/videos | future comparison or weak teacher; keep outside D0/D1 default path |

## 2. Recommended route for our project

### D1a: LIBERO state-rule phase, quick diagnostic

Use simulator state:

```text
EEF pose
source object pose
target/container pose
gripper open/close
source height
source-target distance
```

Detect a coarse 3-stage phase:

```text
before grasp                  -> source, α = 0.9
after grasp before near target -> mixed,  α = 0.5
after near target              -> target, α = 0.1
```

This is the fastest way to check whether phase-conditioned action attribution is worth continuing.

### D1b: demonstration event boundary phase, recommended later main extension

Detect event boundaries per successful trajectory:

```python
t_grasp = first_t(gripper_closing and dist(eef[t], source[t]) < eps_src)
t_lift = first_t(source_z[t] - source_z[0] > eps_lift)
t_near_target = first_t(dist(source[t], target) < eps_tgt)
t_release = first_t(gripper_opening and dist(source[t], target) < eps_tgt)

phase[t < t_grasp] = "source"
phase[t_grasp <= t < t_near_target] = "mixed"
phase[t >= t_near_target] = "target"
```

If all events are stable, expand to five phases:

```text
approach_source -> grasp_source -> lift_or_move -> approach_target -> release
```

If events are missing, fall back to the coarse 3-stage version. Fixed time percentage should only be a negative control.

### D1c: model-internal phase discovery, later research direction

After D0 works, estimate phase from the model itself:

```text
mass_source(t) = sum_i A_act(t, i) · A_lang(source, i)
mass_target(t) = sum_i A_act(t, i) · A_lang(target, i)
```

Then enforce monotonic migration:

```text
source-dominant -> mixed -> target-dominant
```

This is more general than simulator state rules, but should not be the first implementation because early `A_act` may be noisy.

## 3. Related papers and methods

### TACO: Learning Task Decomposition via Temporal Alignment for Control

- Venue: ICML 2018
- Link: https://proceedings.mlr.press/v80/shiarlis18a.html
- PDF: http://proceedings.mlr.press/v80/shiarlis18a/shiarlis18a.pdf
- Core idea: given a weak task sketch, align subtask sequence to unsegmented demonstrations while learning sub-policies.
- Relevance: useful if our task sketch is known, e.g. `approach → grasp → move → place`. We can use TACO-like alignment as a possible D1b baseline or comparison, but it is heavier than simple LIBERO event rules.

### Bottom-Up Skill Discovery from Unsegmented Demonstrations for Long-Horizon Robot Manipulation (BUDS)

- Venue/status: RA-L 2022 / arXiv 2109.13841
- Link: https://arxiv.org/abs/2109.13841
- Project/code: https://ut-austin-rpl.github.io/BUDS-website/ and https://github.com/UT-Austin-RPL/BUDS
- Core idea: build hierarchical task structure from demonstrations through agglomerative clustering; identify recurring skills; train hierarchical imitation policies.
- Relevance: close to D1b if we want data-driven boundaries from unsegmented demos. For our first phase work, it is a reference, not a dependency.

### Learning and Generalization of Complex Tasks from Unstructured Demonstrations

- Venue: IROS 2012
- DOI: 10.1109/IROS.2012.6386006
- Link: https://www.researchgate.net/publication/261353928_Learning_and_generalization_of_complex_tasks_from_unstructured_demonstrations
- Core idea: BP-AR-HMM segments demonstrations, recognizes repeated skills, then uses DMPs for task generalization.
- Relevance: classic unstructured demonstration segmentation. Useful as a conceptual baseline for skill/phase segmentation.

### Online Bayesian changepoint detection for articulated motion models (CHAMP)

- Venue: ICRA 2015
- DOI: 10.1109/ICRA.2015.7139383
- Link: https://doi.org/10.1109/ICRA.2015.7139383
- Core idea: online Bayesian changepoint detection over articulated motion models; detects changes in object articulation/contact relationships.
- Relevance: strong conceptual match for event-boundary detection when contact/articulation changes mark task phases. In LIBERO pick-place, the analogous signals are gripper state, object lift, object-target distance.

### RoboSegNet: Learning Multi-Task Robot Trajectory Segmentation from Visual and Kinematic Streams

- Venue/status: CVPR 2026 project page
- Link: https://berkeleyautomation.github.io/RoboSegNet/
- Core idea: fuse RGB and kinematic/proprioceptive streams to predict trajectory transition boundaries with a DETR-style set prediction objective.
- Relevance: closest modern learned segmentation framework. Good future comparison for D1 learned phase detector, but likely too heavy for first implementation.

### Robo2VLM: Visual Question Answering from Large-Scale In-the-Wild Robot Manipulation Datasets

- arXiv: 2505.15517
- Link: https://arxiv.org/abs/2505.15517
- Core idea: derive ground truth from EEF pose, gripper aperture, force sensing; segment robot trajectories into manipulation phases; generate VQA grounded in robot trajectory data.
- Relevance: very relevant to our D1a/D1b. It supports the idea that proprioceptive signals can define manipulation phases and also connects phase labels to VLM/VQA supervision.

### ProcVLM: Learning Procedure-Grounded Progress Rewards for Robotic Manipulation

- arXiv: 2605.08774
- Project: https://procvlm.github.io/
- Core idea: VLM-based trajectory annotation creates frame-wise subtask stage labels, completion states, remaining actions, and progress scores.
- Relevance: useful for future B-route or D1c comparison. It is progress/reward-model oriented, so do not mix it into first-run D0/D1 unless we explicitly want an external VLM phase teacher.

### ROVER: Recursive Reasoning Over Videos with VLMs for Embodied Tasks

- Venue/status: NeurIPS 2025 project page
- Link: https://rover-vlm.github.io/
- Core idea: recursively decompose long-horizon videos into subtask-local reasoning windows; evaluated on progress estimation, frame-level reasoning, and video QA.
- Relevance: useful if we later want video-only phase/progress reasoning. It belongs to future comparison, not D0.

## 3.1 LIBERO dataset availability for phase labels

LIBERO does not appear to provide explicit phase labels by default. It provides demonstrations as HDF5 trajectories with actions, MuJoCo states, metadata, language instruction, and optionally generated RGB/depth/low-dimensional observations. Therefore LIBERO is suitable for deriving weak D1a/D1b phase labels, but not for directly reading ground-truth `approach/grasp/place/release` labels. See `libero_phase_availability.md` in this folder.

## 4. Recommendation for our docs and experiments

Use this hierarchy:

```text
First VLA run:
D0 only: L_sem + L_contain

D1 diagnostic:
D1a state-rule phase with 3 stages

D1 main extension:
D1b demonstration event boundaries with soft α_t

Future research comparison:
BUDS / TACO / RoboSegNet / Robo2VLM / ProcVLM / ROVER
```

Do not describe D1 as a world-model method. World/VAM methods belong to B-route refiners. D1 is phase-conditioned structure-native action attribution.

## 5. Negative controls

For any `L_phase` result, include:

- reverse phase: target → mixed → source;
- random phase;
- fixed time-percentage phase;
- correct phase but swapped source/target maps;
- `L_phase` without `L_contain`;
- wrong object phrase maps.

## 6. Download status

PDF download via `curl --proxy http://127.0.0.1:7897` failed in the current environment because the proxy/server connection was unavailable. The shell escalation request for downloading was rejected by automatic approval review due external service error. Therefore this folder currently stores method notes and source links, not local PDFs.
