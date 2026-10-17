# Step 4–5 — Full-text verification

Date: 2026-10-17. Read scope: introduction, method, main experiments and limitations/discussion. PDFs/text are local audit cache and are not Git deliverables.

## Do Vision-Language-Action Models Understand Instructions?

- Verified problem: causal footprint of controlled language corruptions in π0.5 and GR00T N1.7 action experts.
- Verified mechanism: attribution patching nominates components; exact activation patching verifies selected layers. Interventions occur after self-attention `o_proj`; language effects are architecture-specific.
- Important evidence: §4 explicitly says it “identify[ies] which layers ... are particularly reactive”; §6 reports early/selected-late π0.5 loci and periodic GR00T effects.
- Limit: diagnostic only; no trainable language-evidence module and no source/target spatial evidence injection. Attribution patching does not correlate with activation patching on π0.5, so our selection must use exact patching as the gate.
- Axes: problem match; mechanism partial; insight match; domain match → 3 exact/near-exact axes, Level 2.

## Encoded but Not in Control

- Verified problem: valid target substitutions show instructed target is encoded but scene-favored behavior still controls action.
- Verified mechanism: scene-preserving intervention, layerwise action lens, target/scene linear probes, attention and representation analyses.
- Evidence: main text and App. D/E show target decodability and layerwise action changes; it does not train a causal-locus residual.
- Limit: establishes the exact failure framing but not our repair.
- Axes: problem match; mechanism partial; insight match; domain match → Level 2.

## LangForce

- Verified problem: language becomes predictable from vision and conditional language–action information collapses.
- Verified mechanism: latent action queries form vision-only prior and vision+language posterior; LLR/PMI term rewards actions that explain language. Inference uses posterior only.
- Evidence: §3.4 Eq. 9 and Tables 1–5; direct QwenGR00T baseline 55.2 vs LangForce 66.5 on SimplerEnv.
- Limit: no causal layer selection, phrase spatial evidence or future-evidence routing; dual branch has extra training cost.
- Axes: problem match; mechanism differ; insight match; domain match → Level 2.

## Text Latent Interpolation

- Verified problem: VLA cannot recombine learned skills for novel instructions.
- Verified mechanism: average per-task text-token hidden states across demonstrations and layers, then add/interpolate them in the text residual stream over time.
- Evidence: §3.2–3.3 Eqs. 2–4; layer-specific intervention analysis exists, but authors keep all layers for best performance rather than selecting action-causal loci.
- Limit: task-average latent, not phrase source/target evidence; inference steering rather than jointly trained zero-init residual.
- Axes: problem partial; mechanism partial; insight match; domain match → Level 3.

## ECT

- Verified problem: instructions retrieve familiar trajectories rather than bind current instruction and scene to the required action.
- Verified mechanism: construct action-valid counterfactual `(s,l,a)/(s',l,a')` pairs through successful replay and sum matched imitation losses in one update.
- Evidence: §5 Eqs. 5–7; π0.5 mean Swap 36→59.
- Limit: architecture unchanged; no causal locus or spatial evidence residual. It requires valid alternative demonstrations and is a mandatory data baseline.
- Axes: problem match; mechanism differ; insight match; domain match → Level 2.

## PosA-VLA

- Verified mechanism from local full-text notes: CLIP task map and gripper query map are supervised by projected EEF/task Gaussian maps; learned map gates DINO/CLIP visual features before a flow action transformer.
- Limit: pose/calibration labels and global forward gating; no causal layer selection or future evidence.
- Axes: problem partial; mechanism partial; insight partial; domain match → Level 3.

## Anchor-Align

- Verified mechanism: all-layer representation anchoring plus frozen-LM-head classification of action-derived direction words on the same observation.
- Limit: language/action semantic consistency does not require common spatial referent evidence or causal-locus injection.
- Axes: problem partial; mechanism differ; insight match; domain match → Level 3.

## FineVLA and VLA Grounder

- FineVLA verifies that fine-grained language data can improve steerability; its contribution is data/annotation/training mixture, not a causal-locus module.
- VLA Grounder optimizes a scene-aware command policy with rollout reward while freezing the VLA; it changes language input, not internal action loci.
- Both are mandatory data/inference baselines but do not subsume the proposed mechanism.

## ProGAL-VLA

- Verified problem: monolithic VLAs treat language as weak conditioning and may act on unverified visual shortcuts or ambiguous referents.
- Verified mechanism: a slow VLM planner outputs a symbolic sub-goal; a 3D entity-centric Grounded State Module constructs tracked entity embeddings; State Alignment Cross Attention binds the symbolic goal to an entity; only the verified goal embedding conditions the fast OpenVLA policy. A Grounding Alignment Contrastive loss aligns symbolic–entity pairs.
- Evidence: §3 Eqs. 1–12 defines the verification bottleneck and `L_action + λL_GAC`; §4 formalizes language influence and an InfoNCE-style entity-information bound; LIBERO-Plus and a custom ambiguity benchmark evaluate language ignorance/entity retrieval.
- Assumptions/scope: uses an external open-vocabulary detector, depth/3D entity graph, asynchronous slow planner and offline sub-goal/entity alignment pairs. It does not use activation-patching-selected action loci, a zero-init residual, or LC-PEA future-delta supervision.
- Refined axes: problem match; core mechanism partial; key insight match; domain match → 3 axes, Level 2 — High Overlap.

## Unresolved high-risk candidate: RoVLA temporal grounding

- Crossref verifies the title `RoVLA: Action-Conditioned Temporal Grounding and Structured Conditioning for Vision–Language–Action Policies`, CASE 2026, DOI `10.1109/CASE69030.2026.11704424`.
- Crossref/OpenAlex exposed no abstract; the arXiv title query returned a different RoVLA paper (`Multi-Consistency Constraints`). No body-level mechanism claim is made here.
- Because the title directly overlaps action-conditioned temporal grounding, it remains a blocking item for any broad LC-PEA “first temporal grounding” claim. The proposed paper must avoid such a broad claim until the CASE PDF is obtained and verified.
