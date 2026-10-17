# Causal-Locus Language-Evidence Residual: scoop-check report

## Verdict

**Level 2 — High Overlap.** No verified paper exactly combines causal-locus selection, phrase-specific source/target spatial evidence injection, zero-initialized action-token residuals and LC-PEA future-change supervision. However, each component and the encoded-but-not-in-control framing have strong direct prior work. ProGAL-VLA additionally supplies a verified symbolic-to-3D-entity grounding bottleneck for OpenVLA. Novelty therefore rests on the precise causal-locus/evidence/future-change binding and its falsification.

## Delta

> Unlike the mechanistic patching study, which localizes language-sensitive action-expert layers without training a repair, and Text Latent Interpolation, which writes task-averaged text latents across residual streams, the proposed method uses legal same-scene instruction patching to select action-causal loci and injects phrase-specific source/target spatial evidence only at those loci through a zero-initialized residual jointly supervised by target-relative future change, aiming to reduce wrong-target rollouts without requiring a new paired action dataset.

## Decomposed claim

- Problem framing: encoded language fails to control target selection.
- Core mechanism: exact patching-selected action loci + gated source/target evidence residual.
- Key insight: decodable language is not behaviorally causal; both locus and referent must be bound.
- Domain: multi-object VLA manipulation.

## Strongest prior work

| Work | What is already covered | What remains different |
|---|---|---|
| Encoded but Not in Control | exact failure framing, action lens, target probes | no trainable repair or evidence residual |
| Do VLAs Understand Instructions? | layer/token activation and attribution patching | diagnostic only; no spatial evidence module |
| LangForce | latent action queries and language–action PMI/LLR | no causal layer selection or phrase spatial evidence |
| ECT | instruction-action binding and strong paired counterfactual remedy | data/loss method, architecture unchanged |
| Text Latent Interpolation | residual text-latent injection and layer intervention | task-average latent, no causal selection or source/target maps |
| PosA-VLA | language/EEF spatial maps gate action features | pose-supervised global gating, no causal action-locus selection |
| Anchor-Align | layerwise anchoring and same-observation action→language loss | semantic consistency, no shared referent evidence |
| ProGAL-VLA | symbolic sub-goal→3D entity verification, contrastive alignment and verified-goal action conditioning | no causal-locus selection, zero-init residual or LC-PEA future delta |

## Decision

Keep the candidate as a **secondary enhancement**, not the primary LC-PEA claim. Run L0 as diagnosis first. Only promote S4 if matched baselines demonstrate that causal locus selection and spatial evidence content are both necessary for rollout gain.

`RoVLA: Action-Conditioned Temporal Grounding and Structured Conditioning for Vision–Language–Action Policies` (CASE 2026, DOI `10.1109/CASE69030.2026.11704424`) remains an unresolved high-risk title because no abstract/PDF body was available from the searched sources. Avoid any broad “first temporal grounding” claim until it is verified.

Full audit: `step1_claim_decomposition.md`, `step2_search_and_triage.md`, `step4_fulltext_verification.md`, `step6_comparison.md`, `step7_verdict.md`.
