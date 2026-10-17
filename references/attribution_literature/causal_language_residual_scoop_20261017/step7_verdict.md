# Step 7 — Verdict and delta

## Verdict

**Level 2 — High Overlap.**

The problem and central insight are already explicit in Encoded but Not in Control, the mechanistic patching study, LangForce and ECT. ProGAL-VLA additionally binds symbolic sub-goals to 3D entities and conditions OpenVLA actions on a verified goal embedding. Activation patching is not novel; residual text-latent intervention is not novel; language-specific action queries, FiLM/cross-attention, verified entity bottlenecks, forward spatial gating and counterfactual training are not novel. The remaining defensible delta is the *binding of three elements*: exact action-causal locus selection, phrase-level source/target spatial evidence, and a zero-initialized action-token residual trained with target-relative future change.

## Delta

> Unlike the mechanistic patching study, which localizes language-sensitive action-expert layers without training a repair, and Text Latent Interpolation, which writes task-averaged text latents across residual streams, the proposed method uses legal same-scene instruction patching to select action-causal loci and injects phrase-specific source/target spatial evidence only at those loci through a zero-initialized residual jointly supervised by target-relative future change, aiming to reduce wrong-target rollouts without requiring a new paired action dataset.

## Reviewer-risk statement

This delta is fragile. If random/all-layer residuals match causal-locus selection, if language-only residuals match spatial evidence residuals, or if ECT/LangForce/ProGAL-VLA wins under matched data/compute, the method contribution collapses to implementation. The unresolved CASE 2026 RoVLA metadata also blocks a broad first-temporal-grounding claim. The paper must not claim first use of activation patching, language adapters, action queries, residual steering, verified grounding bottlenecks, spatial gating or counterfactual language-action training.

## Required matched baselines

1. J4 alone.
2. OpenVLA/OFT native FiLM.
3. Same-parameter random-layer and all-layer residuals.
4. Language-only residual without source/target spatial evidence.
5. Evidence residual without causal selection.
6. Text-latent intervention or closest architecture-compatible residual-steering baseline.
7. LangForce/PMI-LLR or an explicitly documented incompatibility.
8. ECT paired-data baseline when legal counterfactual demonstrations are available.
9. PosA-style forward gating and Anchor-Align/action→language auxiliary where comparable.
10. ProGAL-VLA verified-goal conditioning where its runnable code/data permit, otherwise a clearly scoped architectural comparison.

## Go/No-Go

Freeze S4 as a paper method only if it improves official target-selection/rollout over J4, FiLM, random/all-layer, and language-only/evidence-only ablations; raises `I_bind` and exact activation mediation; and reproduces on a second architecture with architecture-specific locus discovery. Otherwise retain L0 as a diagnostic and LC-PEA as the main method.
