# Step 1 — Claim decomposition

Date: 2026-10-17.

## Research problem

VLA policies may encode the instruction internally while final actions remain dominated by scene/task shortcuts. The goal is to strengthen language control without hard-masking action-relevant visual/state evidence or blindly modifying every layer.

## Proposed novelty

Use same-scene legal instruction swaps and activation patching to identify language residual/head/token locations that causally retarget the action. Only at those validated action loci, insert a zero-initialized residual module that combines phrase hidden states with source/target spatial evidence from the VLM→VLA Bridge-0; train it jointly with LC-PEA and the native action objective.

## Four axes

- Problem framing: encoded-but-not-in-control language grounding in multi-object manipulation; improve action retargeting while preserving primitive execution.
- Core mechanism: probe + activation/head/token patching selects action-causal language loci; source/target spatial evidence is injected only at selected action-token/query positions through a zero-initialized gated residual.
- Key insight: where target identity is decodable is not necessarily where language controls action; module placement should be selected by causal mediation, and content should bind language to spatial referents.
- Application domain: OpenVLA/OFT first, then architecture-adapted π0.5/GR00T/MolmoAct2; LIBERO/Plus and multi-object manipulation.

## Falsification

The claim fails if random-layer or all-layer residual matches the causal-locus residual, if wrong-object/language residual matches correct evidence, if activation mediation rises without target selection/rollout gain, or if a verified prior work already performs causal-locus selection plus referent-evidence injection for VLA actions.
