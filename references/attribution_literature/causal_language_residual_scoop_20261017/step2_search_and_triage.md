# Step 2–3 — Search and abstract triage

Date: 2026-10-17.

## Live-search boundary

Six queries were issued through paper-search. OpenAlex returned relevant results; Semantic Scholar returned HTTP 429; arXiv later returned rate-limit responses; DBLP/Crossref/OpenAlex multi-source runs partly timed out; OpenReview dependency was unavailable. Failed sources are not absence evidence. Search commands and raw output are retained outside the paper claim; seven high-risk candidates were fetched as PDFs where possible.

## High-risk candidates

| Work | Problem | Mechanism from abstract/full text | First-pass overlap |
|---|---|---|---|
| Encoded but Not in Control (2610.06235) | language target is encoded but does not control selection | scene-preserving instruction intervention, layerwise action lens, target/scene probes, attention/UMAP/NMF | 3/4 |
| Do VLAs Understand Instructions? (2610.10178) | locate causal language influence in VLA action experts | attribution patching followed by exact activation patching across π0.5/GR00T layers/tokens | 3/4 |
| LangForce (2601.15197) | vision shortcut collapses conditional information from language | latent action queries, vision-only prior/language posterior, PMI/LLR objective | 3/4 |
| When Instructions Retrieve Trajectories / ECT (2609.39971) | instruction–action binding failure under counterfactual task changes | action-valid paired counterfactual demonstrations and matched-update imitation loss | 3/4 |
| VLAs are Confined yet Capable (2505.03500) | learned skills remain latent and cannot be recombined | task-averaged text latent written to residual stream and temporally interpolated | 2/4 |
| PosA-VLA (2512.03724) | diffuse attention harms precise action | CLIP language maps + EEF maps gate visual features into flow action expert | 2/4 |
| Anchor-Align (2607.13429) | action tuning forgets VLM semantics and language/action diverge | layerwise frozen-VLM anchoring + same-observation action-direction language CE | 2/4 |
| FineVLA (2605.27284) | coarse goal language omits execution factors | fine-grained action-aligned language data/annotator/policy mixture | 1–2/4 |
| VLA Grounder (2607.04517) | frozen VLA is brittle to phrasing | RL-trained upstream language-conditioning policy; downstream VLA frozen | 1–2/4 |

## Other retrieved candidates

Online results also surfaced LangGap, Rephrase Before You Act, Learning What to Say to Your VLA, Paraphrase Consensus, Gaze2Act, Stage-Aware VLA, MoLe-VLA, CroSTAta and goal-representation interfaces. These establish a crowded language-control landscape but are not used for a stronger overlap verdict without equivalent full-text verification.
