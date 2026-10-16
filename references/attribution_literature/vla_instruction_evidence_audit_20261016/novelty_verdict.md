# Novelty verdict

**Level 2 — High Overlap** under the scoop-check mapping; the closest papers match three of four axes, so the overall level is 2. No verified paper in this bounded audit implements the exact crossed interaction estimator, but ECL, LIT, Robust Skills and Shortcut Learning collectively cover the training mechanism, separate interventions, grounding/execution decomposition, and controlled factor-confounding story.

## Reviewer delta

> Unlike ECL, which removes jointly instruction/action-attended tokens under a fixed instruction to counter long-tail action bias, and LIT, which separately tests task-preserving visual changes and goal changes, the candidate crosses legal target instructions with target-specific and matched non-target evidence interventions in the same physical state, quantifying whether action retargeting depends on the corresponding visual referent; this is an evaluation/diagnostic delta, not a distinct training-method claim.

## Decision boundary

The diagnostic interaction is defensible as a measurement contribution. The optional training claim is not: ECL already uses instruction/action evidence selection, counterfactual masking and action supervision; LIT already constrains visual conditioning and tests goal changes. Any future training method needs a separate novelty audit.

## Search provenance

- Attempt 1 through the required port 7897 returned TLS EOF for every live source; retained as `step2_search_records_attempt1_proxy_ssl_failure.json` and logs.
- Direct retry returned arXiv=15 and Crossref=15; DBLP/OpenAlex failed parsing/empty responses, OpenReview dependency was missing, and Semantic Scholar returned HTTP 429. Zero results from failed sources are not absence evidence.
- Live results are in `step2_search_records.json`; all deduplicated triage is in `step3_abstract_triage.*`.
