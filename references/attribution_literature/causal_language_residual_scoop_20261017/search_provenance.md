# Search provenance

Date: 2026-10-17.

Queries:

1. `vision language action language-conditioned adapter action head`
2. `robot manipulation instruction-conditioned routing cross-attention`
3. `vision language action language-action alignment latent query tokens`
4. `VLA latent action queries language instruction`
5. `vision language action language sensitive layers heads`
6. `robot policy language gating FiLM instruction following`

Sources attempted through paper-search: arXiv, DBLP, OpenAlex, OpenReview, Semantic Scholar, Crossref. OpenAlex and Crossref returned results throughout. The initial arXiv batch was rate-limited, but a later single-source query with a 10-second request floor succeeded. `openreview-py` and `feedparser` were initially missing; both were installed on 2026-10-17, after which the OpenReview connector ran anonymously without dependency errors. Its zero result for the tested query reflects the connector's strict all-token match over the configured venue list, not an infrastructure failure.

Semantic Scholar still returns HTTP 429 for anonymous requests and no `SEMANTICSCHOLAR_API_KEY` is configured. Do not repeatedly retry it in one session; add a key later and run it as a single source. DBLP returns HTTP 202 with an HTML browser-verification challenge (`Making sure you're not a bot!`) rather than JSON on both `dblp.org` and `dblp.uni-trier.de`; this is an upstream anti-bot block, not a parser or query result.

## Reliable retrieval workflow

1. Use OpenAlex first for broad recall and structured abstracts.
2. Run arXiv alone, one query at a time, with `ARXIV_MIN_INTERVAL=10`, `PAPER_SEARCH_MAX_ATTEMPTS=1`, and bounded connect/read timeouts.
3. Run Crossref separately for DOI/venue verification.
4. Run OpenReview separately; use broader 1–3-token queries because its connector requires every query token to appear.
5. Use Semantic Scholar only when an API key is configured; anonymous 429 is recorded as missing-source evidence.
6. Treat DBLP as unavailable while its browser challenge is active; substitute OpenAlex/Crossref plus official venue pages.
7. For every high-risk candidate, fetch the PDF directly and verify introduction, method, experiments and limitations. Search-source failure never counts as evidence that a paper does not exist.

Working command template:

```bash
PAPER_SEARCH_TIMEOUT_SECONDS=30 \
PAPER_SEARCH_CONNECT_TIMEOUT_SECONDS=10 \
PAPER_SEARCH_MAX_ATTEMPTS=1 \
ARXIV_MIN_INTERVAL=10 \
/usr/bin/python3.10 /home/bubble/.cc-switch/skills/paper-search/scripts/search_papers.py \
  --query "<focused query>" --start-year 2024 --end-year 2026 \
  --max-papers 10 --sources open_alex
```

Repeat the command with exactly one of `arxiv`, `crossref`, or `openreview`; store stdout/stderr per source and merge/deduplicate only after all completed sources have returned.

Local verified corpus additionally includes PDFs/notes/source for Anchor-Align, PosA-VLA, LIT, ECL, BlindVLA, FOCAL-VLA, π0/OpenVLA/OFT, StarVLA QwenPI/M1/LangForce, ProGAL-VLA, and prior novelty audits. RoVLA: Action-Conditioned Temporal Grounding and Structured Conditioning was recalled through Crossref (DOI `10.1109/CASE69030.2026.11704424`) but Crossref/OpenAlex exposed no abstract and the arXiv title query retrieved a different RoVLA; it remains an unresolved high-risk candidate rather than a verified comparison.
