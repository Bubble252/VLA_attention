# Step 6 — Four-axis comparison

## Proposed candidate

- Problem: language is represented but not behaviorally load-bearing in ambiguous manipulation.
- Mechanism: legal same-scene instruction swaps + exact activation patching choose action-causal language loci; a zero-initialized gated residual injects phrase hidden states and source/target spatial evidence only into action tokens at those loci; joint with LC-PEA.
- Insight: decodability is not control; module placement must be causally selected and its content must bind language to visual referents.
- Domain: multi-object VLA manipulation, OpenVLA/OFT then π0.5.

| Prior work | Problem | Mechanism | Insight | Domain | Matches | Verdict |
|---|---|---|---|---|---:|---|
| Encoded but Not in Control | match | partial | match | match | 3 | Level 2 — High Overlap |
| Do VLAs Understand Instructions? | match | partial | match | match | 3 | Level 2 — High Overlap |
| LangForce | match | differ | match | match | 3 | Level 2 — High Overlap |
| ECT | match | differ | match | match | 3 | Level 2 — High Overlap |
| Text Latent Interpolation | partial | partial | match | match | 2 | Level 3 — Medium Overlap |
| PosA-VLA | partial | partial | partial | match | 1–2 | Level 3/4 |
| Anchor-Align | partial | differ | match | match | 2 | Level 3 — Medium Overlap |
| FineVLA | partial | differ | partial | match | 1 | Level 4 — Low Overlap |
| VLA Grounder | partial | differ | partial | match | 1 | Level 4 — Low Overlap |

The closest verified papers each cover three axes. No verified paper in this bounded audit performs the exact combination of causal-locus selection plus phrase-specific source/target spatial evidence injection plus LC-PEA future-change supervision.
