# Next Forcing archive

## Source

- Paper: https://arxiv.org/abs/2606.11187
- PDF requested by user: https://arxiv.org/pdf/2606.11187
- Official project page: https://gangweix.github.io/next-forcing/
- User-provided code page: https://github.com/gangweix/next-forcing1
- Possible related project URL requiring verification: https://github.com/gangweix/next-forcing
- Model card / metadata: https://huggingface.co/gangweix/next-forcing-base

## Status

`pending_download`: PDF and full code checkout were not copied because the local 7897 proxy was unavailable and sandbox-external execution was rejected by the automatic approval service. The repository provenance placeholder is at `../repos/next-forcing1/README.md`.

## Core method summary

Next Forcing addresses myopic supervision in autoregressive video world models. It adds lightweight chained Multi-Chunk Prediction (MCP) modules that predict multiple future video chunks at several temporal horizons, while the main model remains causal. The method is a temporal prediction/training framework, not a phase-labeling method and not an attribution-alignment method.

Reported official project/model materials describe a 5.1B BF16 base backbone, 30 backbone layers, MCP modules added during post-training, and a larger post-trained model. These details must be rechecked from the downloaded paper/repository before implementation.
