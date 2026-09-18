# V3 Conditioning Sweep

| tag | lambda | teacher temperature | pointing | mass-in-box | IoU |
|---|---:|---:|---:|---:|---:|
| l0p06_t1p00 | 0.06 | 1.00 | 0.2344 | 0.2713 | 0.2653 |
| l0p08_t1p00 | 0.08 | 1.00 | 0.3125 | 0.3244 | 0.2692 |
| l0p12_t1p00 | 0.12 | 1.00 | 0.3125 | 0.3289 | 0.2715 |
| l0p15_t1p00 | 0.15 | 1.00 | 0.2656 | 0.3142 | 0.2711 |
| l0p10_t0p50 | 0.10 | 0.50 | 0.3281 | 0.3314 | 0.2702 |
| l0p10_t0p75 | 0.10 | 0.75 | 0.3594 | 0.3389 | 0.2696 |
| l0p10_t1p25 | 0.10 | 1.25 | 0.4219 | 0.3324 | 0.2723 |
| l0p10_t1p50 | 0.10 | 1.50 | 0.3750 | 0.3408 | 0.2692 |

Best provisional joint mass+pointing candidate: `l0p10_t1p25`. It must still beat the coarse λ=0.1,T=1 anchor on more than one metric and be confirmed with a second seed before selection.
