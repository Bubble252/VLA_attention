# Goal17 execution selection

- status: `PASS`
- best arm: `E2-PRED`
- default arm: `E2-PRED`
- grounding: `PASS`
- E1 settle steps: `32`

| arm | success | released | hold-only | wrong dest |
|---|---:|---:|---:|---:|
| `B0` | 0 | 0 | 0 | 14 |
| `E1-PRED-SETTLE4` | 0 | 0 | 0 | 3 |
| `E1-PRED-SETTLE32` | 2 | 2 | 0 | 3 |
| `E2-GEO` | 2 | 2 | 0 | 3 |
| `E2-PRED` | 3 | 2 | 1 | 3 |

## Gates

- `E1-PRED-SETTLE32`: `True`; failed=[]
- `E2-PRED`: `True`; failed=[]
