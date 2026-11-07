# T4b v12 P0 Lateral-Authority + Release/Settle

- status: `p0_pass`
- gate_passed: `True`
- manifest_sha256: `e9564ca9aef800e618e3fdb5d144892eabe9471815bbd24bae3140583946f45e`
- config_sha256: `ecb501f86b4b795d94429cc17ce21bb2461014b64a54b3b006f2c9d344cef52d`

| arm | cap | success | calibration | confirmation | released | hold-only | released placement | wrong destination | target not reached | retained | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.00 | 0 | 0 | 0 | 0 | 0 | 0 | 14 | 0 | 15 | 0 |
| A05 | 0.05 | 5 | 1 | 4 | 1 | 4 | 2 | 3 | 0 | 15 | 0 |
| A10 | 0.10 | 7 | 1 | 6 | 4 | 3 | 6 | 3 | 0 | 15 | 0 |
| A20 | 0.20 | 7 | 1 | 6 | 4 | 3 | 7 | 3 | 0 | 15 | 0 |

## Gate

| check | passed |
|---|---|
| `b0_baseline_reproduced` | `true` |
| `a10_total_success` | `true` |
| `a10_confirmation_success` | `true` |
| `a10_released_success` | `true` |
| `a10_over_a05` | `true` |
| `a20_overshoot_control` | `true` |
| `stability` | `true` |
