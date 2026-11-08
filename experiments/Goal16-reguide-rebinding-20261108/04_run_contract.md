# Goal16 Run Contract

每个 arm 启动前必须填写。任何一项缺失或与实际运行不一致，该 arm 标为 `INVALID`，不进入 Gate。

| 字段 | 必须满足 |
|---|---|
| `code_commit` | 实际执行 commit；远端代码与本地一致 |
| `config_path` | `configs/vla/goal16_reguide_rebinding.json` |
| `config_sha256` | runner 中的 `EXPECTED_CONFIG_SHA256` |
| `manifest_sha256` | `e9564ca9aef800e618e3fdb5d144892eabe9471815bbd24bae3140583946f45e` |
| `checkpoint_fingerprint` | `747a28690383ca45e795518121e9aed3f4610265c14618a69f203922b4238e20` |
| `statistics_sha256` | `6d5ebb47bc5132df581aae0481d529d04a0821c331859c05a9818d78156c9538` |
| `point_head_path` | `$W/results/star105_goal10p5_20261105/point_head/pilot_5000/point_head.pt` |
| `point_head_sha256` | 启动时计算并写入 |
| `grounding_sidecar_sha256` | 启动时计算并写入 |
| `train_language_counts.white_right` | `0` |
| `train_validation_rollout_overlap` | `0` |
| `init_count` | `15`（3 calibration + 12 confirmation） |
| `seed` | `7` |
| `arms` | `B0`、`R-GEO`、`R-NAIVE`、`R-PRED`、`R-SHUF` |
| `oracle_grounding` | 只用于 `R-GEO`、`R-NAIVE`、`R-SHUF` |
| `predicted_grounding` | 只用于 `R-PRED`，不得静默回退 oracle |
| `control_axes_geo` | x/y/z transport；rotation = base |
| `control_axes_naive` | x/y only；z = base |
| `transport_bounds` | `R-GEO/R-PRED/R-SHUF`: xy `0.20`、z `0.15`；`R-NAIVE`: xy `0.20` |
| `release` | `xy=0.03`、`z=0.05`、streak `3`、settle `4` |
| `max_transports_per_stage` | `2` |
| `gpu` | 只使用 GPU1 |
| `training` | 否；无 backward、无 optimizer、无新 checkpoint |
| `output` | 新目录，不能覆盖 Goal14/v7–v12 |

## 启动断言

```text
assert config_sha256 == EXPECTED_CONFIG_SHA256
assert manifest_sha256 == frozen manifest sha
assert checkpoint_fingerprint == frozen fingerprint
assert statistics_sha256 == frozen statistics sha
assert white_right_train_count == 0
assert train_validation_rollout_overlap == 0
assert len(selected) == 15
assert len(calibration) == 3 and len(confirmation) == 12
assert set(arms) == {B0, R-GEO, R-NAIVE, R-PRED, R-SHUF}
assert r_geo_z_cap == 0.15 and r_geo_xy_cap == 0.20
assert r_naive_z_cap == 0.0
assert release == {xy: 0.03, z: 0.05, streak: 3, settle: 4}
assert predicted_grounding_sha256 is not None for R-PRED
```

## 状态定义

- `PASS`：Run Contract 完整，Gate 全部通过；
- `FAIL`：Run Contract 完整，测量完成，但 Gate 不通过；
- `INVALID`：SHA、split、配置、grounding、评测或运行环境不满足契约，不能判方法失败。

## 逐 arm 前置条件

| arm | 必须已完成 |
|---|---|
| `B0` | Step 0 PASS |
| `R-GEO` | Step 1 单测 + Step 3 smoke PASS |
| `R-NAIVE` | `R-GEO` 已跑完且无运行错误 |
| `R-PRED` | grounding sidecar PASS |
| `R-SHUF` | `R-GEO` 已跑完且无运行错误 |

## 实际结果（2026-11-08）

| 项 | 结果 |
|---|---|
| Run Contract | `PASS` |
| Grounding predictor | `INVALID`（15/15 miss，mean 0.603 m，max 0.623 m） |
| G1 | `PASS` |
| G2 | `FAIL`（原因是 R-PRED grounding INVALID，不是 R-GEO 无信号） |
| 最终状态 | `INVALID` |

详见 `05_pilot_result.md` 和 `run_contract.json`。
