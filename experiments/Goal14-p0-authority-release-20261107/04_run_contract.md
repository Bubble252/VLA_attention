# P0 Run Contract

每个 arm 启动前必须填写。任何一项缺失或与实际运行不一致，该 arm 标为 `INVALID`，不进入 Gate 判读。

| 字段 | 必须满足 |
|---|---|
| `code_commit` | 本地 `main` 的当前 commit；远端代码必须与它一致 |
| `config_path` | `configs/vla/k4_lite_t4b_v12_p0_authority_release.json` |
| `config_sha256` | 与 runner 中 `EXPECTED_CONFIG_SHA256` 完全一致 |
| `manifest_path` | 冻结 15-init manifest |
| `manifest_sha256` | `e9564ca9aef800e618e3fdb5d144892eabe9471815bbd24bae3140583946f45e` |
| `checkpoint` | J2 12500 checkpoint |
| `checkpoint_fingerprint` | `747a28690383ca45e795518121e9aed3f4610265c14618a69f203922b4238e20` |
| `statistics_sha256` | `6d5ebb47bc5132df581aae0481d529d04a0821c331859c05a9818d78156c9538` |
| `metric_amendment_sha256` | `053cc6e1fd9bfe84ec7c4fb6de760a8083f201264466e5a6046272d3fbbd8e52` |
| `task` | `LIVING_ROOM_SCENE5` |
| `prompt` | `put the white mug on the right plate` |
| `inits` | 15 个唯一 init；3 calibration + 12 confirmation |
| `calibration_init_ids` | `init_0e94a4e6b9622c6b`、`init_2edf49999eb35566`、`init_3500df19a4d3b5a1` |
| `seed` | 7 |
| `arms` | `B0`、`A05`、`A10`、`A20`，顺序与配置一致 |
| `control_axes` | 只改 x/y；z 和旋转保持 base policy |
| `release` | 到目标后 release；`release_xy_m=0.03`、`release_z_m=0.05`、streak `3` |
| `settle` | 4 步 |
| `training` | 否；无 backward、无 optimizer、无新 checkpoint |
| `gpu` | 只使用 GPU1 |
| `output` | 新目录，不能覆盖 v7–v11 |
| `errors` | 启动时记录实际值；非零必须审计 |

## 每个 arm 的启动断言

```text
assert config_sha256 == EXPECTED_CONFIG_SHA256
assert manifest_sha256 == frozen manifest sha
assert checkpoint_fingerprint == frozen fingerprint
assert statistics_sha256 == frozen statistics sha
assert len(selected) == 15
assert len(calibration) == 3
assert len(confirmation) == 12
assert arm caps == {B0: none, A05: 0.05, A10: 0.10, A20: 0.20}
assert release == {xy: 0.03, z: 0.05, streak: 3, settle: 4}
```

## 状态定义

- `PASS`：Run Contract 完整，Gate 全部通过；
- `FAIL`：Run Contract 完整，测量完成，但 Gate 不通过；
- `INVALID`：SHA、split、配置、评测、数据或运行环境不满足契约，不能判方法失败。
