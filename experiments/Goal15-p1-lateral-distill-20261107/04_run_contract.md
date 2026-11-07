# P1 Run Contract

每个 arm 启动前必须填写；任何一项缺失或与实际运行不一致，标 `INVALID`，不进入 Gate。

| 字段 | 必须满足 |
|---|---|
| `code_commit` | 本地 `main` 当前 commit，远端代码一致 |
| `config_sha256` | 实际加载的 config SHA |
| `parent_arm` | `R0`、`A10` 或 `R2` |
| `teacher` | `A10`，config SHA 与 P0 一致 |
| `teacher_cache_sha256` | P1-B manifest/数组 SHA |
| `student_backbone` | J2 fingerprint `747a...8e20` |
| `student_param_count` | `R1-CAP` 与 `R2` 差 `<=0.1%` |
| `data_arm` | `cache_train_v2` + P1 teacher labels |
| `train_language_counts` | `white_right = 0` |
| `train_validation_rollout_overlap` | `0` |
| `sampling_weights` | 启动时打印实际值 |
| `effective_batch` | batch × accumulation |
| `lateral_cap` | `0.10` |
| `z_cap` | `0.0` |
| `rotation_cap` | `0.0` |
| `release_mode` | `learned_logit`，阈值来自 validation |
| `settle_steps` | `4` |
| `seed` | `17`（训练） / `7`（闭环） |
| `offline_gate_status` | `PASS / FAIL / INVALID` |
| `closed_loop_gate_status` | `PASS / FAIL / INVALID` |

## 启动断言

```text
assert config_sha256
assert teacher_cache_sha256
assert student_backbone == frozen J2
assert R1_param_count ≈ R2_param_count
assert white_right_train_count == 0
assert train_validation_rollout_overlap == 0
assert lateral_cap == 0.10 and z_cap == 0 and rotation_cap == 0
assert settle_steps == 4
assert all arrays finite
```

## 状态

- `PASS`：契约完整，Gate 通过；
- `FAIL`：契约完整，Gate 不通过；
- `INVALID`：数据、证据、配置、参数量或评测不满足契约，不能判方法失败。
