# Goal18 Run Contract

| 字段 | 必须满足 |
|---|---|
| `config_sha256` | 与实际加载配置一致 |
| `point_head_sha256` | Goal17 `G4-MARGIN` 不变 |
| `train_index_sha256` | 4-prompt train index 不变 |
| `eval_init_count` | 15（3 calibration + 12 confirmation） |
| `eval_init_in_train` | 0 |
| `validation_in_train` | 0 |
| `white_right_eval_in_train` | 0 |
| `training_scope` | 只训练 destination calibration / residual；无完整 action head |
| `oracle_fallback` | false |
| `gpu` | 只使用 GPU1，或明确记录的可用 GPU |
| `output` | 新目录，不覆盖 Goal17 |

## 断言

```text
assert eval_init_count == 15
assert eval_init_in_train == 0
assert validation_in_train == 0
assert white_right_eval_in_train == 0
assert oracle_fallback is False
assert training_scope in {"destination_offset", "destination_residual"}
```

## 状态

- `PASS`：contract 完整，Gate 通过；
- `FAIL`：contract 完整，Gate 不通过；
- `INVALID`：SHA、split、数据覆盖、评测或环境不满足契约。
