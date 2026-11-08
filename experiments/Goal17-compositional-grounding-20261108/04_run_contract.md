# Goal17 Run Contract

| 字段 | 必须满足 |
|---|---|
| `config_sha256` | 与实际加载配置一致 |
| `manifest_sha256` | Goal16 15-init manifest 不变 |
| `eval_init_count` | 15（3 calibration + 12 confirmation） |
| `eval_init_in_train` | 0 |
| `white_right_eval_in_train` | 0 |
| `train_prompt_coverage` | 4 个 prompt 都有 |
| `counterfactual_pairs` | 同图左右/红白配对完整 |
| `margin_enabled` | 主 arm 为 true；baseline 为 false |
| `oracle_fallback` | false |
| `training_scope` | 只训练 grounding head；无 residual/route/action |
| `gpu` | 只使用 GPU1 |
| `output` | 新目录，不覆盖 Goal16 |

## 断言

```text
assert eval_init_count == 15
assert eval_init_in_train == 0
assert white_right_eval_in_train == 0
assert set(train_prompt_ids) == {red_left, red_right, white_left, white_right}
assert oracle_fallback is False
assert gpu == 1
```

## 状态

- `PASS`：contract 完整，Gate 通过；
- `FAIL`：contract 完整，Gate 不通过；
- `INVALID`：SHA、split、数据覆盖、评测或环境不满足契约。
