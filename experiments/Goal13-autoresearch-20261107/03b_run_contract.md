# 03b Run Contract：每个 arm 的启动契约

本契约只回答“这个 arm 是否可解释、可复现”，不回答“方法是否有效”。结果判定在 Gate 中完成。

## 1. 必填字段

| 字段 | 要求 |
|---|---|
| `code_commit` | 实际执行代码的 commit，与记录一致 |
| `config_sha256` | 实际加载 config 的 SHA256 |
| `data_arm` | `D0/D1/D2` |
| `parent_arm` | 上一层 best arm；Stage 0 可为 `null` |
| `train_language_counts` | 每个 prompt 的 episode 数 |
| `heldout_prompt_train_count` | 必须为 0 |
| `train_validation_rollout_overlap` | 必须为 0 |
| `sampling_weights` | 实际 sampler 权重，打印出来，不从 YAML 推断 |
| `carrier_dataset_length` | D1/D2 必须等于实测 effective length；当前为 `4786` |
| `carrier_sampling_weight` | D1/D2 必须记录真实归一化权重；当前为 `0.333333` |
| `carrier_manifest` | D1/D2 必须有 48 条、语言唯一、ID 唯一，且与三个 split 交集为 0 |
| `effective_batch` | `per_device_batch × num_processes × grad_accum` |
| `marker_mode` | `none/oracle/predicted/shuffle` |
| `counterfactual_mode` | `none/rank/suff_nec` |
| `seed` | 固定整数 |
| `grad_norm_ratios` | `slot/swap/suff/nec / action`，训练后补 |

## 2. 状态

- `PASS`：契约满足，可以进入对应 Gate；
- `FAIL`：契约满足但结果未达标；
- `INVALID`：数据、配置、评测或来源不满足契约，不能判方法失败。

`INVALID` 必须在结果文档中保留，不允许静默重跑后覆盖。

## 3. 最小 JSON

```json
{
  "schema": "goal13-run-contract-v1",
  "created_at": "2026-11-07T00:00:00Z",
  "contract_status": "PASS",
  "arm_status": "PENDING_GATE",
  "arm": "D0",
  "parent_arm": null,
  "code_commit": "4507931a625c844404c4a76128fb116536d8ca7c",
  "config_sha256": "...",
  "data": {
    "data_arm": "D0",
    "train_language_counts": {
      "red_left": 27,
      "red_right": 34,
      "white_left": 35
    },
    "heldout_prompt_train_count": 0,
    "train_validation_rollout_overlap": 0,
    "carrier_episode_count": 0,
    "sampling_weights": {}
  },
  "runtime": {
    "per_device_batch": 4,
    "num_processes": 1,
    "grad_accum": 1,
    "effective_batch": 4,
    "seed": 17
  },
  "marker_mode": "none",
  "counterfactual_mode": "none",
  "checks": {},
  "failures": []
}
```

## 4. 数据臂定义

| arm | 训练数据 | 预期 C1 train | 预期 carrier |
|---|---|---:|---:|
| `D0` | C1 三 prompt | 96 | 0 |
| `D1` | D0 + yellow-and-white-right carrier | 96 + carrier | 48（待审计复核） |
| `D2` | D1 + white-bowl-right 低权重 | 96 + carrier + optional | 48 |

`white_right` 在所有 arm 中训练计数必须为 0。

## 5. 构建命令

```bash
python scripts/build_goal13_contract.py \
  --audit "$G13/audit/goal13_audit.json" \
  --split-manifest "$C1/c1_split_manifest_v1.json" \
  --data-arm D0 \
  --output "$G13/contracts/D0_contract.json" \
  --code-commit "$CODE_COMMIT" \
  --per-device-batch 4 \
  --grad-accum 1 \
  --num-processes 1 \
  --seed 17
```

构建器必须 fail closed：缺少 audit、split、counts 或发现交集时，输出 `contract_status=INVALID` 并返回非零退出码。

## 6. Gate 前检查

- 训练日志中的实际 batch、accumulation、world size 与契约一致；
- 训练日志中的实际数据集长度与 audit 一致；
- held-out 评估集和 validation/rollout 无交集；
- 配置文件 SHA 在启动前后一致；
- 训练后补 `grad_norm_ratios` 和 checkpoint SHA；
- 未通过上述检查，状态只能是 `INVALID`。

## 7. Postrun 验证

训练结束后，必须用 checkpoint health 和 postrun verifier 复核，不把“日志没报错”当作 loss 正常：

```bash
/root/starvla_cu124/bin/python /tmp/goal13_tools/check_goal13_checkpoint_health.py \
  --checkpoint "$RUN/final_model/model.safetensors" \
  --output "$G13/contracts/${ARM}_smoke_checkpoint_health.json"

/root/starvla_cu124/bin/python /tmp/goal13_tools/verify_goal13_postrun.py \
  --contract "$G13/contracts/${ARM}_contract.json" \
  --run-dir "$RUN" \
  --train-log "$G13/logs/${RUN_ID}.log" \
  --config-snapshot "$G13/configs/${ARM}_snapshot.yaml" \
  --checkpoint-health "$G13/contracts/${ARM}_smoke_checkpoint_health.json" \
  --output "$G13/contracts/${ARM}_smoke_postrun.json" \
  --expected-steps 2 \
  --repo "$R"
```

状态约定：

- `postrun_status=PASS` 且 `arm_status=SMOKE_PASS`：只说明 smoke 链路可解释；
- `gate_status=NOT_EVALUATED`：不能写成方法有效；
- 任一字段缺失或 checkpoint 非有限：`postrun_status=INVALID`，先修链路。
