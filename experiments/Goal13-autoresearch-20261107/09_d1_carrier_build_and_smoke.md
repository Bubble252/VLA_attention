# 09 D1 Carrier 数据构建、Run Contract 与 2-step Smoke

## 1. 结论

D1 的 carrier 数据已经按 audit 精确抽取并转换成独立 LeRobot 数据集，Run Contract 和 2-step training smoke 都通过：

```text
carrier extraction       = PASS
carrier episodes        = 48
carrier split overlap   = 0
observed dataset length = 11630 (C1) + 4786 (carrier)
sampling weights        = 0.666667 / 0.333333
contract_status         = PASS
2-step postrun          = PASS / SMOKE_PASS
checkpoint tensors      = 248 finite
```

这只证明 D1 数据、mixture、训练链路和契约可解释，**不证明 D1 有效**。D1 是否改善 `white_right` 必须等 2.5K matched run 和同一 15-init pilot。

## 2. Carrier 抽取

来源是 Stage 0 audit 中的 48 条 `yellow_and_white_right` 记录。工具按 `source_shard + source_record_index` 精确读取原始 TFRecord，不按语言重新扫描后猜测记录。

| 检查 | 结果 |
|---|---:|
| 期望 carrier 数 | 48 |
| 实际 carrier 数 | 48 |
| 唯一 episode ID | 48 |
| 与 train 交集 | 0 |
| 与 validation 交集 | 0 |
| 与 rollout 交集 | 0 |
| 语言 | `put the yellow and white mug on the right plate` |
| `white_right` 训练计数 | 0 |

证据：

- [D1_carrier_extract_report.json](results/D1_carrier_extract_report.json)
- [D1_carrier_episodes.jsonl](results/D1_carrier_episodes.jsonl)

关键 SHA256：

```text
carrier_episode_ids_sha256 = bf32459d360bbdcae0366e27de08d000df1e1ff30585aa944881479f809740bb
carrier_tfrecord_sha256     = b7858ac999c48f2e3c7653e6fa0bc9506f7f40057da5e07ca6d51a572dcfe53a
carrier_manifest_sha256     = 616bd9fdf85bb3c90cf53e631ee8c450f3de95f83279d98369e94f896c9c94fe
```

## 3. LeRobot 转换

carrier 被转换成独立数据集 `libero_90_goal13_carrier_lerobot`，没有混入原 C1 dataset，因此历史 `c1_dataset_length=11630` 仍然成立。

```text
episodes      = 48
raw frames    = 5122
effective len = 5122 - 48 × 7 = 4786
total_tasks   = 1
```

训练后真实 loader 观测到：

| 数据集 | effective length | trajectories | sampler weight |
|---|---:|---:|---:|
| `libero_90_gate_lerobot` | 11630 | 96 | 0.666667 |
| `libero_90_goal13_carrier_lerobot` | 4786 | 48 | 0.333333 |

证据：[observed_d1.json](results/observed_d1.json)。

## 4. 为什么是 2:1，不是 50:50

D0 是 C1-only，2.5K steps × batch 4 = 10K samples，全部来自 C1。D1 若 C1/carrier 各 50%，C1 的 effective samples 会直接减半，D1 变差时无法区分“carrier 有害”还是“C1 被稀释”。

因此 D1 使用：

```text
C1 base weight      = 1.0
carrier base weight = 0.5
balance_dataset_weights = false
```

归一化后 C1 占 `2/3`、carrier 占 `1/3`。这仍然会让本轮 C1 effective samples 从 10K 降到约 `6.67K`，所以 D1 的结果必须带着这个限制解释；后续若需要严格隔离数据暴露，应增加 C1 曝光匹配臂或提高 D1 steps，而不是事后改口径。

## 5. Run Contract

```text
arm             = D1
parent_arm      = D0
data_arm        = D1
contract_status = PASS
code_commit     = 4507931a625c844404c4a76128fb116536d8ca7c
config_sha256   = 9dce4f07a037f2e37cbbed16a8a18972be8a05cac57f2befdd801c09b4e9e750
```

新增检查：

- carrier dataset 必须存在且长度等于 `4786`；
- C1/carrier sampler 权重必须分别等于 `0.666667/0.333333`；
- carrier manifest 必须恰好 48 条、语言唯一、episode ID 唯一；
- carrier episode 与 train/validation/rollout 交集必须为 0；
- `white_right` 训练计数必须仍为 0。

证据：[D1_contract.json](results/D1_contract.json)。

## 6. 2-step Smoke

2-step smoke 使用同一 D1 config、同一 A3 初始化、同一 batch/seed 设置，训练日志出现：

```text
STAR105_RUNTIME_BATCH_CONTRACT accum=1 total_batch=4
STAR105_RUNTIME_DATASET_CONTRACT libero_90_gate_lerobot:length=11630:weight=0.666667 libero_90_goal13_carrier_lerobot:length=4786:weight=0.333333
STAR105_RUNTIME_CONFIG_SHA 9dce4f07a037f2e37cbbed16a8a18972be8a05cac57f2befdd801c09b4e9e750
```

postrun 解析到 `max_train_steps=2`、per-device batch `4`、accumulation `1`、seed `17`，并通过：

- final/step checkpoint 存在且内容一致；
- checkpoint 248 个 tensors 全部 finite；
- dataset statistics transitions/trajectories 与契约一致。

证据：

- [D1_smoke_postrun.json](results/D1_smoke_postrun.json)
- [D1_smoke_checkpoint_health.json](results/D1_smoke_checkpoint_health.json)

### 6.1 离线 evaluator 的归一化一致性修复

D1 训练使用 C1+carrier 的 merged dataset statistics，而旧离线 evaluator 只用 C1-only dataset 的 statistics 做反归一化。D0 是单数据集，所以历史结果不受影响；D1 若继续沿用旧路径，会把预测动作按错误统计解释。

因此 D1 评测前增加了：

- 当 `data_mix` 包含多个 dataset 时，用与训练相同的 `get_vla_dataset` 构造 mixture；
- 从 mixture 的 `merged_metadata` 读取训练归一化统计；
- 单数据集路径保持原行为，避免改变 D0 口径。

远端补丁记录在 `server/patches/goal13_d1_eval_mixture_metadata_20261107.patch`。D1 offline smoke 必须先验证该路径确实加载 mixture metadata，再跑完整评测。

## 7. 下一步

1. 完成 D1 2.5K matched training，并重新过 postrun；
2. 用同一 evaluator 跑 D1 offline `limit=2` smoke，再跑完整 118-group offline eval；
3. 用同一 15-init × 4-prompt manifest 跑 D1 pilot；
4. 只有 D1 相对 D0 在 `white_right` 或机制指标上有方向性增益，或给出明确失败解释，才允许取 best data arm 进入 I 层；
5. 不因 smoke PASS 提前启动 interface 或 action 层。
