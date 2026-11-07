# 08 D0 15-init C1 闭环 Pilot

## 1. 结论

D0 matched 2.5K checkpoint 的 15-init × 4-prompt pilot 已完成，契约 `PASS`：

```text
contract_status = PASS
num_init_groups = 15
num_episodes    = 60
seen episodes   = 45
white_right     = 15
```

D0 复现了关键的 held-out 现象：

- `white_right` official success：`0/15`；
- `white_right` strict destination：`0/15`；
- `white_right` first destination：`plate_1 = 7`、`null = 8`；
- 没有 runtime error，60 条 episode 全部完整写出。

因此 D0 可以作为后续 D1 的比较基线。这个结论只说明“历史 `white_right=0/15` 在当前初始化、数据和 evaluator 下可复核”，不说明任何方法有效。

## 2. Seen 指标

| 指标 | D0 matched | 历史 `M0_c1_adapt` | 变化 |
|---|---:|---:|---:|
| seen episode successes | 20 / 45 | 24 / 45 | -4 |
| seen group-any | 13 / 15 | 12 / 15 | +1 |
| seen group-all | 1 / 15 | 4 / 15 | -3 |

分 prompt：

| prompt | D0 success | 历史 success | D0 strict destination | 历史 strict destination |
|---|---:|---:|---:|---:|
| `red_left` | 6 / 15 | 7 / 15 | 10 / 15 | 9 / 15 |
| `red_right` | 6 / 15 | 7 / 15 | 10 / 15 | 11 / 15 |
| `white_left` | 8 / 15 | 10 / 15 | 11 / 15 | 10 / 15 |

解释：

- group-any 与历史只差 1 个 init，满足“seen group-any 不退化”的基线要求；
- episode success 和 group-all 低于历史，说明这次训练的跨 prompt 一致性更弱；
- 不能把这 4 个 episode 的差异写成方法退化，因为它仍是 D0，只换了训练产物；
- D1 必须与当前 D0 使用同一 evaluator、同一 manifest、同一 seed，而不是只与历史数字比较。

## 3. White-right 失败分解

| 项 | D0 matched | 历史 `M0_c1_adapt` |
|---|---:|---:|
| official success | 0 / 15 | 0 / 15 |
| strict destination | 0 / 15 | 0 / 15 |
| first destination `plate_1` | 7 | 7 |
| first destination `null` | 8 | 8 |
| `target_not_reached` | 8 | 8 |
| `wrong_receptacle_or_relation` | 6 | 7 |
| `grasp_failure` | 1 | 0 |

失败结构几乎相同：绝大多数失败不是抓错物体，而是没有把白色 mug 送到右盘；7 次明确进入左盘，8 次没有到达任何目标盘。只有 1 次是 gripper / grasp 级别失败。

这与 65 的假设一致：D0 的瓶颈首先表现为 destination binding / 执行，而不是“看不到白色物体”。但本 pilot 不能区分 binding 和 residual action 的贡献，必须进入 D1 和后续 I 层。

## 4. 证据与口径

- manifest：`$W/vla_workspace/data/jgate_c1/counterfactual_20261027/manifest.jsonl`；
- evaluator：`scripts/star105_c1_rollout.py`；
- server：`deployment/model_server/server_policy.py`；
- seed：7；
- 15 init × 4 prompt = 60 episodes；
- `white_right` 从未进入训练；
- 结果文件包含 official success、first object、first destination、strict destination 和 failure type。

本 pilot 只作筛选，正式结论仍需 30/50 init + 多 seed。

## 5. D0 Gate 判定

| 检查 | 结果 |
|---|---|
| D0 postrun PASS | PASS |
| D0 完整 offline eval PASS | PASS |
| 闭环契约 PASS | PASS |
| `white_right = 0/15` 复现 | PASS |
| seen group-any 不明显退化 | PASS（13/15 vs 12/15） |
| runtime error = 0 | PASS |
| `white_right` 训练计数 = 0 | PASS（Stage 0 audit） |

**D0 baseline Gate：PASS。** 可以进入 D1 数据构建；不得据此进入 I/A/F 层。

## 6. 证据文件

- [star13_d0_matched_2500_summary.json](results/star13_d0_matched_2500_summary.json)
- [star13_d0_matched_2500_c1_pilot_contract.json](results/star13_d0_matched_2500_c1_pilot_contract.json)
- 远端逐 episode：`$G13/c1/star13_d0_matched_2500/episodes.jsonl`
- 远端 rollout summary：`$G13/c1/star13_d0_matched_2500/summary.json`

## 7. 下一步

1. 构建 D1 `yellow_and_white_right` carrier 的 LeRobot 数据；
2. 生成 D1 observed dataset contract，断言 `white_right` 仍为 0；
3. 用同一 A3 初始化、同一 seed、同一 2.5K/effective batch 4 训练 D1；
4. D1 先过 postrun + offline contract，再跑同一 15-init pilot；
5. 只有 D1 相对 D0 有方向性增益或明确失败解释，才允许进入 I 层。
