# 10 D1 Matched 训练、完整离线评测与 C1 Pilot

## 1. 当前结论

D1 的 2.5K matched training 和完整 118-group offline eval 已完成并通过契约：

```text
postrun_status          = PASS / RUN_PASS
offline contract_status = PASS
num_groups              = 118
destination pairs       = 78
object pairs            = 78
```

但离线结果**不支持**“D1 已经优于 D0”：

- destination correct-vs-wrong margin 从 `+0.015105` 降到 `+0.005628`；
- destination correct MAE 从 `0.145168` 变差到 `0.151327`；
- destination margin positive fraction 从 `0.679` 降到 `0.577`；
- object margin 也从 `+0.014345` 降到 `+0.011617`，correct MAE 轻微变差；
- 两个 relation 的 gripper MAE 都上升。

闭环 pilot 随后完成并再次显示：D1 的 `white_right` 仍是 `0/15`，seen 指标反而明显下降。因此 **best data arm 选定为 D0，D1 不进入接口层。** D1 的契约有效，所以这不是 `INVALID`；它是“数据混合方向性不支持”的结果，同时仍受 C1 曝光下降和单 seed 限制。

## 2. Run Contract

| 项 | D0 | D1 |
|---|---:|---:|
| run | `star13_d0_matched_2500` | `star13_d1_matched_2500` |
| data | C1-only | C1 + 48 carrier episodes |
| loader length | 11,630 | 11,630 + 4,786 |
| loader weight | 1.0 | C1 `0.666667` / carrier `0.333333` |
| effective samples | 10,000 | 10,000（C1 约 6.67K） |
| steps / batch / accum | 2,500 / 4 / 1 | 2,500 / 4 / 1 |
| seed | 17 | 17 |
| initialization | A3 12.5K | A3 12.5K |
| config SHA256 | `953e5e16...` | `9dce4f07...` |
| final checkpoint SHA256 | `6df931ae...` | `b1236559...` |

D1 使用独立 carrier 数据集和独立 mixture，但为了避免把 C1 曝光直接减半，没有做 50:50。代价是 D1 的 10K effective samples 中只有约 `6.67K` 来自 C1，而 D0 是 `10K`；所以 D1 变差时，不能把所有差异都归因于 carrier 有害。

证据：

- [D1_contract.json](results/D1_contract.json)
- [D1_matched_postrun.json](results/D1_matched_postrun.json)
- [D1_matched_checkpoint_health.json](results/D1_matched_checkpoint_health.json)
- [D1_offline_full_contract.json](results/D1_offline_full_contract.json)

## 3. 完整离线对比

同一 evaluator、sidecar、validation episodes、seed 17、batch 2 和 78+78 pairs。

### 3.1 总体指标

| relation | arm | correct MAE | wrong MAE | margin | 95% CI | positive fraction |
|---|---|---:|---:|---:|---|---:|
| destination | D0 | 0.145168 | 0.160273 | +0.015105 | [0.007568, 0.023540] | 0.679 |
| destination | D1 | 0.151327 | 0.156956 | +0.005628 | [0.000871, 0.010731] | 0.577 |
| object | D0 | 0.146349 | 0.160694 | +0.014345 | [0.006304, 0.021798] | 0.615 |
| object | D1 | 0.149449 | 0.161066 | +0.011617 | [0.004288, 0.019328] | 0.628 |

D1 的 destination CI 仍大于 0，但下界只剩 `0.000871`，且 correct MAE 变差；这不是可写成“data intervention works”的正信号。

### 3.2 动作维度

| relation | arm | overall amp | translation amp | rotation amp | gripper amp | translation MAE | rotation MAE | gripper MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| destination | D0 | 0.987 | 1.058 | 0.863 | 0.912 | 0.147651 | 0.151361 | 0.119135 |
| destination | D1 | 0.970 | 1.046 | 0.848 | 0.902 | 0.152296 | 0.157647 | 0.129460 |
| object | D0 | 1.006 | 1.075 | 0.927 | 0.860 | 0.160148 | 0.138002 | 0.129992 |
| object | D1 | 0.999 | 1.053 | 0.936 | 0.888 | 0.156627 | 0.142705 | 0.148148 |

没有 NaN、爆炸或 gripper 完全坍缩；但 D1 在 destination 的 translation / rotation / gripper 三个子误差上都更差，object 的 gripper 误差也更差。离线结果更像“动作保真度轻微退化”，而不是“语言条件更会控制动作”。

### 3.3 分语言

destination:

| prompt | D0 margin | D1 margin | D0 CI | D1 CI |
|---|---:|---:|---|---|
| `red_left` | +0.016527 | +0.008645 | [0.007045, 0.026109] | [0.001601, 0.015672] |
| `red_right` | +0.013754 | +0.002763 | [0.004413, 0.024831] | [-0.002102, 0.008459] |

object:

| prompt | D0 margin | D1 margin | D0 CI | D1 CI |
|---|---:|---:|---|---|
| `red_left` | +0.019468 | +0.014247 | [0.009141, 0.031295] | [0.004240, 0.027325] |
| `white_left` | +0.009479 | +0.009119 | [-0.001682, 0.020291] | [-0.001512, 0.020754] |

`red_right` 的 D1 destination CI 跨 0，说明 D1 在两个 seen destination 中的表现也不一致。

## 4. Offline 能说明什么，不能说明什么

能说明：

- D1 evaluator 正确加载了 C1+carrier mixture metadata，反归一化路径与训练一致；
- D1 checkpoint、config、sidecar 和 pair 数量满足契约；
- D1 没有引入 NaN、动作爆炸或评测路径错误。

不能说明：

- 不能说明 D1 改善了 `white_right`；held-out prompt 没有 expert action label；
- 不能说明 data coverage 是瓶颈，因为离线 margin 反而变弱；
- 不能说明 carrier 有害，因为 D1 的 C1 曝光只有 6.67K，且单 seed；
- 不能替代闭环 strict destination 和 wrong-destination。

## 5. C1 Pilot

D1 的 15-init × 4-prompt pilot 使用与 D0 相同的冻结 manifest、seed 7、evaluator 和 GPU1，契约 `PASS`：

```text
contract_status = PASS
num_init_groups = 15
num_episodes    = 60
```

### 5.1 总体对比

| 指标 | D0 | D1 | 变化 |
|---|---:|---:|---:|
| seen episode successes | 20 / 45 | 11 / 45 | -9 |
| seen group-any | 13 / 15 | 9 / 15 | -4 |
| seen group-all | 1 / 15 | 0 / 15 | -1 |
| `white_right` official success | 0 / 15 | 0 / 15 | 0 |
| `white_right` strict destination | 0 / 15 | 0 / 15 | 0 |

D1 没有改变 held-out endpoint，却明显破坏了 seen 多 prompt 一致性。因此没有证据支持“只差数据覆盖”。

### 5.2 分 prompt

| prompt | D0 success | D1 success | D0 strict dest | D1 strict dest | D0 no-destination | D1 no-destination |
|---|---:|---:|---:|---:|---:|---:|
| `red_left` | 6 / 15 | 3 / 15 | 10 / 15 | 3 / 15 | 1 / 15 | 8 / 15 |
| `red_right` | 6 / 15 | 3 / 15 | 10 / 15 | 8 / 15 | 2 / 15 | 3 / 15 |
| `white_left` | 8 / 15 | 5 / 15 | 11 / 15 | 6 / 15 | 2 / 15 | 8 / 15 |
| `white_right` | 0 / 15 | 0 / 15 | 0 / 15 | 0 / 15 | 8 / 15 | 9 / 15 |

D1 的主要退化不是 wrong destination 增加，而是更多 episode 根本没有把物体送到任何盘子。

### 5.3 失败结构

| failure type | D0 | D1 |
|---|---:|---:|
| `target_not_reached` | 16 | 30 |
| `wrong_receptacle_or_relation` | 12 | 9 |
| `predicate_failure` | 10 | 6 |
| `grasp_failure` | 2 | 3 |
| `wrong_object` | 0 | 1 |
| success | 20 | 11 |

`white_right` 内部仍然是 destination / release 类型失败：

| `white_right` 项 | D0 | D1 |
|---|---:|---:|
| target object correct | 13 / 15 | 14 / 15 |
| wrong destination | 14 / 14 | 14 / 14 |
| first destination `plate_1` | 7 | 5 |
| first destination `null` | 8 | 9 |
| `target_not_reached` | 8 | 9 |
| `wrong_receptacle_or_relation` | 6 | 5 |

结论：D1 抓对白杯的比例略升，但送对右盘仍然为零；这不是“看得更准”转化为“做得更准”。

### 5.4 归因边界

D1 变差有两个同时存在的解释：

1. carrier mixture 本身可能扰动动作分布；
2. D1 的 C1 effective samples 从 10K 降到约 6.67K，且只有一个 seed。

因此不能把 seen 退化严格归因于某一条 carrier 数据，也不能把 D1 的负结果扩展成“所有因子数据方法都无效”。但就本实验的目的而言，D1 没有给出值得继续加码的方向性信号，不能用它作为 I 层底座。

## 6. 证据文件

- [D1_offline_full.json](results/D1_offline_full.json)
- [D1_offline_full_contract.json](results/D1_offline_full_contract.json)
- [D1_c1_pilot_summary.json](results/D1_c1_pilot_summary.json)
- [D1_c1_pilot_full_summary.json](results/D1_c1_pilot_full_summary.json)
- [D1_c1_pilot_contract.json](results/D1_c1_pilot_contract.json)
- D0 对照：[star13_d0_matched_2500_offline_full.json](results/star13_d0_matched_2500_offline_full.json)
- D0 pilot 全量对照：[D0_c1_pilot_full_summary.json](results/D0_c1_pilot_full_summary.json)

远端原始日志不提交：

```text
$G13/evals/star13_d1_matched_2500_offline_full.log
$G13/logs/star13_d1_matched_2500.log
$G13/c1/star13_d1_matched_2500/
```

## 7. Gate 判定

| 检查 | 状态 |
|---|---|
| D1 2.5K postrun PASS | PASS |
| D1 offline 118-group contract PASS | PASS |
| D1 offline 方向性优于 D0 | **NO / unclear** |
| D1 15-init pilot contract | PASS |
| D1 `white_right` 方向性增益 | **NO：0/15，与 D0 相同** |
| D1 seen group-any 不退化 | **NO：9/15 vs 13/15** |
| best data arm | **D0** |
| D2 可选臂 | **SKIP：D1 已负向，继续加 carrier 的信息收益低** |
| 以 D0 进入 I 层 | **YES** |

裁决：D1 契约有效但方法方向不支持，保留为负结果；D0 作为唯一 best data arm 进入 `I0/I1/I2/I3`。不得把 D1 的退化单独归因于 carrier，也不得因 D1 失败直接跳到 A2。
