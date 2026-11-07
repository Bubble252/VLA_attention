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

因此 D1 不能凭离线指标进入 best data arm。它是否改变 held-out `white_right`，只能由同口径 15-init 闭环 pilot 判定；在 pilot 完成前，D1 Gate 状态是 `NOT_EVALUATED`，不是 PASS 或 FAIL。

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

D1 的 15-init × 4-prompt pilot 使用与 D0 相同的冻结 manifest、seed 7、evaluator 和 GPU1。结果写入本节后，才做 best data arm 裁决。

待记录：

- seen episode successes；
- seen group-any / group-all；
- `white_right` official success；
- `white_right` strict destination；
- first destination 分布；
- failure type 分布；
- 与 D0 的逐项差值。

## 6. 证据文件

- [D1_offline_full.json](results/D1_offline_full.json)
- [D1_offline_full_contract.json](results/D1_offline_full_contract.json)
- D0 对照：[star13_d0_matched_2500_offline_full.json](results/star13_d0_matched_2500_offline_full.json)

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
| D1 15-init pilot contract | PENDING |
| D1 `white_right` 方向性增益 | PENDING |
| 进入 I 层 | **NO** |

当前只能继续完成 pilot；不得因为 offline contract PASS 进入接口层，也不得把离线退化单独解释成 D1 方法失败。
