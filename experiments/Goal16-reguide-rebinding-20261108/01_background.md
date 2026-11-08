# Goal16：ReGuide-style referent rebinding pilot

**日期**：2026-11-08<br>
**状态**：`PROTOCOL_FROZEN_BEFORE_MEASUREMENT`<br>
**上游**：[67](../../doc/vla_next/67_dependency_repair_and_reguide_pivot_20261108.md)、[65](../../doc/vla_next/65_story_method_selection_20261107.md)、[66](../../doc/vla_next/66_outer_loop_pivot_after_interface_failure_20261107.md)、[P1-C](../Goal15-p1-lateral-distill-20261107/07_p1c_result.md)

## 1. 为什么做这个实验

P1-C 已经在 3 个 seed 上稳定否证“predicted route evidence 能改善 residual 映射”：

| arm | cosine mean | sign mean | MAE mean |
|---|---:|---:|---:|
| `R1-CAP` | `0.7554` | `0.8331` | `0.01684` |
| `R2-ROUTE` | `0.6908` | `0.7912` | `0.02048` |
| `R3-GEO` | `0.8077` | `0.8699` | `0.01689` |

因此本实验不再训练 residual、route 或新的动作头。它只回答一个更前置的问题：

> 当前 C1 `white_right` 的 `0/15`，是否主要是因为冻结策略把执行绑定到了错误目的地，而不是因为它不能执行局部操作？

## 2. 任务与已知失败模式

- suite：`LIVING_ROOM_SCENE5`；
- instruction：`put the white mug on the right plate`；
- target object：`porcelain_mug_1`；
- correct destination：`plate_2`；
- distractor destination：`plate_1`；
- 15 个冻结 init：3 个 calibration + 12 个 confirmation；
- seed：`7`。

Goal14 的 `B0` 在 15 个 init 上 `0/15`；其中 14 个 init 的 first destination 是 `plate_1`，另一个没有可用 destination 标签。由此得到的机制假设是：

1. 冻结策略能够抓取 `porcelain_mug_1`；
2. 它把“放”这个动作绑定到了训练中更常见的 `plate_1`；
3. 只在抓取后加一点横向 residual，不能稳定恢复正确的 referent。

## 3. 本实验要区分的四个层级

| 层级 | 问题 | 本实验中的 arm |
|---|---|---|
| 无干预 | 冻结策略是否仍为 `0/15` | `B0` |
| oracle 语义 + 几何 | 如果目的地指对，并送到可用 pre-place 配置，能否成功 | `R-GEO` |
| oracle 目的地的 naive 执行 | 成功是否只需要“把物体推到正确 xy” | `R-NAIVE` |
| 可部署 grounding | 现有 point head 能否替代 oracle 目的地 | `R-PRED` |
| 语义负控 | 如果故意去错盘子，是否仍会失败 | `R-SHUF` |

## 4. 研究假设

### H-BIND

如果 `R-GEO` 相对 `B0` 产生官方成功和 released success，而 `R-SHUF` 不产生成功，则支持：

> 失败至少部分来自 destination referent binding，而不是只有动作底座能力不足。

### H-GEOM

如果 `R-GEO` 明显优于 `R-NAIVE`，则支持：

> 只把物体送到正确 xy 不够，demonstration-supported pre-place / hand-back 也有作用。

### H-PRED

如果 `R-GEO` 有效而 `R-PRED` 无效，则支持：

> 执行链有 headroom，但当前外部 grounding predictor 是瓶颈。

如果 `R-GEO` 也 `0/15`，则不支持继续扩展 wrapper 主线；按 Goal16 的停止条件写失败裁决。

## 5. 不做什么

- 不训练新的 residual、route 或 action head；
- 不修改 15-init manifest；
- 不使用 `white_right` 训练样本；
- 不用 validation 调参；
- 不把 A10 的“抓取后连续横向 authority”改名为 `R-GEO`；
- 不把 `R-PRED` 的 oracle fallback 冒充 predicted grounding；
- 不在 G2 通过前扩 30/50 或 3 seeds。

## 6. 预期信息价值

本实验是断点定位器，不是论文级成功率结论：

- `R-GEO` 正、`R-PRED` 正：可以进入 30/50 和 3 seeds，并考虑把 grounding 换强；
- `R-GEO` 正、`R-PRED` 负：先修 grounding，不训练新动作头；
- `R-GEO` 负：停止 ReGuide wrapper 主线，转 APT 或任务重选；
- `R-GEO` 与 `R-NAIVE` 同效：增益主要来自机械 transport，不是语义重绑定；
- `R-SHUF` 也有成功：说明指标或 wrapper 逻辑没有真正绑定目的地，结果判 `INVALID`。
