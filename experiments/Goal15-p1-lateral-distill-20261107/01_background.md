# P1：把 P0 的横向 authority 蒸馏成可部署学生

**日期**：2026-11-07
**状态**：`PROTOCOL_FROZEN_NOT_STARTED`；先写协议，不启动训练
**上游**：[65](../../doc/vla_next/65_story_method_selection_20261107.md)、[66](../../doc/vla_next/66_outer_loop_pivot_after_interface_failure_20261107.md)、[Goal14 P0](../Goal14-p0-authority-release-20261107/05_p0_result.md)

## 1. P0 给了什么

P0 在冻结 15-init `white_right` 上通过预注册 Gate：

| arm | total success | confirmation | released success | hold-only |
|---|---:|---:|---:|---:|
| `B0` | 0/15 | 0/12 | 0/15 | 0/15 |
| `A05` | 5/15 | 4/12 | 1/15 | 4/15 |
| `A10` | 7/15 | 6/12 | 4/15 | 3/15 |
| `A20` | 7/15 | 6/12 | 4/15 | 3/15 |

结论是：横向 authority 是一个真实缺口，`A10` 能产生真实 released success；但 `A10` 只比 `A05` 多 `2/15`，`A20` 也没有更好，所以这不是“residual 方法已经成功”。

## 2. P1 要回答什么

P1 只检验 `H-DISTILL`：

> 特权教师 `A10` 生成的横向修正和 release 决策，能否被一个只使用部署时可得输入的 student 学到，并在同一个 `white_right` 闭环上保留一部分收益？

教师 `A10` 可以使用物体和盘子位置；学生不能。学生的部署输入只有：

- 图像；
- 语言；
- proprio；
- base action；
- predicted point / predicted region evidence。

GT object/plate position、GT mask、GT point 只能作为教师标签或 `R3-GEO` 诊断，不能进入部署前向。

## 3. 为什么不能直接上大模型

Goal13 的接口层已经失败：oracle marker 都没有让 action expert 产生正收益。P1 不重复“把 marker 塞进 action expert”的路线，而是：

1. 先把教师标签和 student 的输入输出冻结；
2. 先证明 student 能拟合教师的方向，不先追加接口模块；
3. 只有 offline 方向 Gate 过了，才允许闭环；
4. 闭环只看 `R2-PRED` 相对 `R1-CAP` 的增量，不把参数容量误写成 evidence 收益。

## 4. 一个必须诚实处理的缺口

P0 的 backbone 是 J2/OpenVLA-OFT；Goal13 的 point head 是 Qwen/starVLA 资产，不能直接假定它可以给 J2 提供同分布 predicted point。

因此 P1 先做 `P1-A evidence adapter qualification`：

- 首选：训练/接入一个 J2-compatible point head，并先过 point 质量 Gate；
- 备选：使用冻结 route 权重得到的 source/target centroid 作为可部署 evidence，但 arm 必须改名为 `R2-ROUTE`，不能冒充 `R2-PRED`；
- 如果两种 evidence 都不能过适配 Gate，P1 标 `INVALID`，不能判 `H-DISTILL` 失败。

不允许把旧 Qwen point head 的预测直接接进 J2，然后把它当成“predicted point 已就绪”。

## 5. P1 的 arms

| arm | 输入 | 输出 | 作用 |
|---|---|---|---|
| `R0` | base policy | 无 | 闭环基线 |
| `R1-CAP` | base action + proprio；不含图像/语言 evidence；参数量与 `R2` 对齐 | `delta_xy + release` | 排除“多参数”解释 |
| `R2-PRED` | 图像 + 语言 + proprio + base action + predicted point/region | `delta_xy + release` | 可部署候选 |
| `R3-GEO` | 同 `R2`，point 换成 oracle geometry | `delta_xy + release` | 诊断上界，不部署 |

`R0` 不训练；`R1-CAP`、`R2-PRED`、`R3-GEO` 同数据、同初始化、同 batch、同 step、同 seed。

## 6. 不做什么

- 不做 marker/cross-attention/query-token 的接口注入变体；
- 不启动 `A2/A3/F0`；
- 不把 `R2` 的增益归因于 VLM/语言，除非它同时优于 `R1-CAP`；
- 不把 `R3-GEO` 的 oracle 结果写成方法结果；
- 不把 15 条 rollout 写成论文级显著结论；
- 不在 P1-A/P1-B/P1-C Gate 通过前启动训练。
