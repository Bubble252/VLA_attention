# 01 背景：为什么先做 Stage 0

## 1. 研究问题

前序实验反复出现同一个现象：模型看起来已经定位到正确物体或目标区域，但闭环动作仍然把物体放到错误盘子。65 把这个问题拆成三层：

```text
Data：训练数据是否让 source / destination 因子可辨识？
Interface：证据是否有动作头可读取的显式接口？
Control：动作是否真的把证据当成控制变量？
```

本轮不直接押注完整 FAI-VLA，而是用 C1 的四格任务逐层定位断点。

## 2. 主任务 C1

C1 使用 `LIVING_ROOM_SCENE5` 的红/白杯子和左/右盘子：

| prompt_id | 语言 | 训练可见 |
|---|---|---|
| `red_left` | put the red mug on the left plate | 是 |
| `red_right` | put the red mug on the right plate | 是 |
| `white_left` | put the white mug on the left plate | 是 |
| `white_right` | put the white mug on the right plate | **否，held-out 组合** |

`white_right` 绝不进入训练、验证损失、反事实负样本或采样器。`white_right` 训练计数必须由审计脚本证明为 0。

## 3. 已有证据

前序 C1-only 基线：

- seen official：`24/45`；
- seen group-any：`12/15`；
- seen group-all：`4/15`；
- `white_right` official：`0/15`；
- `white_right` strict destination：`0/15`。

T5/J2 的 `white_right` 失败分类：

- target object correct：`14/15`；
- wrong object：`1/15`；
- wrong destination：`14/15`；
- no destination：`1/15`；
- strict destination：`0/15`。

P1/P2/J4.1 的 point 证据：

- P1 point-in-mask `0.9435`，destination L1 `1.8756 px`；
- P2 predicted `0.215845`，none `0.139848`，predicted 比 none 差 `54.34%`；
- predicted 与 shuffle 只差 `0.001123`；
- J4.1 binding gap `1.681e-4`，entropy `6.2359`。

这些结果支持一个有限但重要的判断：**定位指标变好，不自动等于动作使用证据。** 本地化不是主矛盾，destination binding 和动作控制才是。

## 4. 本轮假设

Stage 0 不检验方法收益，只检验后续实验是否可解释。必须先确认：

1. D0 的三条训练 prompt 计数和历史一致；
2. `white_right` 在训练集中严格为 0；
3. train / validation / rollout 不共享 episode；
4. carrier `yellow_and_white_right` 足够多，且没有把 held-out 组合混入训练；
5. 实际采样权重、batch、config 和 code commit 与计划一致。

只有这些成立，D0/D1 的差异才能被解释为数据层干预，而不是数据泄漏或配置漂移。

## 5. 方法阶梯

| 层 | arm | 问题 |
|---|---|---|
| 数据 | `D0` / `D1` / `D2` | 原 C1、加 carrier、再加白碗是否让因子可辨识 |
| 接口 | `I0` / `I1` / `I2` / `I3` | 无 marker、oracle marker、predicted marker、wrong marker 的差异 |
| 控制 | `A0` / `A1` / `A2` / `A3` | 普通 BC、反事实排序、充分/必要约束、去混淆 |

实验不做 D×I×A 全交叉，而是逐层取 best arm。`F0 = D1 + I2 + A2` 只有在 `A2` 通过后才允许合并；`AC-VLA-lite` 的 phase 机制只有在 `F4` 消融成立后才升级。

## 6. 不做什么

- 不追 `14/15`，主 endpoint 是 `white_right` official success 和机制指标；
- 不在 `white_right`、validation 或 rollout init state 上训练；
- 不把 GT mask/point 作为部署输入；
- 不在 Stage 0 未 PASS 时启动 D0/D1；
- 不因为一次 pilot 失败就换方向，先检查契约和数据；
- 不在没有论文/代码复现材料时实现外部方法。

## 7. Stage 0 完成标准

- [ ] 语言计数审计覆盖全部 RLDS shard；
- [ ] C1 train/validation/rollout 交集为 0；
- [ ] `white_right` 训练计数为 0；
- [ ] carrier 和 `white_bowl_right` 计数可复核；
- [ ] HDF5/sidecar 状态明确，缺失时标记 `weak_sidecar`；
- [ ] Run Contract 输出 `PASS`，否则 `INVALID`；
- [ ] 审计 JSON、契约 JSON、运行日志写入 Goal13 结果目录并 commit。
