# P1-C 3-seed robustness check

**日期**：2026-11-07<br>
**状态**：预注册；不是新的方法 Gate，不改变 P1-C v4 的 FAIL 裁决

## 1. 目的

P1-C v4（seed `17`）显示 `R2-ROUTE` 在方向、符号和 MAE 上均不优于 `R1-CAP`。为避免把单个 seed 的训练噪声误判成 `H-DISTILL` 否证，本检查只回答：

> `R2-ROUTE` 相对 `R1-CAP` 的负增益是否跨 seed 稳定？

## 2. 固定与变量

固定：

- commit `4763365` 的模型、数据、优化器、`configs/vla/p1_lateral_distill_v3.json`；
- 2000 steps、batch `64`、lr `3e-4`；
- P1-B train/validation 标签、`release_target=teacher_release_active`；
- correction gate、release gate、z/rotation 约束和全部指标定义；
- 不跑闭环、不做 validation 调参、不做 post-hoc magnitude calibration。

变量：训练 seed `17 / 27 / 37`。每个 seed 训练 `R1-CAP`、`R2-ROUTE`、`R3-GEO` 三个 arm。

## 3. 预注册判读

主比较是每个 seed 内的配对差：

```text
Δcosine = R2-ROUTE - R1-CAP
Δsign   = R2-ROUTE - R1-CAP
ΔMAE    = R1-CAP - R2-ROUTE
```

判读规则：

- 若 `R2-ROUTE` 在至少 `2/3` 个 seed 上 cosine 和 sign 都低于 `R1-CAP`，且 MAE 更高，则 `H-DISTILL` 的“route evidence 带来增益”不成立；
- 若结果在 seed 间方向不一致，则 P1-C 只能记为“单 seed FAIL、seed 不稳定”，不能升级为方向性否证；
- 无论结果如何，都不能据此启动 P1-D；P1-D 仍只由原协议 Gate 授权。

## 4. 输出

```text
$W/results/K4_lite/T4b/p1_lateral_distill_20261107/p1c_seed27/
$W/results/K4_lite/T4b/p1_lateral_distill_20261107/p1c_seed37/
```

seed `17` 复用 `p1c_pilot_v4`，不重训、不覆盖。
