# P1-C 3-seed robustness 结果

**日期**：2026-11-07<br>
**裁决**：`H-DISTILL-NOT-SUPPORTED`。`R2-ROUTE` 的负增益跨 3 个 seed 稳定，P1-D 继续关闭。<br>
**边界**：只改训练 seed；数据、2000 steps、batch、lr、config、Gate 和 evaluator 均未改变。

## 1. 结果

| arm | cosine mean ± SD | sign mean ± SD | magnitude mean ± SD | MAE mean ± SD |
|---|---:|---:|---:|---:|
| `R1-CAP` | `0.7554 ± 0.0132` | `0.8331 ± 0.0143` | `0.6979 ± 0.0131` | `0.01684 ± 0.00048` |
| `R2-ROUTE` | `0.6908 ± 0.0149` | `0.7912 ± 0.0111` | `0.6953 ± 0.0236` | `0.02048 ± 0.00017` |
| `R3-GEO` | `0.8077 ± 0.0324` | `0.8699 ± 0.0364` | `0.6846 ± 0.0365` | `0.01689 ± 0.00143` |

逐 seed 配对差：

| seed | `Δcos (R2-R1)` | `Δsign (R2-R1)` | `MAE (R2-R1)` |
|---:|---:|---:|---:|
| 17 | `-0.0378` | `-0.0407` | `+0.00308` |
| 27 | `-0.0742` | `-0.0467` | `+0.00361` |
| 37 | `-0.0820` | `-0.0383` | `+0.00424` |

三个 seed 全部满足：

- `R2-ROUTE` cosine 低于 `R1-CAP`；
- `R2-ROUTE` sign 低于 `R1-CAP`；
- `R2-ROUTE` MAE 高于 `R1-CAP`。

符合预注册判据：`R2-ROUTE` 在 `3/3` 个 seed 上方向更差，因此不是单 seed 噪声。

## 2. 几何诊断

`R3-GEO` 在 3/3 个 seed 上 cosine/sign 高于 `R1-CAP`，但：

- magnitude 平均仍只有 `0.6846`；
- seed 间 cosine SD `0.0324`，比两个 baseline 都大；
- MAE 只在 2/3 个 seed 略优于 `R1-CAP`，seed 37 反而更差。

因此 6D route geometry 提供了可复用的方向信息，但没有形成稳定、可部署的完整修正映射；按 P1 协议它仍只是诊断臂，不能替代 P1-D 授权。

## 3. 裁决

1. `H-DISTILL` 的“predicted route evidence 带来增益”不成立：`R2-ROUTE` 稳定劣于参数匹配的 `R1-CAP`。
2. 不因为 `R3-GEO` 的方向指标更好就启动 P1-D；magnitude 与 seed 稳定性仍未过 Gate。
3. 不进入闭环、不扩 30/50、不升级 `D/I/A/F0` 或任何论文近似组件。
4. 下一步应停止 P1 residual 容量路线，按 66 的失败分支处理；若要继续几何方向，必须另写预注册协议，明确 magnitude 校准、seed 数和闭环权限。

## 4. 归档

本地 summary：

```text
experiments/Goal15-p1-lateral-distill-20261107/p1c_v4_summaries/
experiments/Goal15-p1-lateral-distill-20261107/p1c_seed27_summaries/
experiments/Goal15-p1-lateral-distill-20261107/p1c_seed37_summaries/
```

远端：

```text
$W/results/K4_lite/T4b/p1_lateral_distill_20261107/p1c_pilot_v4/
$W/results/K4_lite/T4b/p1_lateral_distill_20261107/p1c_seed27/
$W/results/K4_lite/T4b/p1_lateral_distill_20261107/p1c_seed37/
```
