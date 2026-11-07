# P1-C offline student 结果

**日期**：2026-11-07<br>
**裁决**：`P1_C_FAIL`。`P1-D` 不启动，不跑 15-init 闭环，不升级论文近似组件。<br>
**边界**：这是 offline teacher-label 蒸馏的失败裁决，不是 residual、route 或整个 VLA 方向的最终否证。

## 1. 问题与运行契约

P1-C 要回答的问题是：

> 在冻结的 P1-B A10 标签上，predicted route evidence 是否能把单步 xy 修正和 release 学到可部署的程度，并相对参数匹配的 `R1-CAP` 产生增益？

本次正式裁决使用 commit `4763365` 的实现和 `configs/vla/p1_lateral_distill_v3.json`：

| 字段 | 值 |
|---|---|
| `code_commit` | `4763365` |
| `config_sha256` | `8a43bafd5c950b213715593f7a66ddcf60a77f59f96d266d30e41c65d1d1f4e9` |
| `train_script_sha256` | `4ca18bcbb7a0731f39aa37959841cb0dba0f1847cc596a5de5c3ce4c7e4768dd` |
| `student_module_sha256` | `533f4fda8f9e65006afb7d5636144a22673d468ac807a76298748eb692000e1b` |
| `train_transitions / episodes` | `11,630 / 96` |
| `validation_transitions / episodes` | `1,824 / 15` |
| `white_right` train/validation | `0 / 0` |
| release positives train/validation | `1,442 / 240` |
| `R1-CAP` / `R2-ROUTE` parameters | `741,905 / 741,905` |
| `init_identity_max` | `0.0` |
| z / rotation | 无输出路径，严格为 `0` |

运行前修复了两个实现错误：

1. `evaluate()` 没有返回 `correction_prob`；
2. `evaluate()` 引用了未定义的 `config`，且在构造 correction 指标时漏传 `correction_prob`。

修复后 `tests/test_p1_lateral_distill.py` 通过；旧 v1–v3 结果只保留为调试历史，不参与本次裁决。

## 2. 2000-step pilot 结果

| arm | cosine | dominant sign | magnitude median | all MAE | nonzero MAE | release P/R | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| `R1-CAP` | `0.7415` | `0.8359` | `0.7140` | `0.01749` | `0.03693` | `0.8134 / 0.9083` | FAIL：large bucket |
| `R2-ROUTE` | `0.7036` | `0.7952` | `0.6734` | `0.02057` | `0.04326` | `0.8707 / 0.8417` | FAIL：magnitude ratio |
| `R3-GEO` | `0.8455` | `0.9114` | `0.6946` | `0.01597` | `0.03317` | `0.8706 / 0.9250` | FAIL：magnitude、large bucket |

分桶结果：

| arm | plateau count / MAE | large count / MAE | correction gate P/R | pred zero fraction |
|---|---:|---:|---:|---:|
| `R1-CAP` | `6 / 0.02276` | `41 / 0.05045` | `0.9595 / 0.9078` | `0.5669` |
| `R2-ROUTE` | `6 / 0.03059` | `41 / 0.04915` | `0.9274 / 0.8419` | `0.5844` |
| `R3-GEO` | `6 / 0.02337` | `41 / 0.04222` | `0.9493 / 0.9198` | `0.5565` |

## 3. 机制判读

### 3.1 `R2-ROUTE` 没有相对 `R1-CAP` 产生增益

`R1-CAP` 只使用 base action/proprio，不使用 route evidence；它在 validation 上的 cosine、sign、MAE 和 magnitude 都优于 `R2-ROUTE`。因此本实验不能支持“predicted route evidence 改善 residual 映射”。

这不是“只差一个阈值”：

- `R1-CAP` cosine `0.7415` > `R2-ROUTE` `0.7036`；
- `R1-CAP` sign `0.8359` > `R2-ROUTE` `0.7952`；
- `R1-CAP` nonzero MAE `0.03693` < `R2-ROUTE` `0.04326`；
- `R2-ROUTE` 唯一失败项是 magnitude `0.6734`，但即使把 magnitude 校准到通过，方向指标仍低于 `R1-CAP`。

### 3.2 `R3-GEO` 说明几何方向有信息，但还不足以成为部署方法

`R3-GEO` 的 cosine/sign 明显优于两个 raw-route arm，说明 6D route geometry 比 4096D route token 更接近教师决策方向。但它同时失败 magnitude 和 large bucket，且按协议它只是 oracle/显式几何诊断，不是部署 arm。因此不能据此启动 P1-D。

### 3.3 失败主要在“证据没有转化成增益”，不是 release 学不会

三个 arm 的 release precision/recall 均过线；correction gate 的 precision/recall 也较高。失败集中在连续修正的方向增益和 magnitude，而不是 release 分类。这说明 A10 标签并非完全不可学，但当前 raw route evidence 没有成为教师修正的有效控制变量。

## 4. 证据边界

- 单 seed、2000 steps；不能说“residual 方法在所有 seed 上都失败”。
- P1-B 标签来自 HDF5 expert replay，不是 closed-loop base-policy teacher；`state_source=hdf5_expert_replay`。
- validation 的 plateau 只有 6 条、large 只有 41 条，分桶指标方差较大。
- 本次没有使用 validation 做任何 post-hoc magnitude calibration；所有 Gate 数字都是原始预测。
- 本次没有运行 P1-D 闭环、没有 15-init 结果、没有修改 J2 base policy。

## 5. 裁决

按 P1 协议：

1. `R2-ROUTE` 失败 offline Gate，停止 P1-D；
2. 不扩 30/50、不跑 15-init、不写 residual/authority 方法成功；
3. 不因为 P1-C 失败而自动升级 CompoSlot、AC-VLA phase、CRAFT 或 `F0`；
4. 下一项工作必须是新的、预注册的离线诊断，不得用“多训练一点”或 validation 调参把本次 FAIL 改写成 PASS。

最有信息量的后续离线诊断是：固定 `R1-CAP/R2-ROUTE/R3-GEO`，在 train-only 上拟合一个全局 magnitude calibration，并在独立 validation 上报告 calibration 前后指标，同时跑 3 seeds。该诊断只能回答“失败是否只是尺度校准”，不能替代本 Gate，也不能直接授权闭环。

## 6. 复现

远端输出：

```text
$W/results/K4_lite/T4b/p1_lateral_distill_20261107/p1c_pilot_v4/
```

运行命令：

```bash
$R/server/run_p1c_lateral_distill.sh p1c_pilot_v4 R2-ROUTE 2000
```

本地归档 summary：

```text
experiments/Goal15-p1-lateral-distill-20261107/p1c_v4_summaries/
```

summary SHA256：

| arm | SHA256 |
|---|---|
| `R1-CAP` | `23a765d81b9bf964da1734086a4142b133afd9f6d0797230c7058a1dbe0056cd` |
| `R2-ROUTE` | `8020f9760cdd2062518ba00250e359facc715944d7f984826031c3add492e04a` |
| `R3-GEO` | `82c1f04011892c27384863ff6c05efd6315cc32ac04f276247b664db5db0be4c` |
