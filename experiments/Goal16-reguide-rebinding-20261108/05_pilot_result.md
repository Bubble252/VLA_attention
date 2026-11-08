# Goal16 结果：ReGuide-style referent rebinding pilot

**日期**：2026-11-08<br>
**状态**：`INVALID`（Run Contract PASS；R-PRED grounding 未通过预注册阈值）<br>
**完整结果**：远端 `$W/results/K4_lite/T4b/goal16_reguide_rebinding_20261108/pilot_15init/audit_goal16_reguide_rebinding.json`<br>
**本地摘要**：`pilot_summary.json`、`run_contract.json`、`grounding_summary.json`

## 1. 一句话结论

**动作底座不是当前主要瓶颈。** 冻结策略在 15 个 `white_right` init 上仍然是 `0/15`，但只要把物体送到正确的 `plate_2`，`R-GEO` 能拿到 `2/15`、`R-NAIVE` 能拿到 `4/15`；真正的 deployable 瓶颈是 grounding：point head 对 held-out 的 `white_right` 15/15 都预测到了错误的 `plate_1`。

因此本实验的正式状态是 `INVALID`，不是 `FAIL`：`R-GEO` 的机制信号是正的，但 `R-PRED` 没有满足 Run Contract，G2 不能判过。

## 2. Run Contract

| 项 | 实际值 |
|---|---|
| contract status | `PASS`（实验设置完整） |
| manifest SHA256 | `e9564ca9...f45e` |
| checkpoint fingerprint | `747a2869...8e20` |
| statistics SHA256 | `6d5ebb47...9538` |
| config SHA256 | `34731e83...e502` |
| grounding sidecar SHA256 | `7fb06ba0...e43a` |
| point head SHA256 | `693f0dd8...fce7` |
| init | 15；3 calibration + 12 confirmation |
| seed | 7 |
| GPU | 只使用 GPU1 |
| training | 无 backward、无 optimizer、无新 checkpoint |
| split | `white_right` train = 0；P0 eval 与 train/validation 零重叠 |

## 3. 主表

| arm | total success | calibration | confirmation | released success | hold-only | target dest correct | first wrong dest | transport | hand-back | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0` | 0/15 | 0/3 | 0/12 | 0/15 | 0/15 | 0/15 | 14/15 | 0/15 | 0/15 | 0 |
| `R-GEO` | 2/15 | 1/3 | 1/12 | 2/15 | 0/15 | 10/15 | 3/15 | 15/15 | 14/15 | 0 |
| `R-NAIVE` | 4/15 | 0/3 | 4/12 | 1/15 | 3/15 | 10/15 | 3/15 | 15/15 | 0/15 | 0 |
| `R-PRED` | 0/15 | 0/3 | 0/12 | 0/15 | 0/15 | 0/15 | 14/15 | 15/15 | 14/15 | 0 |
| `R-SHUF` | 0/15 | 0/3 | 0/12 | 0/15 | 0/15 | 0/15 | 14/15 | 15/15 | 14/15 | 0 |

稳定性：五个 arm 均 `0 error`、`0 NaN`、`0 object_flew_out`，`object_retained = 15/15`。

## 4. Gate

### G1：机械有效性

| check | 结果 |
|---|---|
| 五个 arm 各 15 行 | PASS |
| `errors=0`、`nan_count=0`、`object_flew_out=0` | PASS |
| `R-GEO` transport + hand-back `>=12/15` | PASS（15/15 + 14/15） |
| `R-SHUF` 不优于 `B0` | PASS（0 vs 0） |
| 动作幅度有记录 | PASS |

**G1 通过。**

### G2：pilot 效果

| check | 结果 |
|---|---|
| `R-GEO` success `>=1/15` | PASS（2/15） |
| `R-GEO` released success `>=1/15` | PASS（2/15） |
| `R-GEO` wrong-destination 不高于 `B0` | PASS（3 vs 14） |
| `R-PRED` grounding valid | **FAIL（INVALID）** |
| `R-PRED` success `>=1/15` | FAIL（0/15） |
| `R-PRED - B0 >= +1/15` | FAIL（0） |

**G2 未通过，正式状态 `INVALID`。** 原因是 R-PRED 的 grounding 预测不满足预注册阈值，而不是 R-GEO 没有机械信号。

## 5. Grounding 结果

point head 对 15 个 held-out `white_right` init 全部预测到 `plate_1`：

| 指标 | 值 |
|---|---:|
| grounding miss | 15/15 |
| mean destination error | 0.603 m |
| max destination error | 0.623 m |
| 阈值 | 0.150 m |

这不是相机标定或反投影 bug：oracle `plate_2` 的投影点与已知 mask 质心一致；预测点本身落在 `plate_1`。point head 的训练任务只有：

- `red_left`
- `red_right`
- `white_left`

没有 `white_right`。所以它把“white”绑定到了左盘，而不是组合“white + right”。这正是 held-out 组合泛化失败，也是当前 deployable 接口的真实瓶颈。

## 6. 机制解释

### 6.1 支持的部分

- `B0` 复现 `0/15`，`14/15` first destination 是 `plate_1`；
- `R-GEO` 把正确目的地送进可达区域：`target_destination_correct = 10/15`，`2/15` official success；
- `R-SHUF` 去 `plate_1` 仍然 `0/15`，说明“送到正确盘子”不是任意 transport 都能产生成功。

因此 **H-BIND 得到部分支持**：冻结策略的失败至少有一部分来自 destination referent binding，而不只是“不会操作”。

### 6.2 不支持的部分

- `R-NAIVE`（只改 xy，不 hand-back/re-arm）拿到 `4/15`，高于 `R-GEO` 的 `2/15`；
- `R-GEO` 的 3D approach + hand-back + re-arm 没有带来额外收益，反而可能干扰 policy 的后续 release/settle；
- `R-NAIVE` 的 `4/15` 中有 `3/15` 是 hold-only success，说明“到达正确位置”已经发生，缺的是可靠的 release/settle。

因此 **H-GEOM 没有得到支持**：在当前 15-init 上，full 3D wrapper 不如简化的 xy transport。

### 6.3 Deployable 结论

`R-PRED` 完全失败，且失败点在 grounding，不在动作执行：

> 当前 point head 无法把“white”和“right”组合到 held-out 的 `white_right`。即使动作 wrapper 能成功，deployable 版本也没有可用的目的地预测。

## 7. 裁决

按 Goal16 预注册的失败分支：

| 观察 | 本实验 | 裁决 |
|---|---|---|
| `R-GEO = 0/15` | 否（2/15） | 不停止 wrapper 主线 |
| `R-GEO` 正、`R-PRED` 负 | 是 | 先修 grounding，不训练新动作头 |
| `R-GEO ≈ R-NAIVE` | 否，`R-NAIVE` 更好 | 不能把增益写成 3D hand-back；应简化 wrapper |
| `R-SHUF ≈ R-GEO` | 否（0 vs 2） | 语义负控成立，结果不是 INVALID 语义负控 |

**最终裁决：**

1. 保留 `R-GEO` 作为“存在 headroom”的机械上界证据；
2. 不扩展当前 full 3D wrapper；
3. 把 `R-NAIVE`（xy-only transport）作为下一版更简单的执行基线；
4. 下一步的主攻方向是 **compositional grounding**，不是继续调 hand-back。

## 8. 下一步

### P0：修 grounding（最高优先）

当前 point head 的训练任务缺少 `white_right`，因此不能组合属性。下一版必须做以下之一：

- 用 counterfactual 四宫格（`red_left`、`red_right`、`white_left`、`white_right`）训练一个 **组合泛化** 的 grounding head，并在 held-out init 上测；
- 或者用 VLM 做 open-vocabulary target/receptacle 选择，再用几何反投影；
- 或者显式解析 instruction 的 `object` 与 `spatial relation`，用检测/分割模型定位，而不是让一个小 head 从 3 个训练 prompt 里猜。

验收标准：15 个 `white_right` init 上 destination error `<=0.15 m`，且不能使用 oracle fallback。

### P1：简化执行 wrapper

- 以 `R-NAIVE` 为基线：xy transport + explicit release/settle；
- 不再默认使用 3D approach + hand-back + re-arm；
- 目标是把 `R-NAIVE` 的 `3/15` hold-only success 转成 released success；
- 如果简化版仍低于 `R-GEO`，再单独消融 z/approach。

### P2：重新跑 R-PRED

只有在 P0 的 grounding 通过 `<=0.15 m` 后，才把 predicted destination 接回简化 wrapper，重跑 15-init。届时 G2 才有资格判 `PASS/FAIL`。

## 9. 不要做的事

- 不要把 `R-GEO` 的 `2/15` 写成“ReGuide wrapper 已经有效”；
- 不要把 `R-NAIVE` 的 `4/15` 写成“语义 rebinding 成功”，它只是 oracle 目的地 + xy transport；
- 不要用 oracle fallback 救 `R-PRED`；
- 不要在 grounding 未通过前扩 30/50 或 3 seeds；
- 不要继续给 full 3D hand-back 加模块。

## 10. 产物

- `pilot_summary.json`：完整逐 episode 结果；
- `run_contract.json`：实验设置与 SHA 校验；
- `grounding_summary.json`：point head 15-init grounding 结果；
- `predicted_points.jsonl`：R-PRED 使用的 15 个预测目的地；
- 远端 traces：`$W/results/K4_lite/T4b/goal16_reguide_rebinding_20261108/pilot_15init/traces/`。
