# 12 接口层结果：I0 / I1 / I2 / I3

**状态**：接口层已完成正式判读。`I0/I2` 的 500-step continuation、`I1/I2/I3` 的完整 118 groups 均已完成；Run Contract 全部 PASS，但 `G-I1/G-I2` **FAIL**，`W-I` 不开启，`A2/F0` 冻结。<br>
**协议**：[11_i_interface_protocol.md](11_i_interface_protocol.md)<br>
**主数据臂**：`D0`（D1 已关闭，不进入接口层）<br>
**判读规则**：500-step 与 limit=2 只用于链路检查，不能作为方法 Gate。

## 1. I0 matched 500-step

| 字段 | 值 |
|---|---|
| arm | `I0` |
| init | `star13_d0_matched_2500/final_model/model.safetensors` |
| marker | `none` |
| trainable | `action_model` |
| steps | 500 |
| per-device batch | 4 |
| grad accumulation | 1 |
| seed | 17 |
| postrun | `PASS` |
| checkpoint SHA256 | `a3ee3abdfd57fe646504eb360400f56f04aa521e840bed0dd2488e9228281662` |
| config SHA256 | `6045d305f398f1ec65dfd61f53dd70dd7a9c5d7bd59cee70aed2ff32344b38d4` |

训练日志显示数据集长度为 `11630`，可训练参数约 `174.056M`，action loss 从约 `0.105` 收敛到 `0.081`，没有 NaN/Inf。checkpoint health 为 `all_finite=true`。

### 1.1 I0 offline smoke（limit=2）

| 指标 | destination pair | object pair |
|---|---:|---:|
| correct MAE | 0.075088 | 0.075088 |
| wrong MAE | 0.083545 | 0.097130 |
| margin mean | +0.008457 | +0.022041 |
| margin 95% CI | [0.002324, 0.014589] | [0.012588, 0.031494] |
| correct translation MAE | 0.126305 | 0.126305 |
| correct rotation MAE | 0.039397 | 0.039397 |
| correct gripper MAE | 0.028511 | 0.028511 |
| translation amplitude ratio | 1.380193 | 1.380193 |
| rotation amplitude ratio | 1.131786 | 1.131786 |
| gripper amplitude ratio | 1.028511 | 1.028511 |

解释边界：

- 2 个 group 只能说明 evaluator 和动作链路可用；`gate_status=NOT_EVALUATED`。
- I0 没有 marker，所以 `point_results=null` 是预期行为。
- 当前幅度比没有显示 translation/rotation/gripper 塌缩，可继续 I2。

## 2. 契约修复

第一次 I0 postrun 在 `verify_goal13_postrun.py` 崩溃：`expected_raw_transitions` 在 contract 中位于 `data.expected_raw_transitions`，verifier 却只查顶层，读到 `None`。已修为先读顶层、再读嵌套、最后使用 D0 默认值；修复后同一 checkpoint 的 postrun 为 `PASS`。该问题属于工具契约，不是训练失败。

## 3. I2 matched 500-step 与三臂 smoke

| 字段 | 值 |
|---|---|
| arm | `I2` |
| init | D0 matched checkpoint |
| marker | `predicted`，P1 point head |
| trainable | `action_model + point_evidence_projector` |
| steps | 500 |
| trainable parameters | 约 `174.486M` |
| postrun | `PASS` |
| checkpoint SHA256 | `1112960e559964e1f81639d01bfc4e403f795f0262e6051a45b3092c1c1d13e6` |
| config SHA256 | `4fca007051367ad6e916f9881974e1701ee78c5a6cd982bd8d8e78d0a68e9e3f` |

三臂使用同一 I2 checkpoint、同一 354 行 validation sidecar 与 point manifest（SHA `9a3a508d...fab068bf`）。`I1/I2/I3` 的 smoke 都是 `SMOKE_PASS`，并且都产生了 `point_results`。

### 3.1 Point ablation smoke（2 groups）

| 指标 | 数值 |
|---|---:|
| none MAE | 0.073618 |
| predicted MAE | 0.074212 |
| oracle MAE | 0.074632 |
| shuffle MAE | 0.073522 |
| `predicted_minus_none`（`none_mae - predicted_mae`，正=predicted 更好） | -0.000594，95% CI [-0.003436, 0.002247] |
| `predicted_minus_shuffle`（`shuffle_mae - predicted_mae`，正=predicted 更好） | -0.000690，95% CI [-0.001706, 0.000326] |
| `oracle_mae - none_mae`（负=oracle 更好） | -0.001014 |
| predicted point L1 | 1.943 px |
| predicted point-in-mask | 1.000 |

2-group 下没有任何 marker 语义正信号：predicted 略差于 none 和 shuffle，oracle 也没有优于 none。这个结果不能作为正式 Gate，但足以说明完整 118 groups 应优先验证 I1 的 headroom 是否存在。

### 3.2 Pair smoke（2 groups）

| arm | destination margin | 95% CI | object margin | 95% CI |
|---|---:|---|---:|---|
| I1 | +0.007479 | [0.000781, 0.014176] | +0.034932 | [0.020324, 0.049540] |
| I2 | +0.006809 | [-0.000784, 0.014402] | +0.036546 | [0.022472, 0.050619] |
| I3 | +0.007441 | [-0.000179, 0.015062] | +0.036953 | [0.023466, 0.050441] |

`I2` 与 `I3` 的 destination margin 几乎相同，且 CI 都跨 0；这与 point ablation 的负向结果一致。当前只能记为“预注册 smoke 通过、方法方向未证实”。

## 4. 完整 118 groups 结果

`I0/I1/I2/I3` 均使用 118 个 validation groups、78 个 destination pairs 和 78 个 object pairs。四份 contract 的 `contract_status` 均为 `PASS`，`failures=[]`；`gate_status=DESCRIPTIVE` 只表示描述性契约成立，不是 Gate PASS。

### 4.1 destination 主指标

non-gripper MAE 按 7 维动作的 `(7 × correct_mae - gripper_mae) / 6` 重算。这个值用于接口 Gate，不能只看总 MAE。

| arm | destination correct MAE | non-gripper MAE | destination margin | margin 95% CI |
|---|---:|---:|---:|---|
| `I0` none | `0.144959` | `0.149199` | `+0.011656` | `[0.005462, 0.018087]` |
| `I1` oracle | `0.148985` | `0.152426` | `+0.010116` | `[0.004184, 0.017882]` |
| `I2` predicted | `0.149226` | `0.152723` | `+0.010700` | `[0.004433, 0.018365]` |
| `I3` shuffle | `0.148852` | `0.152197` | `+0.010773` | `[0.004424, 0.018104]` |

解读：

- `I1` 相对 `I0` 的 non-gripper MAE 差约 `+2.16%`，不是协议要求的改善 `>=10%`，因此 `G-I1 FAIL`。
- `I2` 的 non-gripper MAE 比 `I3` 差约 `+0.35%`，destination margin 也略低于 `I3`，因此 `G-I2 FAIL`。
- 所有 arm 的 margin CI 都为正，只说明语言配对本身有信号；这不是 marker 有效的证据，因为 `I0` 和 `I3` 也有同样方向。

### 4.2 point ablation 与 point 质量

同一份 `I2` checkpoint 上切换 none/oracle/predicted/shuffle marker，结果如下：

| marker | MAE | MAE 差：marker - none | MAE 差：marker - shuffle |
|---|---:|---:|---:|
| none | `0.148574` | — | — |
| oracle | `0.148958` | `+0.000384`（更差） | — |
| predicted | `0.149074` | `+0.000500`（更差） | `+0.000237`（更差） |
| shuffle | `0.148837` | — | — |

JSON 里的 `predicted_minus_none` 和 `predicted_minus_shuffle` 实际分别计算 `none_mae - predicted_mae`、`shuffle_mae - predicted_mae`，所以正数才表示 predicted 更好；本次两个值都是负数，表示 predicted 的 MAE 反而更高。对应 95% CI 为 `[-0.001544, 0.000681]` 和 `[-0.000603, 0.000157]`，都没有正收益信号。

point head 本身并不差：

| 指标 | predicted | shuffle | oracle |
|---|---:|---:|---:|
| destination L1 | `2.645 px` | `2.654 px` | `0` |
| object L1 | `7.807 px` | `14.440 px` | `0` |
| point-in-mask | `0.8008` | `0.7076` | `1.0` |

因此当前主要问题不是“point head 预测不准”，而是 **action expert 没有把 marker 变成动作控制**。完美 oracle marker 也没有优于 none，进一步排除了“只修 point head 就够”的解释。

## 5. 正式裁决与下一步

### 5.1 Gate 裁决

| Gate | 条件 | 结果 | 裁决 |
|---|---|---|---|
| `G-I1` | `I1` non-gripper MAE 相对 `I0` 改善 `>=10%` | 实际差约 `+2.16%` | **FAIL** |
| `G-I2` | `I2 > I3` | 实际 `I2` 略差于 `I3` | **FAIL** |
| `G-I3` | shuffle 不优于 predicted；none 不优于 predicted | 负控成立，predicted 没有优势 | **FAIL（无正信号）** |
| 幅度检查 | translation/rotation/gripper 不塌缩 | `I0/I1/I2/I3` 均未出现冻结 | PASS（仅链路） |

结论：接口层 Gate 失败，`W-I` 不开启；不能进入 `A2/A3/F0`，也不能把 `AC-VLA-lite`、`CompoSlot` 或 factor loss 升级成论文近似版。

### 5.2 唯一允许的下一步

按 `65` 的停止规则，先做一次 **marker 注入修复**，不增加新 loss、不换数据、不延长训练。

建议新 arm：`I2-XATTN`。

- parent：`I2` 的同一 D0 数据和初始化；
- 训练预算：500 steps、batch 4、seed 17，与 `I2` matched；
- 唯一变量：把当前 common-mode evidence addition 改为 `action_model.evidence_cross_attention: true`；
- 先跑 `I1/I2/I3` limit=2 smoke，再跑完整 118 groups；
- 通过条件仍是 `I1 > I0` 且 `I2 > I3`；若仍失败，只允许再做一次 `evidence_token`/append 类注入修复，随后停止这条接口路线。

在注入修复通过前，不启动 A2，不启动 F0，不把接口组件升级为 AnySlot/CompoSlot 式实现。详细历史结果文件在 `results/star13_i*_offline_full*.json`，接口阈值见 [11_i_interface_protocol.md](11_i_interface_protocol.md)。
