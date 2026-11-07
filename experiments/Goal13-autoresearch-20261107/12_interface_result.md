# 12 接口层结果：I0 / I1 / I2 / I3

**状态**：进行中。`I0/I2` 已完成并通过 smoke；`I1/I2/I3` limit=2 smoke 已完成；尚待完整 118 groups。<br>
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
| predicted - none | -0.000594，95% CI [-0.003436, 0.002247] |
| predicted - shuffle | -0.000690，95% CI [-0.001706, 0.000326] |
| oracle - none | -0.001014 |
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

## 4. 下一步

1. 对 `I1/I2/I3` 依次跑完整 118 groups / 78+78 pairs；
2. 用完整结果判断 `I1 > I0` 是否有动作头 headroom；
3. 若 `I1 <= I0`，停止 A2，先修 marker 注入位置或 action head；
4. 若 `I1 > I0` 但 `I2 <= I3`，修 point head/接口，不进入控制层；
5. 只有接口 Gate 通过，才允许进入 A2/F0。
