# 12 接口层结果：I0 / I1 / I2 / I3

**状态**：进行中。`I0` 已完成并通过 smoke；`I2` 训练中；`I1/I3` 尚未评测。<br>
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

## 3. 下一步

1. 完成 `I2` 500-step 训练与 postrun；
2. 对同一 `I2` checkpoint 依次跑 `I1/I2/I3` 的 limit=2 smoke；
3. 确认三臂都产生 `point_results`，且 oracle/predicted/shuffle/none 的字段完整；
4. 只有 smoke 通过后才扩到完整 118 groups / 78+78 pairs；
5. 未过 `I1 > I0` 与 `I2 > I3` 前，不进入 A2 或 F0。
