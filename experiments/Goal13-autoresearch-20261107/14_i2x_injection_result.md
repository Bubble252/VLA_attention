# 14 I2-XATTN 注入修复结果

**日期**：2026-11-07<br>
**状态**：**FAIL**；接口 Gate 仍未通过，W-I 保持关闭<br>
**协议**：[13_i2x_injection_protocol.md](13_i2x_injection_protocol.md)<br>
**父结果**：[12_interface_result.md](12_interface_result.md)<br>
**运行**：`star13_i2_matched_500-xattn`<br>
**结论**：增加 evidence cross-attention memory 路径没有让动作专家更有效地使用 marker；按协议只剩最后一次 `evidence_token` 注入修复。

## 1. 运行有效性

| 检查 | 结果 |
|---|---|
| config SHA | `6f0f8716090fb95514a99d12bd39f0c269715f0ac750d04136fd9bbaae747de3` |
| preflight | PASS |
| 500-step postrun | PASS |
| checkpoint SHA | `c9898baf54899eba88b5e0b73bee6d8a60f6fe89e6d263dc9fb52351d085d601` |
| checkpoint health | `all_finite=true` |
| trainable parameters | `178.949M`（原 I2 为 `174.486M`） |
| cross-attention 参数 | checkpoint 中存在 7 个 `evidence_context_*` 参数 |
| 完整评测 | `I1/I2/I3` 均为 118 groups、destination/object 各 78 pairs，contract PASS |

该运行使用与 `I2` 相同的 D0 数据、D0 初始化、500 steps、batch 4、seed 17；唯一变量是增加 `evidence_cross_attention: true`。原有的 common-mode addition 仍然保留，因此这不是纯 cross-only 消融。

## 2. 完整 118 groups 结果

non-gripper MAE 按预注册公式 `(7 × correct_mae - gripper_mae) / 6` 计算。

| arm | destination correct MAE | non-gripper MAE | 相对 I0 | destination margin | translation amp | rotation amp |
|---|---:|---:|---:|---:|---:|---:|
| `I0` none | `0.144959092` | `0.149198992` | baseline | `+0.011656230` | `1.0875` | `0.8459` |
| `I1` oracle | `0.150395989` | `0.153917042` | `+3.162%` | `+0.011346647` | `1.096` | `0.853` |
| `I2` predicted | `0.150406316` | `0.153986648` | `+3.209%` | `+0.011738941` | `1.094` | `0.848` |
| `I3` shuffle | `0.150142521` | `0.153697833` | `+3.015%` | `+0.012078720` | `1.093` | `0.848` |

补充数字：

- `I1` 的 non-gripper MAE 比 `I0` 高 `0.00471805`；
- `I2` 的 non-gripper MAE 比 `I3` 高 `0.000288814`，即 predicted 比 shuffle 差约 `0.188%`；
- `I2` 的 destination margin 比 `I3` 低 `0.000339779`；
- 三个修复臂的 translation/rotation/action amplitude 都接近 1，未出现动作幅度塌缩；
- point predicted destination L1 仍为 `2.645 px`，object L1 为 `7.807 px`，说明 point head 精度不是本次失败的直接原因。

## 3. Gate 裁决

| Gate | 条件 | 实际 | 裁决 |
|---|---|---|---|
| `G-I1` | `I1` non-gripper MAE 相对 `I0` 改善 `>=10%` | 比 `I0` 差 `3.162%` | **FAIL** |
| `G-I2` | `I2` non-gripper MAE 优于 `I3`，destination margin 同向 | `I2` 比 `I3` 差 `0.188%`，margin 也更低 | **FAIL** |
| 幅度检查 | translation/rotation/gripper 不塌缩 | 三臂均未塌缩 | PASS（仅链路） |
| 负控检查 | shuffle 不应优于 predicted | shuffle 仍略优于 predicted | **FAIL（无正信号）** |

因此 `W-I` 不开启，`A2/A3/F0` 继续冻结。不能因为 cross-attention 是论文中常见的接口，就把它写成有效组件；本次结果只说明“在当前 matched 配置下，这条额外 cross-attention 路径没有提供可检测的 marker 控制增益”。

## 4. 解释边界

本次结果不能单独证明所有 cross-attention 设计都无效，因为：

- 当前实现是“追加” cross-attention memory，不是替换 common-mode addition；
- cross-attention scale 初始化为 `0.1`，优化路径可能仍然偏弱；
- 只有一个 seed、一个任务和 500-step pilot，不能做统计显著性声明。

但它足以否定当前的修复假设：**在同样数据、初始化、步数和评测下，增加这条路径没有让 `I1 > I0` 或 `I2 > I3` 成立**。因此不能把失败归因于 point head，也不能继续堆更多接口模块。

## 5. 唯一允许的下一步

按 `13_i2x_injection_protocol.md` 的停止规则，只剩一次 `evidence_token=true` 的 query-token 注入修复：

- 保持与 `I2` 相同的 D0 数据、D0 初始化、500 steps、batch 4、seed 17；
- `evidence_cross_attention: false`；
- `evidence_token: true`，把归一化 evidence 作为独立的 future-query token 追加，而不是 common-mode addition；
- 先跑 `I1/I2/I3` limit=2 smoke，再跑完整 118 groups；
- 仍以 `I1 > I0` 且 `I2 > I3` 为唯一接口 Gate。

如果该 arm 仍失败，接口路线停止，不再做第三个注入变体；后续只保留失败诊断和转向另一任务/另一层方法的选择。

## 6. 产物

- `results/I2_matched_500-xattn_{preflight,contract,postrun,checkpoint_health}.json`
- `results/star13_i2_matched_500-xattn_I{1,2,3}_offline_limit2*.json`
- `results/star13_i2_matched_500-xattn_I{1,2,3}_offline_full*.json`
- 远端 checkpoint：`star105_goal13_20261107/checkpoints/star13_i2_matched_500-xattn/final_model/model.safetensors`
