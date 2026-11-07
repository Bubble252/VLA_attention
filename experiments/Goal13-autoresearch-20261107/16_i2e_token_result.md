# 16 I2-ETOKEN query-token 注入结果

**日期**：2026-11-07<br>
**状态**：**FAIL**；最后一次接口修复失败，`W-I` 永久关闭<br>
**协议**：[15_i2e_token_injection_protocol.md](15_i2e_token_injection_protocol.md)<br>
**父结果**：[14_i2x_injection_result.md](14_i2x_injection_result.md)<br>
**运行**：`star13_i2_matched_500-etoken`<br>
**结论**：把归一化 evidence 作为独立 future-query token 追加，没有产生可检测的 marker 控制增益；接口路线到此停止。

## 1. 运行有效性

| 检查 | 结果 |
|---|---|
| config SHA | `5887bcb7dcf3b4216337aa9f7448dd1571399cbd9599ef340e8fa79340fa52a6` |
| preflight | PASS：`evidence_token=true`、`evidence_cross_attention=false` |
| 500-step postrun | PASS |
| checkpoint SHA | `08f88cf48ef893de29f9c65aee233a9d6def94a585f97f0eddb4202f755cee94` |
| checkpoint health | `all_finite=true`，255 tensors |
| trainable parameters | `174.486M`，与原 `I2` 相同 |
| cross-attention 参数 | 不存在（确认没有误开 cross-attention） |
| 完整评测 | `I1/I2/I3` 均为 118 groups、78+78 pairs，contract PASS |

该运行与 `I2` 使用同一 D0 数据、D0 初始化、500 steps、batch 4、seed 17；唯一变量是把 evidence 作为独立 future-query token 追加，而不是 common-mode addition 或 cross-attention memory。

## 2. 完整 118 groups 结果

non-gripper MAE 按 `(7 × correct_mae - gripper_mae) / 6` 计算。

| arm | destination correct MAE | non-gripper MAE | 相对 I0 | destination margin | translation amp | rotation amp |
|---|---:|---:|---:|---:|---:|---:|
| `I0` none | `0.144959092` | `0.149198992` | baseline | `+0.011656230` | `1.0875` | `0.8459` |
| `I1` oracle | `0.145650193` | `0.149181223` | `-0.012%` | `+0.012009065` | `1.076` | `0.836` |
| `I2` predicted | `0.145626649` | `0.149163755` | `-0.024%` | `+0.012110597` | `1.076` | `0.837` |
| `I3` shuffle | `0.145622090` | `0.149166492` | `-0.022%` | `+0.012124856` | `1.076` | `0.837` |

补充数字：

- `I1` 相对 `I0` 的改善只有 `0.012%`，远低于预注册的 `>=10%`；
- `I2` 相对 `I3` 的 non-gripper MAE 仅低 `2.737e-06`（约 `0.002%`），但 destination margin 低 `1.426e-05`，方向不一致；
- 三个臂的 action amplitude、translation、rotation 都接近 1，未出现动作幅度塌缩；
- point predicted destination L1 仍为 `2.645 px`，point head 精度不是本次瓶颈；
- JSON 内部的 `gate` 字段是同一 run 内的 point ablation 布尔值，不是本协议注册的跨 arm Gate，不能拿来替代 `I1 > I0` / `I2 > I3` 判读。

## 3. Gate 裁决

| Gate | 条件 | 实际 | 裁决 |
|---|---|---|---|
| `G-I1` | `I1` non-gripper MAE 相对 `I0` 改善 `>=10%` | 改善 `0.012%` | **FAIL** |
| `G-I2` | `I2` non-gripper MAE 优于 `I3`，destination margin 同向 | MAE 差异约 `0.002%`，margin 反而更低 | **FAIL** |
| 幅度检查 | translation/rotation/gripper 不塌缩 | 三臂均未塌缩 | PASS（仅链路） |
| 动作幅度/契约 | 所有输出来自完整 118 groups | 满足 | PASS（仅链路） |

结论：`I2-ETOKEN` **FAIL**。这不是再跑更久或再换一个 token 拼接方式就能合理修补的问题；按 `15` 的停止条件，接口路线永久关闭，不启动 `A2/A3/F0`，也不再把接口组件升级成 paper-like 版本。

## 4. 解释边界

本次阴性结果支持以下较窄结论：

- 在当前 C1 数据、D0 初始化、500-step matched pilot 下，`evidence_token` 追加没有让 oracle/predicted marker 产生可检测的动作控制差异；
- 三次接口尝试（原 common-mode、cross-attention、query-token）都没有出现 `I1 > I0` 或 `I2 > I3` 的正信号；
- 因此不能再把失败简单归因于“注入方式还不够像论文”，也不能把 point head 精度或动作幅度当作唯一解释。

不能据此声称所有 action-query / cross-attention 设计都无效，因为本实验只覆盖一个任务、一个 backbone、一个 500-step pilot 和一个 seed。但它足以否定当前这条接口路线，后续必须做 outer-loop PIVOT，而不是继续堆接口变体。

## 5. 后续裁决

1. 关闭 `W-I`，停止接口注入实验；
2. 不启动 `A2/A3/F0`，不把 `CompoSlot`、`AC-VLA-lite` 或 factor loss 升级为 paper-like 版本；
3. 保留三组完整 negative result 作为诊断材料；
4. 进入 outer-loop：优先按 `doc/vla_next/53_action_query_investigation_and_next_goal_20261029.md` 评估 action-query/action-head 方向，或按 `65` 的第二场景规则评估 caddy；在写出新的 protocol 和 Gate 前不训练。

## 6. 产物

- `results/I2_matched_500-etoken_{preflight,contract,postrun,checkpoint_health}.json`
- `results/star13_i2_matched_500-etoken_I{1,2,3}_offline_limit2*.json`
- `results/star13_i2_matched_500-etoken_I{1,2,3}_offline_full*.json`
- 远端 checkpoint：`star105_goal13_20261107/checkpoints/star13_i2_matched_500-etoken/final_model/model.safetensors`
