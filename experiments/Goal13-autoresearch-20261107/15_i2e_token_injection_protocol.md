# 15 I2-ETOKEN：最后一次 query-token 注入修复

**日期**：2026-11-07<br>
**状态**：预注册；先 2-step smoke，再 500-step matched pilot，再完整 118 groups<br>
**父协议**：[13_i2x_injection_protocol.md](13_i2x_injection_protocol.md)<br>
**失败依据**：[14_i2x_injection_result.md](14_i2x_injection_result.md)<br>
**对应规划**：`doc/vla_next/65_story_method_selection_20261107.md` 的 `W-I` 最后修复窗口

## 1. 单一假设

`I2-XATTN` 失败后，最后一个合理接口假设是：common-mode addition 和额外 cross-attention memory 都没有给 marker 独立的选择性位置；如果把归一化 point evidence 作为独立的 future-query token 追加，动作 DiT 的 self-attention 可能会把它当作可选择的动作上下文。

本 arm 只检验：

> 把 `point_evidence_projector` 输出的 evidence 向量作为独立 query token 追加，能否让 action expert 真正读取 marker。

如果仍没有 `I1 > I0` 或 `I2 > I3`，停止接口路线，不做第三个注入变体。

## 2. 唯一变量

| 项 | `I2` | `I2-ETOKEN` |
|---|---|---|
| 数据 | `D0` C1 三 prompt | 同 `I2` |
| 初始化 | `star13_d0_matched_2500/final_model/model.safetensors` | 同 `I2` |
| steps | 500 | 500 |
| batch / seed | 4 / 17 | 4 / 17 |
| marker | predicted point | predicted point |
| point head | frozen | frozen |
| trainable | `action_model + point_evidence_projector` | 同 `I2` |
| evidence injection | common-mode addition | `evidence_token: true`；独立追加归一化 evidence token |
| cross-attention | false | false |

该实现不新增网络参数；`evidence_token` 只改变 `_build_future_tokens` 的拼接方式。预期 trainable 参数应与原 `I2` 一样约为 `174.486M`。

## 3. Run Contract

启动前必须满足：

- config 中 `framework.action_model.evidence_token: true`；
- config 中 `framework.action_model.evidence_cross_attention: false`；
- `framework.point_condition.enabled: true`、`injection: evidence`；
- D0 checkpoint SHA、point head SHA、data length `11630`、batch 4、grad accum 1、seed 17 均匹配；
- `GOAL13_EXPECT_EVIDENCE_TOKEN=1` 时 preflight 必须记录 `i2_evidence_token: PASS`；
- `GOAL13_FORBID_EVIDENCE_CROSS_ATTENTION=1` 时 preflight 必须记录 `i2_evidence_cross_attention: PASS`；
- smoke checkpoint 不包含 `action_model.evidence_context_*` 参数；若出现则说明 cross-attention 未关闭，arm 标为 `INVALID`；
- 任何 Run Contract 失败都标为 `INVALID`，不能进入 Gate。

## 4. 执行顺序

### 4.1 2-step smoke

```bash
cd /vepfs-mlp2/c20250405/400040/transfer/vla_attention/vla_workspace/repo/starvla_55_20261030/repo
GOAL13_RUN_TAG=etoken-smoke \
GOAL13_EXPECT_EVIDENCE_TOKEN=1 \
GOAL13_FORBID_EVIDENCE_CROSS_ATTENTION=1 \
bash /tmp/goal13_tools/run_goal13_i_matched.sh I2 /tmp/goal13_tools/goal13_i2_etoken_matched.yaml 2
```

smoke 只回答：配置生效、cross-attention 关闭、loss finite、checkpoint 可加载。不能据此判方法有效。

### 4.2 500-step matched pilot

```bash
GOAL13_RUN_TAG=etoken \
GOAL13_EXPECT_EVIDENCE_TOKEN=1 \
GOAL13_FORBID_EVIDENCE_CROSS_ATTENTION=1 \
bash /tmp/goal13_tools/run_goal13_i_matched.sh I2 /tmp/goal13_tools/goal13_i2_etoken_matched.yaml 500
```

### 4.3 离线评测

```bash
bash /tmp/goal13_tools/run_goal13_i_offline_eval.sh <new_postrun.json> I1 2
bash /tmp/goal13_tools/run_goal13_i_offline_eval.sh <new_postrun.json> I2 2
bash /tmp/goal13_tools/run_goal13_i_offline_eval.sh <new_postrun.json> I3 2
```

limit=2 全通过后，再串行跑完整 118 groups。I1 的 GT point 只作诊断，部署不使用。

## 5. Gate 与停止条件

沿用 [11_i_interface_protocol.md](11_i_interface_protocol.md)：

- `G-I1`：`I1` non-gripper MAE 相对 `I0` 改善 `>=10%`；
- `G-I2`：`I2` non-gripper MAE 优于 `I3`，destination margin 同向；
- 动作 translation/rotation/gripper 幅度不塌缩；
- 所有输出必须来自完整 118 groups / 78+78 pairs。

| 结果 | 裁决 | 下一步 |
|---|---|---|
| `I1 > I0` 且 `I2 > I3` | 最后一次注入修复有效 | 重新打开 `W-I`，进入 A0/A1 控制层 |
| 任一不成立 | 接口路线停止 | 不做第三个注入变体；保留失败诊断，转向第二任务或重新审视动作头设计 |
| 训练/加载/契约失败 | `INVALID` | 只修实现和契约，不判方法失败 |

## 6. 输出

- `16_i2e_token_result.md`
- `results/star13_i2_matched_500-etoken_I{1,2,3}_offline_{limit2,full}.json`
- postrun、checkpoint health、contract、config snapshot
- `research-state.yaml`、`research-log.md` 和 `65` 更新

## 7. Git

实验启动前提交配置、preflight、launcher 和协议；结果出来后提交小 JSON 与结果文档。不要提交 checkpoint、safetensors 或大日志。
