# 13 I2-XATTN：marker 注入修复协议

**日期**：2026-11-07<br>
**状态**：预注册；先 2-step smoke，再 500-step matched pilot，再完整 118 groups<br>
**父协议**：[11_i_interface_protocol.md](11_i_interface_protocol.md)<br>
**失败依据**：[12_interface_result.md](12_interface_result.md)<br>
**对应规划**：`doc/vla_next/65_story_method_selection_20261107.md` 的 `W-I` 修复窗口

## 1. 单一假设

接口层 `G-I1/G-I2` 失败的主要原因是：当前 point evidence 以一个 `[B,D]` common-mode 向量加到所有 `future_tokens` 上，动作查询没有选择性地读取 marker。不是 point head 预测太差，因为 predicted destination L1 只有约 `2.645 px`，而 oracle marker 也没有改善动作。

本 arm 只检验一个假设：

> 把 point evidence 作为 action DiT cross-attention 的额外 encoder memory，能否让 action expert 真正读取 destination/source marker。

如果该假设不成立，不能在同一个 arm 里再加 factor loss、换数据、延长训练或换 backbone。

## 2. 唯一变量

| 项 | `I2` | `I2-XATTN` |
|---|---|---|
| 数据 | `D0` C1 三 prompt | 同 `I2` |
| 初始化 | `star13_d0_matched_2500/final_model/model.safetensors` | 同 `I2` |
| steps | 500 | 500 |
| batch / seed | 4 / 17 | 4 / 17 |
| marker | predicted point | predicted point |
| point head | frozen | frozen |
| trainable | `action_model + point_evidence_projector` | 同 `I2` |
| evidence injection | common-mode addition | 增加 `evidence_cross_attention`，并保留现有 common-mode 路径 |

注意：`evidence_cross_attention` 当前实现是**追加**一个 cross-attention memory 路径，不是替换 common-mode。这个 arm 检验的是“增加可选择性读取的路径是否有信号”，不是纯 cross-only 消融。若它有效，下一轮再单独做 common-mode 关闭消融。

## 3. 代码初始化

D0 checkpoint 是 trainable-only checkpoint，不包含 cross-attention 新增的：

- `action_model.evidence_context_projector.*`
- `action_model.evidence_context_scale`

实际训练加载走 `starVLA/training/trainer_utils/trainer_tools.py` 的 `load_pretrained_backbones`：全模型加载使用 `model.load_state_dict(checkpoint, strict=False)`，因此缺失的新参数会由模型初始化补上，不需要修改 `base_framework.py`。

2-step smoke 已确认这些参数确实进入 checkpoint：

- `action_model.evidence_context_projector.0/1/3.{weight,bias}`
- `action_model.evidence_context_scale`

共 7 个参数；smoke checkpoint 的 trainable 参数为 `178.949M`，而原 `I2` 为 `174.486M`。checkpoint health 为 `all_finite=true`。因此不需要保留 `base_framework.py` 临时补丁，该补丁已撤销，相关 patch 文件已删除。

## 4. Run Contract

启动前必须满足：

- config 中 `framework.action_model.evidence_cross_attention: true`；
- `framework.action_model.evidence_token: false`；
- `framework.point_condition.enabled: true`、`injection: evidence`；
- D0 checkpoint SHA、point head SHA、data length `11630`、batch 4、grad accum 1、seed 17 均匹配；
- `GOAL13_EXPECT_EVIDENCE_CROSS_ATTENTION=1` 时 preflight 必须记录 `i2_evidence_cross_attention: PASS`；
- smoke checkpoint 必须包含上述 7 个 cross-attention 参数，且 checkpoint health 为 `all_finite=true`；
- 出现任何其他 missing/unexpected key，arm 标为 `INVALID`，不能进入 Gate。

## 5. 执行顺序

### 5.1 2-step smoke

```bash
cd /vepfs-mlp2/c20250405/400040/transfer/vla_attention/vla_workspace/repo/starvla_55_20261030/repo
GOAL13_RUN_TAG=xattn-smoke \
GOAL13_EXPECT_EVIDENCE_CROSS_ATTENTION=1 \
bash /tmp/goal13_tools/run_goal13_i_matched.sh I2 /tmp/goal13_tools/goal13_i2_xattn_matched.yaml 2
```

smoke 只回答：配置生效、cross-attention 参数初始化、loss finite、checkpoint 可加载。不能据此判方法有效。

### 5.2 500-step matched pilot

```bash
GOAL13_RUN_TAG=xattn \
GOAL13_EXPECT_EVIDENCE_CROSS_ATTENTION=1 \
bash /tmp/goal13_tools/run_goal13_i_matched.sh I2 /tmp/goal13_tools/goal13_i2_xattn_matched.yaml 500
```

### 5.3 离线评测

使用新 postrun 依次跑：

```bash
bash /tmp/goal13_tools/run_goal13_i_offline_eval.sh <new_postrun.json> I1 2
bash /tmp/goal13_tools/run_goal13_i_offline_eval.sh <new_postrun.json> I2 2
bash /tmp/goal13_tools/run_goal13_i_offline_eval.sh <new_postrun.json> I3 2
```

smoke 通过后跑完整 118 groups。I1/I2/I3 复用同一个新 checkpoint；I1 的 GT point 只作诊断，部署不使用。

## 6. Gate

沿用 [11_i_interface_protocol.md](11_i_interface_protocol.md) 的阈值：

- `G-I1`：`I1` non-gripper MAE 相对 `I0` 改善 `>=10%`；
- `G-I2`：`I2` non-gripper MAE 优于 `I3`，destination margin 同向；
- 动作 translation/rotation/gripper 幅度不塌缩；
- 所有输出必须来自完整 118 groups / 78+78 pairs。

## 7. 停止条件

| 结果 | 裁决 | 下一步 |
|---|---|---|
| `I1 > I0` 且 `I2 > I3` | 注入修复有效 | 进入 A0/A1 控制层 |
| `I1 <= I0` 且 `I2 <= I3` | 注入修复仍失败 | 只允许再做一次 `evidence_token=true` 的 query-token 注入；随后停止接口路线 |
| 训练/加载失败 | `INVALID` | 修实现，不判方法失败 |
| 幅度塌缩 | `INVALID` | 修训练/评测链路 |

## 8. 输出

- `14_i2x_injection_result.md`
- `results/star13_i2_matched_500-xattn_I{1,2,3}_offline_full.json`
- postrun、checkpoint health、contract、config snapshot
- `research-state.yaml` 和 `research-log.md` 更新

## 9. Git

实验启动前提交配置、preflight、launcher 和协议；结果出来后提交小 JSON 与结果文档。不要提交 checkpoint、safetensors 或大日志。
