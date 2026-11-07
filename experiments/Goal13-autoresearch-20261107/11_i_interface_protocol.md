# 11 接口层协议：I0 / I1 / I2 / I3

## 1. 目的

D0/D1 已经关闭数据层：D0 是 best data arm，D1 契约有效但方向为负。接口层只回答一个问题：

> 当显式空间证据已经存在时，动作专家能不能把它变成动作控制？

这不能只看 point-in-mask 或 point L1。point head 可以很准，而动作仍然不用它；上一轮 M3 的 Gate P2 正是在这里失败。因此接口层必须把“证据质量”和“动作读取”拆开。

## 2. 四个 arm 的精确定义

| arm | 初始化 | 训练 | 推理 marker | 用途 |
|---|---|---|---|---|
| `I0` | D0 matched checkpoint | 2.5K action-only continuation | none | matched 无 marker 对照 |
| `I1` | `I2` checkpoint | 不单独训练 | GT object/destination point | oracle 诊断，不可部署 |
| `I2` | D0 matched checkpoint | 2.5K action + point-evidence projector | P1 predicted point | 候选接口 |
| `I3` | `I2` checkpoint | 不单独训练 | predicted point 做 batch shuffle | 语义负控 |

关键约束：

- `I0` 和 `I2` 必须使用同一份 D0 数据、同一 seed、同一 batch/steps、同一 evaluator；
- `I1`/`I3` 不训练新参数，只复用 `I2` checkpoint，避免把“更多训练”误写成接口收益；
- `I1` 的 GT point 只用于诊断，绝不能进入部署输入；
- `I3` 的 shuffle 只改变 marker 与样本的对应关系，不改变图像、语言或专家动作。

## 3. 为什么 I2 要训练 action expert

旧 M3 只训练 `point_evidence_projector`，冻结 action expert。结果是 predicted 与 shuffle 几乎一样，不能证明 marker 真的进入动作。接口层需要给动作专家一个学习读取 marker 的机会，所以：

- 冻结 Qwen/VLM；
- 冻结 P1 point head；
- 冻结 point tokenizer；
- 训练 `action_model`；
- `I2` 同时训练 `point_evidence_projector`；
- `I0` 只训练 `action_model`，不创建或读取 marker。

这不是“换了一个大方法”，而是把接口 headroom 和 marker 质量分开。`I0` 多训练了 action expert，因此它和 `I2` 是 matched 训练预算对照，而不是原 D0 checkpoint 的直接替代。

## 4. 训练与评测预算

### 4.1 Pilot

先跑 500 steps 的 `I0` 和 `I2`：

```text
data_arm      = D0
init          = star13_d0_matched_2500/final_model/model.safetensors
steps         = 500
batch         = 4
grad_accum    = 1
seed          = 17
```

500 steps 只用于检查：

- loss finite；
- action/projector 梯度非零；
- `predicted != none`；
- rotation/gripper 不塌缩；
- oracle/predicted/shuffle 的动作差异不是纯噪声。

### 4.2 Formal pilot

只有 500-step 方向不坏，才扩到 2.5K。正式接口 Gate 使用完整 118 groups / 78+78 pairs 的 offline evaluator；15 条 closed-loop pilot 不替代它。

## 5. 评测接口

所有 arm 使用同一份 validation sidecar、validation episodes 和 seed。对每个 batch 计算：

| 指标 | 用途 |
|---|---|
| `correct_mae` / `wrong_mae` | 语言条件是否正确改变动作 |
| `destination margin` | destination 因子是否被使用 |
| `object margin` | source 因子是否被使用 |
| `non-gripper MAE` | 接口主 Gate，排除 gripper 单维塌缩 |
| `translation/rotation/gripper amplitude` | 检查动作是否被压扁 |
| `point_in_mask` / point L1 | 证明 marker 本身的证据质量，不单独作为 Gate |

`I1` 评测时把 GT point 注入 `point_tokenizer -> projector -> action_model` 的同一条路径，不改变其他输入。`I3` 评测时对同一 batch 的 predicted point 做固定 roll。

## 6. Gate

### G-I0 契约

- [ ] `I0` 与 `I2` 的 config SHA、data length、sampling weight、batch、seed 全部匹配；
- [ ] `I0` marker = none，`I2` marker = predicted；
- [ ] `I1`/`I3` 的 parent 指向同一个 `I2` checkpoint；
- [ ] 所有 postrun 都是 PASS，checkpoint finite；
- [ ] 训练/评测日志没有 NaN、shape error 或 hidden fallback。

### G-I1 动作头 headroom

- [ ] `I1` 的 non-gripper MAE 相对 `I0` 改善 `>= 10%`；
- [ ] `I1` 的 destination/object margin 同向；
- [ ] translation、rotation、gripper 的 amplitude 不塌缩；
- [ ] 如果 `I1 <= I0`，结论是“动作头不能读 marker”，停止，不进入 A2。

### G-I2 predicted marker

- [ ] `I2 > I3`（non-gripper MAE、destination margin 至少一个主指标同向）；
- [ ] `I2` 的 point-in-mask / point L1 抽检通过；
- [ ] `I2` 相对 `I0` 的方向与 `I1` 一致，允许幅度较小；
- [ ] 若 `I1 > I0` 但 `I2 <= I3`，结论是“point head/接口预测是瓶颈”，先修 point，不上控制层。

### G-I3 负控

- [ ] shuffle marker 不优于 predicted；
- [ ] zero/none 不优于 predicted；
- [ ] 失败不是由某个动作维度完全冻结造成；
- [ ] 正式结论必须用完整 118 groups，不应只报 limit=2 smoke。

## 7. 状态与停止条件

| 观察 | 解释 | 下一步 |
|---|---|---|
| `I1 > I0`，`I2 > I3` | 动作头能读 marker，预测 marker 也保留语义 | 进入 A0/A1 |
| `I1 > I0`，`I2 ≈ I3` | 动作头有 headroom，但 point 预测/接口不可用 | 修 point head 或 marker 表示 |
| `I1 ≈ I0` | 动作头不能读 marker，或 projector 没进动作 | 不改 A2，先改注入位置/训练解冻范围 |
| 全 arm 都退化 | 2.5K continuation 本身伤害动作，或契约有误 | 先查 I0，不判方法失败 |
| 动作幅度/rotation/gripper 塌缩 | 评测或训练目标有问题 | 标记 INVALID，先修链路 |

接口层只决定“能不能进入控制层”。即使 `I2` 有正信号，也不能直接宣称 FAI-VLA；必须继续 A 层。

## 8. Run Contract 记录字段

```yaml
arm: I0 | I1 | I2 | I3
data_arm: D0
parent_arm: D0 | I2
marker_mode: none | oracle | predicted | shuffle
marker_source: none | gt_point_manifest | point_head
init_checkpoint: ...
init_checkpoint_sha256: ...
point_checkpoint: ...
point_checkpoint_sha256: ...
trainable_modules: ...
diagnostic_only: true | false
```

`I1` 必须标 `diagnostic_only: true`，`I3` 必须标 `diagnostic_only: true` 且 `parent_arm: I2`。任何字段缺失，arm 状态为 `INVALID`，不能进入 Gate 判读。
