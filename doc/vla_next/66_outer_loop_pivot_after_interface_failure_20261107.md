# 66 接口路线失败后的 Outer-loop PIVOT

**日期**：2026-11-07<br>
**状态**：PIVOT 协议；不启动训练，先做闭环 authority 上界复现<br>
**输入**：`65`、Goal 9 / `55`、K4-lite T4b v7–v11、Goal12、Goal13 的 D/I 结果<br>
**一句话**：当前最先要验证的不是“再加一个 evidence token”，而是 **横向控制 authority 是否足够，以及这份 authority 能否从特权教师蒸馏成可部署策略**。

**P0 更新（2026-11-07）**：P0 已完成，`A10` 在 15-init `white_right` 上取得 `7/15` official success、`4/15` released success、`6/12` confirmation success，预注册 Gate 全部通过。结果见 [Goal14 P0](../experiments/Goal14-p0-authority-release-20261107/05_p0_result.md)。这只允许写 P1 protocol，不直接启动训练。

---

## 1. 裁决先行

### 1.1 不再做什么

以下路线当前全部关闭，不能原样重跑：

- 不再做 marker、cross-attention、query-token 的接口注入变体；
- 不再用 `I2-XATTN`、`I2-ETOKEN` 或其他“更像论文”的接口替换来绕过 `I1 > I0` / `I2 > I3`；
- 不启动 `A2/A3/F0`，不把 CompoSlot、AC-VLA-lite 或 factor loss 升级成论文近似版；
- 不把“离线 MAE 改善”直接写成动作能力提升；
- 不追已经接近 ceiling 的 `14/15`。

### 1.2 下一阶段只保留两条假设

| 编号 | 假设 | 先验依据 | 一旦不成立说明什么 |
|---|---|---|---|
| `H-AUTH` | 低 z 段的横向 authority 不足，是 `white_right` 闭环失败的主要剩余瓶颈 | v8.1 延迟释放只移动 `0.1mm`；v9 继续控制 142 步只改善 `0.08mm`；v10 的 `S10` 在 3 个 calibration init 上拿到 hold-only `3/3` | 不是“再来一点 residual”能解决；必须换任务或重做执行链 |
| `H-DISTILL` | 特权教师生成的横向修正，可以被可部署的视觉/语言/动作学生网络蒸馏出来 | v10 已给出有效 authority 量级；v2.2 证明旧 supervision 只学到幅度、没学到方向 | 当前部署输入不足以预测修正，或数据分布仍不匹配；停止 residual 容量路线 |

`caddy` 不是第三条假设。它只在 `H-AUTH/H-DISTILL` 出现可复核正信号后，作为第二场景验证；现在没有 caddy 的 headroom、point/sidecar 和 failure atlas，不能拿它替代 C1 的机制发现。

---

## 2. 现有证据为什么支持 PIVOT

### 2.1 数据层和接口层都没有产生正信号

| 证据 | 结果 | 能说明什么 |
|---|---|---|
| `D0` 15-init pilot | `white_right` official `0/15`、strict destination `0/15`；first destination 为 `plate_1=7`、`null=8` | 0/15 可复现，失败主要是没到目标盘或放错盘 |
| `D1` carrier + matched pilot | `white_right` 仍 `0/15`；seen group-any 从 `13/15` 降到 `9/15`；offline destination margin 从 `+0.015105` 降到 `+0.005628` | 只补 carrier 没有解决 held-out；不能让 C1 曝光下降后还宣称数据有效 |
| Goal13 `I0/I1/I2/I3` | `I1` 比 `I0` 差约 `2.16%`；`I2` 不优于 `I3` | oracle marker 都没让动作专家产生正收益 |
| `I2-XATTN` | `I1` 比 `I0` 差约 `3.162%`；`I2` 比 `I3` 差约 `0.188%` | 增加 cross-attention 记忆路径没有让 marker 进入动作 |
| `I2-ETOKEN` | `I1` 相对 `I0` 仅改善 `0.012%`；`I2` 相对 `I3` 仅低约 `0.002%`，margin 反向 | 第三种接口注入也没有正信号 |

### 2.2 容量控制已经排除了“action query 自己有效”的解释

Goal 9 的 216 条共享 probe：

| arm | MAE | 相对 G1 | 说明 |
|---|---:|---:|---|
| `G1` cross-attention bridge | `0.190666` | — | 有额外容量 |
| `G1c` 参数匹配、完全不看视觉/语言 | **`0.187739`** | **+1.54%** | 容量控制反而更好 |
| `G2` predicted evidence | `0.191140` | `-0.25%` | 证据没有增量 |
| `G3` oracle evidence | `0.196194` | `-2.90%` | 完美证据也没有正收益 |

三臂闭环 `G0/G1/G2` 都是 `0/15`。因此当前不能再说“bridge 的离线改善来自 VLM evidence”；更准确的解释是：**额外容量和残差微调能降一点 MAE，但没有把证据变成控制。**

### 2.3 闭环执行已经排除了 release/settle 作为唯一瓶颈

| 诊断 | 关键数字 | 结论 |
|---|---|---|
| v8 `D3` margin gate + settle | release `3/3`、released placement `3/3`、official success `1/3` | 释放和 settle 可以执行，但仍差毫米级横向误差 |
| v8.1 deferred release | release 后额外 4 步只改变约 `0.1mm` | 不是“释放太早” |
| v9 持续横向控制 | `init_2edf` 再走 142 步，xy 只改善 `0.08mm` | 低 z 段进入平台期 |
| v10 authority sweep | `S10`（cap `0.10`）在 3 个 calibration init 上 hold-only `3/3`；`S20/S40` 出现过冲 | authority 有阈值，约 `0.10` 有效，`>0.20` 不安全 |
| v11 三维位姿 | `XYZ10` 从 `XY10` 的 `3/3` 退化为 `0/3`；`ZR10` 为 `0/3` | z 和姿态不能替代横向 authority |

需要明确一个边界：v10 的 `S10` 是 **hold-only success**，不是 released placement。它只证明“横向能动起来并进入成功半径”，不证明最终释放后的操作成功。因此 P0 必须把 v10 的横向 authority 与 v8/v8.1 的 release/settle 组合起来重新测。

---

## 3. 为什么不是继续修接口，也不是先换 caddy

### 3.1 接口路线的上限已经被测过

- point head 不是主要瓶颈：predicted destination L1 约 `2.645 px`、point-in-mask `0.8008`；
- 即使 oracle marker 也没有改善动作；
- 参数匹配控制说明 bridge 的离线收益来自容量；
- 三种注入形式都没有 `I1 > I0` 或 `I2 > I3`。

因此继续做“第四种 marker 拼接”没有可辩护的因果假设，只会增加实验数量，不会增加信息量。

### 3.2 caddy 现在只能做 headroom audit

STUDY_SCENE1 caddy 的数据量更大、四个 compartment 更均衡，但当前：

- 没有本项目的低分证据；
- 没有现成 point/sidecar、rollout manifest 和 failure atlas；
- 不知道普通 checkpoint 是否在 caddy 上已经 ceiling；
- 失败可能混入长程控制，而不只是 destination binding。

所以 caddy 的正确位置是“C1 出现正信号后的第二场景”，或“C1 authority 假设被否证后的 fallback”，不是当前第一训练任务。

---

## 4. 两条新假设的可检验形式

### 4.1 `H-AUTH`：横向 authority 上界

**陈述**：在同一 15-init `white_right` 评测集上，把 v10 的 bounded `x/y` authority 与 v8/v8.1 的 release/settle 组合后，能产生真实的 released success，而不只是 hold-only success。

**预测**：

- `A10`（fine cap `0.10`）的 official success 明显高于 `B0`；
- `A05` 因低于静摩擦/接触阈值，success 明显较低；
- `A20` 可能在某些 init 上更快，但会出现过冲，不能稳定优于 `A10`；
- released success 不能为 0；否则只是重复 v10 的 hold-only。

**否证条件**：

- `A10` 没有超过 `B0` 至少 `3/15`；
- 或 12 个未参与 v10 calibration 的 init 上少于 `2/12`；
- 或 released success 少于 `2/15`；
- 或出现明显 NaN、动作爆炸、物体飞出、夹爪/释放错误。

### 4.2 `H-DISTILL`：可部署蒸馏

**陈述**：用 `A10` 作为特权教师，在 plateau 状态上生成横向修正和 release 标签，可以训练一个只使用部署时可得的图像、语言、proprio、base action 和 predicted point 的学生网络，并保留教师的一部分闭环收益。

**预测**：

- `R2-PRED` 的闭环 official success 高于参数匹配、但不看视觉/语言的 `R1-CAP`；
- `R2-PRED` 在 12 个 confirmation init 上至少有 `2/12`；
- `R3-GEO`（oracle geometry，只诊断）不显著低于 `R2-PRED`；否则说明预测点/几何接口仍是瓶颈；
- 学生不出现 z/rotation 漂移，动作幅度和夹爪时序不塌缩。

**否证条件**：

- `R2-PRED` 不优于 `R1-CAP`；
- 或 `R2-PRED` 的闭环 success 少于 `2/15`；
- 或离线方向 cosine `< 0.50`、dominant-axis sign `< 0.75`；
- 或 release 预测导致 released success 下降；
- 或 seen group-any 低于 `10/15`。

---

## 5. Stage P0：先做闭环 authority 上界复现（不训练）

### 5.1 目的

只回答一个问题：**把 v10 的横向 authority 与 v8/v8.1 的 release/settle 组合后，`white_right` 能否出现真实的 released success？**

这一步不训练、不改 VLM、不改 action head。它决定 `H-AUTH` 是否值得进入蒸馏。

### 5.2 数据与评测

- 任务：C1 `LIVING_ROOM_SCENE5`；
- prompt：`put the white mug on the right plate`；
- 评测集：现有冻结 15-init manifest，manifest SHA 必须与 v10 一致；
- v10 已用过的 3 个 init 标为 `calibration`，另外 12 个标为 `confirmation`；
- 不把 3 个 calibration init 的结果写成 15 个独立 confirmation；
- seed：7；同一 evaluator、同一 success 半径、同一 release/settle 口径；
- `white_right` 仍然不进入任何训练。

### 5.3 Arms

| arm | 横向 authority | release | settle | 作用 |
|---|---|---:|---:|---|
| `B0` | 无 | 沿用 base policy | 0 | 复现 `0/15` 基线 |
| `A05` | fine cap `0.05` | 到目标后 release | 4 步 | 低于阈值的负控 |
| `A10` | fine cap `0.10` | 到目标后 release | 4 步 | 主上界臂 |
| `A20` | fine cap `0.20` | 到目标后 release | 4 步 | 过冲/上限控制 |

控制器只改 `x/y`，z 和旋转保持 base policy；不引入新的学习模块。`A10` 的控制器口径必须冻结，不能看结果后改 cap、gain、release 半径或 settle 步数。

### 5.4 指标

必须同时报告：

- `official_success`；
- `released_success`（有 release 且 official success）；
- `hold_only_success`（不能算成功）；
- `released_placement`；
- `wrong_destination`；
- `target_not_reached`；
- `object_retained`；
- `release` 次数与 release step；
- `final_xy`、`release_xy`、`release_z`；
- `translation/rotation/gripper amplitude`；
- `error`、`NaN`、`flew_out`。

### 5.5 预注册 Gate

| 检查 | 通过条件 |
|---|---|
| `B0` 基线复现 | official success `0/15`，或差异有可复核解释 |
| `A10` 总量 | official success `>=5/15` |
| `A10` 独立确认 | 12 个未参与 v10 的 init 中 official success `>=2/12` |
| `A10` 真实释放 | released success `>=4/15`，不能只靠 hold-only |
| `A05` 负控 | `A10 > A05`，至少多 `2/15` |
| `A20` 过冲控制 | `A20` official success 不超过 `A10 + 1/15`；若超过，先审计过冲、飞出和 release 失败类型，不能直接扩大 cap |
| 稳定性 | 0 error、0 NaN、0 飞出、object retained `>=14/15` |
| 契约 | manifest、checkpoint、config、evaluator SHA 全部一致 |

**P0 PASS**：`H-AUTH` 获得强支持，允许写 P1 protocol；仍不允许直接训练。<br>
**P0 FAIL**：`H-AUTH` 被否证或不够稳健；停止 residual 容量路线，转 caddy zero-shot headroom audit 或重新写 outer-loop PIVOT。

---

## 6. Stage P1：可部署横向 authority 蒸馏（P0 PASS 后才启动）

### 6.1 先写 protocol，再写代码

P1 必须先落盘：

- 教师标签定义；
- 学生输入/输出；
- 训练/验证/rollout split；
- 参数匹配 control；
- 离线 Gate、闭环 Gate、停止条件；
- 预计 GPU 时间与 checkpoint 路径。

没有 P1 protocol 和 Run Contract，不启动训练。

### 6.2 教师与标签

教师：P0 的 `A10` 控制器，只在训练/标签生成时使用物体和盘子位置。特权几何、GT mask/point **不能进入部署前向**。

建议标签：

```text
teacher_action = [delta_x, delta_y, release_logit]
```

- `delta_x/delta_y` 是相对 base action 的 bounded 横向修正；
- `release_logit` 由教师 release 触发得到；
- z、rx、ry、rz 固定为 0；
- 标签采样必须覆盖 v9 的 plateau 区间和 v2.2 失败的大修正桶；
- 同一图像要配正确/错误 destination 的语言对，避免只记住 `right -> plate_2`；
- `white_right` 仍然完全隔离，不能进入训练或验证 loss。

### 6.3 学生与 arms

| arm | 输入 | 输出 | 作用 |
|---|---|---|---|
| `R0` | base policy | 无 | 闭环基线 |
| `R1-CAP` | 只含 base action/proprio，不含视觉/语言 evidence；参数量与学生匹配 | `delta_xy + release` | 排除“多参数”解释 |
| `R2-PRED` | 图像 + 语言 + proprio + base action + predicted point | `delta_xy + release` | 可部署候选 |
| `R3-GEO` | 同 `R2`，但 point 换成 oracle geometry | `delta_xy + release` | 诊断上界，不部署 |

所有 arms 同数据、同初始化、同 batch、同 step、同 seed；先 smoke，再 pilot。`R1-CAP` 的参数量必须与 `R2-PRED` 对齐，不能再用 Goal 9 那种事后补 control 的方式。

### 6.4 离线 Gate

- 修正方向 cosine `>=0.50`；
- dominant-axis sign accuracy `>=0.75`；
- magnitude ratio median 在 `0.70–1.30`；
- plateau/大修正桶不能比 validation 更差；
- release precision/recall 有明确报告，不能用 total accuracy 掩盖稀有 release；
- z/rotation 输出严格为 0 或不超过 `1e-8`；
- 所有指标必须在 eval 模式复算，不能使用 dropout 开启时的随机监视值。

### 6.5 闭环 Gate

- `R2-PRED` official success `>=3/15`；
- 12 个 confirmation init 中 `>=2/12`；
- `R2-PRED` 比 `R1-CAP` 至少多 `2/15`；
- released success `>=2/15`；
- seen group-any 不低于 `10/15`；
- wrong-destination 不超过 `B0`，不能把 wrong destination 换成 no-destination 来刷分；
- 0 error、0 NaN、0 飞出、动作幅度不塌缩。

### 6.6 失败分支

| 观察 | 解释 | 下一步 |
|---|---|---|
| 离线方向好，闭环无提升 | 学生学会了标签但执行/释放不稳 | 查 release 与 authority，不扩大网络 |
| 离线方向差 | 预测点/几何到动作的映射不可学 | 改输入或改显式 subgoal，不继续调容量 |
| `R2` 不如 `R1-CAP` | 增益来自容量，不是 evidence | 判 `H-DISTILL` 不成立 |
| `R3` 明显优于 `R2` | predicted point 是瓶颈 | 回到 point/几何接口，不再加 action token |
| 闭环有 `>=3/15` | 出现真实正信号 | 扩 30/50、3 seeds，再做 caddy 第二场景 |

---

## 7. 统一指标与解释边界

| 指标 | 用途 | 不能替代什么 |
|---|---|---|
| official success | 唯一主 endpoint | 不能只看 release 或 placement |
| released success | 防止 hold-only 冒充成功 | 不能替代 official success |
| released placement | 机制/执行诊断 | 不能写成 official success |
| strict destination | 目的地绑定诊断 | 不能替代闭环成功 |
| correct-vs-wrong action margin | 证据是否影响动作的中间判据 | 不能替代闭环 |
| offline cosine/sign/magnitude | 学生是否学到方向 | 不能替代 rollout |
| seen group-any | 非退化 Gate | 不能替代 held-out |

15 条 rollout 只作方向性筛选；`>=3/15` 不是论文级显著结论。出现正信号后才扩 30/50 和 3 seeds。

---

## 8. Run Contract

每个 arm 启动前必须填写：

| 字段 | 要求 |
|---|---|
| `code_commit` / `config_sha256` | 与实际执行一致 |
| `parent_arm` | `B0` 或 P0 最佳臂 |
| `manifest_sha256` | 与冻结 15-init manifest 一致 |
| `train_language_counts` | `white_right` 必须为 0 |
| `train_validation_rollout_overlap` | 必须为 0 |
| `sampling_weights` | 实际打印 |
| `effective_batch` | batch × accumulation |
| `teacher_checkpoint` / `teacher_config_sha256` | P1 必填 |
| `student_param_count` | `R1-CAP` 与 `R2-PRED` 必须匹配 |
| `lateral_cap` / `z_cap` / `rotation_cap` | 逐项记录，不能只写总 bound |
| `release_mode` / `settle_steps` | 口径必须冻结 |
| `seed` | 固定并可复现 |
| `gate_status` | `PASS / FAIL / INVALID` |

`INVALID` 表示契约不满足，不能用来判方法失败；只有契约 PASS 且 Gate FAIL 才是方法失败。

---

## 9. 资源、数据与安全边界

- 只使用 GPU1；训练、评测、rollout 串行；
- P0 只做闭环诊断，不启动 optimizer；
- `white_right` 训练计数永远为 0；
- validation/rollout init state 不进入训练；
- 特权物体/盘子位置、GT mask、GT point 只用于教师标签或诊断，不进入部署前向；
- 不修改 `references/repos/starvla`；
- checkpoint、视频、safetensors、pt 和大日志不提交；
- 不覆盖 v7–v11、Goal 9、Goal13 的旧结果与 gate。

---

## 10. 外部论文方法的材料门槛

P0/P1 的 v10 控制器、v8 release/settle 和现有 residual 都是本项目资产，不触发外部论文材料门槛。

只要 P1 或后续 arm 借用任何外部论文的模块、loss、数据构造、评测协议或训练 recipe，就必须按 `65` 第 `13.1/13.6` 节执行：

1. 下载论文 PDF 到 `papers/{slug}.pdf`，抽取 `papers/{slug}.txt`；
2. 在摘要、方法、实验、脚注、附录、补充材料和 OpenReview 中搜索代码链接；论文正文中的链接不能漏；
3. 找到官方仓库后 clone 到 `references/repos/{slug}`，记录 `repo_found_in`、URL、commit、license、是否 official；
4. 读完 README、config、loss、data pipeline、train/eval 入口；
5. 写 `references/paper_reading/{slug}.md`，明确 `faithful_reproduction | adapted_reproduction | pseudocode`；
6. 缺任何一项，该 arm 标为 `INVALID`，不进入训练或 Gate。

不能先凭印象写一版，再回头补论文和代码。

---

## 11. 决策树与停止条件

```text
P0：B0 / A05 / A10 / A20 的 15-init authority 上界
  ├─ PASS：H-AUTH 成立，写 P1 protocol
  │    └─ P1：R0 / R1-CAP / R2-PRED / R3-GEO
  │         ├─ R2 闭环正信号：扩 30/50 + 3 seeds + caddy
  │         └─ R2 无正信号：H-DISTILL 不成立，停止 residual 容量路线
  └─ FAIL：H-AUTH 不稳健
       ├─ 先做 caddy zero-shot headroom audit
       └─ 仍无 headroom：重新写 outer-loop PIVOT，不继续堆接口
```

停止条件（满足任一即停）：

1. P0 的 `A10` 未达到总量或 confirmation Gate；
2. P1 的 `R2-PRED` 不优于 `R1-CAP`；
3. 数据/评测契约持续 `INVALID`；
4. 连续 3 个 matched arm 失败且下一轮 PIVOT 已写清；
5. 12 个训练/eval arms 预算用完；
6. 用户要求停止。

任何一次修复都必须先写清假设、变量、baseline、预测、Gate、停止条件和预计 GPU 时间；最多允许同一假设 2 次修复。

---

## 12. 下一阶段 Goal 提示词

```text
你在 /home/bubble/类脑计算/VLM终局 工作。先完整阅读：
- doc/vla_next/39_t4b_v10_verdict.md
- doc/vla_next/44_t4b_v11_verdict.md
- doc/vla_next/51_t4b_v2_2_verdict.md
- doc/vla_next/55_goal9_result_report_20261030.md
- doc/vla_next/65_story_method_selection_20261107.md
- doc/vla_next/66_outer_loop_pivot_after_interface_failure_20261107.md
- experiments/Goal13-autoresearch-20261107/ 下的 D/I/XATTN/ETOKEN 结果

目标：执行 66 的 outer-loop PIVOT。当前 W-I 永久关闭，不启动 A2/A3/F0，不再做 marker、cross-attention、query-token 或 action-query bridge 变体。

第一阶段只做 P0：闭环 authority 上界复现，不训练。

P0 设置：
- 任务：C1 LIVING_ROOM_SCENE5；
- prompt：put the white mug on the right plate；
- 冻结 15-init manifest，manifest SHA 必须与 v10 一致；
- v10 已用的 3 个 init 标为 calibration，另外 12 个标为 confirmation；
- seed 7；同一 evaluator、同一 success 半径、同一 release/settle 口径；
- arms：B0 无 authority；A05 fine cap 0.05；A10 fine cap 0.10；A20 fine cap 0.20；
- 只改 x/y，z 和旋转保持 base policy；
- A10 使用到目标后 release + 4 步 settle；hold-only success 不算 released success；
- 不使用 GT mask/point/object position 进入部署前向；P0 的 authority 只作诊断，privileged geometry 不得写成方法结果。

P0 预注册 Gate：
- B0 复现 white_right 0/15，或差异有可复核解释；
- A10 official success >=5/15；
- A10 在 12 个 confirmation init 中 official success >=2/12；
- A10 released success >=4/15；
- A10 比 A05 至少多 2/15；
- A20 official success 不超过 A10 + 1/15；若超过，先审计过冲、飞出和 release 失败类型，不得直接扩大 cap；
- 0 error、0 NaN、0 飞出、object retained >=14/15；
- manifest/checkpoint/config/evaluator SHA 一致。

每个 arm 启动前写 Run Contract，状态只能是 PASS / FAIL / INVALID。INVALID 不能判方法失败。

P0 PASS 后允许自动进入 P1，但必须先写 P1 protocol 并 commit，再训练：
- 教师：P0 的 A10，只在标签生成时使用特权几何；
- 学生：部署输入为图像、语言、proprio、base action、predicted point；输出 bounded delta_xy + release logit；z/rotation 为 0；
- arms：R0 base；R1-CAP 参数匹配但不看视觉/语言；R2-PRED 可部署 predicted evidence；R3-GEO oracle geometry 仅诊断；
- 同数据、同初始化、同 batch、同 step、同 seed；
- 离线 Gate：cosine >=0.50、dominant-axis sign >=0.75、magnitude ratio 0.70-1.30、plateau/大修正桶不能更差、release precision/recall 必须报告；
- 闭环 Gate：R2 official success >=3/15、confirmation >=2/12、R2 比 R1-CAP 至少多 2/15、released success >=2/15、seen group-any 不低于 10/15、wrong-destination 不超过 B0；
- P1 失败按 66 的失败分支处理，不继续调容量或换一个 token。

P0 FAIL 时停止 P1，先做 STUDY_SCENE1 caddy 的 zero-shot headroom audit；caddy 没有 headroom 就重新写 outer-loop PIVOT。

任何外部论文方法进入实验前，必须按 65 第 13.1/13.6 节下载论文 PDF/TXT、全文查找正文/脚注/附录/OpenReview 中的代码链接、clone 官方仓库并 pin commit、记录 license、写 paper_reading；缺材料则 arm 为 INVALID。不能凭印象实现外部方法。

资源与 git：
- 只用 GPU1，训练/评测/rollout 串行；
- 不占用 GPU0；
- 不修改 references/repos/starvla；
- 不提交 checkpoint、视频、safetensors、pt 和大日志；
- 每次只 git add 本次相关文件，不使用 git add -A；及时 commit，不 push 除非用户明确要求；
- 先写 P0 protocol 和 Run Contract，再执行；不要直接启动 P1 训练。

输出：
- P0 protocol、run contract、逐 episode 结果、P0 verdict；
- 明确 PASS/FAIL/INVALID；
- 报告 official success、released success、hold-only、released placement、wrong-destination、target-not-reached、final xy、release step；
- 如果 PASS，给出 P1 protocol 和下一阶段命令；如果 FAIL，给出 H-AUTH 的否证证据和 caddy/PIVOT 建议。
```

---

## 13. Git

本次只新增本文并更新 `65` 的 PIVOT 交叉引用；不提交 checkpoint、视频、safetensors、pt 和大数据。提交命令：

```bash
cd /home/bubble/类脑计算/VLM终局
git add doc/vla_next/65_story_method_selection_20261107.md \
        doc/vla_next/66_outer_loop_pivot_after_interface_failure_20261107.md
git diff --cached --check
git commit -m "docs(vla): add outer-loop pivot after interface failure"
```

不自动 push；需要同步远端时再执行 `git push origin main`。
