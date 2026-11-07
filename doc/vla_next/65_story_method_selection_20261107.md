# 65 故事、主任务与方法选择

**日期**：2026-11-07<br>
**状态**：升级规则与外部论文准入规范已落盘；截至 2026-11-07，Goal13 接口 Gate 为 FAIL，`W-I` 关闭，不启动 `A2/F0`<br>
**输入**：61–64 的审计、C1/J4.1/Goal11/Goal12/G2 的已有结果、2026 年近邻论文<br>
**一句话**：主任务继续用 C1，但论文故事必须是 **“证据为什么没有控制动作”**，而不是再讲一次 “我们的 grounding 更准”。方法采用 **Data → Interface → Control 三级因果阶梯**；FAI-VLA 只有在第三级成立时才能作为主方法。

---

## 1. 裁决先行

### 1.1 推荐结论

| 项目 | 结论 |
|---|---|
| 主任务 | C1 `LIVING_ROOM_SCENE5`：`red/white mug × left/right plate` |
| 主 held-out | `put the white mug on the right plate`，训练计数必须为 0 |
| 主指标 | `white_right` official success；第一目标是从 `0/15` 到 `>=1/15` |
| 机制指标 | strict destination、wrong-destination rate、correct-vs-wrong action margin、factor sufficiency/necessity |
| 第一备选 | STUDY_SCENE1 caddy 四隔间任务，作为第二场景/外部验证 |
| 主故事 | **VLA 已经能把证据编码进表征，但证据没有成为动作的控制变量；我们用 factorial 数据和三层干预定位断点。** |
| 主方法候选 | 暂称 `Data -> Interface -> Control` 因果阶梯；最终方法名不抢跑，第三级通过后再称 FAI-VLA |
| 完整候选结果 | `D1` 因子覆盖 + `I2` predicted marker + `A2` counterfactual action constraint |
| 不应该做的事 | 继续在旧 C1 三 prompt 上单加 point/evidence token；追 `14/15`；把 marker 或反事实监督本身说成新贡献 |

### 1.2 故事的一句话版本

> 我们发现一个 VLA 可以“看见正确目标”却仍然把物体放到错误盘子。已有工作分别证明了 grounding gap、显式空间 marker、反事实监督和因子数据策略都有效；但它们没有回答一个更窄、更可归因的问题：**数据覆盖、显式空间接口、动作侧因果约束，究竟哪一层把 encoded evidence 变成了 action control？** 我们在一套受控的 factorial split 上逐层加入这三个干预，并把“因子对动作是否充分且必要”作为可检验的中间判据。

这句话的好处是：

1. 不声称“首次发现 grounding gap”；
2. 不声称“marker 或反事实监督是全新的”；
3. 即使最终只有数据层有效，也仍然是一个清楚的实证结论；
4. 如果第三层有效，才自然升级为方法论文。

### 1.3 调查方法与证据边界

本文不是“拍脑袋选方向”，而是按以下顺序做的：

| 步骤 | 使用的方法/skill | 产出 |
|---|---|---|
| 1 | 本地实验证据审计 | C1/J4.1/Goal11/Goal12/G2 的失败类型、0/15、14/15、wrong-destination、first-target 等数字 |
| 2 | `paper-search` 多查询检索 | 2024–2026 的 VLA 组合泛化、反事实监督、空间 marker、action query、因子偏置近邻 |
| 3 | `scoop-check` 的分解方式 | 把候选贡献拆成 problem framing / core mechanism / key insight / application domain，逐项找覆盖 |
| 4 | `idea-evaluator` 的五维评分 | Higher / Faster / Stronger / Cheaper / Broader + fatal-flaw 审计 |
| 5 | 任务数据审计 | C1 四 prompt、carrier、caddy、KITCHEN_SCENE2 的 episode 计数和 held-out 可行性 |

检索侧的主要命中包括 `Encoded but Not in Control`、`CRAFT`、`AnySlot`、`Scale Up Strategically`、`CounterAlign`、`CAST`、`LangForce`。2025–2026 的条目大多是 arXiv 预印本，论文数值只用于判断机制方向，不当作本项目可复现结论。

证据边界：

- 本轮没有启动训练、rollout 或 checkpoint 审计；
- C1 的远端 RLDS 统计来自前序只读扫描，正式训练前必须重新跑 Run Contract；
- 近邻论文的“已覆盖”判断只针对公开摘要/方法段，不声称是完整 prior-art 检索；
- 因此本文的裁决是“当前最值得执行的方案”，不是“已经证明唯一正确”。

---

## 2. 为什么主任务仍是 C1

### 2.1 C1 不是“普通低分任务”，而是天然的四格干预

C1 的训练与 held-out 可以写成标准的 `2 × 2` 因子表：

| source | left plate | right plate |
|---|---|---|
| red mug | 已见 | 已见 |
| white mug | 已见 | **held-out** |

这比“随便找一个 0/15 任务”更适合讲因果故事，因为：

- 同一初始状态可以换四个 prompt；
- source 与 destination 可以分别 swap；
- 失败几乎都在 destination，而不是抓取；
- 已有 evaluator、rollout manifest、P1 point head 和失败分类。

已有结果中最关键的一组数字是：

- `white_right` official success：`0/15`；
- target object correct：`14/15`；
- wrong destination：`14/14`。

这说明主失败不是“看不见白杯”，而是“抓对白杯后没有把右盘当成控制目标”。这正是“encoded but not in control”的干净实例。

### 2.2 C1 的局限必须写进故事

C1 只有四个 prompt，而且 `white porcelain mug -> right` 在真实数据中完全没有。它适合做**机制发现**，不足以单独支撑“通用组合泛化”主张。因此：

- C1 是 development task / mechanism task；
- STUDY_SCENE1 caddy 是第二场景；
- LIBERO-CF / PRO / Plus 只做外部构念效度；
- 最终论文不能说“方法在通用 LIBERO 上全面领先”，只能说“在受控 factorial split 上，某层干预使 held-out 组合出现了可复核提升”。

### 2.3 为什么不换成 caddy 作为第一主任务

STUDY_SCENE1 caddy 有 371 条 episode，四个隔间关系也更均衡，但它目前缺少：

- 本项目已有的零样本 headroom 证据；
- 现成 point head / sidecar；
- 同一 init 的四 prompt 评测链；
- 与 J4.1/Goal12 可比的失败分类。

所以 caddy 是**第一备选和外部验证**，不是第一天的主任务。只有当 C1 数据不可用、S0 直接 ceiling，或 C1 连续失败且需要验证“不是单任务特例”时才切换。

---

## 3. 文献裁决：哪些话不能再说

### 3.1 直接近邻

| 论文 | 已经覆盖什么 | 我们不能声称什么 |
|---|---|---|
| **Encoded but Not in Control** `2610.06235` | 语言被编码但没有控制动作；attention / action lens 诊断 | 不能声称首次发现 “encoded but not control” |
| **CRAFT** `2610.00524` | 同图反事实指令 + skill representation transfer，改善未见 skill 组合 | 不能把“反事实指令 + 动作监督”本身当新方法 |
| **AnySlot** `2604.10432` | 用户给定 slot marker，goal-conditioned VLA 做 slot-level placement | 不能把“显式视觉 marker”本身当新贡献 |
| **Scale Up Strategically** `2607.21582` | factor bias、FDR/FDH、bias-aware data collection | 不能把“因子诊断 + 按因子补数据”当新方法 |
| **CounterAlign** `2608.21740` | 反事实负监督 + discriminator/AWR | 不能把 correct-vs-wrong 动作排序本身当新方法 |
| **CAST** `2508.13446` | 反事实标签改善 instruction following | 不能把反事实标签当新数据贡献 |
| **CAG / When Vision Overrides Language** `2602.17659` | training-free 反事实推理和双分支选择 | 不能把推理时反事实选择当主方法 |
| **LangForce** `2601.15197` | 条件互信息坍缩；latent action queries + PMI | 不能把“语言条件互信息不足”当首次发现 |
| **AC-VLA** `2607.15714` | 指令分解、轨迹对齐、组合数据混合 | 不能把 phase/子任务分解本身当新方法 |
| **InternVLA-M1 / Pointing-VLA** | 空间 grounding 预训练、点/区域读入动作 | 不能把“where to act 再 how to act”当新框架 |

### 3.2 可辩护的 delta

我们不把单点模块当贡献，而是把贡献写成一个**受控的因果分解**：

> 已有工作分别证明数据覆盖、显式空间接口和反事实动作监督可以各自改善组合泛化；但没有在同一任务、同一 backbone、同一评测下，把三者作为逐层干预，判断“encoded evidence 在哪一层失去对动作的控制”。我们进一步把这一步操作化为 factor-wise action sufficiency/necessity，并要求 predicted slots 在部署时独立满足该判据。

这条 delta 的成立条件很严格：

- `D0/D1` 必须 matched；
- `I1/I2` 必须 matched；
- `A1/A2` 必须 matched；
- `A2` 必须相对 `I2` 有增益；
- 必须有一个第二场景或外部 benchmark 同向；
- 不能只在 seen prompt 上有效。

如果这些条件不成立，本文应降级为“诊断/实证研究”，不能包装成新方法论文。

---

## 4. 方法选择：四级方案对比

### 4.1 路线 1：Scale-Up 式因子数据

**做什么**：用 FDR/FDH 或简单 factorial 统计找欠覆盖因子，按因子补数据，普通 BC 训练。

**对应故事**：失败主要来自数据覆盖不足。

**优点**：最便宜、最容易出工程结果；不需要改 action head。

**缺点**：`Scale Up Strategically` 已经覆盖；单独做只能当 baseline 或 workshop 结果。

**结论**：必须作为 `D1` 臂保留，不能作为主方法。

### 4.2 路线 2：AnySlot 式显式 marker

**做什么**：VLM/point head 预测 source/destination，渲染成固定 marker，action expert 直接读图像 marker。

**对应故事**：失败主要来自证据被 pooling 压扁，显式空间接口即可解决。

**优点**：工程直观，和已有 P1 point head 最匹配；oracle marker 能快速判断 action head 是否有 headroom。

**缺点**：AnySlot 已覆盖 marker；如果只做这个，新颖性不足。

**结论**：作为 `I2` 主接口，必须有 oracle `I1` 和 none/shuffle 负控。

### 4.3 路线 3：CRAFT/CounterAlign 式反事实动作监督

**做什么**：同图换 instruction，构造 correct-vs-wrong 动作对；用 ranking / discriminator / skill representation transfer 迫使动作依赖语言。

**对应故事**：失败主要来自监督目标只要求“拟合专家动作”，没有要求“错误语言产生更差动作”。

**优点**：已有工作显示这是高概率有效的机制；C1 四 prompt 天然支持。

**缺点**：直接复用新颖性低；如果只做 C，审稿人会问和 CRAFT/CounterAlign 的区别。

**结论**：作为 `A1` 臂和强 baseline；只有再加 factor-wise 判据才可能成为主方法。

### 4.4 路线 4：Data -> Interface -> Control 因子化因果阶梯

**做什么**：在 B 的 predicted marker 上，增加 factor-wise swap、sufficiency、necessity、selectivity；每个因子有独立 token / 独立干预路径。

**对应故事**：失败主要来自动作没有把每个语言因子当成独立控制变量。

**优点**：故事最完整；能把“看对”与“用对”分开；能给出机制图和失败解释。

**缺点**：实现最多；如果 `A2` 不胜 `I2`，方法贡献不成立；需要第二场景。

**结论**：作为论文主候选，但必须 stage-gated，不能一上来全量训练。

### 4.5 评分与裁决

按 `idea-evaluator` 的五维框架做保守评分：

| 方案 | Higher | Faster | Stronger | Cheaper | Broader | 裁决 |
|---|---:|---:|---:|---:|---:|---|
| 路线 1：数据覆盖 | 6 | 8 | 5 | 8 | 5 | 只能做 baseline |
| 路线 2：marker | 6 | 7 | 6 | 7 | 6 | 工程臂，不单独成论文 |
| 路线 3：反事实动作 | 7 | 6 | 7 | 6 | 6 | 强 baseline / 快速结果 |
| 路线 4：D→I→A | 7 | 5 | 7 | 5 | 6 | **Accept with Revisions** |

**裁决**：执行路线 4 的故事框架，但按路线 1→2→3 的顺序逐层加干预。这样即使最终方法没有胜出，也能得到清楚的归因结论；如果路线 4 胜出，再把它写成方法论文。

---

## 5. 核心方法：Evidence-to-Control Ladder

### 5.1 三段因果链

```text
语言 + 图像
   │
   ├─ D：数据是否让每个因子可辨识？
   │      训练覆盖 red/white × left/right 的足够组合
   │
   ├─ M：证据是否有可被动作头读取的显式接口？
   │      source point / destination point -> fixed visual marker
   │
   └─ C：动作是否真的把每个因子当成控制变量？
          factor swap + sufficiency + necessity + selectivity
```

### 5.2 三个中间判据

对每个因子 `f ∈ {source, destination, relation}`：

**Sufficiency（充分性）**

```text
correct slot 的动作损失 < null slot 的动作损失
```

含义：有正确证据时，动作应比没有证据时更好。

**Necessity（必要性）**

```text
wrong slot 的动作损失 > correct slot 的动作损失
```

含义：错误证据必须让动作变差，而不是“随便换个条件都一样”。

**Selectivity（选择性）**

```text
只换 destination，动作变化应集中在 transport / release 相关阶段；
只换 source，动作变化应集中在 approach / grasp 相关阶段；
换无关因子，动作变化应尽量小。
```

这里不预设“destination 只改 xy”。正确做法是记录完整 action chunk 和 phase 的响应，再报告哪些维度/阶段实际变化。此前把 destination 简化成“横向修正量”只是一种任务直觉，不是方法假设。

### 5.3 训练目标

最终候选 `A2` 的目标函数：

```text
L = L_BC
  + λ_slot  * L_slot_grounding
  + λ_swap  * L_factor_swap
  + λ_suff  * L_sufficiency
  + λ_nec   * L_necessity
  + λ_div   * L_slot_diversity
```

其中：

- `L_slot_grounding` 只在训练时用 GT mask/point，部署不使用；
- `L_factor_swap` 用同图/同轨迹的因子替换构造；
- `L_sufficiency` 和 `L_necessity` 作用在 action expert 的 flow/动作损失上；
- `L_slot_diversity` 防止三个 slot 塌成一个全局向量。

必须同时记录：

```text
slot_grad_norm / action_grad_norm
swap_grad_norm / action_grad_norm
suff_grad_norm / action_grad_norm
nec_grad_norm / action_grad_norm
```

只看 loss 数值不足以下结论；上一轮 retention 的教训就是“损失项存在”不等于“函数级效果存在”。

---

## 6. 实验阶梯：每一臂只回答一个问题

### 6.1 数据层

| arm | 训练数据 | 要回答的问题 |
|---|---|---|
| `D0` | 原 C1 三 prompt：red-left、red-right、white-left | 现有 0/15 是否能在 matched 复现 |
| `D1` | `D0` + yellow-and-white-right carrier | 单纯补“白色源物体 -> 右盘”覆盖是否足够 |
| `D2`（可选） | `D1` + white-bowl-right 低权重 | 更宽的白色物体因子是否有额外收益 |

`D0` 必须存在。没有它，就无法判断提升来自“新方法”还是“只补了一条数据”。

### 6.2 接口层

| arm | 条件 | 要回答的问题 |
|---|---|---|
| `I0` | 无 marker，普通 BC | 与 D 层匹配的基线 |
| `I1` | oracle marker（GT point） | action head 是否有读取显式空间条件的能力 |
| `I2` | predicted marker（P1） | 预测出来的空间证据是否能达到 oracle 的一部分收益 |
| `I3` | shuffle marker / wrong marker | marker 是否只是“多了一个视觉输入”，而不是真的有语义 |

`I1` 是诊断臂，不是部署方法；`I2` 才是候选方法。

### 6.3 控制层

| arm | 条件 | 要回答的问题 |
|---|---|---|
| `A0` | `I2` + 普通 BC | predicted marker 本身是否已经足够 |
| `A1` | `A0` + counterfactual action ranking | correct instruction 的动作是否优于 wrong instruction |
| `A2` | `A1` + sufficiency/necessity | 每个因子是否成为动作的必要/充分条件 |
| `A3` | `A2` + Cofact/去混淆正则（可选） | 视觉捷径是否仍然存在 |

### 6.4 结果解释表

实验不是 `D × I × A` 全交叉。每一层只把上一层的 best arm 带入下一层，否则单卡预算会被组合爆炸吃掉：

```text
D0 vs D1
  -> 取 best data arm
  -> I0 vs I1 vs I2 vs I3（同一 data arm）
  -> 取 best interface arm
  -> A0 vs A1 vs A2（同一 interface arm）
```

这样每个比较只改变一个层，故事才可归因。

| 观察 | 故事解释 | 下一步 |
|---|---|---|
| `D1 > D0`，`I2` 不再提升 | 主要瓶颈是数据覆盖/因子偏置 | 保留 D1；可做 data paper；不要把 marker 写成主贡献 |
| `I1` 有收益、`I2` 没有 | action head 有 headroom，但预测器/接口不可用 | 修 P1 或 marker 接口；不要直接加 A2 |
| `I2` 有收益、`A0` 无额外收益 | 显式空间接口已经足够 | 方法可以是 marker + factorial data；A2 只作机制分析 |
| `A2 > A0`，且 suffice/necessity 为正 | 动作侧因果约束是必要层 | 进入 FAI-VLA 主方法候选 |
| 所有层都无收益 | 任务/假设/执行链有问题 | 先查 data contract 和 rollout；必要时切 caddy |

这张表是本文“故事讲明白”的核心：每个结果都有唯一解释，不允许事后改故事。

---

## 7. 主任务与第二场景

### 7.1 C1 主实验

默认设置：

```text
场景：LIVING_ROOM_SCENE5
训练：red-left / red-right / white-left（+ 可选 carrier）
held-out：white-right
评测：15 init × 4 prompt；先 pilot，再 30/50 条确认
```

硬规则：

- `put the white mug on the right plate` 训练计数永远为 0；
- validation/rollout init state 不进入训练；
- 训练时 GT mask/point 只做 loss，不作为部署输入；
- 每个 arm 先过 Run Contract，`INVALID` 不进入 Gate 判读。

### 7.2 caddy 第二场景

只在 Stage 1 出现正信号后启动，用来回答“方法是否只对 C1 有效”。

```text
STUDY_SCENE1 caddy：
front / left / right / back compartment
```

它的优势是数据量更大、四个 destination 更均衡；缺点是 source 固定、没有现成 point/sidecar，因此首轮成本高。它适合做“destination factor 的第二个验证”，不适合替代 C1 的机制发现。

### 7.3 外部验证

优先级：

1. C1 caddy 第二场景；
2. LIBERO-CF target swap；
3. LIBERO-PRO / LIBERO-Plus Language/Layout；
4. U5 source-swap 只做诊断，不做训练主表。

---

## 8. Gate 与 Run Contract

### 8.1 每个 arm 必须填写的字段

| 字段 | 要求 |
|---|---|
| `code_commit` / `config_sha256` | 与实际执行一致 |
| `data_arm` | `D0/D1/D2` 明确 |
| `parent_arm` | 本层继承的上一层 best arm |
| `train_language_counts` | 每个 prompt 的 episode 数 |
| `heldout_prompt_train_count` | 必须为 0 |
| `train_validation_rollout_overlap` | 必须为 0 |
| `sampling_weights` | 实际打印，不只看 config |
| `effective_batch` | batch × accumulation |
| `marker_mode` | `none/oracle/predicted/shuffle` |
| `counterfactual_mode` | `none/rank/suff_nec` |
| `seed` | 固定并可复现 |
| `grad_norm_ratios` | slot/swap/suff/nec 与 action 的比值 |

状态只能是：

- `PASS`：可以进入 Gate；
- `FAIL`：配置正确但结果未达标；
- `INVALID`：数据/配置/评测不满足契约，不能判方法失败。

### 8.2 Gate

**G0 数据**

- [ ] `D0` 三 prompt 计数与历史一致；
- [ ] `white_right` 训练计数 = 0；
- [ ] carrier 至少 40 episodes，100 帧人工抽检通过；
- [ ] train/validation/rollout init 交集 = 0；
- [ ] 实际 sampling weights 打印且与计划一致。

**G1 数据层**

- [ ] `D0` 复现 `white_right = 0/15` 或解释差异；
- [ ] `D1` 相对 `D0` 的 paired gain 有方向性；
- [ ] seen group-any 不退化超过预注册阈值。

**G2 接口层**

- [ ] `I1` oracle marker 相对 `I0` 的 non-gripper MAE 改善 >=10%；
- [ ] `I2` 至少达到 `I1` 收益的一部分；
- [ ] `I2` 优于 shuffle/none；
- [ ] 动作幅度、rotation、gripper 不塌缩。

**G3 控制层**

- [ ] `A2` 相对 `A0` 的 held-out success 至少 `+5pt`，或 suff/nec margin 明确为正；
- [ ] destination swap 的动作响应大于无关因子；
- [ ] wrong-destination 不增加；
- [ ] 至少 2 个 seed 同向；
- [ ] `A2` 不胜 `I2` 时，不能声称 factor-wise 贡献。

**G4 外部**

- [ ] caddy 或 LIBERO-CF 至少一个同向；
- [ ] 30/50 条确认，而不是只靠 15 条 pilot；
- [ ] 失败分类显示 wrong-destination 下降，而不是变成 no-destination。

---

## 9. 资源与执行顺序

```text
Stage 0：数据/契约/抽检             0.5–1 day
  ↓
Stage 1a：D0 vs D1                     0.5–1 day
Stage 1b：best D + I0/I1/I2/I3 小 pilot 1–1.5 days
  ↓
Stage 2：A0/A1/A2                     1–2 days
  ↓
Stage 3：30/50 确认 + caddy/外部      3–5 days
```

规则：

- 只用 GPU1；训练/评测串行；
- 每阶段先 offline limit=2 smoke，再 pilot；
- 所有 checkpoint/日志/manifest 写远端结果目录；
- 不修改 `references/repos/starvla`；
- 不提交 checkpoint、视频、safetensors、pt 和大数据；
- 不 commit/push/建分支，除非用户明确要求。

---

## 10. 主要风险与处理

| 风险 | 影响 | 处理 |
|---|---|---|
| C1 carrier 没有 GT mask | I2 只能 pseudo point | 标记 `weak_sidecar`，I1 只做 oracle 诊断；必要时下载 HDF5 |
| carrier 被学成“第三个物体” | D1 无效 | 加 white-bowl-right 低权重；检查 source swap 响应 |
| P1 point head 未见 carrier | I2 预测差 | 先做 point-in-mask / destination L1 抽检；不行先补 point 数据 |
| action head 不读 marker | I1 无收益 | 改通道拼接/FiLM/cross-attn；I1 失败就不要上 A2 |
| A2 只在 seen 有效 | 方法不成立 | 不扩 30/50；回退 I2 或只写诊断 |
| 只有 15 条 pilot | 结论不稳 | 15 条只作筛选；正式结论用 30/50 + 3 seeds |
| 单场景过拟合 | 论文 scope 太窄 | caddy 或 LIBERO-CF 同向后才写“泛化” |
| 因子判据只停在 loss | 伪机制 | 必须报告 action sensitivity、phase response 和闭环 |

---

## 11. 最终建议

### 11.1 现在做什么

先做 **Stage 0 + Stage 1 的 D0/D1，再在 best data arm 上做 I0/I1/I2/I3**，不要直接上 A2。

理由：

- `D0` 决定旧结果是否可信；
- `D1` 回答“是不是只差数据”；
- `I1` 回答“动作头有没有能力读显式空间条件”；
- `I2` 回答“预测器能不能把这份能力兑现”；
- 只有 `I1/I2` 有信号，`A2` 才值得做。

### 11.2 什么时候升级为 FAI-VLA

同时满足以下三条才升级：

1. `I2` 至少复现 `I1` 的一部分收益；
2. `A2` 相对 `A0` 有 held-out 增益或明确的 suff/nec margin；
3. 第二场景/外部 benchmark 同向。

否则：

- 如果 `D1` 有效：写成“因子数据覆盖”实证结果；
- 如果 `I2` 有效但 `A2` 无效：写成“显式空间接口足够”的系统结果；
- 如果都无效：写失败分析并切换 caddy，不要继续堆模块。

### 11.3 一句话给负责人

**我们要做的不是“再加一个 binding 模块”，而是用 C1 这个受控四格任务，证明证据到底卡在数据、接口还是动作控制；FAI-VLA 只是第三层通过后的方法名，不是现在就要押注的结论。**

### 11.4 什么时候可以把组件升级成“论文级方案”

组件升级不是看“代码写完了”或“loss 降了”，而是看它是否已经通过可复核的因果 Gate。建议分成四级：

| 级别 | 形态 | 进入条件 | 可以写成什么 |
|---|---|---|---|
| L0 | 诊断/假设 | 有 matched baseline 和预注册指标 | 机制分析、负结果、workshop 诊断 |
| L1 | 单层组件 | D、I、A 中至少一层相对上一层有稳定增益 | 数据/接口/控制三选一的实证论文 |
| L2 | 完整方法候选 `F0` | `A2` 通过，且 D/I/A 的增益能叠加 | FAI-VLA 方法稿候选 |
| L3 | 论文级完整方案 | L2 + 3 seeds + 第二场景/外部 benchmark + 完整消融 | 主会方法论文 |

#### 11.4.1 完整 FAI-VLA 的合并时机

只有 `A2` 通过后，才允许把 `AC-VLA-lite + CompoSlot/marker + factor sufficiency/necessity` 合成完整 `F0`：

```text
F0 = D1（因子数据覆盖）
   + I2（predicted marker）
   + A2（factor-wise sufficiency/necessity）
```

`F0` 不是“再训练一个大杂烩”，而是合成后的主方法臂。必须同时跑以下消融：

| arm | 组合 | 回答什么 |
|---|---|---|
| `F0` | D1 + I2 + A2 | 完整方法是否成立 |
| `F1` | D0 + I2 + A2 | 去掉因子数据覆盖是否变差 |
| `F2` | D1 + I0 + A2 | 去掉显式 marker 是否变差 |
| `F3` | D1 + I2 + A0 | 去掉动作侧因果约束是否变差 |
| `F4` | D1 + I2 + A2 去掉 phase/AC-VLA-lite | phase/阶段监督是否必要 |

合并 Gate：

- `F0` 必须优于 `F1/F2/F3` 中至少两个；
- `A2` 的增益不能只来自更多参数或更长训练；
- `F4` 不胜 `F0` 才可声称 phase/AC-VLA-lite 有贡献；
- `F0` 只在 seen 有效、held-out 无增益时，不得写成完整方法。

#### 11.4.2 什么才算“论文级”

`F0` 通过后，还需要以下证据才能从“实验方案”升级成“论文级方案”：

- [ ] 主表同时报告 held-out `white_right`、seen group-any、wrong-destination、strict destination；
- [ ] 30/50 条确认 + 3 seeds；
- [ ] 第二场景 caddy 或 LIBERO-CF 至少一个同向；
- [ ] 与 CRAFT / CounterAlign / AnySlot / Scale-Up 式 baseline 的可比对照；
- [ ] `F0/F1/F2/F3/F4` 消融完整；
- [ ] 机制图包含 suff/nec/selectivity 和失败分类；
- [ ] 复现实验包含 config SHA、commit、环境、随机种子和失败日志；
- [ ] 如果代码/数据不能公开，必须写清 release boundary，不能假装可复现。

如果只通过 L1：

- 只有 D 层有效：写成“因子数据覆盖”的实证结果，不硬造 FAI-VLA；
- 只有 I 层有效：写成“显式空间接口足够”的系统结果，不硬加 causal loss；
- 只有 A 层有效：才进入 `F0` 合并和 FAI-VLA 方法稿。

#### 11.4.3 组件级升级时间表

“升级到更像论文”不是把更多模块塞进来，而是把当前 lite 版替换成原文的完整机制。每个组件都必须先通过本层 Gate，再升级；升级前必须完成第 13.1 节的检查清单和第 13.6 节的论文/代码准入流程。

| 组件 | 当前 lite 版 | 论文近似版 | 最早升级时点 | 升级前必须看到的证据 |
|---|---|---|---|---|
| 因子数据 `D1/D2` | 按 prompt 补数据 + 固定采样权重 | FDR/FDH 式因子诊断、bias-aware sampling、预算对齐的混合训练 | Stage 1a 的 `D0/D1` matched smoke/pilot 后；只有 `D1 > D0` 才升级 | `D1 > D0`；seen group-any 不明显退化；`white_right` 训练计数仍为 0 |
| 显式接口 `I1/I2` | 单点/fixed marker，P1 pseudo point | AnySlot 式 slot marker、区域/HSV 提取、slot-level spatial token | `I1 > I0` 且 non-gripper MAE 改善 >=10%，同时 `I2 > I3` 后；`I1` 无效先修 action head | marker 真正进入动作；`I2` 不塌缩；point-in-mask 抽检通过 |
| 反事实动作 `A1/A2` | ranking + sufficiency/necessity loss | CRAFT/CounterAlign/CAST 式配对构造、negative mining、discriminator/AWR（逐项按原文核对） | `A1 > A0` 后才升级；完整 `A2` 只在 held-out 和 margin 同向时升级 | correct-vs-wrong margin 为正；wrong-destination 不升高；suff/nec 梯度比可解释 |
| `AC-VLA-lite` phase | phase 分段 + 简化轨迹对齐 | AC-VLA 原文 instruction decomposition、trajectory aligner、混合任务训练 | `A2` 通过且 `F4` 证明 phase 有贡献后才升级；`A2` 前禁止合并 | `F4` 不优于 `F0`；phase 消融成立；不是更多参数或更长训练造成的假增益 |
| `CompoSlot` | `I2` predicted marker | slot-centric compositional representation + slot diversity | `I2 > I0/I3` 稳定成立后；`A2` 前只作为 I 层组件升级 | slot 不塌缩；wrong slot 明显更差；部署不使用 GT mask/point |
| `A3` 去混淆（可选） | 可选去混淆正则 | CofactVLA 式 language-masked/双分支反事实推理 | `A2` 通过，且 wrong-destination/visual shortcut 仍明显时 | wrong-destination 下降；seen 不退化；增益不能与更多参数混淆 |
| 完整 `F0` FAI-VLA | `D1 + I2 + A2` | 完整 AC-VLA + CompoSlot + factor constraints | `A2` PASS 且 D/I/A 增益可叠加后，才进入 `F0` | `F0` 优于 `F1/F2/F3` 中至少两个；`F4` 消融成立；第二场景同向 |

升级顺序固定为：

```text
D Gate 通过 -> 只允许升级数据组件
I Gate 通过 -> 只允许升级接口/marker 组件
A Gate 通过 -> 才允许升级 A/F0 组件
F4 消融成立 -> 才允许升级 AC-VLA-lite 的 phase 机制
```

任何组件升级都必须作为新 arm 重新登记 `parent_arm`、config SHA、论文/仓库来源和 Gate，不能静默替换旧实现。为了“更像论文”而提前升级、但没有本层 Gate 证据的 arm，标为 `INVALID`，不得写入主结论。

#### 11.4.4 升级窗口：什么实验之后才允许更像论文

这里的“什么时候”以实验 Gate 为时钟，不按日历计算。没有对应 Gate 的 PASS，组件就冻结在当前 lite 版；`smoke PASS` 只说明接口能跑通，不允许据此升级方法。一个组件只有在下列窗口开启后，才允许替换成更接近论文原版的实现。

| 窗口 | 开启条件（全部满足） | 最早允许升级到什么 | 仍然禁止 |
|---|---|---|---|
| `W-D` 数据组件 | `D0` vs `D1` 使用同一 Run Contract 完成 pilot；`D1` 在 held-out 或因子可辨识指标上优于 `D0`；seen group-any 无明显退化；`white_right` 训练计数仍为 0 | FDR/FDH 式因子诊断、bias-aware sampling、预算对齐的混合训练 | 同时加 marker/反事实；用 pilot 结果直接宣称论文级结论 |
| `W-I` 接口组件 | 固定 best data arm；`I1 > I0`，证明动作头确实读取 marker；`I2 > I3`，证明 predicted marker 不塌缩；point-in-mask 抽检通过 | AnySlot/CompoSlot 式 slot marker、区域/HSV 提取、slot-level spatial token | 把 GT mask/point 放进部署输入；在 `I1` 无效时硬上 `I2` |
| `W-A` 动作组件 | 固定 best interface arm；`A1 > A0`；随后 `A2` 在 held-out 和 suff/nec margin 上同向 | CRAFT/CounterAlign/CAST 式配对构造、negative mining、discriminator/AWR（逐项按原文核对） | `A2` 未过就合并 phase；用更多参数或更长训练冒充因果增益 |
| `W-F` 完整方法 | `A2` PASS；`F0` 至少优于 `F1/F2/F3` 中两个；`F4` 消融显示 phase 有贡献；第 13.1/13.6 节论文/代码检查完成 | 完整 `F0`、AC-VLA phase、CompoSlot 组合版 | 在 3 seeds + 第二场景前写“L3/主会方法论文已成立” |

**当前执行状态（2026-11-07，Goal13；接口层已完成判读）**：

- `D0` 契约 PASS，固定为 best data arm；`D1` 契约也 PASS，但结果方向为负，`white_right` 仍为 `0/15`，C1 seen group-any 从 `13/15` 降到 `9/15`。`W-D` 关闭，`D2` 跳过，数据组件保持 lite 版。
- 接口层已完成 `I0/I2` 的 500-step continuation 和 `I1/I2/I3` 的完整 118 groups 离线评测；所有 arm 的 Run Contract 为 PASS，但 `gate_status=DESCRIPTIVE` 只表示结果可描述，不表示 Gate 通过。
- `G-I1` **FAIL**：`I1` oracle marker 没有优于 `I0`；`G-I2` **FAIL**：`I2` predicted marker 没有优于 `I3` shuffle。完整数字如下：

| arm | destination correct MAE | non-gripper MAE（由 7 维动作重算） | destination margin | 结论 |
|---|---:|---:|---:|---|
| `I0` none | `0.144959` | `0.149199` | `+0.011656` | matched baseline |
| `I1` oracle | `0.148985` | `0.152426` | `+0.010116` | 比 `I0` 差约 `+2.16%`，无动作头 headroom |
| `I2` predicted | `0.149226` | `0.152723` | `+0.010700` | 不优于 shuffle |
| `I3` shuffle | `0.148852` | `0.152197` | `+0.010773` | 负控不劣于 predicted |

- point head 本身不是主要瓶颈：predicted point 的 destination L1 约 `2.645 px`、object L1 约 `7.807 px`，point-in-mask 为 `0.8008`；但 point ablation 中 oracle 的 MAE 比 none 差 `0.000384`，predicted 的 MAE 比 shuffle 差 `0.000237`（JSON 字段 `predicted_minus_shuffle=-0.000237` 的实际定义是 `shuffle_mae - predicted_mae`，正数才表示 predicted 更好），完美 oracle marker 也没有改善动作。因此当前结论是“marker 没有被 action expert 有效读取”，不是“point 预测太差”。
- `W-I` **不开启**，接口组件冻结在 lite 版；`W-A`、`W-F` 以及 `A2/A3/F0` 均不得启动。不能因为论文里使用 cross-attention、slot 或 factor loss，就跳过 `I1 > I0` 直接升级。
- 下一步只允许一次**注入修复** arm，例如 `I2-XATTN`：保持与 `I2` 相同的 `D0` 数据、初始化、500 steps、batch 4、seed 17，只把 marker 注入方式改为 `action_model.evidence_cross_attention: true`；先跑 `I1/I2/I3` limit=2 smoke，再跑完整 118 groups。若仍无 `I1 > I0` / `I2 > I3`，只允许再做一次 `evidence_token`/append 类注入修复，随后停止这条接口路线，不再用更长训练或“更像论文”掩盖失败。
- `D1` 的负结果必须保留为解释材料，不能因为流程上 PASS 就改写成“方法失败”或“数据无用”。接口数字见 `experiments/Goal13-autoresearch-20261107/12_interface_result.md`，数据数字见 `10_d1_matched_result.md`。

**现在能否升级为论文近似版（2026-11-07）**

| 组件 | 当前层级 | Gate 状态 | 现在能否升级 | 解锁条件 |
|---|---|---|---|---|
| 数据配方 | `D0` lite | `W-D` 关闭 | 否 | `D1` 需在 held-out 或因子可辨识指标上优于 `D0`，且 seen 不退化 |
| point/marker 接口 | `I0/I1/I2/I3` lite | `W-I` 关闭；`G-I1/G-I2 FAIL` | 否 | 先完成一次注入修复并重新通过 `I1 > I0`、`I2 > I3` |
| 反事实动作 | 未启动 | `W-A` 关闭 | 否 | 接口 Gate PASS 后，先证明 `A1 > A0` |
| `AC-VLA-lite` phase | 未合并 | `W-F` 关闭 | 否 | `A2` PASS，且 `F4` 消融证明 phase 有贡献 |
| `CompoSlot` | 未升级 | `W-I` 关闭 | 否 | `I2` 相对 `I0/I3` 稳定有增益，slot 不塌缩 |
| `A3` 去混淆 | 可选，未启动 | 未进入 | 否 | `A2` 通过且仍有 visual shortcut |
| 完整 `F0` | 未合并 | 未进入 | 否 | `A2` PASS，`F0 > F1/F2/F3` 中至少两个，第二场景同向 |

执行时间顺序固定为：

```text
Stage 0: D0 Run Contract PASS
  -> D0 matched pilot
  -> D1 carrier + matched pilot
  -> W-D 开启：允许升级数据组件
  -> I0/I1/I2/I3 pilot
  -> W-I 开启：允许升级接口/marker 组件
  -> A0/A1/A2 pilot
  -> W-A 开启：允许升级动作组件
  -> F0/F1/F2/F3/F4
  -> W-F 开启：允许合并完整方法
  -> 30/50 + 3 seeds + 第二场景：才允许写 L3
```

冻结和回滚规则：

- Gate 为 `FAIL` 或 `INVALID` 时，对应组件冻结在 lite 版；先修数据、契约或接口，不得用“更像论文”掩盖失败。
- 每个升级必须是独立 arm，登记 `parent_arm`、config SHA、Gate、论文来源、官方仓库 URL 和 pin commit；不能直接覆盖旧实现。
- 升级后的第一次运行只算 smoke，必须与 parent 使用同一数据、evaluator、batch 和 seed；若退化，回滚到 parent，并把升级标记为失败证据。
- pilot 只能支持“是否进入下一层”的工程决策；论文级结论必须等 30/50 条确认、3 seeds 和第二场景/外部 benchmark。

**一句话**：`AC-VLA-lite`、`CompoSlot` 和 `factor sufficiency/necessity` 不是永久绑在一起；它们是三个可独立升级的组件，只有 `A2` 通过后才允许合并成完整 FAI-VLA。

#### 11.4.5 升级批次与材料门槛

下面的“批次”是执行顺序，不是日历承诺。只有上一批的 Gate 已经 PASS，且该批组件的外部材料包已经完整落盘，下一批才允许启动。当前（2026-11-07）处于 `P0`：`W-I` 关闭，只允许做一次接口注入修复，不允许把任何组件换成“论文近似版”。

| 批次 | 触发条件 | 允许升级的组件 | 启动前必须完成的材料检查 |
|---|---|---|---|
| `P0` 当前 | `G-I1/G-I2 FAIL`；`W-I` 关闭 | 无；只允许 `I2-XATTN` 或 `evidence_token` 注入修复 | 修复本身如果借用外部 cross-attention/query-token 设计，也要先下载对应论文 PDF/TXT、追到正文中的仓库链接、clone 并 pin commit，再写 `paper_reading` |
| `P1` 接口 | 注入修复后重新满足 `I1 > I0` 和 `I2 > I3`，且 point-in-mask 抽检通过 | `I2` marker -> AnySlot/CompoSlot 式 slot marker、区域/HSV 提取、slot-level spatial token | AnySlot/CompoSlot 论文 PDF/TXT 已下载；全文搜索到仓库链接或明确记录 `official_code_missing=true`；官方仓库已 clone、pin commit、记录 license 和入口文件 |
| `P2` 动作 | 固定 best interface arm 后，`A1 > A0`，且 `A2` 的 held-out 指标与 suff/nec margin 同向 | `A1/A2` -> CRAFT/CounterAlign/CAST 式配对构造、negative mining、discriminator/AWR | 每引入一个外部 loss 或 pair 构造，都要单独下载对应论文和官方代码；`paper_reading` 必须逐项写出“原文做法 / 我们做法 / 差异 / 风险” |
| `P3` 完整方法 | `A2` PASS，且 `F0` 至少优于 `F1/F2/F3` 中两个，`F4` 消融显示 phase 有贡献 | `AC-VLA-lite` phase + `CompoSlot` + factor constraints 合并为 `F0` | AC-VLA、CompoSlot 及其他被合并组件的材料包全部齐全；每个升级 arm 登记 `parent_arm`、config SHA、论文来源、仓库 URL、pin commit |
| `P4` 论文级 | `L2` 成立，并完成 30/50 条确认、3 seeds、第二场景/外部 benchmark | 不再新增组件；只做复现、消融和证据补强 | 所有 baseline、消融和外部方法都能追到论文、仓库 commit 和运行命令；缺材料的 arm 只能标 `INVALID` |

“材料包齐全”不是一句口头要求，而是以下四件套同时存在：

1. `papers/{slug}.pdf` 和 `papers/{slug}.txt` 已在本机；
2. 已在论文摘要、方法、实验、脚注、附录、补充材料和 OpenReview 附件中搜索过代码链接，不能只搜首页；
3. 找到的官方仓库已 clone 到 `references/repos/{slug}`，并记录 `repo_found_in`、`repo_url`、`repo_commit`、`license`、`official`；
4. `references/paper_reading/{slug}.md` 已写完，明确 `faithful_reproduction | adapted_reproduction | pseudocode` 以及我们与原文的差异。

固定执行顺序只能是：

```text
下载论文 PDF/TXT
  -> 全文定位论文正文中的代码链接
  -> clone 官方仓库并 pin commit
  -> 读完 config / loss / data pipeline / train-eval 入口
  -> 写 paper_reading 和 repositories.yaml
  -> 再改我们自己的代码
  -> 最后才启动新 arm
```

不能先凭印象写一版实现，再回头补论文和代码；也不能用“论文里大概是这样”代替本地材料。缺少四件套中任何一项，该 arm 的 Run Contract 必须是 `INVALID`，不进入 Gate 判读。详细操作见第 13.6 节。

---

## 12. 下一阶段 Goal 提示词

### 12.1 固定阶梯版（默认）

```text
你在 /home/bubble/类脑计算/VLM终局 工作。请先阅读：
- doc/vla_next/61_vla_idea_viability_audit_and_pivot_20261107.md
- doc/vla_next/62_fai_vla_fusion_plan_20261107.md
- doc/vla_next/63_result_first_fai_staged_plan_20261107.md
- doc/vla_next/64_main_task_selection_20261107.md
- doc/vla_next/65_story_method_selection_20261107.md

目标：只做 Stage 0 和 Stage 1 的数据层与接口层，不要直接启动 A2/FAI-VLA 全量训练。

故事锁定：
VLA 可能已经“看见”正确目标，却没有让证据控制动作。我们要用 Data -> Interface -> Control 三级阶梯定位断点，而不是继续加 point token 或追 14/15。

主任务：
- C1 LIVING_ROOM_SCENE5，held-out 固定为 put the white mug on the right plate；
- 训练计数必须为 0；
- D0：原 C1 三 prompt；D1：+ yellow_and_white_right carrier；D2 可选 white_bowl_right 低权重；
- I0：无 marker；I1：oracle marker；I2：P1 predicted marker；I3：shuffle/wrong marker 负控。

必须做：
1. 先写 Run Contract：data_arm、language counts、heldout 计数、sampling weights、effective batch、seed、code/config SHA、marker_mode、grad norm ratios；不满足则 INVALID。
2. 先跑 D0 复现，再跑 D1；先 offline limit=2 smoke，再小 pilot。
3. 取 best data arm 后，再跑 I0/I1/I2/I3；I1 只作诊断，I2 才是候选；不做 D×I 全交叉。
4. 记录 official success、strict destination、wrong-destination、first-target、correct-vs-wrong action margin、suff/nec/selectivity。
5. 允许迭代，但每次迭代前写假设、变量、baseline、Gate 和停止条件；不得事后改阈值。

硬约束：
- 只用 GPU1，训练/评测串行；
- 不修改 references/repos/starvla；
- 不 commit、不 push、不建分支；
- GT mask/point 只进训练 loss，不进部署输入；
- validation/rollout init state 不进训练；
- 15 条 pilot 只作筛选，正式结论用 30/50 + 3 seeds。
- 任何外部论文方法在使用前，必须按第 13.0/13.1/13.6 节下载论文 PDF/TXT、找到并下载官方代码仓库、pin commit、写 paper_reading；缺材料则该 arm 为 INVALID。

停止/转向：
- D0 不能复现 0/15：先查 data/rollout，不判方法失败；
- D1 无效但 I1 有效：进入接口层；
- I1/I2 无效：不要上 A2，先修 marker 接口；
- I2 有效而 A2 无效：回退 I2，不写 factor-wise 方法；
- C1 数据不可用或 S0 ceiling：按 64 的触发条件转 STUDY_SCENE1 caddy。

输出：stage0_data_report.md、stage1_interface_result.md，明确 PASS/FAIL/INVALID，并给出是否进入 A2 的裁决。
```

### 12.2 受约束 AutoResearch 版（允许自主迭代）

这个版本适合希望 agent 自己跑实验、自己查论文、自己调整方案的情况。它不是“无限自主”，而是“在预注册协议、预算和停止条件下自主”。

```text
你是 /home/bubble/类脑计算/VLM终局 的受约束 AutoResearch agent。你的目标不是无限跑实验，而是在给定预算内，尽可能得到一个可复核的正信号，或得到一个能解释失败的高质量负结果。

先读：
- doc/vla_next/61_vla_idea_viability_audit_and_pivot_20261107.md
- doc/vla_next/62_fai_vla_fusion_plan_20261107.md
- doc/vla_next/63_result_first_fai_staged_plan_20261107.md
- doc/vla_next/64_main_task_selection_20261107.md
- doc/vla_next/65_story_method_selection_20261107.md

总故事：
VLA 可能已经“看见”正确目标，却没有让证据控制动作。你要用 Data -> Interface -> Control 三级阶梯，判断断点到底在数据覆盖、显式接口，还是动作侧因果约束；不要继续追 14/15。

主任务：
- C1 LIVING_ROOM_SCENE5，held-out 固定为 put the white mug on the right plate；
- white_right 训练计数必须为 0；
- 默认数据臂：D0 原 C1 三 prompt；D1 加 yellow_and_white_right carrier；D2 可选 white_bowl_right；
- 默认接口臂：I0 无 marker；I1 oracle marker；I2 P1 predicted marker；I3 shuffle/wrong marker；
- 默认动作臂：A0 普通 BC；A1 反事实动作排序；A2 sufficiency/necessity；A3 可选去混淆。

一、工作记忆

在 experiments/Goal13-autoresearch-20261107/ 下维护：
- research-state.yaml
- research-log.md
- findings.md
- literature/survey.md
- to_human/
- experiments/{hypothesis-slug}/protocol.md
- experiments/{hypothesis-slug}/results/
- experiments/{hypothesis-slug}/analysis.md

research-state.yaml 至少包含：
- current_stage
- current_hypothesis
- best_arm
- parent_arm
- contract_status
- primary_metric
- key_metrics
- budget_used
- next_action
- blockers
- open_literature_questions

每轮开始先读 research-state.yaml 和 findings.md，恢复状态；每轮结束必须更新它们。

二、两循环节奏

INNER LOOP（单个可证伪实验）：
1. 只选一个最高优先级假设；
2. 先写 protocol：假设、改什么、baseline、预测、指标、Gate、停止条件、预计 GPU 时间；
3. 协议写入文件并记录时间戳/哈希；在结果出来前不得改 Gate；
4. 先 smoke，再 pilot；跑完先做数据/NaN/动作幅度/基线复现 sanity check；
5. 记录 CONFIRMATORY 与 EXPLORATORY 结果，不能混写；
6. 更新 research-state.yaml 和 research-log.md。

OUTER LOOP（每 3–5 个小实验，或结果异常/卡住时）：
1. 汇总最近结果，按 D/I/A 层聚类；
2. 问“为什么有效/为什么无效”，而不是只报数字；
3. 更新 findings.md 的 Current Understanding / Patterns / Lessons / Open Questions；
4. 如果两次连续失败或出现反直觉结果，必须回到文献；
5. 用 paper-search / scoop-check 查新论文，写入 literature/；
6. 最多提出 2 个新假设；不能一次改五个变量；
7. 明确选择 DEEPEN / BROADEN / PIVOT / CONCLUDE，并写理由。

三、实验顺序

不要做 D×I×A 全交叉。按阶梯取 best arm：

D0 vs D1
-> 取 best data arm
-> 在同一 data arm 上跑 I0/I1/I2/I3
-> 取 best interface arm
-> 在同一 interface arm 上跑 A0/A1/A2

允许根据结果跳过一层，但必须满足：
- 有 PASS 的 matched 结果；
- 在 research-log.md 写清为什么跳过；
- 不能因为一次 0/15 就跳层；
- 跳过会被标记为 EXPLORATORY，不能作为主结论。

四、什么算“出效果”

主 endpoint：white_right official success。
- 第一阶段：0/15 -> >=1/15，只作为继续门槛；
- 正式确认：30/50 条 + 3 seeds；
- 不能只看 loss、MAE、point-in-mask 或 attention；
- 机制必须同时看 strict destination、wrong-destination、correct-vs-wrong action margin、suff/nec/selectivity；
- seen group-any 不能明显退化。

五、Run Contract

每个 arm 启动前必须写：
- code commit / config SHA
- data_arm / parent_arm
- train_language_counts
- heldout_prompt_train_count（必须为 0）
- train_validation_rollout_overlap（必须为 0）
- sampling weights
- effective batch
- marker_mode
- counterfactual_mode
- seed
- grad norm ratios

状态：
- PASS：进入 Gate；
- FAIL：配置正确但结果未达标；
- INVALID：数据/配置/评测不满足契约，不能判方法失败。

六、文献与方向切换

出现以下情况必须查文献：
- 连续 2 个 matched arm 失败；
- 结果与已有论文相反；
- 发现 marker / counterfactual / factor data 可能已被覆盖；
- 当前任务 data 不足或 baseline ceiling。

文献动作：
- 用 paper-search 查 3 组不同措辞的问题；
- 用 scoop-check 拆 problem / mechanism / insight / domain；
- 保存到 literature/survey.md；
- 任何准备借用的方法，必须按第 13 节（含 13.6 的固定操作）下载论文 PDF/TXT、查正文中的 code/project page、克隆官方仓库并 pin commit；没有代码就标记 official_code_missing；
- 在 references/paper_reading/{slug}.md 写“原文做法 / 我们做法 / 差异 / 风险”，未完成不得进入训练；
- 只提出最多 2 个可执行的新假设；
- 不得把“论文里有效”直接当作本项目有效。
- 任何组件从 lite 升级到论文近似版，必须先满足第 11.4.4 节的 W-D/W-I/W-A/W-F 窗口；没有对应 Gate PASS，只能保持 lite，不能因为“论文这样做”就提前升级。

允许的方向切换：
- 数据不可用 -> 先修数据；
- D0 不能复现 -> 先查 contract/rollout；
- D0/D1 都没有 headroom -> 换 held-out 或 caddy；
- I1 无效 -> 修 marker 接口，不急着上 A2；
- I2 有效、A2 无效 -> 回退 I2，不写 factor-wise 方法；
- 连续 3 个 PASS contract 的 matched arm 失败 -> 触发 outer-loop PIVOT，写清证据后再换任务/方法。

禁止：
- 因为一次 0/15 就换方向；
- 看到结果后改主 endpoint；
- 把 exploratory 结果写成 confirmatory；
- 隐藏与已有工作重叠的文献；
- 没读论文、没查官方代码就凭印象实现外部方法；
- 只报告成功 arm，不报告失败/INVALID arm。

七、预算与停止

- 最多 12 个 training/eval arms；
- 同一假设最多 2 次修复尝试；
- 每阶段先 smoke，再 pilot；正式 30/50 只在 pilot 有方向性信号后跑；
- 只用 GPU1，训练/评测串行；
- 不修改 references/repos/starvla；
- 不 commit、不 push、不建分支，除非用户明确要求；
- GT mask/point 只进训练 loss，不进部署输入；
- validation/rollout init state 不进训练；
- 不提交 checkpoint、视频、safetensors、pt 和大数据。

停止条件（满足任一即停）：
1. white_right 出现可复核 >=1/15，且机制指标同向：扩 30/50 + 3 seeds；
2. 预算用完；
3. 数据/评测契约持续 INVALID；
4. 连续 3 个 matched arm 失败且 outer-loop 已给出 PIVOT 结论；
5. 第二场景/外部 benchmark 也失败，主故事不再成立；
6. 用户要求停止。

八、输出

每个阶段输出：
- stageX_protocol.md
- stageX_result.md
- run_contract.json
- analysis.md
- findings.md 更新

最终输出：
- 一页给负责人看的结论：当前证据支持“数据 / 接口 / 控制”哪一层；
- 主表和失败表都要有；
- 明确 PASS/FAIL/INVALID；
- 如果正向：给出 30/50 与第二场景计划；
- 如果负向：给出高质量负结果和下一步 PIVOT，不许用“再调调参”掩盖。

现在开始：先建立 research-state.yaml、research-log.md、findings.md，然后执行 Stage 0 的 D0/D1 数据审计与 Run Contract。不要直接启动 A2 全量训练。
```

---

## 13. 外部论文方法与代码复现规范

这条是硬规则：**只要准备借用某个论文的模块、loss、数据构造、评测协议或训练 recipe，就必须先把论文和代码参考材料下载到本地，读完实现后再动手。** 不能只看摘要、二手解读或博客就开始写。

### 13.0 适用范围与不可协商的红线

这条规则适用于“直接移植”和“只借一个思想”两种情况；只要某个设计来自某篇论文，就必须留下可追溯材料，不能凭记忆实现。

- 论文必须实际下载到本地，至少保留 `papers/{slug}.pdf` 和抽取出的 `papers/{slug}.txt`；只保存网页链接、摘要或聊天记录不算完成。
- 必须找到并下载对应代码仓库到 `references/repos/{slug}`；仓库链接可能出现在论文正文的方法段、实验设置、脚注、附录、补充材料或 OpenReview 附件中，**不能只搜论文首页就下结论说“没有代码”**。
- 如果正文链接指向项目页、OpenReview 或 HuggingFace，必须继续追到实际 GitHub/GitLab 仓库；找不到时才允许标记 `official_code_missing=true`。
- 进入训练前，本地必须同时具备四类材料：论文 PDF、抽取文本、官方/明确标注非官方的代码 checkout、`references/paper_reading/{slug}.md` 阅读笔记。缺任何一项，该 arm 标为 `INVALID`，不得进入 Gate。
- 必须按第 13.5 节记录仓库链接在论文中的位置、pin 住的 commit 和 license；不能把一个第三方复现仓库写成“官方实现”。
- 下载或克隆失败时，先尝试 7897 代理或国内镜像，并把 URL、失败原因和替代来源写入 `references/download_failures.log`；不能用“当时打不开”作为跳过复现记录的理由。

### 13.1 使用前必须完成的检查清单

- [ ] 下载论文 PDF 到 `papers/{slug}.pdf`，不能只保存链接；
- [ ] 抽取正文到 `papers/{slug}.txt`，便于全文搜索；
- [ ] 在摘要、引言、方法、实验、脚注、附录、补充材料、参考文献和 OpenReview 附件中搜索 `github.com`、`gitlab`、`code`、`implementation`、`project page`、`supplementary`、`huggingface`、`zenodo`；不能只搜论文首页或只看二手解读；
- [ ] 记录仓库链接出现在论文的哪个位置，例如 `paper §4.2`、`appendix A.1`、`footnote 2`；如果链接指向项目页/OpenReview/HuggingFace，继续追到实际代码仓库，不能停在项目页；
- [ ] 找到官方仓库后，实际克隆到 `references/repos/{slug}`，执行 `git rev-parse HEAD` 并在 `references/repositories.yaml`、`references/repo_versions.md` 中记录 commit；
- [ ] 记录 license、依赖、数据集版本、预处理、超参数、训练步数；
- [ ] 阅读官方代码中的 README、config、requirements、loss、数据 pipeline、训练/评测入口和模型接口，不只读 README；
- [ ] 写 `references/paper_reading/{slug}.md`，记录“原文做法 / 我们做法 / 差异 / 风险”；
- [ ] 若没有官方代码，明确标记 `official_code_missing=true`，只能按论文 pseudocode 复现，不能声称 faithful reproduction；
- [ ] 若下载失败，先试 7897 代理或国内镜像，并把失败原因记录进 `references/download_failures.log` 和对应阅读笔记；
- [ ] 方法进入训练前，先过上面的清单；未过清单的 arm 标为 `INVALID`，不进入 Gate。

### 13.2 本地参考库约定

```text
papers/                         # 论文 PDF 与抽取文本
references/papers/              # 需要长期保留的论文材料
references/repos/{slug}/        # 官方代码或复现仓库，pin commit
references/paper_reading/{slug}.md  # 论文方法、代码结构、与我们的差异
literature/survey.md            # 所有候选论文的摘要与相关性
```

规则：

- 不修改 `references/repos/starvla`；
- 不把外部仓库直接拷贝进主训练代码，优先在我们自己的 adapter / loss / data pipeline 中实现；
- 使用外部方法时，代码注释必须写清来源 `paper + repo + commit`；
- 论文方法和我们的实现不完全一致时，必须在 `paper_reading/{slug}.md` 写 `differences`，不能把差异藏起来；
- 如果论文没有代码，优先找作者项目页、OpenReview supplement、附录；仍找不到就标记缺失。

### 13.3 当前需要重点核对的外部方法

| 方法 | 本地论文 | 代码仓库状态 | 使用前必须确认 |
|---|---|---|---|
| CRAFT `2610.00524` | 已有 PDF/TXT | 待查正文与项目页 | skill representation、counterfactual pair、stop-gradient 细节 |
| AnySlot `2604.10432` | 已有 PDF/TXT | 待查正文与项目页 | marker 渲染、HSV 提取、低层 policy 训练数据 |
| Scale Up Strategically `2607.21582` | 已有 PDF/TXT | 待查正文与项目页 | FDR/FDH、bias-aware sampling、预算对齐 |
| Encoded but Not in Control `2610.06235` | 已有 PDF/TXT | 正文出现代码链接，需确认是否对应方法 | 诊断指标、attention 归因、干预协议 |
| AC-VLA `2607.15714` | 未下载 | 未查 | 指令分解、trajectory aligner、混合训练 |
| CounterAlign `2608.21740` | 未下载 | 未查 | negative pair、discriminator、AWR |
| CAST `2508.13446` | 未下载 | 未查 | counterfactual label 生成与筛选 |
| LangForce `2601.15197` | 未下载 | 未查 | latent action queries、PMI/LLR loss |
| CAG `2602.17659` | 未下载 | 未查 | 双分支反事实推理 |
| CofactVLA `2608.04396` | 未下载 | 未查 | language-masked branch、去混淆投影 |

这张表不是为了现在全部实现，而是防止“看到一篇论文就凭印象加模块”。每个方法进入实验前，先把对应行补完整。

表中的“待查/未下载”只是审计起点，不是使用许可。真正进入训练前，对应行必须至少补齐为：`paper_downloaded=true`、`repo_url`（找不到则 `official_code_missing=true`）、`repo_commit`、`repo_found_in`、`paper_reading`、`reproduction_level`；任何一项为空，该外部方法只能用于文献讨论，不能进入 arm。

### 13.4 代码复现的最低证据

一个外部方法只有在以下条件满足后，才能进入我们的训练矩阵：

- 论文方法段和附录读完；
- 官方代码（如有）读完并 pin commit；
- 实现的 loss / 数据流 / 输入输出与论文对齐，差异有记录；
- 先跑一个最小 smoke，确认不是接口错误；
- 与我们 baseline 使用同一数据、同一 evaluator、同一 batch/seed；
- 结果分为 `faithful_reproduction` 和 `adapted_reproduction`，不能混写。

### 13.5 从论文找官方代码的固定顺序

论文中的仓库链接经常不在摘要，而在方法段、实验设置、脚注或附录里。固定按以下顺序找，找不到才允许标记缺失：

1. 对 `papers/{slug}.txt` 全文搜索 `github.com`、`gitlab`、`code`、`implementation`、`project page`、`supplementary`、`openreview`、`huggingface`、`zenodo`；
2. 检查首页脚注、方法段末尾、实验设置、附录和补充材料；
3. 检查 OpenReview 的 supplementary material、arXiv HTML/ancillary files、作者项目页和实验室主页；
4. 如果有多个链接，优先作者/官方实验室仓库；复现仓库必须明确标记为非官方；
5. 克隆后 pin commit，并至少阅读 README、config、requirements、loss、data pipeline、train/eval 入口；
6. 如果论文正文有链接但仓库已失效，记录原 URL、失效证据和替代来源；找不到就写 `official_code_missing=true`；
7. 没有完成上述步骤的方法，不得进入训练，也不得在论文里写“按原文实现”。

每个外部方法的来源记录至少包含：

```yaml
- id: {slug}
  paper:
    arxiv_id: "{arxiv_id}"
    pdf: "papers/{slug}.pdf"
    txt: "papers/{slug}.txt"
  repo:
    url: "https://github.com/org/repo.git"
    found_in: "paper §4.2 / appendix A.1 / footnote 2"
    local_path: "references/repos/{slug}"
    commit: "{git rev-parse HEAD}"
    license: "{license}"
    official: true
  paper_reading: "references/paper_reading/{slug}.md"
  reproduction_level: "faithful | adapted | pseudocode"
  official_code_missing: false
```

这张记录同时更新到 `references/repositories.yaml`；下载/归档状态更新到 `references/download_manifest.md`。论文方法进入实验前，`paper_reading/{slug}.md` 必须已经写完“原文做法 / 我们做法 / 差异 / 风险 / 复现等级”。

### 13.6 一次外部方法准入的固定操作

只要某个 arm 使用了外部论文的模块、loss、数据构造、评测协议或训练 recipe，就先完成本节，再改我们自己的训练代码。不能先用“论文里大概是这样”写一版，再回头补材料。

#### 13.6.1 下载论文并全文找代码

```bash
cd /home/bubble/类脑计算/VLM终局
SLUG=acvla                 # 例如 acvla / composlot / craft / counteralign
ARXIV_ID=2607.15714        # 替换成论文的真实 arXiv ID
mkdir -p papers references/papers references/repos references/paper_reading literature

# 下载 PDF，并抽取可全文搜索的文本
curl -L --fail --retry 3 "https://arxiv.org/pdf/${ARXIV_ID}" -o "papers/${SLUG}.pdf"
pdftotext -layout "papers/${SLUG}.pdf" "papers/${SLUG}.txt"

# 不要只搜首页；正文、脚注、附录、补充材料和参考文献都要一起搜
rg -n -i 'github\.com|gitlab|code|implementation|project page|supplementary|openreview|huggingface|zenodo' \
  "papers/${SLUG}.txt"

# 如果论文给了多个仓库，逐个打开确认；优先作者/官方实验室仓库
```

找到链接后，必须记录它出现在论文的哪个位置，例如 `paper §4.2`、`appendix A.1`、`footnote 2` 或 `OpenReview supplementary`。如果链接只指向项目页、OpenReview 或 HuggingFace，继续追到实际 GitHub/GitLab 仓库；不能停在项目页。

#### 13.6.2 克隆并 pin 官方代码

```bash
git clone <repo_url> "references/repos/${SLUG}"
git -C "references/repos/${SLUG}" remote get-url origin
git -C "references/repos/${SLUG}" rev-parse HEAD
git -C "references/repos/${SLUG}" log -1 --format='%H %ad %s' --date=iso
```

克隆后至少阅读以下内容，不能只读 README：

- `README`、`requirements` / `environment`、`config`；
- 论文对应模块的模型定义和 forward 路径；
- loss、数据 pipeline、训练入口、评测入口；
- 与论文公式/伪代码对应的变量、超参数和默认值。

如果找到的是第三方复现仓库，必须在记录里写 `official: false`，并在文件名或目录名上标 `-unofficial`；不能写成“官方实现”。如果论文没有代码，写 `official_code_missing: true`，只能按论文伪代码实现并标 `pseudocode`，不得声称 `faithful_reproduction`。

#### 13.6.3 写完阅读记录后才能进入 arm

`references/paper_reading/{slug}.md` 至少包含以下六段：

```markdown
# {paper title}
## 1. 原文做法
## 2. 官方代码结构与关键路径
## 3. 我们的做法
## 4. 与原文的差异
## 5. 风险与失败模式
## 6. 复现等级与证据边界
```

然后在 `references/repositories.yaml`、`references/repo_versions.md` 和 `references/download_manifest.md` 中补齐 `paper_downloaded`、`repo_url`、`repo_found_in`、`repo_commit`、`license`、`paper_reading`、`reproduction_level`。这些字段任意一项缺失，该 arm 只能标为 `INVALID`，不能进入训练或 Gate 判读。

#### 13.6.4 硬阻断条件

以下情况一律不允许启动新 arm：

- 只有论文链接、摘要或二手解读，没有本地 PDF/TXT；
- 只搜了论文首页，没有检查正文、脚注、附录和 OpenReview 附件；
- 论文正文或补充材料给了代码链接，但没有继续追踪到实际仓库；
- 仓库没有 clone、没有 pin commit，或没有记录 license；
- 没有写 `paper_reading/{slug}.md`，或没有写与原文的差异；
- 把第三方复现仓库写成官方实现；
- 因为没有官方代码，就把“按论文思路实现”写成 `faithful_reproduction`。

这些都按 Run Contract 的 `INVALID` 处理，不进入 Gate；不能用“先跑一下看看”绕过。

---

## 14. 证据索引

- [现有 idea 可行性审计](61_vla_idea_viability_audit_and_pivot_20261107.md)
- [FAI-VLA 融合方案](62_fai_vla_fusion_plan_20261107.md)
- [Result-First 分阶段规划](63_result_first_fai_staged_plan_20261107.md)
- [主任务选择调查](64_main_task_selection_20261107.md)
- [最新 binding 文献调查](59_latest_vla_binding_survey_and_pivot_20261106.md)
- [低分 binding 任务调查](60_low_score_binding_tasks_and_method_choice_20261106.md)
- [Code-as-Policy 结果](../../experiments/G2-gpt-gear-bridge/code_as_policy_results.md)
- [Goal12 N6 状态](../../experiments/Goal12-counterfactual-action-20261106/04_run_status.md)
- [Goal11 Gate B](../../experiments/Goal11-gate-b-k5cf-20261106/04_gate_b_result.md)

外部条目：

- Encoded but Not in Control `2610.06235`
- CRAFT `2610.00524`
- AnySlot `2604.10432`
- Scale Up Strategically `2607.21582`
- CounterAlign `2608.21740`
- CAST `2508.13446`
- CAG `2602.17659`
- LangForce `2601.15197`
- AC-VLA `2607.15714`
- InternVLA-M1 `2510.13778`
- Pointing-VLA `2608.23138`
- CofactVLA `2608.04396`

---

## 15. Git

本次文档改动只涉及本文件；不提交 `papers/`、`references/` 的工作区变化、checkpoint、视频或大数据。确认后按以下方式提交：

```bash
cd /home/bubble/类脑计算/VLM终局
git status --short
git add doc/vla_next/65_story_method_selection_20261107.md
git diff --cached --check
git commit -m "docs(vla): add component upgrade batch and material gate"
```

需要同步到远端时，再执行 `git push origin main`；本次不自动 push。
