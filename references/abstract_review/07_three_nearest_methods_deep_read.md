# 三篇最相近工作的原文精读：如何参考、如何比较、还剩什么问题

日期：2026-09-16。本文按用户要求分析方法与实验设计，不是全文翻译。阅读范围为三篇本地 PDF 的方法、实验、相关消融与实施附录；不声称已运行官方代码。所有数值都是论文报告，不是本项目结果。页码统一为 PDF 页面序号。

| 标识 | 原论文与本地版本 | 本次重点证据 |
|---|---|---|
| DB | Don't Blind Your VLA: Aligning Visual Representations for OOD Generalization，2510.25616v1 | p.3–9 正文；p.11–13 附录 |
| AA | Generalizable VLA Finetuning via Representation Anchoring and Language-Action Alignment，2607.13429v1，方法名 Anchor-Align | p.3–11 方法/结果；p.19–22 实施；p.27–29 诊断 |
| PA | PosA-VLA: Enhancing Action Generation via Pose-Conditioned Anchor Attention，2512.03724v2 | p.3–8 正文；p.10–12 实施/局限 |

对应原文件均在 ../papers/。证据段落 ID 与原文页提取保存在 `three_methods_source_map.json`。本次还人工查看 PA p.4–5、AA p.4 的渲染页面，避免 PDF 分栏使公式串行错误。没有据此确认线上最新版本或代码发布状态。

## 1. 首先纠正此前过度简化的理解

三篇都与我们接近，但监督对象不同：

| 问题 | Don't Blind | Anchor-Align | PosA-VLA | 我们尚待验证的 D0 |
|---|---|---|---|---|
| 希望保住什么 | 局部视觉语义表征 | 整个 VLM 的视觉和文本表征及动作方向语义 | 任务交互区域与末端执行器的空间关注 | 当前输出实际依赖的语言条件视觉证据 |
| 教师/标签 | 独立冻结视觉 encoder，主要 C-RADIOv3 | 同底座冻结 VLM + demonstration 自动生成的方向词 | 示范 EEF 位姿、夹爪事件、相机投影 | Lavender 词图 + 内部语言/动作输出归因 |
| 监督位置 | VLA 中间 decoder 层的视觉 token | 所有 decoder 层视觉/文本 token；另监督 pre-action state 的语言输出 | CLIP cross-attention 产生的双通道 anchor maps | 选定视觉 token 上的输出条件空间归因 |
| 与动作的连接 | 表征正则间接影响策略 | 同一观测预测动作和方向词 | map 直接乘到视觉特征，再输入 flow action head | 拟通过归因损失反传到策略，而不是新增前向视觉门控 |
| inference 改动 | 正则教师不必运行 | frozen anchor 和辅助监督不作为动作标签输入 | attention/gating 是策略前向的一部分 | 理想情况移除外部教师，不新增必需感知器 |

不能再把 Don't Blind 说成“只对齐视觉编码器”：其较优配置是 **VLA backbone 中间层 → 独立视觉 encoder**。也不能把 Anchor-Align 说成任意语言—动作热力图一致性：它主要用可解释的方向词监督语言头。PosA 的 pose 主要是生成训练标签的来源，不等于部署时输入未来 GT grasp pose。[DB-METHOD、AA-METHOD、PA-MAPS]

三个轴已经有工作：**表征保持、输出语义一致性、前向空间门控**。我们可以探索的是输出条件空间证据约束能否提供额外价值，但这些三篇没有直接出现我们当前公式，不等于整个领域从未做过。[本项目推论]

## 2. Don't Blind Your VLA：最应该参考的是“监督位置”和“投影器逃避”

### 2.1 原方法如何工作

冻结视觉教师从同一图像抽 patch features；学生取 VLA 某个内部层对应的视觉 token，经 projector 对齐维度，计算逐 patch 相似度，再与动作 token 的自回归交叉熵相加。对应原文 Eq.(7)–(10)，p.5。[DB-METHOD]

下面为保持含义的简化记法，不是额外提出的损失：

```text
z_i = frozen_vision_teacher(image)_i
u_i = P(h_i^l)
L_DB = -(1/N) Σ_i cos(u_i, z_i)
L_total = L_action + λ L_DB
```

关键区别：目标 z_i 来自图像，而不是每个语言对象的词图；学生监督的是高维特征，不是对 action output 求梯度得到的 attribution。即使 patch 是空间化的，feature alignment 仍不等于输出归因对齐。

### 2.2 最值得照着做的四个消融

1. **Backbone2Enc vs Enc2Enc**：p.8 Table 5 中，中间 backbone token 对齐教师的 semantic/vision 指标较好。作者据此强调不能只保护前端 encoder。这是模型内部监督位置的直接先例。[DB-ABLATION]
2. **冻结 projector vs 可训练 projector**：作者解释，可训练 projector 可能代替 backbone 吸收损失，从而没有纠正模型内表征。Table 6 的 frozen MLP semantic 0.61、trainable MLP 0.54，但并非每一指标都显著或最佳，不能将冻结写成普遍规律。[DB-ABLATION]
3. **早/中/晚层**：Table 7 和附录 Table 12 比较多个位置，较优配置选第 16 层；不能直接将 16 搬给 Qwen 或 flow expert，要按功能/深度重新选。[DB-ABLATION、DB-CONFIG]
4. **教师/损失选择**：比较 C-RADIOv3、DINOv2、SigLIP、Theia，以及 cosine、L2、contrastive。Table 9 记录 60k steps、LoRA rank 32、λ=0.2、冻结 MLP 和第 16 层。这些是该 OpenVLA 设置，不是我们的冻结超参数。[DB-CONFIG]

### 2.3 benchmark 的巧思和证据边界

VL-Think 固定 carrot 搬运，改变目标 board 的 shape、color、traffic、laundry、weather、arrow、public-info、parity。作者尽量降低抓取难度，观察“是否懂目标概念”；另用视觉/语义/执行三类 OOD 和 ImageNet-100 linear probe。[DB-EVAL]

我们可借鉴**同一运动技能、不同目标语义**：同一场景两个外观不同但可抓取性相似的物体，只换指令，比较目标选择。这样比把复杂长程失误一概解释为 grounding 差更明确。但 board VQA 与实际搬运仍是不同任务，不能直接把两个准确率差值当纯遗忘量。

论文自己在 p.7 指出 Align 对 Color/Shape 有改善，其他较少出现的概念恢复有限。Table 1 的 PosChangeTo 从 0.23 变为 0.20，因此不能说所有 OOD 均提升。[DB-RESULT]

### 2.4 我们应怎样参考

- 建一个 **DB-style feature alignment** 对照，配同一个学生、同一动作头和数据。若没复现全部官方 projector/预处理，只叫 inspired baseline。
- 区分“教师更强”和“监督对象更好”：C-RADIO feature 与 Lavender map 对比同时改变教师与损失，不能独自证明归因优于 feature。需再用同源 frozen VLM 的 feature/map 做成对实验，或明确承认混杂。
- 增加 **bridge-only trainable / backbone updates** 的检查。热力图头能对齐不代表动作头使用同一证据；报告 action branch 梯度是否收到对齐损失。
- 原文未清楚给出 frozen MLP 的初始化/先训练再冻结流程；正文维度和 Table 9 projector dimension 也需代码核对。不能擅自假设随机固定 MLP 就等价于完整方法。
- 作者从可视化推断 attention sink 缓解；我们仍需数值 sink 指标和受控输入干预。漂亮 attention 不是独立因果证据。

## 3. Anchor-Align：它已经给出一个非常强的“语言和动作一致”基线

### 3.1 Anchor 的对象比 Don't Blind 更宽

冻结一份**同底座 VLM**，在同一机器人观测的 images/text 上逐层对齐学生 hidden states，位置集合覆盖视觉与文本 token。Eq.(2) 是平方 Frobenius norm，Eq.(3) 再平均 decoder 层。[AA-METHOD]

```text
L_anchor = (1/|D|) Σ_l ||H_student^l[m] - H_frozen^l[m]||_F²
m = 视觉与文本 token 位置
```

由于同底座 hidden dimensions 对应，不需要像 Don't Blind 那样把 4096 维映射到异构视觉教师。论文实施附录使用 MSE；其 sum/mean reduction 必须核对代码，否则 token 数变化会改变有效 λ。

### 3.2 Align 到底对齐什么

把 GT action chunk 的前三个平移分量求均值，过滤近静止样本，取绝对值最大轴的符号，映射到六个方向词。使用最后一个 instruction/pre-action token hidden state，经可训练 W_proj 和冻结原 LM head，预测该词的 vocabulary logits，计算 CE。[AA-METHOD]

```text
v̄ = mean_k(a_GT[k, translation])
j* = argmax_j |v̄_j|
y_dir = direction_word(j*, sign(v̄_j*))
h_pre = last instruction token hidden state
logits = frozen_LM_head(W_proj h_pre)
L_align = CE(logits, y_dir)
L_total = L_action + λ_anchor L_anchor + λ_align L_align
```

因此它让语言输出“向右”与连续动作“向右”共用同一观测与语义标签，**没有要求两路依赖相同图像位置**。两路方向一致仍可能共享错误 shortcut；反过来，方向一致也可能已足以改善策略，必须实际比较才能证明 D0 值得做。

### 3.3 具体实现可以参考什么

- 默认学生是 **VLA-Adapter + Prismatic-Qwen2.5-0.5B**，bridge-attention regression head；不是主表上的每个 competitor 都接入同一种方法。真实实验另做 **StarVLA + Qwen2.5-VL 3B + GR00T FM-DiT flow head**；附录另有不同 StarVLA/OFT head 配置，应分开记录。[AA-SETUP]
- p.19：默认所有 24 decoder 层、LoRA r=64、10k steps、batch 32、λ_anchor=0.1、λ_align=0.02。额外 W_proj 为 896→896，原 LM head 冻结。[AA-CONFIG]
- 训练 chunk 阈值 τ=0.15 与 p.27 per-frame 诊断阈值 μ=0.001、连续 5 帧同向，不是一组可互换参数。移植 DROID 必须按动作归一化、坐标系和频率重新校准。[AA-CONFIG、AA-DIAGNOSTIC]
- p.20 在其配置中训练耗时 1.28→1.64 s/iteration（+28%）。冻结教师仍有算力和显存成本；我们的二阶归因训练也必须记录，不能只匹配 step 数。[AA-COST]

### 3.4 消融设计比只加随机图更强

p.8 Table 3（PRO / Plus）报告：BC 61.0/85.1，Align-only 65.9/88.6，Anchor-only 68.1/87.3，完整 71.9/90.3。说明输出语义监督与表征保持各有贡献，完整优于单项。[AA-ABLATION]

Shuffle 使用固定方向词置换；Scatter 用固定无运动语义词作六类标签。两者仍是可学习分类任务，不是每步随机噪声。其退化为“语义是否有用”提供证据，比仅打乱全部图更细。[AA-CONTROLS]

**批判性阅读**：原文称这些对照保持 loss magnitude/gradient norm step-for-step；仅替换 frozen LM head 下的目标 token 并不能数学上保证相同梯度。我们借鉴其对照意图，同时实际记录 loss/gradient scale，而不照搬这个强断言。

### 3.5 benchmark 应如何借鉴

LIBERO ID 放附录，重点在 PRO（rephrase/object swap/position swap）、Plus（七维鲁棒性）、CALVIN ABC→D；真机验证同场景替换语义目标。Position swap 即使完整方法也仅 22.6%，应保留其难度信息。[AA-SETUP、AA-RESULT]

对我们的“语言条件区域”主张，PRO 的换物体/换位置和同图换指令可能比只做背景扰动更有鉴别力。Plus 检验无关变化鲁棒性，PRO 检验新指令/目标关系适应，两者不应被一个总分替代。DROID 首轮仍是离线真实图像评估，不是同等闭环证据。

## 4. PosA-VLA：与我们的 C 更直接重合，也暴露 D0 的过强假设

### 4.1 不是给现成 action attention 随便加一个高斯

CLIP 文本与图像 patch 经过 cross-attention 生成任务图 M_task；另用 “gripper” 文本 query 得到 M_end，两张图拼为 M。训练标签由示范 pose 得到：

- 夹爪开/合事件时的 EEF 位置对应任务交互区域，投影后构造较宽高斯 F_task。
- 每个时刻 EEF 当前位置投影产生较窄高斯 F_end。

这需要可靠相机—机器人变换，分别处理 head/wrist view。[PA-MAPS]

```text
F_task(i,j) = exp(-((i-u)²+(j-v)²)/(2σ_task²))
F_end (i,j) = exp(-((i-u)²+(j-v)²)/(2σ_end²))
σ_end < σ_task
M = [M_task, M_end]
```

注意原文事件图如何从稀疏事件传播到非事件训练帧写得不够明确，Algorithm 1 又简化为当前 pose 投影。不能自行假设“全程用未来最终 grasp 点”就是其官方算法；这个问题需读代码才能做忠实复现。

### 4.2 三个损失与前向连接

空间监督用 focal loss；从标签图 >0.7 的区域提取视觉特征，再与语言特征拼接，以同 caption/task ID 作为 batch pair 正负关系，使用 sigmoid BCE contrastive loss。[PA-LOSS]

```text
L_f      = FocalLoss(M, [F_task,F_end])
L_anchor = α L_f + (1-α) L_c       # 文中 α=0.5
F_ref    = M ⊙ F_DINO
L_total  = L_flow + λ L_anchor    # 文中 λ=1.0
```

**空间图实际参与前向动作计算**：门控 DINOv2 特征后交给 Flow Matching Transformer。Figure 3 又单独显示 action-transformer 最后层平均 cross-attention；anchor map 和 action attention 是两类图，不应混称同一归因图。[PA-LOSS、PA-MAPS]

推理不需要提供 demonstration GT 高斯，但仍运行 learned anchor/gating 网络。论文说无需额外 segmentation/grounding，并不意味着不需要 CLIP、DINOv2 或前向视觉模块。

### 4.3 结果与实施中值得注意的细节

- Table 4：Basic 74.9，去 anchor loss 29.5，去 contrastive 57.8，去 EEF attention 55.6。这支持“夹爪区域也是必需信息”，不能把 source/target 以外证据一律当错。[PA-ABLATION]
- 正文写 anchor-only 20k pretraining，再进入联合训练；σ_task 为短边 1/10、σ_end 为 1/15，action chunk 30，附录 flow inference 10 steps。不能用我们的 H=1 结果直接对照这些数字。[PA-CONFIG]
- 正文的统一训练表述与附录 baseline 配置不完全一致：附录 π0 30k steps/单 GPU，OFT 100k/8 GPU，SmolVLA 200k 等。因此这是系统级比较，不是所有模型严格同计算预算的 paired comparison。[PA-CONFIG]
- LIBERO 附录将 DINOv2 改为 CLIP，报告合成图下的适配差异。Table 7 Average 95.1，高于未标星 OFT 94.2，但低于过滤失败示范的 OFT* 96.8；不能简化为击败所有配置。[PA-SIM]
- 局限明确提到推/滑等夹爪状态不变的动作和重遮挡问题。事件高斯不是普适 affordance 真值。[PA-LIMIT]

### 4.4 对我们最直接的设计后果

1. EEF/target 高斯 C 路线已有明确相邻方案，适合对照/标签诊断，不宜单独作为 novelty。
2. 把 `A_act ⊂ A_lang(source) ∪ A_lang(target)` 写成硬正确性要求太强。夹爪、障碍、接触边缘可能是合法动作证据。先比较 object-only 与 object+EEF 的软约束；不马上加入未标定的第三教师。
3. 增加 **gating vs attribution regularization** 对照：相同语义图用于门控前向特征，和用于约束输出梯度，哪个更有价值？这能直接区分“我教模型看哪里”与“我约束动作依赖哪里”。
4. 如果我们主张更少 pose/标定依赖，要在不用投影标签时证明这一优点，而不是一边依赖 simulator masks 一边称弱监督。

## 5. 推荐的实验比较：先拆贡献，再扩模型

以下是待确认设计，不替用户冻结骨干；所有组在同一初始化、训练样本、图像增强、action head、chunk、优化配置下比较。教师离线和在线成本另列。

| 组 | 配置 | 排除的替代解释 |
|---|---|---|
| C0 | 原生 BC/SFT | 起点 |
| C1 | DB-style 单中层 patch feature alignment | 只是保留视觉表征 |
| C2 | AA-style 全层 vision+text anchor | 只是保持整个 VLM |
| C3 | C2 + 同观测方向词 CE | 已有输出语义对齐足够 |
| C4 | Lavender-style 语义图监督，不含 D0 | 增益来自外部语义教师 |
| C5 | C4 + D0 输出归因约束 | 动作空间证据约束的增量 |
| C6 | C3 + D0（只在公平定义 A_lang 后） | 已保持表征并对齐语义时，D0 是否仍有价值 |
| C7 | 相同空间 prior 做前向 gating | 不用高阶归因也能得到同样效果 |
| C8 | pose/EEF 高斯辅助图，标定可用时 | 普通几何锚定能否解释收益 |

无需一次跑九组。建议先 C0/C1/C4/C5 的小规模试验；若 C5 有增量，再做 C2/C3/C6 对抗强相邻方法，C7/C8 按最终主张选择。此顺序是实施建议，不把先跑的弱组当成最终充足 baseline。

**识别教师混杂**：C1 与 C4 既换监督形态又换教师，不能独立判断 heatmap 优于 feature。增加同源教师的 feature/map 对照，或者把研究结论限定为整套方法组合；均须显式报告。

## 6. 原方案需要重新审计的公式和术语

### 6.1 不把 loss 梯度叫作唯一动作依据

动作损失的梯度在拟合良好时可能趋于零，图反映的是误差敏感性。动作输出的梯度才直接描述局部输出敏感性，但向量动作必须逐分量计算/聚合，并处理单位尺度和正负抵消。flow 模型要固定/平均采样噪声与时间步。

因此 D0 先分别记录 A_act-output 与 A_act-loss，验证干预关系后再确定训练目标；不能把两者写成完全等价。

### 6.2 当前 containment 的图归一化需要修正候选

若 A_lang 是和为 1 的空间概率，而支持区域含 M 个均匀 patch，那么 A_lang≈1/M；即使动作完全位于合法区域，原来的 Σ A_act(1−A_lang) 仍约为 1−1/M，优化会偏向最高单点，不严格代表“越界质量”。

可考虑把语言图转换成独立的 soft support mask M_allow∈[0,1]，动作图才做概率归一化，并在训练时对 M_allow stop-gradient。一个候选为：

```text
q_i = normalized nonnegative action attribution
L_leak = max(0, Σ_i q_i (1 - stopgrad(M_allow_i)) - ε)
```

这只是建议，不是已验证新公式。ε 给合法越界留余量；mask 面积、阈值和信心必须只在验证集选择；全零动作归因应跳过/标记，不能偷偷归一化成成功样本。增加 object-only、object+EEF、面积匹配随机 mask 对照，防止语义图扩散或 action 梯度饱和使损失虚降。

### 6.3 A_lang 要真正有语言条件

在视觉 token 排在 instruction 前、使用严格 causal mask 的 decoder 中，视觉 token hidden state 不一定看到后续语言。直接拿其 self-attention 画词图可能不成立；应以具体 phrase/answer score 为目标反传，或核实独立语言 query 的计算路径。新增 heatmap 分支也要证明它与动作使用的特征/参数有实际耦合。

### 6.4 训练复杂度不能隐藏

若 L_contain 依赖参数相关的梯度归因，再对模型参数训练通常涉及二阶导数。DB 和 AA 主要是普通前向特征监督；我们的新方法如果训练显著昂贵，必须提供成本—收益对照。detach attribution 再监督不会自动优化原归因，应核对梯度路径。

## 7. 这三篇给我们的最有价值的研究问题

推荐把待验证主张收窄为：

> 当 VLA 的视觉语言表征已被保持、语言输出与动作方向也已得到监督时，动作预测是否仍依赖与任务不匹配的视觉证据？若存在这种现象，对输出条件空间归因加约束是否能带来独立的 OOD 收益？

先找这种差距的实证，再决定是否投入 D0。用同场景目标交换、同目标背景变化、夹爪/障碍区域干预，并同时报告正确目标选择、动作误差、闭环成功与归因变化。正常 proprioception 依赖不是捷径；去掉全部状态后失败不构成坏机制的证据。

VLM 阶段验证教师定位与梯度/坐标接口；VLA 阶段才验证动作依赖。DROID 必須使用，但目前只承担离线真实数据证据，闭环成功交给 LIBERO/真实机器人。SpikingBrain 后置，骨干组合仍待用户选择。

## 8. 精读后的选择建议

- **最快可信起步**：参考 Don't Blind 建同模型 feature baseline，同时验证我们的桥接是否真正更新策略；冻结/可训练 projector 必须做小试验。
- **最强语义对照**：Anchor-Align 的同观测方向词监督；不能跳过它，只与纯 BC 比较就宣称语言—动作连接创新。
- **最直接空间对照**：PosA 的任务+EEF 双图与前向门控；若不依赖 pose，明确比较标注成本与适用任务，而不宣称无监督完成它的功能。
- **尚不能下结论**：D0 是否更好、是否需要复杂 future teacher、是否跨所有骨干普适。原文阅读把可检验差异变清楚，尚未代替实验。
