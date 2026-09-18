# GradientAdapter：当前版本是否合适、是否完备

## 结论先行

当前 GradientAdapter 适合作为跨模型主线的 **G0 interface-gradient baseline**，因为它不依赖标准 cross-attention，而是直接问“为了提高当前 phrase/action target score，视觉 token 的哪些激活重要”。但它不是最完备的最终归因系统，也不能仅凭当前 F0 结果声称已完成跨模型或 VLA 验证。

正确定位是：保留 G0 作为默认可训练 adapter；将 direct attention、integrated gradient 和 occlusion 分别作为比较、稳健化审计和因果验证，而不是把所有机制塞入一个高成本 loss。

## 当前代码到底计算了什么

`scripts/run_qwen_v3_smoke.py` 的实际路径为：

1. Qwen visual encoder 将图像转为 visual embedding；
2. embedding 被插入语言 decoder 输入 token 序列；
3. 插入后的 multimodal embedding `e_vis` 被 `detach().requires_grad_(True)`；
4. 用 teacher-forced caption 中 phrase token 的 log-probability sum 作为标量 `s_phrase`；
5. 计算：

$$
A_i=\left|\sum_c\frac{\partial s_{phrase}}{\partial e_{i,c}}e_{i,c}\right|.
$$

6. 将 `A` 与已 bridge 到 Qwen token grid 的 SD phrase map 做 spatial KL。

这意味着当前 map 是 **视觉—语言接口 token 的 output-conditioned local attribution**。它不是：

- 原始视觉 encoder 自注意力图；
- Lavender 的 phrase-to-vision attention；
- 完整积分梯度；
- 对图像像素的因果遮挡分数。

视觉 encoder parameter 在当前 V3 中不通过 semantic loss 更新；这与“只调语言 LoRA、避免破坏视觉底座”的实验约束一致。现有 `visual` hook / `requires_grad` 设置只是历史接口审计遗留，实际核心梯度对象应明确记录为 `e_vis`，而不是视觉 encoder weight。

## 为什么 G0 仍然合适

| 属性 | 当前 G0 interface-gradient | 对多模型适配的价值 |
|---|---|---|
| 输出条件性 | 直接依赖 phrase score | 能回答特定词为何被输出，而非一般视觉显著性 |
| 结构依赖 | 不要求标准 cross-attention | 可适配 visual-prefix decoder、cross-attention VLM 与 action decoder |
| token grid | 梯度落在模型真实 visual token 位置 | 可与动态 resolution 的 grid metadata 对接 |
| 参数更新 | 通过二阶梯度更新 LoRA | 让空间监督作用于会影响输出的语言/融合路径 |
| 教师复用 | 同一 SD phrase cache | 可跨 Qwen、LLaVA、Prismatic、InternVL 等共享 teacher target |

因此它比“对所有模型强行读取 attention”更适合作为普适主线。

## 当前不完备的六个方面

### 1. 局部梯度不等于全局因果贡献

`gradient×activation` 是 target 附近的一阶局部近似。若 score 饱和、attention path 有强非线性或 token 间交互很强，梯度可能低估真实影响。当前没有 integrated-gradient 或 multi-baseline 审计。

### 2. 单接口层，不是多层 attribution

G0 只在插入 decoder 前的 visual embedding 上取梯度。它适合跨模型统一，但无法判别某个中间融合层是否更适合作为空间证据。当前不应宣称“所有层级归因都已对齐”。

### 3. phrase occurrence 有歧义

当前通过 caption 中 phrase 的字符位置和 tokenizer offset 定位 target。若同一 phrase 在一句 caption 中重复，可能选错 occurrence；Flickr Entities 本来有 phrase index/entity id，后续 manifest 应保存精确 character span 或 token span，而不是仅依赖字符串第一次出现。

### 4. KL 的质量分布和区域几何不是同一目标

当前 `KL(Norm(T_SD) || Norm(A))` 主要约束相对质量分配。Pointing、mass 和 IoU 的 sweep 不稳定说明：mass distribution matching 不能自动产生紧凑、几何正确的 top region。继续调 scalar 不能解决这个目标错配。

### 5. 还没有 student-side causality audit

已完成的是 teacher map 的 correct/wrong-word/wrong-image/random 控制。还缺少：根据学生 map 遮掉 top-k 区域后，phrase score 是否比遮掉 bottom-k 或随机区域下降更多。没有这一步，学生热图只能说是 optimization proxy，不能说“faithful”。

### 6. 尚未完成跨架构和 VLA target 验证

当前真实训练只在 Qwen2.5-VL 上。VLA 中 `s_phrase` 必须被替换为明确的 action target，例如 delta-EEF action token log-probability、连续 action head 某维 score 或 expert-action NLL 的负值；不同 action architecture 应通过相同 `SpatialEvidenceAdapter` 接口记录 target 定义。

## 最完备但仍可执行的版本

不建议直接把所有项加进训练。建议采用逐层证据链：

| 层级 | 模块 | 用途 | 是否每个 batch 训练 |
|---|---|---|---|
| G0 | interface gradient×activation | 跨模型默认训练图 | 是 |
| L0 | Lavender-adapted direct attention map | 检验“为何不是直接 attention” | 是，作为独立 baseline |
| G1 | 4–8 step integrated gradients，blank-image embedding baseline | 审计 G0 是否受局部梯度饱和影响 | 否；小验证集离线运行 |
| G2 | selected-layer hidden-state attribution | 检查空间证据位于哪层融合 | 否；小验证集离线运行 |
| C0 | top-k / bottom-k / random occlusion | 因果 faithfulness | 否；每个 checkpoint 的评估 |
| B0 | BlindVLA-style feature retention | 排除视觉遗忘解释 | 是，独立 baseline 或组合组 |

G0 是方法的通用训练接口；L0/B0 是竞争或正交基线；G1/G2/C0 是让论文结论可信的审计工具。它们分工明确，避免训练成本和创新点一起膨胀。

## 推荐的下一步判断门

只有满足以下链条，才称 GradientAdapter 是“足够完备的主方法”：

1. 在至少两个融合结构不同的 VLM 上，G0 可通过 token/grid/phrase capability audit；
2. 在至少三 seed 上，G0 相对 V1 在 grounding 指标有稳定收益；
3. `G0 > L0`，排除“直接 attention MSE 已经足够”；
4. `B0+G0 > B0`，排除“仅仅视觉表征保持”；
5. G0 top-k occlusion 比 bottom-k/random occlusion 更大幅降低 phrase score；
6. VLA 中用 action target 定义替换 phrase target 后，同样趋势在 LIBERO/DROID action evaluation 出现。

当前状态满足第 1 项的 Qwen 子集和 teacher 控制的一部分，但尚未满足 2–6。因此应把当前方法称作 **promising universal adapter candidate**，而不是最终完备方案。

## 最终方法与实验组件的裁决

### 当前应冻结的主方法

主方法应保持最小且可解释：

$$
L_{total}=L_{task}+\lambda_{sem}L_{G0\leftrightarrow SD},
$$

其中 `G0` 是 output-conditioned interface gradient×activation map，SD 是离线 phrase map teacher。词筛选、token-grid coordinate bridge、teacher temperature 和 warm-up 属于该损失的实现配置，不是独立方法模块。当前最有希望的候选配置为 `lambda=.10, T=1.25`；仍需多 seed 验证后才冻结。

这保持了论文的核心主张：**不是让模型模仿原始 attention，而是让与目标输出有关的视觉证据具有语言条件空间一致性。**

### L0 必须比较，但不属于最终方法

L0 是 Lavender-adapted direct attention-map MSE。它与主方法使用相同 SD cache、phrase filter、坐标桥、训练数据、LoRA budget 和 held-out protocol。它回答唯一关键问题：

> 如果直接对齐 attention 已足够，为什么还需要 output-conditioned attribution？

所以 `G0 > L0` 是我们核心叙事必需的实验；L0 是强 baseline，不应并入主方法。

### B0 与 B0+G0 的正确位置

B0 是 BlindVLA-style feature retention，回答“是否只是避免视觉遗忘”。它也是 baseline，而非默认核心模块。

`B0+G0` 应作为组合消融组：

$$
L_{total}=L_{task}+\lambda_{ret}L_{B0}+\lambda_{sem}L_{G0\leftrightarrow SD}.
$$

只有当多 seed 下 `B0+G0 > B0`，且没有损伤 task performance，才把 B0+G0 作为最终部署配置或“可选 retention stabilizer”。当前 V4 没有稳定优于 V2，因此**现在不应把 B0 写进最终方法定义**。即使以后组合有效，论文核心仍是 G0；B0 只是独立贡献的控制和稳定化项。

### G1、G2、C0 是什么，是否进训练

| 组件 | 含义 | 是否进入每个 batch 训练 | 最终论文位置 |
|---|---|---|---|
| G1 | integrated gradients：从 blank-image / reference visual embedding 到真实 embedding 的路径积分归因 | 否；成本是多次前反传，适合小验证集 | attribution robustness audit / appendix |
| G2 | selected hidden-layer attribution：对若干融合层分别取 target-conditioned map | 否；先用于找最有语义的层 | layer-selection ablation；只有审计明确某层更好才将单层替换 G0 interface layer |
| C0 | top-k、bottom-k、random occlusion：遮挡不同区域并测 phrase/action score 的下降 | 否；离散遮挡更适合作因果测量 | faithfulness evaluation 主文或 appendix |

它们不应与 G0 一起堆进训练，因为这会同时增加二阶梯度、多层权重和离散扰动，导致无法判断哪个模块产生收益。正确顺序是：G0 训练；G1/G2/C0 验证 G0 是否可信；验证后最多把 G2 选出的**一个**层作为新的 G0 提取位置。

### 最终实验表应如何读

| 组 | 目的 | 是否属于主方法 |
|---|---|---|
| V1 | caption SFT | 否，基础 baseline |
| B0 | BlindVLA-style retention | 否，视觉遗忘 baseline |
| L0 | Lavender-adapted attention MSE | 否，attention matching baseline |
| G0 | output-conditioned GradientAdapter | 是，核心方法 |
| B0+G0 | 测试互补 / 可能的稳定化部署配置 | 取决于 `B0+G0 > B0` 是否成立 |
| G1/G2/C0 | 鲁棒性、层选择、因果 faithfulness 审计 | 否，验证工具 |

## IoU 优化路线与可证伪验收门

当前不能承诺 IoU 一定达到某个预设数值。原因是当前语义损失优化的是整张归因分布的 KL，而报告的 IoU 是对 top-k 支持集取外接框后得到的离散几何指标；两者并不等价。F0 中 SD 教师本身的 held-out IoU 约为 `0.325`，而 V1 约为 `0.268`。因此合理目标首先是缩小学生与教师之间的几何差距、稳定超过匹配 baseline，而不是无条件超过教师。当前 64 条 held-out、少量训练步数和两 seed 的结果也不足以给出保证。

下一轮按以下顺序做，每一步都有独立的停止条件：

1. **先做几何审计，不改模型。** 在验证划分上冻结训练后再选择 map-to-box 规则，报告 top-10/20/30% 支持集、固定阈值、soft-IoU 和中心/面积误差。分别记录 SD 教师、V1、V3 的上限与误差；测试平移、旋转、模糊和错词图，确认当前 IoU 是否对支持集阈值或外接框过于敏感。测试集只使用冻结规则，不能用来调阈值。
2. **加入教师派生的几何目标。** 保留当前 `L_KL`，一次只增加一个项：
   - `L_rank`：要求教师 top-quantile 区域的学生归因高于 bottom-quantile 区域；
   - `L_moment`：匹配归因分布的中心与二阶矩，约束目标位置和尺度；
   - `L_JS`：用对称分布距离替代单向 KL，减少教师过宽区域对学生的拉平效应。
   首轮比较 `KL`、`KL+rank`、`KL+moment`，固定 `lambda=0.10, T=1.25`，每组至少 3 个 seed。训练只使用教师图产生的伪几何量，不使用 Flickr 的 GT box；GT box 只用于额外的 supervised upper-bound 对照，避免把方法变成有监督检测器。
3. **再处理空间分辨率。** 将学生和教师先映射到共同的 32×32（必要时 64×64）坐标，再下采样到模型 token grid；同时保存原始图、重采样图和 top-k 支持集。若低分辨率量化是瓶颈，再评估一个冻结或强正则化的轻量 spatial adapter，不能先引入可自由变形的高容量模块。
4. **最后才恢复组合比较。** 几何目标在 G0 单独有效后，再重复 `L0`、`B0` 和 `B0+G0`。否则无法区分 IoU 增益来自 output-conditioned attribution、attention 拟合还是 retention。

建议的 F0 验收标准分两级：

- **最低可行证据：** 相对匹配的 semantic-off baseline，3 个 seed 的中位 IoU 至少提高 `+0.01`，且 pointing/mass-in-box 不下降超过 `0.005`；paired bootstrap 的下界不应系统性偏负。
- **强证据：** 至少闭合当前 baseline 到教师之间 50% 的差距。以 `0.268` 和 `0.325` 为暂定参考，对应 IoU 约 `0.296`；该数值是研究门槛，不是保证值，需在几何审计后按冻结的 metric 重新计算。

如果经过 `KL+rank`、`KL+moment`、共同分辨率和 3-seed 复验仍不能通过最低门槛，就应停止声称“语义对齐提升 IoU”，转而报告 pointing/mass 的收益和 IoU 失败分析，而不是继续无止境调 λ。这样才能保证结论可信，即使最终数值没有达到期望。
