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
