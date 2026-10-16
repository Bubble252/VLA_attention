# Step 1 — 新候选创新性拆解

## Research problem

现有 VLA 可能在语言改变时仅响应词向量、固定位置或显著物体，而没有让动作变化依赖新指令所指定目标的视觉证据。单独测试换词或单独测试遮挡不能识别二者的交互。

## Novelty claim under audit

在相同或严格匹配的物理场景和机器人状态下，联合操纵合法目标指令与各候选目标的视觉证据，测量并可选地训练 instruction–evidence interaction，使 VLA 的动作重定向同时依赖正确语言目标与相应目标视觉信息。

## Four atomic axes

- **Problem framing**：将 instruction following、visual target use 和 primitive execution 拆开；主要单位是同场景目标交换/证据干预下的成对动作响应与闭环选择，而非单张 grounding 图。
- **Core mechanism**：对合法 instruction A/B 及 target-A/target-B/非目标视觉干预形成二因子设计，计算预注册的动作交互量；训练变体只在存在各目标合法动作监督时约束正确交互，保留策略完整视觉通路。
- **Key insight**：换词导致动作变化并不能证明模型看了正确对象；目标遮挡导致动作变化也不能证明模型按语言选择对象。只有 instruction 与对应视觉证据的交互才排除两类单因素 shortcut。
- **Application domain**：OpenVLA/OFT 与 LIBERO 多候选 pick/place；若标准数据不支持合法配对，则降为 controlled diagnostic/evaluation，不将伪反事实动作加入训练。

## Reviewer-visible falsifier

若在 matched baseline 中，新候选的 interaction、correct-target selection 或 rollout 不能超过 wrong-map、instruction-shuffle、uniform/full-image、RS-CL/inverse-dynamics 对照，或提升由新增 demonstrations/augmentations 解释，则核心主张失败。

## Candidate identity lock

本审计不把旧的“语言区域 pooling + action-chunk distance + triplet loss”改名为新候选。任何根据审计结果提出的新方法必须另列，不能回填并抬高本候选的 novelty level。
