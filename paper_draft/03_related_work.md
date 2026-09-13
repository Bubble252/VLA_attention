# 3. Related Work 组织稿

## 3.1 Diffusion-guided VLM grounding

Lavender 使用 Stable Diffusion 的词级 cross-attention 作为外部空间教师，并对选定层进行监督。本文继承其 `word → region` 先验，但把学生对象从同构 cross-attention 扩展为统一的 `A_lang`：SpikingBrain 可读取 full-attention，Qwen/LLaVA 使用输出梯度归因，OpenVLA 使用 action-token 或 adapter 归因。关键区别是教师图相同，学生归因接口按模型结构实现。

## 3.2 Attention visualization 与 Transformer attribution

Attention rollout/flow 聚合多层权重，Transformer relevance propagation 和 Grad-CAM/Integrated Gradients 将输出依赖传播到特征。它们提供结构线索或输出条件代理，但 attention 不必然等价于决策解释。因此本文把 attention 放在证据层级的低位，gradient×activation 放在通用接口，patch occlusion 作为外部校准。

## 3.3 Attention sink、register 与图像伪热点

StreamingLLM 和 Vision Transformers Need Registers 说明特殊 token 或固定位置可能吸收大量权重。本文不提出新的 sink-removal 算法，而是在 P3 做 raw/sink-free 双轨诊断，报告特殊 token、边缘 patch、空间熵和干预一致性。若 sink-free 仅改善视觉观感而不改善 intervention agreement，不将其写成方法收益。

## 3.4 VLM/VLA grounding 与 action attribution

Qwen-VL、LLaVA、OpenVLA/OFT、π0、LingBot-VLA 等模型在视觉—语言—动作结构、动作 token、flow/action expert 和多帧输入上存在差异。本文不把任意一个模型的 attention 图当作通用教师，而是统一到图像坐标后比较 `A_lang` 和 `A_act`。已有动作可视化结果需要用 occlusion、语言反事实和 state 遮挡重新校准。

## 3.5 Manipulation phase 与未来模型

TACO、BUDS、CHAMP、RoboSegNet、Robo2VLM、ProcVLM 和 ROVER 提供从轨迹、视觉和 proprioception 推断阶段/进度的思路。本文 D1 首先采用 LIBERO event boundary 的弱 phase；WAM/VAM、LingBot-VA、Next Forcing 只在 B 路线生成未来归因 `R_future`，不参与 D0 默认损失。

## 3.6 研究空白

现有路线分别解决 grounding、解释、阶段分割或未来预测，尚未在本文的统一协议下同时回答：

1. 不同 VLM/VLA 的内部证据能否映射到同一个可审计空间图？
2. 语言 grounding 约束能否改善动作归因，而不是只改善图像问答？
3. 热力图是否与动作干预和跨场景控制一致？

本文围绕这三个问题组织实验，而不是把多个 teacher 简单叠加。
