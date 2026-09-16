# 1. Idea、类脑结合与研究定位

## 1.1 研究问题

机器人需要同时解决三个问题：指令中的对象在哪里、当前动作依赖哪些视觉证据、动作执行后是否朝目标推进。论文只把第一个问题的语义教师和第二个问题的动作归因作为首轮主线；第三个问题由 WAM/VAM 的 `R_future` 作为后期扩展。

核心研究问题：

> 在没有统一 cross-attention 的异构 VLM/VLA 中，能否用外部词级空间先验和模型自身的输出归因，建立可验证的语言—动作空间一致性，并改善真实视觉分布下的机器人动作预测？

## 1.2 原始想法到最终方法

原始 Lavender/SpikingBrain 想法是对 selected attention layers 施加扩散图 MSE。现在将其提升为模型无关主线，SpikingBrain 只保留为后续类脑扩展。最终收敛为三步：

1. Lavender/Stable Diffusion 产生 `T_sem(w,x)`；
2. 学生模型用原生 attention、gradient×activation 或 action-token attribution 产生 `A_lang`；
3. VLA 用 action loss gradient×activation 产生 `A_act`，并以 `L_contain` 约束其落在语言相关区域。

因此贡献不是“设计一张更好看的 attention 图”，而是定义一套可跨架构审计的空间证据接口。

## 1.3 类脑结合的准确边界

类脑结合只在三个可检验的结构层面发生，不声称复制生物神经系统：

- **主线不预设类脑结构**：OpenVLA/OFT、π0 系或其他成熟 VLA 先承担可复现实验；模型只需提供视觉 token 和动作输出归因接口。
- **类脑扩展**：SpikingBrain 的稀疏脉冲表征、局部—全局层级和 GLA 递归状态作为后续结构消融，检验空间归因约束能否迁移到非标准注意力架构。
- **局部—全局层级**：window/SWA 承担局部建模，full-attention block `[7,15,23,31]` 提供跨物体整合，GLA 提供递归状态；只在可解释的 full-attention 或 gradient proxy 上施加监督。
- **时间整合**：VLA 的动作归因随 `approach → grasp → move → place` 改变；D1 用 phase soft target 表示这种动态，而不是把所有时间点压成静态图。

类脑部分服务于方法选择：它解释为什么要做层级选择、局部/全局消融和时序 phase 分析；它不是额外的生物学论断。

## 1.4 创新边界

已知工作已经覆盖 attention rollout、Transformer relevance、Grad-CAM、diffusion grounding、VLA action token 可视化和 occlusion。本文的差异轴是：

1. 用同一空间坐标桥比较异构 VLM/VLA 的输出条件归因；
2. 将语言归因和动作归因连接成 `T_sem → A_lang → A_act`，而不是只解释单一输出；
3. 在脉冲模型层级结构中选择性监督，并用干预实验校准热力图；
4. 用 DROID 检查方法是否跨越 LIBERO 的视觉捷径。

若实验只能证明单模型 grounding 改善，则论文应降级为“结构感知 VLM 归因对齐”，不宣称普适 VLA 方法。
