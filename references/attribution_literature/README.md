# 热力图与空间归因文献调研

检索范围：VLM 语言 grounding、Transformer 归因、VLA 动作归因、attention sink 与干预验证。在线 API 本轮受到 Semantic Scholar 429 和部分网络限制，以下优先收录已有本地 PDF/仓库或高置信经典工作；新增条目的版本和代码需在下载后复核。

## 方法谱系

| 类别 | 代表工作 | 证据形式 | 对本项目的启发 |
|---|---|---|---|
| Attention rollout | Abnar & Zuidema, *Quantifying Attention Flow in Transformers* (2020) | 跨层 attention 传播 | 可作基线，但不能当作因果证据 |
| Attention explanation | Jain & Wallace, *Attention is not Explanation* (NAACL 2019) | attention 与输入扰动对照 | 支持我们必须加入 occlusion/intervention |
| Transformer attribution | Chefer et al., *Transformer Interpretability Beyond Attention Visualization* (CVPR 2021) | relevance propagation/梯度 | 可作为 gradient×activation 之外的归因对照 |
| ViT attribution | Chefer et al., *Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers* (ICCV 2021) | 双模态 relevance map | 与 VLM 语言到图像归因最接近 |
| Gradient saliency | Grad-CAM、Integrated Gradients | 输出对视觉特征的梯度 | 适合 Qwen/LLaVA 等无统一 cross-attention 模型 |
| CLIP/VLM grounding | CLIP-ES、CLIP-Surgery、DenseCLIP 系列 | 文本-图像相似度/类别响应图 | 可作为不改模型的 grounding baseline |
| Diffusion teacher | Lavender (2025) | Stable Diffusion 词级 cross-attention | 我们的 `T_sem` 教师来源 |
| Sink/register | *Vision Transformers Need Registers* (ICLR 2024)；StreamingLLM (2023) | 特殊 token/边缘 patch 吸收权重 | 支持 raw/sink-free 双轨审计 |
| Intervention | RISE、Meaningful Perturbations、Occlusion sensitivity | 遮挡后输出变化 | 作为 VLA 动作归因校准证据 |

## VLA 相关处理原则

公开 VLA 工作通常展示 action token、cross-modal attention 或 feature visualization，但不同骨干的 attention 语义并不一致。OpenVLA/OFT、π0/openpi、LingBot-VA 等更适合作为动作接口和结构参照；本项目不把它们的 attention 图直接当教师，而是统一计算 `A_act`，并用 patch intervention 检查其决策相关性。

## 本项目必须吸收的实验

1. raw attention、rollout、gradient×activation 三种图并列；
2. head-wise/layer-wise 与平均图并列；
3. patch occlusion 生成 `Δa` 校准图；
4. 语言反事实、state 遮挡和背景/相机扰动；
5. sink/register mask 前后分别报告，不把 sink removal 本身包装成核心创新。

完整检索结果和失败信息记录在仓库根目录 `allinone.md`；下载目标见 `references/attribution_literature/download_targets.md`。
