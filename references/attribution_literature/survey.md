# Attention 热力图与空间归因调研

## 研究问题

1. 哪些方法把 attention 或 hidden-state 梯度转换为视觉空间归因图？
2. 哪些工作证明 attention 图不等于决策因果证据？
3. VLA 中如何校准动作相关区域，并处理 sink/register？

## 文献谱系

| 分支 | 代表工作 | 图的来源 | 证据强度 | 对本项目的作用 |
|---|---|---|---|---|
| Attention rollout/flow | Abnar & Zuidema, *Quantifying Attention Flow in Transformers* (2020) | 跨层 attention 传播 | 结构线索 | 作为 rollout baseline |
| Attention 可靠性批评 | Jain & Wallace, *Attention is not Explanation* (NAACL 2019) | 与输入扰动和替代 attention 对比 | 反事实证据 | 直接支持 occlusion/反事实实验 |
| Relevance propagation | Chefer et al., *Transformer Interpretability Beyond Attention Visualization* (CVPR 2021) | relevance propagation、梯度和 attention 组合 | 输出相关归因 | 可作为 gradient×activation 的高级对照 |
| 双模态 Transformer 解释 | Chefer et al., *Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers* (ICCV 2021) | 文本 token 到图像 token relevance | 输出相关归因 | 与 `A_lang` 最接近的先验 |
| Gradient saliency | Grad-CAM、Integrated Gradients | 输出对视觉特征的梯度 | 输出相关代理 | 适配 Qwen/LLaVA 等无统一 cross-attention 模型 |
| CLIP/VLM grounding | CLIP-Surgery、DenseCLIP 等 | 文本/类别响应或相似度图 | grounding 代理 | 做不改模型的 VLM baseline |
| Diffusion teacher | Lavender | Stable Diffusion 词级 cross-attention | 外部语义教师 | 产生 `T_sem` |
| Sink/register | StreamingLLM；*Vision Transformers Need Registers* (ICLR 2024) | 特殊 token 或固定 patch 权重 | 稳定性诊断 | raw/sink-free 双轨审计 |
| Interventional attribution | RISE、Meaningful Perturbations、occlusion sensitivity | 遮挡后的输出变化 | 干预证据 | 生成 VLA `Δa` 校准图 |

## 对 VLA 的直接启发

VLA 的 action token、flow action expert、连续 delta EEF head 和多模态 hidden state 并不共享一种 attention 语义。因而不能把某个骨干的 attention 图直接当作另一个骨干的教师。最稳妥的统一接口是：

```text
视觉 token → 输出条件归因 → 图像坐标图 → 干预校准
```

对动作输出，建议至少报告：

```text
A_act_grad = |∂L_action/∂h_i ⊙ h_i|
A_act_occ  = ||a(x) - a(x_mask_i)||₂
```

其中 `A_act_occ` 是外部校准图；attention 权重只作为结构诊断，不作为唯一因果证据。

## 对当前方案的结论

1. `T_sem → A_lang` 的方向与双模态 relevance/grounding 文献一致，但我们的跨骨干统一坐标桥仍是需要实证证明的部分。
2. `A_act` 不能默认取 action attention；对没有标准 cross-attention 的模型，应使用 action loss gradient×activation，并用 patch occlusion 做校准。
3. sink-free 是质量控制而不是主创新：先比较 raw map、sink-free map 和 intervention map，再决定是否让 sink-free 图进入 `L_sem` 或 `L_contain`。
4. VLM 阶段应报告 grounding 指标和 intervention agreement；VLA 阶段应报告动作差异图与目标区域的 pointing/IoU、成功率和 shortcut 反事实。

## 局限

本轮在线 API 检索受到 Semantic Scholar 429、代理/DNS 限制；因此没有把无法复核的 2025–2026 VLA 新论文写成已确认事实。新增论文下载目标在 `download_targets.md`，下载后应记录版本、代码 commit 和 SHA256。
