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

## VLA 候选创新性审计（2026-10-16）

- 完整报告：[`vla_novelty_audit_20261016/report.md`](vla_novelty_audit_20261016/report.md)
- 结论：原“语言区域汇聚 + action-chunk 距离 triplet”候选为 **Level 2 — High Overlap**。RS-CL 已在 VLA 中以 proprioception 距离塑造对比表征，PSE 已以行为相似性定义表征距离，FOCAL-VLA 已在子任务区域汇聚几何/未来运动教师特征。因此原候选保留为低成本机制对照，不冻结为论文主创新。
- 审计边界：尚未发现逐项完全相同的方法；多源检索存在 arXiv SSL、DBLP 解析、OpenReview 依赖和 Semantic Scholar 429 失败，不能将未检出解释为不存在相关工作。
- 原文核验：RS-CL、PSE、inverse-dynamics VLA、FOCAL-VLA、TCN、DBC，以及用户提供的 Copper-Policy。PDF 与全文提取保留为本地审计缓存，不进入 Git 或飞书。
- 后续提出的“指令重定向 × 目标视觉干预”是独立新候选，尚未完成自己的创新性审计，不因原候选被否定而自动获得新颖性。

## Instruction–Evidence 最终选题审计（2026-10-16）

- 审计目录：[`vla_instruction_evidence_audit_20261016/`](vla_instruction_evidence_audit_20261016/)
- 最终结论：**B — Freeze as diagnostic/mechanistic paper**。冻结问题为“VLA 的动作重定向是否真正依赖语言所指定 referent 的视觉证据”。不冻结新的训练 loss。
- 创新性：宽泛候选为 **Level 2 — High Overlap**。ECL 已做 instruction/action evidence selection + counterfactual masking + action supervision；LIT 已分别做 visual/goal interventions；Robust Skills 已拆分 target selection 与 primitive execution。剩余差异是二因子 crossed interaction estimator，适合作为诊断测量。
- 数据：解析官方 130 个 LIBERO task 和十个 scene-compatible pair；当前 432 条 RLDS 仅覆盖 Spatial。十个候选 pair 均没有 exact shared init state + 双端合法动作 inventory，因此只支持 diagnostic-only intervention。
- 最小实验与否定条件：见 [`next_minimal_experiment.md`](vla_instruction_evidence_audit_20261016/next_minimal_experiment.md) 与 [`falsification_gate.md`](vla_instruction_evidence_audit_20261016/falsification_gate.md)。

## Causal-Locus Language-Evidence Residual 审计（2026-10-17）

- 审计入口：[`causal_language_residual_scoop_20261017/REPORT.md`](causal_language_residual_scoop_20261017/REPORT.md)
- Verdict：**Level 2 — High Overlap**。语言“encoded but not in control”、VLA activation patching、latent action query/PMI、text-latent residual steering、空间图前向 gating 和 paired counterfactual training 均已有直接近邻。
- 剩余候选差异：用合法同场景换指令的 exact activation patching 冻结 action-causal loci，再将 Bridge-0 的 phrase-specific source/target spatial evidence 以零初始化残差写入这些 loci，并与 LC-PEA 的 target-relative future-change 联训。后续补检发现 ProGAL-VLA 已使用 symbolic sub-goal→3D entity verification、对比对齐和 verified-goal action conditioning，进一步限制宽泛 grounding/action 创新表述。
- 决策：保持为 LC-PEA 后置的二级增强。只有相对 FiLM、随机层、全层、language-only、evidence-only、LangForce/ECT/PosA 等强对照仍提高 target selection 和官方 rollout，才升级为论文方法；否则 L0 只作诊断。
- 检索边界：OpenAlex/Crossref 可用；arXiv 已通过单源 10 秒限速恢复；`openreview-py`/`feedparser` 已安装并验证 connector 可运行。Semantic Scholar 因没有 API key 持续匿名 429；DBLP 返回浏览器 anti-bot HTML challenge。两者使用 OpenAlex/Crossref/官方 PDF 替代召回并保留缺口。RoVLA temporal-grounding CASE 2026 只有 DOI 元数据、正文待核验。PDF 与全文提取仅作本地审计缓存，不进入 Git/飞书。
