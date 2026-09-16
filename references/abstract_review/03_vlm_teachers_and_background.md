# 逐篇归纳 C：VLM、教师、归因基础和后置材料

阅读范围：本地摘要/首页概要；P26 摘要在第 2 页，P52 是报告幻灯片而非带摘要的论文。所有方法效果为作者主张，不是本项目结果。摘要能支持研究定位，不能替代具体方法、指标和代码复核。

## P03 — Attention Distillation: A Unified Approach to Visual Characteristics Transfer

来源：Attention Distillation.pdf，第 1 页摘要，11 页。

**摘要归纳**：使用预训练 diffusion self-attention 特征迁移风格、外观与纹理；构造 attention distillation loss，在 latent space 优化生成图像，并把损失结合到采样 guidance。

**借鉴**：教师信号、被监督对象和优化位置必须分清。可参考选择中间 attention 特征作为监督的设计。

**重合/边界**：不是 VLA attention distillation，也不是机械臂动作热力图。不能因文件名把它当作与 D0 相同的机器人方法。

**筛选**：低优先级跨领域参考，首轮不进入动作 baseline。

## P08 — FullFlow

来源：FullFlow.pdf，第 1 页摘要，36 页。

**摘要归纳**：把预训练 text-to-image rectified-flow 模型变成双向 vision-language generator，仅训练 LoRA 与轻量文本头；图像保持 continuous flow，文本采用 discrete insertion process，两模态独立时间步支持生成、描述、联合采样和 VQA。

**借鉴**：扩散模型可能提供丰富视觉语义，但要明确是冻结教师还是另造生成式 VLM。该文的 matched trainable parameters、LoRA rank、wall-clock 对照值得用于我们的训练成本表。

**重合/边界**：与 diffusion→VLM 知识利用动机相邻，训练目标是联合生成，不等于空间归因监督或 VLA 控制。

**筛选**：教师背景，不因它另开生成模型训练主线。

## P17 — OREO.pdf 实际为 OREOS

来源：OREO.pdf，第 1 页标题、摘要，8 页；本地内容研究 oriented place recognition with 3D LiDAR scans（arXiv:1903.07918）。

**摘要归纳**：CNN 从 LiDAR scan 提取紧凑描述符，检索相邻位置并估计 yaw 差，用 triplet loss 和 hard negative mining 训练，在 NCLT/KITTI 做长期室外定位。

**借鉴**：仅能抽象借鉴 hard-negative 和位置/方向联合表征，和本项目动作语义归因关联弱。

**重合/边界**：这不是此前讨论的 object-aware policy regularization OREO。不能用它的文件存在证明对应机器人正则论文已归档，也不能把两者 benchmark 混淆。

**筛选**：从当前核心引用中排除，保留文件并标错配；如要比较机器人 OREO，另核对原论文。

## P23 — SpikingBrain 中文技术报告

来源：SpikingBrain_Report_Chi.pdf，第 1 页摘要，38 页。

**摘要归纳**：报告类脑脉冲模型的线性/混合线性架构、转换训练、脉冲编码和国产 GPU 工程，涉及 7B/76B、长序列效率与稀疏性。

**借鉴**：仅作为未来异构结构适配的背景。若后续做效率，必须区分脉冲活动率、真实 kernel 加速与硬件能耗。

**重合/边界**：摘要主要是语言模型与系统效率；不能据其 TTFT/sparsity 宣称 VLA 更准或更节能。不能把“生物启发”当作当前归因方法的实验依据。

**筛选**：用户已确认后置，有空再做，不列任何首轮阻塞条件。

## P24 — SpikingBrain 英文技术报告

来源：SpikingBrain_Report_Eng.pdf，第 1 页摘要，40 页。

**摘要归纳**：同一模型家族的英文技术报告，强调 MetaX 系统、adaptive spiking neurons、linear/hybrid-linear、模型转换和长上下文效率。

**借鉴**：未来核对公式与代码可与 P23 对读；指标应按不同任务和平台分别解释。

**重合/边界**：与 P23 是同一研究家族，不能当两篇独立实验来证明普适性。

**筛选**：后置材料，和 P50 合并文献记录。

## P26 — WordCon: Word-level Typography Control in Scene Text Rendering

来源：WordCon.pdf，第 2 页无标题摘要；首页是 Figure 1，12 页。

**摘要归纳**：构建词级排版控制数据，用 grounding 的文本—局部图像对应训练 T2I；hybrid PEFT、latent masked loss 与 joint-attention loss 控制不同词的字体属性和空间解耦。

**借鉴**：词级对齐必须核对多 token span 和相邻词串扰；对 source/target 词图可加交换词、同类多实例的负控。

**重合/边界**：不是机器人动作或 world model。文中的印刷词区域和机器人目标物体区域语义不同，attention loss 形式相似不说明任务相同。文件内 DOI 有占位符，正式引文不能照抄占位 DOI。

**筛选**：辅助读物，不进入 VLA baseline 主表。

## P28 — Efficient Streaming Language Models with Attention Sinks

来源：attention_sinks_2309.17453.pdf，第 1 页 A BSTRACT，21 页。

**摘要归纳**：滑动 KV cache 移除初始 token 后，长流式生成性能会退化；保留初始 sink token 的 KV 能恢复稳定性。StreamingLLM 保留 sink，并研究预训练时引入专用占位 sink。

**借鉴**：注意力高不等于语义重要，但 sink 可能承担数值/分布功能。空间可视化中排除非视觉 token 与改变模型前向注意力是两件事，必须分开。

**重合/边界**：本论文支持 sink 存在及其功能，不支持默认删除视觉边缘 patch，更不证明所有高值 gradient 是 sink。此前文档的默认 mask 建议需收紧。

**筛选**：归因质量控制必读；首轮 raw 前向不改，校正图仅作为对照，验证后才决定训练处理。

## P29 — DeepSeek-VL2

来源：deepseek_vl2_2412.10302.pdf，第 1 页摘要，28 页。

**摘要归纳**：采用 dynamic tiling 视觉编码与 DeepSeekMoE/MLA 语言组件，覆盖 VQA、OCR、文档表格图表与 visual grounding，多种激活参数规模。

**借鉴**：多 crop/tile 的视觉 token 必须重映射，不能直接 reshape；MoE/router 与 MLA latent 不是词到像素 attention。

**重合/边界**：这是通用 VLM，不能当 VLA action baseline。选择依据应为接口差异和可审计程度，而非只看参数量。

**筛选**：Qwen+LLaVA 之后的 VLM 扩展，不阻塞首轮。

## P31 — Gated Linear Attention Transformers with Hardware-Efficient Training

来源：gla_2312.06635.pdf，第 1 页摘要，23 页。

**摘要归纳**：在 matrix-valued recurrent states 的线性 attention 上引入 data-dependent gating，并设计 I/O-aware Flash Linear Attention 实现，评估语言模型质量、速度和长度外推。

**借鉴**：非 softmax attention 的状态需要通过输出条件归因映射，不应当作空间概率图。训练可微接口与高阶梯度是否支持仍需实际检查。

**重合/边界**：计算架构论文，不能直接支持 VLA 空间对齐效果。

**筛选**：SpikingBrain 等异构结构扩展时再深入 kernel，不列首轮主实验。

## P33 — When Less is More: 8-bit Quantization Improves Continual Learning in Large Language Models

来源：int8.pdf，第 1 页摘要，7 页。

**摘要归纳**：比较 FP16/INT8/INT4 与 replay buffer 对语言任务序列中的遗忘与新任务学习影响，提出量化噪声可能提供正则化的解释。

**借鉴**：量化、replay 和可训练模块是训练混杂因素，baseline 应匹配；若未来讨论 VLA 遗忘，可独立检查精度差异。

**重合/边界**：结果来自语言持续学习，不能推广成“INT8 VLA 一定更泛化”或“量化代替空间监督”。标题中的收益是研究条件下的报告。

**筛选**：低优先级，不因量化新开主线。

## P34 — Diffusion Instruction Tuning / Lavender

来源：lavender_2502.06814.pdf，第 1 页摘要/图 2，38 页。

**摘要归纳**：SFT 时对齐 VLM 的 text-vision attention 与冻结 Stable Diffusion 的对应图；图 2 描述用三层 ConvNet 变换学生图并用 MSE 正则，报告 Llama-3.2-11B、MiniCPM 系列及 ID/OOD 视觉语言任务。

**借鉴**：固定教师缓存、词 span 和空间桥；核心 baseline 包括原始 SFT、Lavender-style map alignment、我们的 output-attribution alignment。报告参数和教师离线成本。

**重合/边界**：外部扩散空间教师、层选择和注意力 MSE 都已有；我们的 novelty 不能停在换 backbone 或加动作头。教师也可能定位错误，IoU 不能只用教师图自身评估。

**筛选**：核心教师与最直接 VLM 方法 baseline；摘要的 0.13M 训练量不等于 Flickr1k 归因图子集，数据具体角色必须查正文和仓库。

## P45 — The Prism Hypothesis / Unified Autoencoding

来源：prism.pdf，第 1 页摘要，12 页。

**摘要归纳**：分析 semantic encoder 与 pixel encoder 的特征频谱，提出用 frequency-band modulator 统一语义结构与像素细节的 autoencoding 表征，服务生成与重建。

**借鉴**：teacher latent 是否保留无关纹理，可能影响 B 的鲁棒性；可对语义/像素目标做简单对照。

**重合/边界**：频谱观察是特定表征研究，不是“低频必然语义、高频必然无用”的普遍定理。不能据此删除抓取边缘等高频细节。

**筛选**：B 表征选择的背景，首轮不增加频域损失。

## P46 — Qwen2.5-VL

来源：qwen25_vl_2502.13923.pdf，第 1 页摘要，23 页。

**摘要归纳**：提升定位、文档与视频理解，采用原生动态分辨率 ViT/window attention 和时间编码，支持 points/bboxes 等结构输出。

**借鉴**：适合作为 VLM 首轮候选；区分模型直接输出 bbox 的准确率与我们从输出归因得到的 pointing/IoU，二者不能混为同一指标。

**重合/边界**：有 window/global 层级不等于类脑独有特性；层选择应由实际接口和验证集决定，不搬用 SpikingBrain 的层号。

**筛选**：与 LLaVA-1.6/OneVision 形成跨系列 VLM 组合，具体 checkpoint 待用户确认。

## P47 — Qwen2-VL

来源：qwen2_vl_2409.12191.pdf，第 1 页摘要，52 页。

**摘要归纳**：Naive Dynamic Resolution 让输入分辨率对应可变视觉 token 数，M-RoPE 联合表达文本、图像、视频位置，讨论模型/数据 scaling。

**借鉴**：token 数、crop、时间和位置编码应进入归因图元数据，防止测试图尺寸变化导致假峰值。

**重合/边界**：提供架构基线；同一系列多个代际不等于多种独立归因结构的普适性证据。

**筛选**：若主跑 Qwen2.5/3，只作历史接口参考。

## P48 — Qwen3-VL

来源：qwen3_vl_2511.21631.pdf，第 1 页摘要，42 页。

**摘要归纳**：dense/MoE 系列、长 interleaved 多模态上下文和强化推理，升级 interleaved-MRoPE 与视觉层特征使用等架构设计。

**借鉴**：多层注入时 A_lang 与 A_act 的梯度 hook 要记录接入层；多图/视频时保留 image/view/time 身份。

**重合/边界**：长上下文能力与机器人动作 grounding 是不同结论。不能以模型新代际自动保证归因更忠实。

**筛选**：与 Qwen2.5 二选一先跑，另一个只在接口/结果有明确价值时加。

## P50 — SpikingBrain: Spiking Brain-inspired Large Models

来源：spikingbrain_2509.05276.pdf，第 1 页摘要，40 页。

**摘要归纳**：与 P24 同标题和研究家族，强调 linear/hybrid-linear、脉冲编码、转换训练与 MetaX 长上下文效率。

**借鉴/重合**：与 P23/P24 合并理解；具体文件是否重复由 SHA256/正文比较决定，不能重复计为三项独立支持。

**筛选**：已确认后置，仅未来 appendix/结构迁移候选。

## P51 — Toward Visual Grounding: A Survey

来源：visual grounding survey.pdf，第 1 页摘要，30 页。

**摘要归纳**：梳理 referring expression comprehension、phrase grounding、grounded pretraining、MLLM grounding、generalized 和 giga-pixel grounding；统一任务设置、数据集和指标以支持公平比较。

**借鉴**：明确我们测的是词区域定位、回答证据还是输出行为忠实性。Flickr30k Entities 与 RefCOCOg 可为前者提供标注；不存在对象、多实例和关系指令另设测试。

**重合/边界**：grounding 准确不等于 action faithfulness；我们不能把两者用单个 heatmap IoU 包办。

**筛选**：选数据和指标的必读地图；本目录没有 LLaVA PDF，若最终选该骨干，需补独立论文来源，不虚称本轮已读。

## P52 — 脑启发的新型混合模型架构（石润林）

来源：中文同名 PDF，34 页报告幻灯片，无摘要。已读第 1–3 页题目/概览/假设，第 4–8 页 attention head 行为，第 9–16 页 RoPE/频率重要性分析；不是完整论文精读。

**概要归纳**：讨论短时精确时序与长期检索的分工，分析检索头/位置头、GQA 下 RoPE 频率重要性及层间特异，再导向层内混合架构。首页声明为作者本科毕设展示，观点不代表机构。

**借鉴**：选择监督头应依据任务功能与干预，而不是只挑 entropy 最低或图最好看的 head；按层/头报告差异可能比全层平均更有意义。

**重合/边界**：这些假设和观察不能直接作为 VLA 机制定论，也不能用报告中的类脑比喻替代实验。我们尚未核验其中所有图表和消融。

**筛选**：保留为后置结构分析启发；用户已要求 SpikingBrain 不做主线，此报告不改变该约束。
