# 逐篇归纳 B：WAM/VAM 与动态教师

范围：本地摘要或无 Abstract 标题的首页概要；未复现作者实验。页码为 PDF 页。以下判断以本地版本为准，不表示最新线上版本或代码可用性已经核验。文中“借鉴”均为本项目推论。

## P07 — Fast-WAM

来源：FAST-WAM.pdf，第 1 页摘要，13 页。标题问题为 Do World Action Models Need Test-Time Future Imagination?

**摘要归纳**：拆分训练时视频联合建模和推理时显式未来生成。保留 video co-training，却跳过 test-time future prediction；受控变体显示移除视频训练的损失更大。摘要列 LIBERO、RoboTwin、真机以及低延迟结果。

**借鉴**：B 路线必须拆成“训练期未来教师”和“推理期世界模型”两条。先缓存冻结教师图，部署只保留 VLA，才能检验是否需要在线想象。

**重合/限制**：未来建模可改善策略已不是新结论。若只加未来训练损失，很容易落入已有范式；我们需要比较相同 future supervision 下，有无当前空间归因约束的额外收益。

**选择**：高优先级 B 对照，记录训练总步数、视频监督量、推理时延，不能只比较一次动作预测速度。

## P09 — GenRL

来源：GenRL.pdf，第 1 页摘要，27 页。

**摘要归纳**：对齐 foundation VLM/video-language 表征与生成世界模型的 latent space，仅依赖视觉数据完成连接。将语言/视觉提示转为 latent target，在世界模型想象中通过 RL 学行为，面向 locomotion 和 manipulation。

**借鉴**：语言语义目标需要落到动力学表征，不能把语义相似度直接当任务进展；可启发 R_future 的目标标量设计。

**重合/限制**：跨语义与世界模型 latent 对齐已有先例，但它是 imagination-based RL，不是我们的离线 BC + 空间归因。data-free policy learning 不能理解为整个方法无数据。

**选择**：思想参照，首轮不引入 RL 训练管线。

## P11 — ImageWAM

来源：ImageWAM.pdf，第 1 页无标题摘要，19 页。

**摘要归纳**：认为完整未来视频带来冗余 token、外观细节与长时误差，改用预训练图像编辑模型表达当前到目标的变化。推理仍做 image-editing denoising，但不解码目标图像；flow action expert 使用其 KV cache。摘要报告仿真/真机、计算和延迟对照。

**借鉴**：R_future 可从目标变化表征取得，未必需要多帧视频。候选对照为静态语义图、单目标 latent、短时未来 latent，检验时间信息是否真的必要。

**重合/限制**：论文已讨论编辑 cache 的 task-relevant attention。“编辑模型注意力更聚焦”本身不能作我们的创新，也不能从不解码图像推断不运行生成网络。

**选择**：B 路线轻量候选；与 Fast-WAM 在推理阶段的计算不同，需单独计费。

## P12 — LaWAM

来源：LaWAM.pdf，第 1 页摘要，23 页。

**摘要归纳**：在预训练视觉 foundation model 的 latent space 学 latent action 模型，复用前向 decoder 预测未来观测特征，以紧凑 latent visual subgoal 条件化策略，避免像素视频生成。摘要列 LIBERO、RoboTwin 和真机。

**借鉴**：B 的未来目标可定义在语义 latent，而非整帧像素差。给 R_future 做目标接近度、语义变化与像素差三类对照。

**重合/限制**：latent action 不自动等于真实 7D delta EEF。未来表示是否接受我们的候选控制动作、如何关联物理时间，仍需接口核验。

**选择**：高优先级未来表征参照；先离线教师，不重新训练全套 LaWAM。

## P13 — Latent-WAM

来源：Latent-WAM.pdf，第 1 页摘要，29 页。

**摘要归纳**：研究自动驾驶轨迹规划。SCWE 蒸馏几何知识并压缩多视角图像；DLWM 使用因果 Transformer 从历史视觉与运动表征预测未来世界状态。摘要 benchmark 是 NAVSIM v2 和 HUGSIM。

**借鉴**：多视角坐标、压缩 latent 与动力学分开设计；借鉴 matched-data/parameter 对照。

**重合/限制**：名字与 LaWAM 接近，领域和动作空间不同；不能作为机械臂 LIBERO/DROID 的直接强基线。

**选择**：跨领域方法启发，低优先级，不扩展驾驶实验。

## P14 — MV-WAM

来源：MV-WAM.pdf，第 1–2 页摘要，20 页。

**摘要归纳**：联合视觉预测、动作生成和 value estimation；用跨模态 causal mask 分层关联 future video、action 与 value，辅以 manifold-aware optimization 和 progress-value regulation，通过 value-guided rollback 处理执行偏差。摘要报告 RoboTwin random 和双臂真机任务。

**借鉴**：未来视频与动作是否一致需要独立的进展评分，不应只看二者 attention 相似；value 评分可能成为 S^k 候选。

**重合/限制**：它已覆盖未来动作一致性和进展反馈，不能把“未来正确才动作正确”作为新概念。回滚依赖闭环状态与执行，不是 DROID 离线数据可以直接测的指标。

**选择**：B 路线高相关工作；方法复杂，首轮可只做受控评分器/归因实验，不默认为完整复现。

## P16 — MotionEnhancer

来源：MotionEnhancer.pdf，第 1 页摘要，26 页。

**摘要归纳**：从视频扩散模型蒸馏运动先验，通过 attention alignment 增强 VLM 的细粒度运动理解。MHS 选择 motion-sensitive heads，MTTI 识别 motion-salient text tokens，两者为无需新增训练参数的选择模块。摘要称在两类 motion-level video understanding benchmark 上有一致提升。

**借鉴**：教师头与词不是无差别平均；可以用验证集上的运动相关性选教师头/动词，并与随机头、同数量头比较。

**重合/限制**：这是最接近“把 Lavender 从图像扩展成视频运动教师”的工作。单纯视频教师+注意力对齐已不足以构成我们的新意。其 motion understanding 结果也不能直接证明机器人控制有效。

**选择**：VLM 动态扩展必读；摘要没有完整 benchmark 名称，需正文查明再冻结评估。

## P20 — Robust-WAM

来源：Robust-WAM.pdf，第 1 页无标题摘要，13 页；自动扫描误命中第 3 页的 abstract representation，阅读采用首页。

**摘要归纳**：保留已有 VAE 空间视频生成通路，给动作流加入 semantic foresight alignment。learnable query 对齐真实未来帧的语义特征，并采用对应 action token 的位置编码建立时间对应；旨在利用 VGM 预训练又减少外观偏差，评估多个 WAM 的 ID/OOD 和真机。

**借鉴**：语义监督可加在动作流，不一定重建世界模型；query 与未来/action step 时间对应必须明确。R_future 应区分 teacher 用真实未来监督与部署使用预测未来的可用性。

**重合/限制**：与 B 的“未来语义监督动作”高度重合。我们的潜在区别是从未来目标回溯到当前空间证据，而不是仅对齐 hidden states；必须实验验证，不能现在宣布未被覆盖。

**选择**：B 的最高优先级相邻工作；加入直接 semantic-feature alignment baseline。

## P25 — World Action Models: A Survey — Dream Less, Act More

来源：WAM_survey.pdf，第 1 页概要，57 页；自动抽取的 Abstract 命中目录，已人工纠正。

**摘要归纳**：区分 broad world model、video generation、action-grounded video model、VLA 和 WAM；一条轴按 rendered futures、latent futures、video-generation-free action reasoning 分类，另一条轴按 predictive substrate、backbone、action coupling、deployment regime 拆分。强调控制价值与时延/算力/数据的权衡。

**借鉴**：选 B 模型时记录“预测什么、如何接动作、训练时还是推理时使用、需要哪些标签”。不再用 WAM/VAM 名称粗略归为一类。

**重合/限制**：综述提供组织框架，不提供我们的算法有效性证据；它收录方法需回溯原论文。

**选择**：作为 WAM 文献地图，保留多种未来教师选项，不预设昂贵视频是最强教师。

## P36 — LingBot-VA 2.0 / Native Video-Action Pretraining

来源：lingbot_va2_2607.08639.pdf，第 1 页概要，29 页。

**摘要归纳**：从具身需求出发设计 semantic visual-action tokenizer、原生 causal pretraining、稀疏 MoE 和异步闭环；并行预测未来 latent 与执行动作，并用最新观测重新 grounding。

**借鉴**：教师 tokenizer 优化语义/动作还是像素重建，会影响 R_future 的质量；预测到执行间的观测陈旧问题必须纳入闭环分析。

**重合/限制**：这是基础模型系统设计；不能把我们的轻量归因正则与其从头预训练直接比较。没有摘要证据表明它直接提供可靠区域教师。

**选择**：后期 teacher/系统参照，与旧 LingBot-VA 分开版本记录。

## P37 — LingBot-VA / Causal World Modeling for Robot Control

来源：lingbot_va_2601.21998.pdf，第 1 页概要，31 页。

**摘要归纳**：自回归扩散联合学习预测帧和动作，使用 MoT 视觉动作交互、真实观测反馈的闭环 rollout、异步预测执行；评估仿真与真机，关注长时域、数据效率与新配置泛化。

**借鉴**：有动作条件的世界模型才有资格解释候选动作后果；比较不同候选时保持观测与随机噪声一致。

**重合/限制**：联合预测并不自动形成可解释的 R_future，也不意味着视频预测可靠就能控制成功。

**选择**：B 的可审计入口候选；继续核验归因所需梯度和未来目标接口。

## P38 — LingBot-Video

来源：lingbot_video_2607.07675.pdf，第 1 页概要，51 页。

**摘要归纳**：面向具身的 MoE DiT 视频预训练；数据增加操作、导航、第一人称视频，训练加入物理合理性与任务完成等奖励维度，区别于仅追求美观的视频生成。

**借鉴**：筛选教师不能只看 FVD 或生成质量，要核对接触、物体状态与任务进展。

**重合/限制**：首页概要没有证明模型接受我们的 delta EEF 候选动作；在动作接口未确认前只能称视频先验，不能称动作反事实模拟器。

**选择**：视频教师备选，优先审计再决定是否下载权重。

## P39 — LingBot-VLA 2.0

来源：lingbot_vla2_2607.06403.pdf，第 1 页概要，20 页；标题 From Foundation to Application: Improving VLA Models in Practice。

**摘要归纳**：扩大机器人与人类视频预训练，扩展头部、腰部、移动底盘与灵巧手动作空间，并使用视频语义与深度几何先验辅助预测动力学；评估 GM-100 和跨平台长程移动操作。

**借鉴**：未来视频/深度辅助任务是现成对照，不能把加入两种教师本身称为核心创新；记录多 embodiment action schema。

**重合/限制**：它的全身动作不能未经说明硬转成七维。主线仍保持单臂 delta EEF 范围，若采用此骨干要限定兼容子设置。

**选择**：成熟 VLA 候选之一，是否比 OpenVLA/OFT、π0、MolmoAct2 更适合，需要相同数据与接口审计。

## P40 — LingBot-VLA / A Pragmatic VLA Foundation Model

来源：lingbot_vla_2601.18692.pdf，第 1 页概要，20 页。

**摘要归纳**：扩大真实双臂数据，系统考察跨任务/平台适配与成本；摘要描述四平台、每平台 100 任务和统一 post-training 示范量，还介绍高吞吐训练实现。

**借鉴**：把 sample efficiency、训练吞吐和任务成功同时记录；比较基线时匹配 adaptation 数据量，而非只对照各论文最佳结果。

**重合/限制**：大规模基础模型/工程路线，不能据其泛化成绩推断我们的归因方法一定有效。

**选择**：VLA 候选，原版与 2.0 不混用指标。

## P41 — LingBot-VLA v2 的另一份文件

来源：lingbot_vla_v2_2607.06403.pdf，第 1 页概要，20 页。

**摘要归纳**：与 P39 同标题、同 arXiv 版本和摘要内容。其借鉴与重合沿用 P39，不当作独立文献来增加覆盖数。是否字节完全重复见 inventory 的 SHA256。

**选择**：保留用户原文件，引用时合并为同一研究。

## P42 — Next Forcing

来源：next_forcing.pdf，第 1 页摘要，18 页。

**摘要归纳**：为 causal world modeling 增加多未来 chunk 的去噪目标，辅助模块沿因果链连接主模型多层特征，以更密集的多时域监督改善收敛；推理可保留辅助模块并行预测。摘要报告不同帧率、RoboTwin、PhyWorld 和视频生成结果。

**借鉴**：R_future 若引入多时域，必须明确短/中/长的物理时间与权重；比较 same-horizon 重复、随机 horizon、wrong-action，并匹配额外训练成本。

**重合/限制**：它提出多 chunk 训练，不是 phase 标签方法，也没有在摘要中证明未来归因可监督当前空间证据。我们之前“未来更强即可解决 shortcut”的说法过于宽泛。

**选择**：B 后期候选；先证明单时域未来目标归因有价值，再增加多时域。
