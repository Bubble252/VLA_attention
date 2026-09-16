# 逐篇归纳 A：VLA、空间监督与视觉捷径

阅读范围：本地 PDF 的摘要，必要时补读引言和实验设置。下面的效果均为作者报告，未由我们复现。每节的“借鉴”和“重合”是本项目分析，不是原论文原话。P 编号对应 inventory.json；页码为 PDF 文件页码。更细的实验预算、checkpoint 和许可证必须在复现前核验。

## P01 — 3D HAMSTER

来源：3D-hamster.pdf，第 1 页摘要，8 页。标题为 Bridging Planning and Control in Hierarchical Vision Language Action Models through 3D Trajectory Guidance。

**摘要归纳**：2D 规划路径与低层 3D 点云策略存在表示不匹配；把路径像素直接反投影到其下方表面，可能产生错误深度。论文为 VLM 加入深度编码和稠密深度重建任务，直接输出有尺度的 3D waypoint，再交给点云控制器。作者在轨迹预测、仿真和真机报告收益。

**可借鉴**：坐标桥不是简单 resize。我们应区分二维“证据在哪里”和三维“动作怎么执行”；多相机图不能拼成一个没有视角标记的平面。可增加深度歧义或遮挡案例。

**重合与边界**：与 C 路线的投影/空间先验相邻；其输出是显式规划轨迹，我们的归因是对模型输出依赖的解释或约束，二者不等价。

**选择建议**：若保留 RGB-only 归因主线，将其作为表示边界和相关工作；若转做 3D planning，则需纳入同传感器、同控制器 baseline。摘要未足以确定所有 benchmark 协议，正文待精读。

## P02 — AVA-VLA: Improving Vision-Language-Action Models with Active Visual Attention

来源：AVA-VLA.pdf，第 1 页摘要，11 页。

**摘要归纳**：单帧、无历史的 VLA 难以处理部分可观测性；该方法以循环状态近似历史 belief，再结合语言与历史重加权当前视觉 token。在 LIBERO、CALVIN 和真实双臂任务评估。

**可借鉴**：D1 应与“历史状态调节关注区域”比较，不能仅以 source→target 热力图迁移作为创新。应匹配输入帧数和历史长度，设置有记忆但无对齐、无记忆但有对齐的对照。

**重合与边界**：与动态视觉注意力和 phase 叙事重合较强；它通过历史驱动前向视觉处理，当前 D 通过输出条件归因施加约束。需验证后者有额外效果。

**选择建议**：D0 首轮作为 related work；若主张时序动态归因，则升级为高优先级比较。CALVIN 可检验长时程，但不必与 DROID 同时开新数据管线。

## P04 — Breaking the Vision–Action Shortcut / LIT

来源：Breaking the Vision-Action Shortcut.pdf，第 1 页摘要；第 5 页实验设置/Table I，第 6 页 Table II，第 8 页 Table III；共 10 页。此前详细笔记的部分页码不准确，以本次 PDF 定位为准。

**摘要归纳**：先不输入图像，以语言、机器人状态和 action chunk 终点 SE(3) 位姿预训练动作专家；随后用受终点位姿重建监督的 latent interface 作为唯一视觉通道。目的在于保留空间信息，同时减少视觉捷径。

**模型巧思**：π0.5 与 MolmoAct2 分别覆盖 shared attention 和逐层 KV conditioning；FAST-WAM 与 ImageWAM 覆盖训练期未来建模和推理期保留生成表征。是接口覆盖，不是单纯增加模型数量。

**baseline 巧思**：同 backbone、随机初始化动作专家、同数据和总优化步数；它明确没有直接微调已有 policy checkpoint，因此不能将其基线分数当作官方 checkpoint 水平。移除组件与替代解释分开：移除 Stage 1、移除 pose loss、恢复直接视觉访问，以及 aggregation-only、staged-training-only、pose-supervision-only。

**benchmark 巧思**：LIBERO 40 任务×50 rollouts；LIBERO-Plus 10,030 个实例、每实例固定 seed 一次评估，按七维扰动做不加权平均。原始 LIBERO 训练、Plus 零样本测试。Table II 存在 language 维度下降，不能把平均提升写成所有维度提升。

**借鉴与重合**：它与我们共享“约束视觉信息如何参与动作”的问题。我们要通过相同初始化下的 BC、语义对齐、containment、等预算辅助监督、LIT 类通道约束，证明贡献不只是额外正则。对照应同时检验对无关变化稳定、对目标变化敏感。

**选择建议**：列为核心相邻工作；LIBERO-Plus 可作为主 OOD 闭环候选，DROID 保留为真实数据离线泛化。独立多 seed 和置信区间可补强该论文的评估。

## P05 — BridgeVLA

来源：BridgeVLA.pdf，第 1 页摘要，39 页。

**摘要归纳**：先将 VLM 预训练成 2D 图像输入、2D heatmap 输出；VLA 阶段把点云投影为多视角图像，在输出动作之前预测热力图，保持输入输出空间结构的一致性。摘要列出 RLBench、COLOSSEUM、GemBench 与真机实验，并报告少样本能力。

**可借鉴**：建立“显式预测空间图”baseline，用它区分我们的输出条件归因是否有必要。若决定做点云/多视角，RLBench+COLOSSEUM 的 ID/OOD 配对有价值。

**重合与边界**：与“先 VLM 空间任务再 VLA”的叙事高度相邻，不能再泛称这条训练顺序是空白。BridgeVLA 的 heatmap 是预测与控制接口；我们的图应定义为特定输出目标的归因，并证明其约束产生独立价值。

**选择建议**：必读方法细节；RGB 与点云方法不要直接排名。若不使用深度，先报告相关工作差异和可比的 heatmap 辅助头，不能称简化实现为完整 BridgeVLA。

## P06 — Don't Blind Your VLA

来源：Don't Blind Your VLA.pdf，第 1 页摘要/图 1，13 页。

**摘要归纳**：系统检查 VLM 转为 VLA 后视觉语义表征的保持程度；使用 hidden-state probe、attention 和针对性任务比较 VLA 与原始 VLM，并以表征对齐减轻退化。首页图示提到中层特征与教师 embedding 对齐，评估使用 Simpler-based 泛化设置。

**可借鉴**：VLA 微调前后必须用同一 grounding/视觉能力保留集；对照冻结视觉编码器、纯 BC、简单 feature anchoring 与我们的方法。

**重合与边界**：与“保持语言视觉能力，再改善动作”动机重合强。我们的区别不能只是从 feature 换成 heatmap，必须说明输出条件空间约束为什么优于普通表征保持。

**选择建议**：作为保留能力实验的首要参考；具体 teacher、层和归一化策略需正文核对，摘要不证明其与我们的损失等价。

## P10 — Generalizable VLA Finetuning / Anchor-Align

来源：Generalizable VLA Finetuning.pdf，第 1 页摘要，39 页；标题为 Generalizable VLA Finetuning via Representation Anchoring and Language-Action Alignment。

**摘要归纳**：纯 BC 会使预训练表征漂移，而互联网图文 cotraining 把语言和动作损失放在不同观测上，仍可能失配。Anchor-Align 一方面蒸馏冻结 VLM 的逐层表征，另一方面把动作转换为离散运动方向标签，在同一机器人观测上联合训练语言和动作。摘要包含两种 VLA、xArm7 真机及 LIBERO-PRO、LIBERO-Plus、CALVIN。

**可借鉴**：设置同一观测上的语言目标，明确 A_lang 的标量目标和动作目标如何配对。增加 representation-anchor-only、language-action-label-only 和二者结合的比较。

**重合与边界**：它已经明确研究语言—动作对齐和表征保持，因此我们不能再声称二者之间无人连接。可探索的差异是“同一图像坐标上的输出归因约束”，仍需方法与实验核实。

**选择建议**：最高优先级竞争方法之一；不能只用随机教师这种弱负控来代替它。摘要的两种 VLA 未在此确认型号，不凭记忆填入。

## P15 — MolmoAct2

来源：MolmoAct2.pdf，第 1 页摘要，51 页。

**摘要归纳**：以空间/具身推理 VLM Molmo2-ER 为底座，发布机器人数据、FAST tokenizer，通过逐层 KV conditioning 接入连续 flow action expert；另有按变化区域更新深度的 Think 版本。摘要明确包含经过质量筛选的 MolmoAct2-DROID 数据以及仿真和真机 benchmark。

**可借鉴**：模型选择应覆盖 conditioning 结构。该模型可作为与 π0 系 shared-attention 不同的候选；DROID 质量过滤思路可减轻数据清洗工作，但要核对 split、过滤标签和预训练重叠。

**重合与边界**：这是骨干、数据与部署路线，不是对我们的归因损失的直接替代。强空间底座可能缩小我们增益，也更适合检验方法是否只对弱模型有效。

**选择建议**：保留为第二 VLA 候选，与 OpenVLA/OFT、π0/π0.5、LingBot-VLA 逐一核对权重和训练接口后由用户选择；不把“论文说开放”当作本地已可运行。

## P18 — PosA-VLA

来源：PosA-VLA.pdf，第 1 页摘要，19 页；完整标题为 Enhancing Action Generation via Pose-Conditioned Anchor Attention。

**摘要归纳**：将冗余运动归因于对无关区域的分散感知，通过 pose-conditioned supervision 锚定视觉注意力，改善动作精度和执行效率；声称无需辅助分割或 grounding 网络。首页图示比较抓取过程中末端与目标的距离。

**可借鉴**：C 路线的 EEF/pose 区域教师不能当作新贡献；应对比 pose anchor 与语言空间先验。指标增加到达目标所需步数、路径长度和成功条件下的执行耗时，避免只看终点成功率。

**重合与边界**：与“动作相关区域监督”最直接相邻。是否使用相同投影、相同 attention 和训练目标尚需方法级核查；不能凭摘要断定它已做我们的 containment。

**选择建议**：最高优先级精读和 baseline 候选；若选择我们纯语义教师路线，要说明不依赖 pose 标签的价值及代价。

## P19 — ReVLA

来源：ReVLA.pdf，第 1 页摘要及引言，7 页。

**摘要归纳**：检查三种机器人基础策略在视觉 OOD 下的表现；以 OpenVLA 视觉编码器的深度回归能力退化为遗忘证据，通过模型合并式的逐步 backbone reversal 恢复预训练视觉能力。引言明确使用 SIMPLER-based OOD 物体与 distractor 评估。

**可借鉴**：将视觉 probe 与动作结果并列；不能因为 backbone 很强就默认视觉能力保留。作为更简单的恢复/冻结 baseline 思路，区分我们的教师究竟增加知识还是仅阻止遗忘。

**重合与边界**：共同关注视觉泛化，但 ReVLA 调整参数恢复路线，当前方案约束空间证据。摘要的相对提升不能和不同任务上的成功率百分点直接比较。

**选择建议**：保留为恢复能力对照；SIMPLER 是 benchmark 替代选项，不与 LIBERO-Plus 同时强制展开。

## P21 — Shortcut Learning in Generalist Robot Policies

来源：同名 PDF，第 1 页摘要，29 页。

**摘要归纳**：分析 OXE 等混合机器人数据中的捷径，指出子数据集内部多样性不足和子数据集之间分布差异导致的 fragmentation。提出采集建议，并在不增加大规模采集的条件下用数据增强缓解捷径，涉及 π0 仿真与真机。

**可借鉴**：给 DROID 设计按环境分组的 split，并记录任务、相机、采集者之间是否共线；为 D0 加入等预算视觉增强 baseline。不能把增益全部解释成语义归因，如果普通增强已能做到同样效果。

**重合与边界**：我们声称“降低视觉捷径”必须超过该类数据层面的修复；DROID 规模大本身不是泛化证据。

**选择建议**：重点读取增强方法和分组方式；operator/object split 仅在真实元数据可靠时使用，不预设 DROID 提供完整对象身份。

## P22 — Spatial-Aware VLA Pretraining / VIPA-VLA

来源：Spatial-Aware VLA Pretraining.pdf，第 1 页摘要，12 页。

**摘要归纳**：利用人类操作视频中的 3D visual/action 标注，在机器人学习前训练视觉—物理对齐；VIPA-VLA 以双编码器引入 3D-aware 特征，缩小二维视觉与三维动作之间的差距。

**可借鉴**：把“空间表征更强”和“证据使用更合理”分开比较。可用冻结深度/几何特征的辅助监督检查收益是否仅来自额外空间信息。

**重合与边界**：与先空间后动作的路线相邻，但其数据、3D 编码器和额外预训练很不同。不使用相同资源时不做无条件性能排名。

**选择建议**：后期几何扩展；首轮不额外建立 Hand3D 类预训练管线。

## P27 — Zero-Shot Visual Generalization

来源：Zero-Shot Visual Generalization.pdf，第 1 页摘要，21 页。

**摘要归纳**：把 disentangled representation 和 associative memory 从简单 RL 环境推广到更复杂操作，并扩展到 Diffusion Policy 模仿学习；还从 equivariance 出发构造对二维平面旋转不变的策略。

**可借鉴**：加入视觉增强/表征解耦或冻结视觉的轻量 OOD 对照；视角变化时必须检查动作坐标如何变换，不能把整图旋转后仍沿用旧动作标签当作物理正确的数据。

**重合与边界**：与抗无关视觉变化的目标重合，方法不使用我们语言归因链。摘要中的二维旋转性质不能泛化到任意 3D 相机位姿变化。

**选择建议**：作为 OOD protocol 与 Diffusion Policy 非语言控制基线的参考，详细任务范围待正文复核。

## P30 — Diffusion Policy

来源：diffusion_policy_2303.04137.pdf，第 1 页摘要，22 页。

**摘要归纳**：用条件去噪扩散表达视觉动作策略，结合 receding-horizon 控制、视觉条件和时间序列 diffusion transformer，面向多模态与高维动作。摘要报告四类 benchmark 的 15 个任务。

**可借鉴**：连续动作归因必须固定采样噪声和去噪步数；多种正确动作的存在意味着单一专家轨迹 L2 误差不等于真实控制能力。chunk、时域、频率都应匹配。

**重合与边界**：属于动作生成基础方法。若作为不含语言的 baseline，可检验语言是否真的提供收益，但不能冒称与 VLA 完全同输入。

**选择建议**：可选控制基线；若首轮模型已支持 continuous head，不必为齐全再复现所有 DP benchmark。

## P32 — HAMSTER

来源：hamster.pdf，第 1 页排印为 A BSTRACT，29 页。

**摘要归纳**：高层 VLM 从 RGB 与语言产生粗 2D EEF 路径，再由 3D-aware 低层策略执行。层级解耦有助利用便宜的 off-domain 数据，如无动作视频、草图或仿真，并跨 embodiment、视觉与语义差异迁移。

**可借鉴**：清楚区分任务语义、空间中间量和低层执行；对照显式 path/heatmap 中间任务，检验我们归因正则是否比增加中间监督更值得做。

**重合与边界**：也是语义到动作的空间桥，但输出路径是可执行指导，不是忠实归因。其高层/低层系统成本和传感器不能忽略。

**选择建议**：与 P01 联读；若仍选择端到端 VLA 主线，只保留方法比较而不强制迁移到层级控制。

## P35 — LIBERO

来源：libero_2306.03310.pdf，第 1 页摘要，44 页。

**摘要归纳**：原始任务是机器人终身学习，区分 declarative 和 procedural 知识迁移，研究架构、算法、顺序及预训练；提供可扩展程序生成和四个任务 suite，摘要记载共 130 任务。

**可借鉴**：区分原论文 suite 定义与 VLA 常用 40 任务评估协议；明确使用哪个版本、训练示范和评估 seed。单 pick-place 是 smoke，不能代表完整 suite 泛化。

**重合与边界**：benchmark 而非归因方法。仿真可提供可控状态与干预；不能据此宣称 DROID 自带同样 phase 或成功标签。

**选择建议**：保留闭环主验证；LIBERO-Plus 是单独 OOD 候选，不能以本文件已下载为由宣称 Plus 已部署。

## P43 — OpenVLA（原文件损坏，已获取补读副本）

来源：openvla_2406.09246.pdf。pdftotext 报 Couldn't find trailer dictionary、Invalid XRef、Top-level pages object is wrong type；此次不能读取有效摘要。

**归纳状态**：已通过指定 7897 代理从 https://arxiv.org/pdf/2406.09246 获取可读的 v3 补读副本（37 页），见 source_recovery.json。原损坏文件保留，不以新版本覆盖原始来源；下述摘要来自补读副本第 1–2 页。

**摘要归纳**：OpenVLA 以 Llama 2 和融合 DINOv2/SigLIP 的视觉编码器构建 7B VLA，用 970k 真实机器人 demonstrations 训练。摘要强调开放权重/代码、多机器人控制和参数高效微调；在其评估条件下比较 RT-2-X 与 Diffusion Policy，并研究量化部署。

**借鉴/重合**：双视觉编码器意味着归因应分别标注语义/视觉分支和融合位置；比较原生 policy 微调与随机动作头初始化是不同问题，应分表。它提供可适配骨干而非空间归因监督，不能把其预训练收益当成我们的贡献。

**选择建议**：不因此排除 OpenVLA；先修复来源，再检查原始 token 动作与 OFT 连续/并行动作实现的区别，不能把 OpenVLA/OFT 视为一种动作接口。

## P44 — π0: A Vision-Language-Action Flow Model for General Robot Control

来源：pi0_2410.24164.pdf，第 1 页摘要，17 页。

**摘要归纳**：将预训练 VLM 与 flow matching 动作架构结合，利用多平台示范训练通用策略，考察语言指令、上层 VLM 提示和新技能微调；关注复杂灵巧任务。

**可借鉴**：作为连续生成动作头候选；动作输出归因与 flow loss 归因应分开，因为前者解释决策、后者解释拟合误差。固定噪声、时间步和采样策略后再比较空间图。

**重合与边界**：它提供底座，不是归因对齐方法；本文件是 π0，不能据其摘要代替 π0.5 的具体实现和实验协议。

**选择建议**：保留 π0 系与 OpenVLA/OFT、MolmoAct2、LingBot-VLA 的多种组合，让用户按复现门槛和结构差异选择。

## P49 — Qwen-RobotManip Technical Report

来源：qwen_robotmanip_2606.17846.pdf，第 1 页摘要，44 页；副标题 Alignment Unlocks Scale for Robotic Manipulation Foundation Models。

**摘要归纳**：针对多源动作数据异构、昂贵和多样性不足，在 representation、motion、behavior 三个维度统一对齐，并利用 human-to-robot 合成和数据清洗扩大预训练。摘要强调普通 benchmark 可能不能反映预训练质量，因此采用 RoboCasa365、LIBERO-Plus、EBench、RoboTwin-Clean2Rand、RoboTwin-IF、RoboTwin-XE 等 OOD 设置。

**可借鉴**：把物体/场景鲁棒性、指令跟随、跨 embodiment 分为不同实验问题；统一 delta EEF 不能只统一维度，必须核对相机/世界坐标、控制频率和运动定义。

**重合与边界**：泛称“统一语言视觉动作对齐”并不新；我们的空间归因目标需要与它的数据/动作表示对齐区分。大规模合成数据贡献不能直接移植为小数据正则收益。

**选择建议**：强候选/参照，先审计 checkpoint 和 benchmark 可运行性；不要求首轮复现其全套数据规模。
