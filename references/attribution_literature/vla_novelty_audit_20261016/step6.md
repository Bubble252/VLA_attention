# Step 6 — 四轴比较

2026-10-16T12:05:27.430883+00:00

- **Proposed work**
  - Title: 语言目标条件下的动作可分辨视觉表征（原候选）
  - Date: 2026-10-16，未发表
  - Source: provisional_recommendation.json
  - Problem framing: VLA 状态混淆与任务相关视觉证据
  - Core mechanism: 冻结 source/target map pooling + action-chunk distance 正负对 + triplet loss，原生策略通路保留
  - Key insight: 控制差异应体现为任务相关视觉表示差异
  - Application domain: OpenVLA/OFT、LIBERO；尚未实验

- **RS-CL**
  - Title: Contrastive Representation Regularization for Vision-Language-Action Models
  - Date: 2025-10 / PDF v4 2026-05
  - Source: https://arxiv.org/abs/2510.01711
  - Problem framing: VLA 表征缺乏控制状态敏感性
  - Core mechanism: summary token + proprio 距离 soft weights + InfoNCE + view cutoff，与原动作损失联合
  - Key insight: 物理状态距离可塑造控制相关表征
  - Application domain: GR00T/RoboCasa/LIBERO/真机
  - Verified venue: ICML 2026（PDF 首页）
  - Closest-passage evidence: §2.1–2.2 Eq.(2)–(5)；Table 2 LIBERO；Table 3 比较 TCN
  - Assumptions & scope: 使用 proprio 距离，非我们的语言区域 action-chunk triplet；不能说 exact duplicate。
  - Refined overlap (problem/mechanism/insight/domain): match / partial / match / match
  - Verdict: 3/4 match，Level 2。partial 不自动算 match。

- **PSE**
  - Title: Contrastive Behavioral Similarity Embeddings for Generalization in Reinforcement Learning
  - Date: 2021-01 / PDF v2 2021-03
  - Source: https://arxiv.org/abs/2101.05265
  - Problem framing: 跨环境行为泛化
  - Core mechanism: policy similarity metric 的当前动作分布差异+未来状态递归；contrastive metric embeddings
  - Key insight: 行为相似决定表示相似
  - Application domain: RL/LQR/jumping/Distracting Control
  - Verified venue: ICLR 2021（PDF 首页）
  - Closest-passage evidence: §3 Eq.(2)；§4 Eq.(4)；Fig.3 联合 RL/IL 与辅助损失
  - Assumptions & scope: 需要专家/近似最优策略，保留未来行为；有限 H=8 action L1 不继承其理论。
  - Refined overlap (problem/mechanism/insight/domain): partial / match / match / partial
  - Verdict: 2/4 match，Level 3。partial 不自动算 match。

- **Inverse dynamics VLA**
  - Title: Mitigating State Aliasing in Vision-Language-Action Models via Inverse Dynamics Learning
  - Date: 2026-05
  - Source: https://arxiv.org/abs/2605.29577
  - Problem framing: VLA 相似视觉对应不同动作的状态混淆
  - Core mechanism: 当前/未来编码器特征预测中间 action chunk，附加 pseudo time reversal
  - Key insight: 直接动作监督视觉编码器可减轻混淆
  - Application domain: CALVIN ABC-D/SimplerEnv；LIBERO probing
  - Verified venue: arXiv preprint
  - Closest-passage evidence: §3.1 inverse action loss；§3.2 PTR 的非精确反演声明；§4 和 frozen encoder probes
  - Assumptions & scope: 逆动力学非 triplet；未来观测只用于训练辅助头。
  - Refined overlap (problem/mechanism/insight/domain): match / partial / match / match
  - Verdict: 3/4 match，Level 2。partial 不自动算 match。

- **FOCAL-VLA**
  - Title: FOCAL-VLA: Subtask-Guided Geometry Distillation and Implicit World Modeling for Vision-Language-Action Models
  - Date: 2026-09
  - Source: https://arxiv.org/abs/2609.21228
  - Problem framing: 当前子任务所需空间与未来动态理解
  - Core mechanism: 子任务实体 mask pooling VGGT/Track4World 教师特征，geometry/motion queries 条件化动作
  - Key insight: 全场景蒸馏包含无关信息，任务区域应随子任务改变
  - Application domain: π0.5/LIBERO/真机
  - Verified venue: arXiv preprint
  - Closest-passage evidence: §III-B Eq.(1)–(2)；§III-C Eq.(3)–(4)；§III-D 原生 context 路径保留
  - Assumptions & scope: 依赖离线子任务语义和实体 masks；绝对教师特征蒸馏，不是动作距离监督。
  - Refined overlap (problem/mechanism/insight/domain): partial / partial / match / match
  - Verdict: 2/4 match，Level 3。partial 不自动算 match。

- **Copper**
  - Title: Copper-Policy: Focus on the Representation for Robust Robot Manipulation
  - Date: 2026-09
  - Source: https://arxiv.org/abs/2609.32779
  - Problem framing: 学习有利于控制的紧凑世界表征
  - Core mechanism: 意图条件 online/EMA compactor + future flow prediction + 当前帧细节动作通路
  - Key insight: 任务进展与执行细节互补，target space 随 policy 学习
  - Application domain: LIBERO/Plus/RoboTwin/真机
  - Verified venue: arXiv preprint（用户本地副本）
  - Closest-passage evidence: §3.2 Eq.(2)–(3)；§3.4–3.5 asymmetric interface；§4.5 Table 7
  - Assumptions & scope: 未来条件为 task intention，非任意候选动作后果；完整架构不是训练期 triplet。
  - Refined overlap (problem/mechanism/insight/domain): partial / differ / match / match
  - Verdict: 2/4 match，Level 3。partial 不自动算 match。

- **TCN**
  - Title: Time-Contrastive Networks: Self-Supervised Learning from Video
  - Date: 2017-04 / PDF v3 2018-03
  - Source: https://arxiv.org/abs/1704.06888
  - Problem framing: 视角不同但语义相同/外观相近但功能不同的操作状态
  - Core mechanism: 同步多视角 positives + temporal negatives + triplet margin
  - Key insight: 不变性与功能敏感性需同时学习
  - Application domain: 机器人模仿/倒水/人体姿态
  - Verified venue: arXiv preprint（本次 PDF 未显示正式 venue）
  - Closest-passage evidence: Abstract；§III-A triplet inequality；§III-B/C reward 与 imitation 使用
  - Assumptions & scope: 同步视角/时间负对，非 action-chunk 距离；有单视角变体。
  - Refined overlap (problem/mechanism/insight/domain): partial / partial / match / match
  - Verdict: 2/4 match，Level 3。partial 不自动算 match。

- **DBC**
  - Title: Learning Invariant Representations for Reinforcement Learning without Reconstruction
  - Date: 2020-06 / PDF v2 2021-04
  - Source: https://arxiv.org/abs/2006.10742
  - Problem framing: 像素控制抗无关背景
  - Core mechanism: latent 距离拟合 reward difference + transition Wasserstein distance
  - Key insight: 保留行为信息无需像素重建
  - Application domain: 视觉 MuJoCo/CARLA RL
  - Verified venue: ICLR 2021（PDF 首页）
  - Closest-passage evidence: §3 bisimulation Definition 2；§4 encoder loss；§3 partial observability 假设
  - Assumptions & scope: 用 rewards/transitions，不等价于即时 action distance，理论不能借用。
  - Refined overlap (problem/mechanism/insight/domain): partial / partial / match / partial
  - Verdict: 1/4 match，Level 4。partial 不自动算 match。
