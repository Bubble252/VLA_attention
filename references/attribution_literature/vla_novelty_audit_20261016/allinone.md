# 检索完整记录

# Step 2 — 检索与去重

2026-10-16T12:05:27.430883+00:00

查询：
- vision language action state aliasing
- robot policy representation metric learning
- language conditioned action similarity contrastive representation

年份 2020–2026，各源每查询上限 3。结果：arxiv=6, dblp=0, open_alex=9, openreview=0, semantic_scholar=0, crossref=9。24 原始记录，24 标题去重记录；补充 6 篇（3 篇 model-recall、2 篇上一轮近邻/用户文献、1 篇引用追溯），合计 30。

arXiv 第一查询 SSLError/SSLEOFError；DBLP 三查询 JSON 解析失败；OpenReview: openreview not installed；Semantic Scholar 三查询 HTTP 429。其余 API 有返回。失败和低召回不构成不存在近邻的证据。

TCN 2017 是历史机制近邻，显式超出 API 年份保留。原始摘要及来源保存在 search_raw.json / audit_records.json。

### 1. Robot Trains Robot: Automatic Real-World Policy Adaptation and Learning for Humanoids

- Title: Robot Trains Robot: Automatic Real-World Policy Adaptation and Learning for Humanoids
- Date: 2025-08-17
- Problem framing: 机器人保护与真实 humanoid 学习
- Core mechanism: 机械臂教师/潜变量适应
- Key insight: 主动保护支持真实训练
- Application domain: humanoid RL
- Overlap score: 0/4，摘要筛选值，非最终判定
- Source: http://arxiv.org/abs/2508.12252v2；arxiv

### 2. Learning to Discern: Imitating Heterogeneous Human Demonstrations with Preference and Representation Learning

- Title: Learning to Discern: Imitating Heterogeneous Human Demonstrations with Preference and Representation Learning
- Date: 2023-10-22
- Problem framing: 异质示范质量评估
- Core mechanism: 轨迹嵌入与偏好质量评估
- Key insight: 分离示范质量与风格
- Application domain: 离线模仿学习
- Overlap score: 1/4，摘要筛选值，非最终判定
- Source: http://arxiv.org/abs/2310.14196v1；arxiv

### 3. Untangling Dense Knots by Learning Task-Relevant Keypoints

- Title: Untangling Dense Knots by Learning Task-Relevant Keypoints
- Date: 2020-11-10
- Problem framing: 绳结解开
- Core mechanism: 关键点与几何规划
- Key insight: 任务几何结构支持控制
- Application domain: 机器人绳索
- Overlap score: 1/4，摘要筛选值，非最终判定
- Source: http://arxiv.org/abs/2011.04999v1；arxiv

### 4. On the Similarities of Embeddings in Contrastive Learning

- Title: On the Similarities of Embeddings in Contrastive Learning
- Date: 2025-06-11
- Problem framing: 对比嵌入几何
- Core mechanism: 相似度理论与辅助方差损失
- Key insight: 批大小影响正负对几何
- Application domain: 一般表征学习
- Overlap score: 1/4，摘要筛选值，非最终判定
- Source: http://arxiv.org/abs/2506.09781v2；arxiv

### 5. Weakly-Supervised 3D Medical Image Segmentation using Geometric Prior and Contrastive Similarity

- Title: Weakly-Supervised 3D Medical Image Segmentation using Geometric Prior and Contrastive Similarity
- Date: 2023-02-04
- Problem framing: 医学分割
- Core mechanism: 几何与对比损失
- Key insight: 低对比边界的表征区分
- Application domain: 医学图像
- Overlap score: 0/4，摘要筛选值，非最终判定
- Source: http://arxiv.org/abs/2302.02125v1；arxiv

### 6. MameLoshnLM: Yiddish Language Model and Evaluation Benchmark

- Title: MameLoshnLM: Yiddish Language Model and Evaluation Benchmark
- Date: 2026-08-06
- Problem framing: 意第绪语建模
- Core mechanism: 语料与继续预训练
- Key insight: 低资源语言数据质量
- Application domain: NLP
- Overlap score: 0/4，摘要筛选值，非最终判定
- Source: http://arxiv.org/abs/2608.05850v2；arxiv

### 7. Mitigating State Aliasing in Vision-Language-Action Models via Inverse Dynamics Learning

- Title: Mitigating State Aliasing in Vision-Language-Action Models via Inverse Dynamics Learning
- Date: 2026-05-28
- Problem framing: VLA 控制状态混淆
- Core mechanism: 当前未来视觉→动作 chunk 的逆动力学辅助任务
- Key insight: 动作监督塑造视觉区别
- Application domain: VLA/CALVIN/SimplerEnv
- Overlap score: 3/4，摘要筛选值，非最终判定
- Source: https://doi.org/10.48550/arxiv.2605.29577；open_alex

### 8. Comparison of Vision Language Action using goal state as action tokens

- Title: Comparison of Vision Language Action using goal state as action tokens
- Date: 2026-02-02
- Problem framing: goal state action token 方法比较
- Core mechanism: 比较条目，未核实为原创论文
- Key insight: 目标状态与低层动作区别
- Application domain: VLA
- Overlap score: 1/4，摘要筛选值，非最终判定
- Source: https://doi.org/10.48366/r1582110；open_alex

### 9. ALFRED-L: Investigating the Role of Language for Action Learning in Interactive Visual Environments

- Title: ALFRED-L: Investigating the Role of Language for Action Learning in Interactive Visual Environments
- Date: 2022-01-01
- Problem framing: 交互环境中的语言与动作
- Core mechanism: 摘要仅元数据，方法待核实
- Key insight: 语言作用需隔离测量
- Application domain: 交互具身任务
- Overlap score: 1/4，摘要筛选值，非最终判定
- Source: https://doi.org/10.18653/v1/2022.emnlp-main.636；open_alex

### 10. SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Models

- Title: SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Models
- Date: 2025-06-21
- Problem framing: 空间动作表示
- Core mechanism: 空间感知表示与自适应 action grids
- Key insight: 空间编码支持迁移
- Application domain: VLA
- Overlap score: 1/4，摘要筛选值，非最终判定
- Source: https://doi.org/10.15607/rss.2025.xxi.011；open_alex

### 11. Learning Representations Via a Robust Behavioral Metric for Deep Reinforcement Learning

- Title: Learning Representations Via a Robust Behavioral Metric for Deep Reinforcement Learning
- Date: 2022-01-01
- Problem framing: 行为相关 RL 表征
- Core mechanism: robust behavioral metric；摘要缺失，未原文核实
- Key insight: 与动作距离学习存在潜在邻近
- Application domain: RL
- Overlap score: 2/4，摘要筛选值，非最终判定
- Source: https://doi.org/10.52202/068431-2656；open_alex

### 12. Deployment-time reliability of learned robot policies

- Title: Deployment-time reliability of learned robot policies
- Date: 2026-03-11
- Problem framing: 部署可靠性
- Core mechanism: 摘要缺失，方法未核实
- Key insight: 暂不能判断具体重合
- Application domain: robot policy
- Overlap score: 1/4，摘要筛选值，非最终判定
- Source: https://doi.org/10.25740/cp016xz0883；open_alex

### 13. ConSERT: A Contrastive Framework for Self-Supervised Sentence Representation Transfer

- Title: ConSERT: A Contrastive Framework for Self-Supervised Sentence Representation Transfer
- Date: 2021-01-01
- Problem framing: 句子表示
- Core mechanism: 自监督对比框架
- Key insight: 文本增强用于语义表征
- Application domain: NLP
- Overlap score: 0/4，摘要筛选值，非最终判定
- Source: https://doi.org/10.18653/v1/2021.acl-long.393；open_alex

### 14. DiffCSE: Difference-based Contrastive Learning for Sentence Embeddings

- Title: DiffCSE: Difference-based Contrastive Learning for Sentence Embeddings
- Date: 2022-01-01
- Problem framing: 句子表示
- Core mechanism: difference-based contrastive learning
- Key insight: 文本表征变化
- Application domain: NLP
- Overlap score: 0/4，摘要筛选值，非最终判定
- Source: https://doi.org/10.18653/v1/2022.naacl-main.311；open_alex

### 15. Self-Guided Contrastive Learning for BERT Sentence Representations

- Title: Self-Guided Contrastive Learning for BERT Sentence Representations
- Date: 2021-01-01
- Problem framing: 句子表示
- Core mechanism: 自引导对比学习
- Key insight: 文本表示迁移
- Application domain: NLP
- Overlap score: 0/4，摘要筛选值，非最终判定
- Source: https://doi.org/10.18653/v1/2021.acl-long.197；open_alex

### 16. Grounding Hierarchical Vision-Language-Action Models Through Explicit Language-Action Alignment

- Title: Grounding Hierarchical Vision-Language-Action Models Through Explicit Language-Action Alignment
- Date: 2026-06
- Problem framing: 层级 VLA 对齐
- Core mechanism: 摘要缺失，标题提示 explicit language-action alignment
- Key insight: 不能仅凭标题确定机制
- Application domain: VLA
- Overlap score: 1/4，摘要筛选值，非最终判定
- Source: https://doi.org/10.1109/cvprf71909.2026.00889；crossref

### 17. Vision-Language-Action and Vision Language Models for Robot Manipulation: A Comprehensive Review Towards Real-World Applications

- Title: Vision-Language-Action and Vision Language Models for Robot Manipulation: A Comprehensive Review Towards Real-World Applications
- Date: 未核实
- Problem framing: VLA 综述
- Core mechanism: 综述，非原方法
- Key insight: 分类与挑战
- Application domain: VLA
- Overlap score: 1/4，摘要筛选值，非最终判定
- Source: https://doi.org/10.20944/preprints202606.0400.v1；crossref

### 18. Transforming the vision into action

- Title: Transforming the vision into action
- Date: 2021-09-17
- Problem framing: 非技术章节噪声
- Core mechanism: 无相关方法证据
- Key insight: 不适用
- Application domain: 非目标领域
- Overlap score: 0/4，摘要筛选值，非最终判定
- Source: https://doi.org/10.1017/9781009024648.008；crossref

### 19. Peer Review #3 of "Dynamic guided metric representation learning for multi-view clustering (v0.2)"

- Title: Peer Review #3 of "Dynamic guided metric representation learning for multi-view clustering (v0.2)"
- Date: 2022-03-08
- Problem framing: 聚类论文审稿记录
- Core mechanism: 非研究论文原文
- Key insight: 不适用
- Application domain: 多视图聚类
- Overlap score: 0/4，摘要筛选值，非最终判定
- Source: https://doi.org/10.7287/peerj-cs.922v0.2/reviews/3；crossref

### 20. Peer Review #3 of "Dynamic guided metric representation learning for multi-view clustering (v0.1)"

- Title: Peer Review #3 of "Dynamic guided metric representation learning for multi-view clustering (v0.1)"
- Date: 2022-03-08
- Problem framing: 聚类论文审稿记录
- Core mechanism: 非研究论文原文
- Key insight: 不适用
- Application domain: 多视图聚类
- Overlap score: 0/4，摘要筛选值，非最终判定
- Source: https://doi.org/10.7287/peerj-cs.922v0.1/reviews/3；crossref

### 21. Peer Review #2 of "Dynamic guided metric representation learning for multi-view clustering (v0.1)"

- Title: Peer Review #2 of "Dynamic guided metric representation learning for multi-view clustering (v0.1)"
- Date: 2022-03-08
- Problem framing: 聚类论文审稿记录
- Core mechanism: 非研究论文原文
- Key insight: 不适用
- Application domain: 多视图聚类
- Overlap score: 0/4，摘要筛选值，非最终判定
- Source: https://doi.org/10.7287/peerj-cs.922v0.1/reviews/2；crossref

### 22. A Qualitative Similarity Framework for the Interpretation of Natural Language Similarity Expressions

- Title: A Qualitative Similarity Framework for the Interpretation of Natural Language Similarity Expressions
- Date: 2021
- Problem framing: 语言相似性表达
- Core mechanism: 属性空间框架
- Key insight: 定性语义相似关系
- Application domain: 语言知识表示
- Overlap score: 0/4，摘要筛选值，非最终判定
- Source: https://doi.org/10.1007/978-3-030-69823-2_4；crossref

### 23. CLOP-DiT: Structured-Metadata-Conditioned Single-Cell Latent Generation via Contrastive Language-Omics Pretraining and Diffusion Transformers

- Title: CLOP-DiT: Structured-Metadata-Conditioned Single-Cell Latent Generation via Contrastive Language-Omics Pretraining and Diffusion Transformers
- Date: 未核实
- Problem framing: 单细胞生成
- Core mechanism: 对比跨模态与扩散
- Key insight: 元数据生成条件
- Application domain: 生物信息
- Overlap score: 0/4，摘要筛选值，非最终判定
- Source: https://doi.org/10.64898/2026.03.26.714457；crossref

### 24. Enhancing SCGAN's Disentangled Representation Learning with Contrastive SSIM Similarity Constraints

- Title: Enhancing SCGAN's Disentangled Representation Learning with Contrastive SSIM Similarity Constraints
- Date: 2024-05-14
- Problem framing: 一般解耦表示
- Core mechanism: contrastive SSIM；摘要缺失
- Key insight: 未核实
- Application domain: 一般视觉
- Overlap score: 0/4，摘要筛选值，非最终判定
- Source: https://doi.org/10.1109/icee63041.2024.10668118；crossref

### 25. Contrastive Representation Regularization for Vision-Language-Action Models

- Title: Contrastive Representation Regularization for Vision-Language-Action Models
- Date: 2025-10 / PDF v4 2026-05
- Problem framing: VLA 表征缺乏控制状态敏感性
- Core mechanism: summary token + proprio 距离 soft weights + InfoNCE + view cutoff，与原动作损失联合
- Key insight: 物理状态距离可塑造控制相关表征
- Application domain: GR00T/RoboCasa/LIBERO/真机
- Overlap score: 3/4，摘要筛选值，非最终判定
- Source: https://arxiv.org/abs/2510.01711；引用追溯：inverse-dynamics 文献 [45]

### 26. Contrastive Behavioral Similarity Embeddings for Generalization in Reinforcement Learning

- Title: Contrastive Behavioral Similarity Embeddings for Generalization in Reinforcement Learning
- Date: 2021-01 / PDF v2 2021-03
- Problem framing: 跨环境行为泛化
- Core mechanism: policy similarity metric 的当前动作分布差异+未来状态递归；contrastive metric embeddings
- Key insight: 行为相似决定表示相似
- Application domain: RL/LQR/jumping/Distracting Control
- Overlap score: 2/4，摘要筛选值，非最终判定
- Source: https://arxiv.org/abs/2101.05265；model-recall，PDF 已核验

### 27. FOCAL-VLA: Subtask-Guided Geometry Distillation and Implicit World Modeling for Vision-Language-Action Models

- Title: FOCAL-VLA: Subtask-Guided Geometry Distillation and Implicit World Modeling for Vision-Language-Action Models
- Date: 2026-09
- Problem framing: 当前子任务所需空间与未来动态理解
- Core mechanism: 子任务实体 mask pooling VGGT/Track4World 教师特征，geometry/motion queries 条件化动作
- Key insight: 全场景蒸馏包含无关信息，任务区域应随子任务改变
- Application domain: π0.5/LIBERO/真机
- Overlap score: 2/4，摘要筛选值，非最终判定
- Source: https://arxiv.org/abs/2609.21228；上一轮检索，本轮 PDF 核验

### 28. Copper-Policy: Focus on the Representation for Robust Robot Manipulation

- Title: Copper-Policy: Focus on the Representation for Robust Robot Manipulation
- Date: 2026-09
- Problem framing: 学习有利于控制的紧凑世界表征
- Core mechanism: 意图条件 online/EMA compactor + future flow prediction + 当前帧细节动作通路
- Key insight: 任务进展与执行细节互补，target space 随 policy 学习
- Application domain: LIBERO/Plus/RoboTwin/真机
- Overlap score: 2/4，摘要筛选值，非最终判定
- Source: https://arxiv.org/abs/2609.32779；用户本地 PDF

### 29. Time-Contrastive Networks: Self-Supervised Learning from Video

- Title: Time-Contrastive Networks: Self-Supervised Learning from Video
- Date: 2017-04 / PDF v3 2018-03
- Problem framing: 视角不同但语义相同/外观相近但功能不同的操作状态
- Core mechanism: 同步多视角 positives + temporal negatives + triplet margin
- Key insight: 不变性与功能敏感性需同时学习
- Application domain: 机器人模仿/倒水/人体姿态
- Overlap score: 2/4，摘要筛选值，非最终判定
- Source: https://arxiv.org/abs/1704.06888；model-recall，PDF 已核验，显式扩展检索年份

### 30. Learning Invariant Representations for Reinforcement Learning without Reconstruction

- Title: Learning Invariant Representations for Reinforcement Learning without Reconstruction
- Date: 2020-06 / PDF v2 2021-04
- Problem framing: 像素控制抗无关背景
- Core mechanism: latent 距离拟合 reward difference + transition Wasserstein distance
- Key insight: 保留行为信息无需像素重建
- Application domain: 视觉 MuJoCo/CARLA RL
- Overlap score: 1/4，摘要筛选值，非最终判定
- Source: https://arxiv.org/abs/2006.10742；model-recall，PDF 已核验


## 检索解释

相关簇：行为度量（PSE/DBC/RS-CL）、状态混淆（逆动力学）、任务空间蒸馏（FOCAL）、未来表征（Copper）。本次只是 bounded novelty search；不能从小规模且多源失败的检索推断领域趋势或可靠作者引文排行。引用数不作为新颖性证据；噪声记录完整保留。