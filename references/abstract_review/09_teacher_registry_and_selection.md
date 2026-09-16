# 当前 Teacher Registry：选择、作用和不可混用的边界

模型候选扩大后，teacher 不能再用“一个大模型 teacher”统称。当前 registry 按监督对象和是否参与部署分层。

## 1. T_sem：语义空间教师（VLM 首轮唯一主教师）

### T_sem-A：Lavender / Stable Diffusion 词级图（首选）

- 输出：`word/span → image region` 的词级 spatial map；
- 用途：VLM grounding、VLA 训练中的语言归因锚点；
- 优点：与原始项目直接连续，已有公开代码和 attention map 思路；
- 限制：不是动作、接触或动力学教师；扩散 attention 也不自动是真值；
- 负控：错词、错图、随机图、同面积平滑图、教师不确定样本跳过。

**当前建议**：VLM 首轮只使用 T_sem-A，先和 DB-style feature alignment 分开比较。

### T_sem-B：冻结 VLM/视觉表征 teacher（强 baseline，不是主教师）

来源：Don't Blind 的 C-RADIOv3/DINOv2/Theia 路线，以及 Anchor-Align 的 frozen VLM copy。

- 输出：patch/hidden feature；
- 用途：检查收益是不是单纯 representation retention；
- 不应称为 word-region teacher，除非额外建立 phrase-conditioned 目标；
- 对应 baseline：DB-feature、AA-anchor。

### T_sem-C：VLM 自身离线归因 ensemble（sanity check）

Qwen2.5-VL、InternVL3.5、Ovis2.5、LLaVA-OneVision 可以提供答案/phrase score gradient attribution。它们不是外部真值：若拿它们当 teacher，要隔离冻结 checkpoint 和学生 checkpoint，避免 teacher/student 共享错误。

## 2. T_spatial：显式空间/几何教师

### T_spatial-A：Pose/EEF anchor（PosA-VLA 相邻）

- 输入：EEF 3D pose、相机外参、gripper event；
- 输出：task Gaussian + end-effector Gaussian；
- 用途：C 路线 sanity check、前向视觉 gating baseline；
- 局限：推/滑/遮挡和相机标定会破坏标签；不能当 universal ground truth。

### T_spatial-B：BridgeVLA/3D heatmap

- 输入：点云投影/多视角；
- 输出：2D heatmap 或 3D guidance；
- 用途：如果我们选择 3D VLA 方向的强 baseline；
- 当前状态：不进入 RGB-only 首轮主教师。

## 3. T_retention：表征保持教师

### T_retention-A：Don't Blind feature teacher

冻结 C-RADIOv3/DINOv2/Theia，对齐 OpenVLA 中层 patch features；代码默认 layer 16、cosine、alignment coefficient 0.2、frozen projector，但 projector 参数组和恢复链路需审计。

### T_retention-B：Anchor-Align frozen VLM copy

同一输入上逐 decoder layer 对齐 vision/text hidden states，并用同一观测的动作方向词监督。它是最强 representation/action semantic baseline；不能将其全部归入“普通 feature teacher”。

## 4. T_phase：阶段/动作相关弱教师

### T_phase-A：LIBERO state-rule

由 EEF/source/target 距离、gripper、object height 派生 source/mixed/target；首轮诊断。

### T_phase-B：成功 demonstration event boundaries

由 grasp、lift、near-target、release 事件生成 soft phase；后期 D1 主扩展。

### T_phase-C：PosA pose events

利用 gripper change 时刻和 EEF 投影构造 task/end maps；作为 C baseline，不直接称 D1。

## 5. T_future：WAM/VAM 未来教师

### T_future-A：world/video model future target attribution

给定当前观测、语言和候选动作，WAM/VAM 预测 future latent/frame；对明确的 future score `S^k` 求梯度或做干预，得到 `R_future`。必须加入 wrong-action、wrong-future、random-horizon 和低置信度回退。

候选参考：FAST-WAM、ImageWAM、LaWAM、Robust-WAM、MV-WAM、LingBot-VA、Next Forcing。

### T_future-B：latent subgoal / semantic foresight

使用 LaWAM/Robust-WAM 类未来语义表征。它不是视觉 heatmap 真值；必须把未来表征和当前图像 patch 坐标桥接后才可与 `T_sem` 组合。

### T_future-C：WAM/video backbone attention

只作结构诊断。视频模型 attention 不是 action causal map，生成质量也不能替代动作干预。

## 6. 当前建议的 teacher 组合

| 阶段 | 主教师 | 强 baseline | 后置扩展 |
|---|---|---|---|
| VLM | T_sem-A Lavender | T_retention-A/B | T_sem-C |
| VLA-D0 | T_sem-A → `A_lang` | T_retention + PosA-inspired gating | T_phase-A |
| VLA-D1 | T_sem-A + T_phase-B | phase-only / Anchor-Align-style direction | T_future-A |
| B 路线 | T_future-A/B | static `T_sem`、direct future-feature alignment | Next Forcing multi-horizon |

## 7. 不允许的混用

- 不把 frozen visual feature 当作词级语义真值；
- 不把 EEF Gaussian 当作完整 action causal map；
- 不把 action-direction label 当作空间归因；
- 不把 WAM 预测质量当作成功动作证明；
- 不把模型学生的离线 attribution ensemble 当无偏 teacher；
- 不把 Sink-free 预处理当 teacher；
- 不把 SpikingBrain 的结构层号迁移给其他 VLM。

## 8. 仍需用户筛选

- [ ] VLM 首轮：Prismatic-7B、Qwen2.5-VL-7B、InternVL3.5、Ovis2.5 中选两个；
- [ ] VLA 首轮：OpenVLA/OFT、π0、π0.5、MolmoAct2 中选两个；
- [ ] benchmark 首轮：LIBERO + SimplerEnv；RoboTwin 是第二阶段或 WAM 扩展；
- [ ] 主教师冻结为 Lavender；DB/AA/PosA 作为 baseline，而不是混合 teacher；
- [ ] SpikingBrain 保持后置。
