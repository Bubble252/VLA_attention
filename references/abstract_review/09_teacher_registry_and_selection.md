# 当前 Teacher Registry：选择、作用和不可混用的边界

模型候选扩大后，teacher 不能再用“一个大模型 teacher”统称。当前 registry 按监督对象和是否参与部署分层。

## 1. T_sem：语义空间教师族（VLM 首轮重点）

### T_sem-A：Lavender / Stable Diffusion 词级图（已具备的可复现基线）

- 输出：`word/span → image region` 的词级 spatial map；
- 用途：VLM grounding、VLA 训练中的语言归因锚点；
- 优点：与原始项目直接连续，已有公开代码和 attention map 思路；
- 限制：不是动作、接触或动力学教师；扩散 attention 也不自动是真值；
- 负控：错词、错图、随机图、同面积平滑图、教师不确定样本跳过。

它不预先被认定为唯一或最好的主教师；它是多扩散教师评估的第一个可复现基线。

### T_sem-D：PixArt-α / PixArt-Σ / Playground-v2.5（候选扩散教师）

这三个候选用于避免把 `T_sem` 绑定到单一 Stable Diffusion 架构、text encoder 或 cross-attention 实现。每个候选独立产生语言条件二维图 `T_sem^d(w,x)`，不在校准前直接平均。

| 候选 | 必须审计 | 当前状态 |
|---|---|---|
| Stable Diffusion / Lavender | token span、denoising step、层、CFG conditional branch、空间 resize | 可复现基线 |
| PixArt-α | 文本 token→图像 token attention、tokenizer、块/头/步选择 | 待代码接口审计 |
| PixArt-Σ | 与 α 之间的 tokenizer、训练分辨率和 attention 模块差异 | 待代码接口审计 |
| Playground-v2.5 | 权重/许可、attention hook、CFG 与文本编码器路径 | 待权重/接口审计 |

若一个候选不能导出语言条件二维图，它不能叫 `T_sem`；最多做 `T_retention` feature teacher 或独立 gradient proxy。

### 多扩散教师选择协议

1. 固定 phrase、token span、输入图、seed、resolution、prompt template 和 CFG，分别提取 `T_sem^d`；
2. 在独立校准集测 phrase pointing、IoU、无效词率、跨 seed/增强一致性、对象词错误率；
3. 主实验先分别报告 `SD`、`PixArt-α`、`PixArt-Σ`、`Playground-v2.5`；
4. 仅当两个以上教师各自可靠且错误不高度一致，才比较 confidence-weighted ensemble；
5. ensemble 权重仅用校准集冻结，必须保留 best-single、uniform ensemble、wrong-map 负控；
6. 教师不一致可以标为低置信样本，不得用归一化强造尖峰。

候选 ensemble 仅作消融：

```text
T_sem^ens = Normalize(Σ_d c_d · Valid_d · T_sem^d)
```

`c_d` 和 `Valid_d` 在校准后冻结；token 不可映射、图面积过小或目标词无效则 `Valid_d=0`。

### “校准后冻结”与 best-single 的精确定义

这里的“校准”不训练扩散模型，也不在最终测试集上选老师。它是一次**教师选择和提图配置选择**：

```text
候选：SD / PixArt-α / PixArt-Σ / Playground-v2.5
配置：每个模型的 text span、attention block、denoising step、CFG 分支、map 聚合方式
数据：带 region 标注的独立 calibration split
指标：pointing、IoU、无效词率、跨 seed 稳定性
```

`best-single` 指整个校准协议中分数最好的**一个固定教师配置**，例如：

```text
PixArt-Σ + object-token span + blocks {b1,b2} + steps {t1,t2} + conditional CFG branch
```

而不是每个训练样本根据区域真值挑一个“最漂亮”的教师图。选定后固定三类内容：

1. 扩散模型参数始终冻结；
2. teacher identity 和提图配置冻结；
3. 若使用 ensemble，ensemble 权重和有效性门槛冻结。

随后才在独立的 VLM/VLA 训练集上生成 `T_sem`，最终测试集只评估，不能重新选择 teacher、层、step、阈值或权重。若所有候选都未通过最低 teacher-quality 门槛，则停止语义图监督，不把低质量 ensemble 硬塞进训练。

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

### VLM 与 VLA 的使用范围

P4 VLM 阶段可使用 `T_retention-A` 的 DB-style patch feature alignment，检验空间教师是否超过通用视觉表征保持；不使用 action loss、方向词或 `A_act`。`T_retention-B` 的 Anchor-Align direction label 需要 demonstration action，因此只作为 VLA 阶段强 baseline。

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
| VLM | 每个经校准的 T_sem-A/D 单教师 | T_retention-A/B | T_sem-C、confidence ensemble |
| VLA-D0 | 冻结的 best-single `T_sem` → `A_lang` | T_retention + PosA-inspired gating | T_phase-A、teacher ensemble |
| VLA-D1 | best-single `T_sem` + T_phase-B | phase-only / Anchor-Align-style direction | T_future-A |
| B 路线 | T_future-A/B | static `T_sem`、direct future-feature alignment | Next Forcing multi-horizon |

### VLA 首要顺序：先与 BlindVLA 比较

`T_retention` 是 VLA 强 baseline，不是 VLM grounding 的前置教师。P4 先独立冻结词级 `T_sem`；当前方法进入 VLA OOD 主结论前，再完成：

```text
B0 native SFT/BC
B1 T_retention-A (DB-style patch feature alignment)
B2 best-single T_sem spatial alignment
B3 B1 + B2
B4 B3 + D0 action-evidence constraint
```

固定相同学生、数据、动作头、预算与增强。只有 B4 超过 B3，且正确 `T_sem` 超过错教师负控，才可声称动作证据约束超过表征保持；否则将它降级为诊断或相关工作分析。多扩散教师 ensemble、D1、T_future 均在此 gate 之后。

## 7. 不允许的混用

- 不把 frozen visual feature 当作词级语义真值；
- 不把多个扩散模型未经校准的 attention 平均后称为更可靠教师；
- 不把 EEF Gaussian 当作完整 action causal map；
- 不把 action-direction label 当作空间归因；
- 不把 WAM 预测质量当作成功动作证明；
- 不把模型学生的离线 attribution ensemble 当无偏 teacher；
- 不把 Sink-free 预处理当 teacher；
- 不把 SpikingBrain 的结构层号迁移给其他 VLM。

## 8. 仍需用户筛选

- [ ] VLM 首轮：Prismatic-7B、Qwen2.5-VL-7B、InternVL3.5、Ovis2.5、LLaVA-OneVision 中选两个；
- [ ] VLA 首轮：OpenVLA/OFT、π0、π0.5、MolmoAct2 中选两个；
- [ ] benchmark 首轮：LIBERO + SimplerEnv；RoboTwin 是第二阶段或 WAM/VLA 扩展；
- [ ] 用校准集从 Stable Diffusion、PixArt-α、PixArt-Σ、Playground-v2.5 中选择 best-single；DB/AA/PosA 作为 baseline，而不是混合 teacher；
- [ ] SpikingBrain 保持后置。
