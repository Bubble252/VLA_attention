# 5. Experiments 与预期结论

## 原文精读后的基线补充（待筛选）

详见 [Don't Blind / Anchor-Align / PosA-VLA 原文分析](../references/abstract_review/07_three_nearest_methods_deep_read.md)。增加三类强对照候选：中层 patch-feature alignment、全层 frozen-VLM anchoring + 同观测方向词监督、任务/EEF 双图前向 gating。它们分别排除表征保持、输出语义一致和空间门控的替代解释。

VLM 阶段独立运行 `V0 checkpoint / V1 SFT / V2 DB-style feature retention / V3 T_sem → A_lang / V4 V2+V3`，先证明词级空间 grounding 是否超过通用 feature retention。这里的 V2 是 BlindVLA 思想的 VLM adaptation，不是完整 VLA 论文复现。VLA 阶段才用同一 VLA 学生比较 BC、DB-style feature alignment、`L_sem`、D0；正结果后增加完整 Anchor-Align-style 及其上叠加 D0 的实验。不同教师同时改变监督形态时需注明混杂。冻结与可训练 bridge、合法 EEF/障碍区域、动作输出梯度与动作损失梯度差异、二阶梯度成本均为训练前审计项。

本文后续原有 containment 公式尚为历史候选，不以 L1 概率图直接称“允许区域 mask”；归一化与 soft support/泄漏余量的修正建议见专题第 6 节，待小样本验证后再冻结。SpikingBrain 后置，主线模型组合继续由用户筛选。

### 首要比较：当前方法是否超过 BlindVLA-style feature retention

在 VLA 的任何 OOD 扩展前，首个动作结论必须来自下列同预算 paired comparison；它不替代 VLM 阶段的词级 grounding 验证：

```text
B0  native SFT/BC
B1  DB-style intermediate patch feature alignment
B2  fixed best-single diffusion T_sem → A_lang
B3  B1 + B2
B4  B3 + D0 soft action-evidence leakage constraint
B5  B3 + D0-CF counterfactual action-evidence support (candidate; only after offline gate)
```

所有组固定 backbone/checkpoint、训练 episodes、train/val/test split、LoRA、action head、action chunk、图像增强、优化步数和随机种子；教师额外前向、离线 map 生成和二阶梯度成本单独报告。B1 的代码实现参照 BlindVLA，若未完整修复其 projector/optimizer/恢复链路，只写 `DB-style inspired baseline`。B5 不是首轮必需组：只有冻结 B0 产生的动作条件 patch-occlusion 图在不同 mask 算子下稳定、且能区分正确动作与错动作/背景控制时，才进入训练比较。

判定顺序：先以 P4 的 VLM grounding 决定是否把 best-single `T_sem` 接入 VLA；再以动作输出归因、目标/背景干预和 ID/OOD 动作指标判断 B4 是否超过 B3。若 B4 只令图更集中、却不改善干预一致性或 OOD，则不能作为主贡献。B5 检验动作条件反事实证据能否有依据地扩展语言实体区域，覆盖夹爪、接触边缘或目标开口等非名词线索；它必须通过正确动作、错动作、随机 patch、背景 patch、mask 算子控制，并在动作相关指标上优于 B4。若该 gate 失败，patch occlusion 只保留为评测干预，不当作训练教师。若 B1 已经覆盖 B3/B4 的收益，论文主张退回 representation retention；若 B2/B3 有收益而 B1 无收益，才说明词级空间教师值得保留。

## 5.1 实验总表

本实验表借鉴 *Breaking the Vision–Action Shortcut* 的四个设计原则：异构结构覆盖、architecture-matched paired baseline、ID/OOD 成对评估、反事实与组件拆分。具体借鉴分析见 `references/paper_reading/breaking_vision_action_shortcut_notes.md`。

| 实验 | 数据/模型 | 主要问题 | 预期结论 |
|---|---|---|---|
| E0 接口与坐标桥 | 合成棋盘格、Flickr30k | token/grid/window 恢复是否正确 | 峰值位置恢复误差低于预设阈值，否则停止后续训练 |
| E1 Sink 与归因诊断 | OpenVLA/Qwen/LLaVA；SpikingBrain 后置 | 热力图是否被 sink 或固定热点污染 | raw/sink-free 差异按模型报告，不把图像美观当收益 |
| E2 VLM grounding | Flickr30k/Entities；Qwen、LLaVA，SpikingBrain 后置 | `T_sem → A_lang` 是否改善 grounding | 正确教师优于错词/错图/随机教师；至少两个结构不同模型成立 |
| E3 VLM 证据校准 | 同 E2 | attention/gradient 与干预是否一致 | intervention agreement 高于 attention-only；若不成立，降低解释性主张 |
| E4 LIBERO 离线 VLA | 一个 pick-place | D0 是否改善单步动作和归因 containment | action MAE 不升高，`A_act` 越界比例下降 |
| E5 LIBERO rollout | 同 E4 | D0 是否改善闭环控制 | 成功率和接近/抓取/放置子阶段至少一项改善 |
| E6 DROID 离线泛化 | pick-place 子集；7D delta EEF | 是否跨真实视觉分布有效 | D0 相对 baseline 的收益不只出现在 LIBERO；按 camera/object/scene/operator 划分 |
| E7 D1 phase | LIBERO/DROID 轨迹 | `source→mixed→target` 是否存在且有益 | 正确 phase 优于反向/随机/固定百分比；否则保留 D0 |
| E8 B WAM/VAM | DROID 离线 future attribution | 未来归因是否比静态图更接近动作成败 | 只有通过 horizon/动作负控才进入主结果 |
| E9 跨骨干 | OpenVLA/OFT、π0 系、Qwen/LLaVA；SpikingBrain 后置 | 方法是否依赖某个骨干 | 统一归因接口在至少两种结构上有效 |

### VLM benchmark 的固定分工

```text
Teacher calibration: Flickr30k Entities calibration split
ID spatial grounding: Flickr30k Entities test
Semantic referring transfer: RefCOCOg official split
Visual task-preserving OOD: Flickr30k Entities test perturbations
VLM retention diagnostic: VL-Think/SimplerEnv static screenshot QA
```

主结论来自带 phrase-region 标注的 Flickr30k Entities；RefCOCOg 检验更复杂指代，但不与 Flickr 分数合并成平均榜；图像扰动保持原 region 标注不变，单列报告每种 photometric corruption；VL-Think style QA 检验概念保留，不替代空间定位。

### Lavender-style 能力评测轨道

几何 grounding 只能回答“模型的空间证据是否落在目标区域”。为了检验方法是否像 Lavender 一样真正改善 VLM 能力，增加一条独立的下游能力轨道。这里复刻的是 Lavender 的**评测逻辑和分组方式**，不是声称完整复现其 20 个 benchmark 或原始模型结果。

#### 评测分层

| 层级 | 首轮数据集 | 指标 | 要回答的问题 |
|---|---|---|---|
| Caption | COCO Captions、Flickr30k captions | CIDEr、BLEU-4、METEOR、ROUGE-L | 语义对齐是否改善描述，而不是只改变热力图 |
| General VQA | VQAv2、OK-VQA、ScienceQA | 官方 accuracy / exact match | 一般视觉问答和知识推理是否保持或提升 |
| Fine-grained/OCR | TextVQA、DocVQA、OCRBench、InfoVQA | 官方 accuracy / ANLS 或 benchmark 默认分数 | 细粒度文字、文档和局部区域能力是否提升 |
| Hallucination/robustness | POPE、HallucinationBench、HatefulMemes | 官方 accuracy、F1 或 hallucination rate | 是否减少视觉幻觉，而非只提高语言流畅度 |
| Broad perception | MME、MMBench、MMStar、MMMU | 官方分数 | 方法是否损伤通用感知和多学科推理 |
| OOD | WorldMedQA-V、固定视觉扰动集 | 多语言/医学 VQA accuracy；按扰动类型分组 | 是否具有跨领域和视觉分布的有限泛化 |

#### 首轮与完整套件

首轮不直接跑完整 20 项，而采用固定的最小套件：`COCO Captions + VQAv2 + TextVQA + POPE + MME + WorldMedQA-V`。它覆盖 caption、一般问答、细粒度/OCR、幻觉、综合感知和 OOD 六类能力。F1-10k 方向性结果通过后，再加入 `OK-VQA、DocVQA、OCRBench、MMBench、MMStar、MMMU、ScienceQA、InfoVQA、HatefulMemes`，形成扩展套件。

Lavender 论文中的“20 benchmark”用于说明其覆盖面；我们的主表只报告实际运行、版本冻结且能公平比较的项目。没有可复现 evaluator 或预算不足的项目放入附录，不用缺失项计算平均分。

#### 统一比较协议

所有能力套件都使用 `V0 原始 checkpoint / V1 caption SFT / V2 feature retention / V3 T_sem→A_lang / V4 V2+T_sem→A_lang` 的 paired checkpoint。每个模型固定 prompt、解码参数、图像分辨率、few-shot 设置、评测脚本和 evaluator；不同模型的原生 leaderboard 分数不混入主表。

每个 benchmark 同时报告：

1. 绝对分数；
2. 相对 V1 的绝对增量和相对增幅；
3. V3 相对 V1、V4 相对 V2 的 paired difference；
4. 按能力类别的 macro average，但只在该类别所有项目都完成时计算；
5. 训练数据与测试集的来源重叠审计。

Lavender 还观察了数据规模、训练步数、attention MSE 与下游分数的关系。我们对应记录训练样本数、wall-clock、教师 cache 成本、显存和 loss 曲线，并画 `semantic-map loss / attribution metric` 与下游分数的相关图；相关性只作分析，不作为因果证据。

#### 能力与几何结果如何合并解释

两条轨道必须分开报告，再用预先定义的判定表解释：

| 几何 grounding | Lavender-style 能力 | 结论 |
|---|---|---|
| 提升 | 提升或保持 | 最强证据：空间监督带来真实能力增益 |
| 提升 | 下降 | 归因约束过强或损伤语言能力；不能宣称方法有效 |
| 不变/下降 | 提升 | 只能声称下游 SFT/正则收益，不能声称 grounding 改善 |
| 不变/下降 | 不变/下降 | 停止扩大 teacher 或 loss 组合，先检查接口与数据 |

正确教师、错词、错图和随机图控制也要跑能力套件中的小子集，优先选择 `TextVQA、POPE、VQAv2`。如果任意空间图都带来相同能力增益，说明收益可能来自正则化或训练预算，而不是词级语义。

#### OOD 和定性分析

WorldMedQA-V 必须保持完全未参与训练、teacher calibration 和阈值选择。报告总体分数、语言分组、问题类型分组和错误案例；固定视觉扰动集报告每类 corruption，而不只报平均值。定性图同时展示输入图、教师图、学生 `A_lang`、预测答案、正确答案和错误类型，延续 Lavender 的可视化方式，但不把好看的图当作定量证据。

因此，VLM 阶段的主结论需要满足两条相互独立的证据：`V3/V4` 在 Flickr30k Entities 上改善或至少保持几何 grounding，并在 Lavender-style 能力套件上相对对应 baseline 有稳定收益或无显著损伤。只满足其中一条时，论文主张必须相应收窄。

### 首轮训练监督与 Lavender 的可比性

VLM 主表的 V1--V4 使用图像 caption SFT，而不把 Entities box 或 referring-expression 作为训练标签：固定 caption prompt，监督原始 Flickr30k caption。Lavender 的基础监督也是 image-to-caption SFT，并将 Stable Diffusion 的 per-caption-token attention 作为额外 MSE 信号。我们的 V3/V4 在相同 caption token 上以语言条件空间归因 `A_lang` 对齐 `T_sem`；V2/V4 的 retention 项也不改变任务、数据或生成目标。

因此，Entities 的人工 phrase-box 只在独立 calibration/test 中选择、评价和反证教师图。若将来训练 “Where is the red ball?” 一类 referring prompt，必须单列为额外数据形式消融，不能与 V0--V4 主表混合，否则性能变化无法归因于归因对齐方法。

### F1 结果的预期结论与反证解释

我们不把任一单项提升当作 idea 成立。F1-10k 预先要求五层证据：`V3 > V1` 证明 semantic attribution 在 SFT 之上有独立价值；`V4 > V2` 证明其不被 visual retention 完全解释；正确教师优于错词/错图/随机图证明词级空间语义被使用；`V3 > V3-attn-proxy` 证明 output-conditioned attribution 优于直接 attention/rollout；独立 Flickr test 与固定 visual-OOD 同趋势才支持有限的 OOD 主张。

若 C1 不成立，主张退回为无增益；若 C2 不成立，主张退回为 retention 替代实现；若 C3 不成立，主张退回为一般空间正则；若 C4 不成立，不宣称 attribution 的必要性；若 C5 不成立，只报告 ID 效果。Lavender exact 与完整 BlindVLA policy 比较用于补强近邻定位，不是这五条核心可证伪判据的替代品。

### 快速 idea validation 与正式主实验的分层

在投入 F1-10k 前，先用 F0v2-256 训练结果和一个完全 held-out 的 64--128 image-phrase test subset 做方向性验证：恢复 V0--V4 checkpoint，统一计算 pointing/mass-in-box/IoU，再加入 wrong-word、wrong-image、random-map 三类负控。只有 `V3 > V1`、`V4 > V2`、正确教师优于负控且 held-out 趋势一致，才把方法推进到 F1-10k；该阶段不宣称最终泛化或论文显著性。

完整 Lavender / BlindVLA 源码比较安排在 F1 后：当前阶段只采用其已审计且可迁移的机制。Lavender exact 需要显式 cross-attention VLM，BlindVLA exact 需要 OpenVLA/OFT policy/action benchmark；把任一源码强行直接迁移到 Qwen caption F0 会混淆学生图定义或任务层级，不能作为公平结论。

### 5.0 最终冻结协议：主次指标、选择时点和结论层级

正式主矩阵固定为：

```text
V0 = 原始 Qwen checkpoint，仅评测
V1 = 完整 Flickr30k caption SFT
V2 = V1 + 冻结 DINOv2 ViT-L/14 retention
V3 = V1 + 单一 best-single T_sem → A_lang
V4 = V2 + 同一 T_sem → A_lang
```

V3/V1 和 V4/V2 是唯一主比较。正确教师、错词、错图、随机图和 raw-attention proxy 是机制对照；DVD-style feature distillation、完整 BlindVLA policy、teacher ensemble 和其他 loss family 放在独立消融/附录。VLM 主矩阵不使用动作标签、`A_act` 或 VLA success。

正式训练预算冻结为完整有效 train manifest 的 1 epoch、seed 17/29/41、无 test-based early stopping。选择 1 epoch 的工程理由是固定一次完整数据遍历和相同更新预算，而不是声称它是最优 epoch 数；若所有组在 epoch 末共同 underfit，只执行预先登记的全组 2-epoch sensitivity run。teacher 在正式训练前用约 1000 张独立 validation calibration 图选择，候选为 SD1.5、PixArt-α、PixArt-Σ、Playground-v2.5，按 pointing、mass/soft-IoU、无效词率和跨 seed 稳定性排序，只选一个 best-single。test、RefCOCOg、OOD 和下游能力分数不得参与 teacher 或阈值选择。

主指标是 Flickr30k Entities phrase pointing；mass-in-box、soft-IoU、calibrated IoU、top-k IoU、entropy、增强一致性、patch intervention agreement、官方 caption/VQA/OCR/幻觉 benchmark、RefCOCOg 和 visual OOD 全部记录。正式 benchmark 使用官方 split、官方输入格式和官方 evaluator。成功分四级：主指标显著且达到预设实用增量；主指标不劣但收敛/总成本改善；几何增益有限但正确 teacher、干预和 OOD 机制证据成立；能力 benchmark 在不损伤 grounding 的情况下有稳定提升。任何意外指标提升保留并解释，但不能替换主判定。

这也给出 VLM→VLA 的论文叙事：VLM 证明语言条件空间归因是可迁移的监督接口；VLA 将同一接口接到 action-loss gradient×activation，检验 `A_act` 是否落在 `A_lang` 目标区域并随阶段转移。LIBERO 负责机制和闭环，LIBERO-Plus/LIBERO-PRO 负责可控 OOD，DROID 负责真实离线泛化，SimplerEnv 负责视觉/语义保留诊断，RoboTwin 后置。首轮不增加额外 benchmark。

跨模型声明只在 Qwen 与一个通过 P1 的非 Qwen 骨干（优先 LLaVA-OneVision，否则 Prismatic）都完成 V1/V3 同协议复现后成立；只有 Qwen 时，论文只能声称 Qwen 实例验证。

## 5.1.1 OOD 假设与分维度报告

论文不声称解决所有 OOD。D0 的可检验假设仅针对语言无关视觉捷径与语言空间重新 grounding：背景/光照/干扰物变化时保持动作稳定，目标对象/属性/位置/语言变化时相应改变动作。

| OOD 轴 | 实例 | 主要证据 | 方法预期 | 失败时的解释 |
|---|---|---|---|---|
| Nuisance visual | background、lighting、texture、distractor、sensor noise | LIBERO-Plus、SimplerEnv、DROID 场景/相机分组 | 少依赖无关区域，保持任务相关反应 | 若提升只在 ID，不能称快捷方式缓解 |
| Semantic/referential | rephrase、object swap、颜色/属性、多实例、position swap | LIBERO-PRO、VL-Think 风格保留集 | 正确重定向 source/target | 若不随目标变化，稳定可能是忽略语言 |
| State | robot initial state、有限 proprioception 改变 | paired state intervention | 区分合理控制依赖与 state shortcut | 屏蔽全部 state 后失败不等于发现 shortcut |
| Camera/geometry | viewpoint、crop、有限视角变化 | Plus/SimplerEnv/DROID camera split | 对已校准变化更稳定 | 不能据此声称学会 3D 几何 |
| Dynamics/embodiment | 接触、长时序、新动作空间/平台 | D1/B、RoboTwin、CALVIN、真机 | 后期单独检验 | 不属于 D0 默认能力 |

主表必须显示逐轴结果，不能只报平均 OOD。每个轴至少包含 baseline、正确教师、错词/错图/随机教师；视觉与语义变化还需包含对应反事实。DROID 的结论只写“离线真实数据泛化”，闭环 success 仅由 LIBERO/真机支撑。

## 5.2 基线与消融

每个骨干必须使用 paired baseline：相同 pretrained backbone、相同数据、相同动作接口、相同训练预算和 native action objective。不得把不同模型的原生结果直接混成一张性能表。

### VLM

Don't Blind 现为本阶段重点参照。加入 **原始 checkpoint（仅评估） / 普通 SFT / DB-style patch feature alignment / Lavender map alignment / teacher-to-output-attribution / feature+attribution** 的候选矩阵，具体执行顺序见 [官方代码审计第 9 节](../references/abstract_review/08_blindvla_code_audit.md)。用 VL-Think 风格保留集与区域标注集分开测概念保留和空间定位；官方 attention ratio 与 raw attention 分开画、共享颜色标尺并报告扰动一致性。

- no alignment；
- Stable Diffusion/Lavender、PixArt-α、PixArt-Σ、Playground-v2.5 各自的 attention-to-attention 或可用 teacher-to-attribution；
- 校准集冻结的 best-single 与 confidence-weighted ensemble；
- teacher-to-attribution；
- 按所选主线结构比较早/中/晚层与同数量随机层；SpikingBrain 固定层号后置；
- raw/sink-free；
- 正确教师、错词、错图、随机图；
- attention、gradient×activation、occlusion。

### VLA

- behavior cloning baseline；
- `L_sem only`；
- `L_sem + L_contain`（D0）；
- D0 + D1 phase；
- D0 + C state-rule diagnostic；
- D0 + B future refiner；
- B/C without D；
- 只做 latent/aggregation 或只做 phase/pose proxy 的替代解释；
- action expert 蒸馏负控；
- wrong phase、wrong horizon、wrong action、random teacher。

### 借鉴机制的最小可执行消融

这些机制不能一次性叠加；每个机制先作为独立对照或诊断，再决定是否进入后续阶段。

| 机制 | 首次出现 | 实现 | 要回答的问题 | 是否进入主方法 |
|---|---|---|---|---|
| BlindVLA retention drift | B1/G1 | 记录层级 feature drift、collapse、sink mass、VL-Think 保留；后期可做 drift-weighted `L_ret` | 语义/动作增益是否只是视觉表征保持 | 否，B1 是强 baseline |
| LIT paired budget | G1 | 共享初始化、数据、action head、训练步数和 evaluator | 比较是否公平 | 是评测规则，不是 loss |
| LIT task-preserving visual intervention | G1/G2 | distractor、blur、lighting/background/camera 改变，instruction/state 不变 | 无关视觉变化是否改变动作 | 否，机制评测 |
| LIT goal-changing intervention | G1/G2 | 保持图像和 state，只改变目标语言 | 模型是否响应真正的目标变化 | 否，机制评测 |
| Next Forcing multi-layer attribution | layer ablation | 固定早/中/后层聚合 `A_lang/A_act` | 单层选择是否导致归因不稳定 | 否，稳定性消融 |
| `E-LIT-prior` | G1 通过后 | image-free action prior 后接视觉训练 | 收益来自 action prior 还是空间归因 | 否，独立 baseline |
| `R_future` | G2 通过后 | short/mid/long future attribution，wrong-future 负控 | 当前动作归因是否具有未来一致性 | 后期 WAM/VAM 扩展 |

禁止首轮同时使用 retention、semantic、phase、future 和 pose 五类损失。否则即使 success 上升，也不能判断提升来自哪条机制链路。

### 推荐的主方法版本

```text
Ours-1: action objective + T_sem -> A_lang + L_sem + D0 A_act containment
Ours-2: Ours-1 + BlindVLA-style L_ret
Ours-3: Ours-2 + R_future consistency
```

Ours-1 是主方法候选；Ours-2 只用于证明语义归因在视觉表征保持之上仍有增量；Ours-3 只在 WAM/VAM 后期扩展。完整 LIT latent bottleneck、完整 Next Forcing MCP、EEF 高斯动作 teacher 和多 teacher ensemble 不进入首轮主方法。

## 5.3 指标

### Grounding

- pointing accuracy；
- phrase-region IoU；
- pointing game top-1；
- attribution entropy；
- cross-augmentation consistency；
- intervention agreement。

### 动作

- 7D delta EEF MAE/RMSE；
- gripper accuracy；
- trajectory smoothness；
- `A_act` 落在 `A_lang` 并集外的质量比例；
- patch occlusion 与 `A_act` 的 rank correlation；
- LIBERO success rate；
- DROID 场景/物体/相机划分下的离线误差。

## 5.4 每一步结论门槛

1. **E0 失败**：坐标桥或窗口逆变换错误，不能继续。
2. **E1 显示 sink 很强**：保留双轨并启用 mask；如果 sink 只影响可视化，不改变主损失。
3. **E2 正确教师无优势**：检查教师图质量和词 span，暂停 VLA；不能靠调大 `λ` 掩盖。
4. **E3 attention 与干预不一致**：论文只保留 gradient/intervention 结论，不宣称 attention 解释。
5. **E4 D0 降低动作误差且 containment 改善**：进入 E5；若 grounding 改善但动作无效，检查 action head 归因接口。
6. **E5 成功率下降**：降低 `λ_contain`、检查过强约束和 state 输入；D0 不能作为成功方法。
7. **E6 DROID 无收益**：主结论限定为 LIBERO 机制，不声称真实泛化，并优先排查数据转换和分布差异。
8. **OOD 平均改善但目标反事实失败**：不声称语言 grounding 改善；模型可能只是变得不敏感。
9. **视觉扰动稳定但背景/目标 occlusion 同样无影响**：不声称动作证据正确，模型可能忽略视觉或依赖 state。
10. **E7 phase 负控无差异**：D1 退回诊断，D0 保持主线。
11. **E8 future attribution 无额外信息**：B 放入 future work，不增加 world-model loss。
12. **E9 只在一个骨干有效**：限定模型适用范围，解释失败原因；不自动改回 SpikingBrain 主线。

## 5.5 预期主表结构

| 模型/数据 | Baseline | `L_sem` | D0 | D1 | D0 + B | 归因干预一致性 |
|---|---:|---:|---:|---:|---:|---:|
| OpenVLA/OFT / LIBERO | 待测 | 待测 | 待测 | 待测 | 后置 | 待测 |
| OpenVLA/OFT / DROID | 待测 | 待测 | 后置 | 后置 | 后置 | 待测 |
| π0 系或第二 VLA / DROID | 待测 | 待测 | 待测 | 后置 | 后置 | 待测 |
| SpikingBrain / 后置扩展 | 待测 | 待测 | 后置 | 后置 | 后置 | 待测 |

所有“预期”在实际结果前都保持为假设，不能写成已验证结论。

### VLA 当前工程证据边界（2026-10-06）

缓存与 B0 的工程链路已经打通：三类 teacher cache 通过审计，OpenVLA/OFT P1、30-step action smoke、独立 checkpoint restore 和官方 LIBERO episode loop 均可运行。B0 未充分训练 checkpoint 的两个 rollout 为 0/2，不能作为方法性能结论。当前证据只支持“训练与评测接口可复现、归因与 teacher cache 可以接入”，尚不支持“语义归因改善 VLA 成功率”。严格 same-image phrase-role swap、面积匹配随机图、正式 matched B0–B4 训练和三 seed 统计仍是机制与性能结论的前置条件。

### 严格 teacher control 的新增证据

在 OpenVLA/OFT 上，same-image phrase-role swap 与面积匹配随机图均能通过 20-step finite interface smoke；两者使用相同模型 revision、manifest、seed、semantic cache 和损失权重。该结果排除了“控制定义或数据接口无法运行”的工程问题，但不等价于正确 teacher 在正式训练或 paired validation 指标上已经显著优于负控。正式论文结论仍必须来自固定验证集、独立测试集和多 seed 的统计比较。


### VLA 前置状态（2026-10-06）

缓存审计和 B0–B4 工程接口链已经打通，但当前不把 smoke 或两条 rollout engineering 记录当作正式 LIBERO 性能结果。正式结果前必须完成 rollout demonstration provenance 与训练集的严格隔离审计；因此当前阶段的结论边界是“实现可运行、教师和归因接口可接入、控制组可执行”，而不是“方法已经提升成功率”。


### Rollout 结论边界

rollout inventory 的唯一性不能替代 demo provenance。当前只证明 split episode ID 互斥和 rollout init state 去重，尚未证明 rollout 初始状态不来自训练 demo；因此 rollout 结果只能作为工程可运行性证据。


### B0 offline preflight 证据

50-step B0 offline preflight 在固定 manifest、OpenVLA/OFT checkpoint 和 7D delta EEF action schema 下通过，所有训练 loss、gradient 和 held-out prediction 均为 finite。该结果仅支持“训练接口和离线评测链路可运行”，不支持性能提升或 rollout 泛化结论。


### B0 restore 证据

独立 checkpoint restore 达到 exact prediction equality，continuation 的 loss/gradient、学习率和 scheduler 均可复现。这证明训练状态可恢复，但仍不代表方法在 LIBERO rollout 上取得性能提升。


### B1–B4 接口证据边界

B1–B4 的 20-step smoke 证明接口可运行，但 B1 使用了不同版本的 C-RADIO manifest，不能直接与当前冻结 cache 作为同一正式实验比较；B2–B4 的 semantic manifest 已一致。


### C-RADIO teacher provenance 修正

正式记录以直接计算的 `cache_manifest.jsonl` SHA 为准。此次只修复 audit metadata 并重跑接口 smoke，不把旧 manifest SHA 混入正式比较。


### VLA smoke 统一结论边界

统一审计确认 B0–B4 链路和控制组可运行，但这些结果不支持正式 benchmark 或 rollout 泛化结论。论文中必须把 rollout provenance 未验证作为当前限制，直到获得可追溯的 demo-to-init-state 证据。

### P6 seed17 正式 offline action prediction（2026-10-08）

OpenVLA/OFT + LIBERO 的 B0–B4 seed17 已完成 10,080 optimizer updates，并在 held-out offline action prediction 上恢复 `checkpoint_10080` 评测。该评测只比较 action prediction error，不报告 LIBERO rollout success。完整表见 `experiments/P6-vla-b0-b4/results/formal_offline_action_summary_20261008.md`。

| 组 | action MSE ↓ | action MAE ↓ | translation MSE ↓ | rotation MSE ↓ | gripper acc ↑ |
|---|---:|---:|---:|---:|---:|
| B0 原生 action baseline | 0.187634 | 0.308525 | 0.257857 | 0.141033 | 0.769343 |
| B1 BlindVLA-style retention | 0.170122 | 0.293686 | 0.223477 | 0.136108 | 0.786573 |
| B2 semantic attribution | 0.270470 | 0.383775 | 0.318018 | 0.149714 | 1.000000 |
| B3 retention + semantic attribution | 0.204097 | 0.326322 | 0.266259 | 0.142188 | 0.815345 |
| B4 B3 + containment | 0.249770 | 0.384150 | 0.314967 | 0.149551 | 1.000000 |

当前 seed17 的直接结论是：B1 是最强 offline action baseline，说明 BlindVLA-style 视觉表征保持在这个设置下确实有效；B2 相对 B0 恶化，B3 相对 B1 恶化，B4 相对 B3 继续恶化，因此不能声称 semantic attribution 或 containment 已经提升 VLA action prediction。B4 相对 B2 的 action MSE 略低，只能说明 containment 在已恶化的 semantic 分支上有弱修正，不能说明 D0 成为有效方法。

该结果要求后续叙事收紧：VLA 阶段当前不是“方法已胜出”，而是“强 baseline 已跑通，语义/动作归因项暴露出 loss scale、teacher map、动作证据定义或 offline 指标不匹配的问题”。下一步必须完成 held-out semantic controls、action evidence support mass、wrong-word/wrong-image/random/area-matched random 对照和 patch/action evidence 干预，再决定是修正 `L_sem/L_contain` 进入第二轮，还是把 VLA 主张降级为机制诊断和 future work。multi-seed 与 rollout provenance 仍是正式性能结论的前置条件。


## 2026-10-09 P6 VLA seed17 offline result update

P6 seed17 post-train evaluation now has both held-out action prediction and held-out controls. Action prediction favors B1, the BlindVLA-style visual representation retention baseline. The semantic attribution / containment branches B2/B3/B4 do not improve held-out action MSE over B1. Controls provide only partial mechanism support: correct semantic maps are stronger than role-swapped and area-matched random maps, but they are not consistently stronger than wrong-image maps. This means the current method should not be claimed as a final VLA improvement yet; it supports a revised next iteration focused on instance-sensitive semantic teachers and phase/action-conditioned soft evidence alignment. No rollout success is claimed from this stage.

## 2026-10-13 更新：VLA 实验结论边界与下一轮假设

P6 的结果使 VLA 部分的论证边界更清楚：当前最强证据不是 semantic containment，而是 BlindVLA-style 视觉表征保持。B1 在 held-out offline action prediction 上最强；B2/B3/B4 没有稳定 action 收益。因此论文中不能把 P6 B3/B4 写成最终有效方法，只能作为“静态 containment 不足”的负结果和设计动机。

后续 VLA 论证分两步：

1. **闭环工程可运行性**：用 LIBERO pick-place task0 做最小 rollout smoke。B0/B1 各 1 episode 已完整跑通 evaluator，但 success=0，不作为 benchmark。扩展到 B0/B1/B3/B4 每组 5 episodes 后，只分析失败模式。
2. **P7-lite 机制修复**：以 B1 为基线，将 semantic loss 从“落在支撑区域内”改为判别式 ranking/KL：correct map 的 action evidence score 要高于 wrong-image、role-swapped、area-matched random。若该目标能提升 correct-vs-control margin 且不显著损伤 action MSE，再考虑进入正式训练和 rollout。

预期结论不再写成“semantic containment 一定提升 VLA”，而是更稳健地写成：视觉表征保持是强必要 baseline；语言条件空间归因若要服务 VLA，需要从静态 containment 转向可判别、阶段化或动态证据目标。P7-lite 是验证这一修正假设的最小实验。

### 2026-10-13 rollout smoke 实际结果

B0/B1/B3/B4 每组 5 episodes 的 LIBERO pick-place task0 smoke 已完成，所有 episode 都完整跑完，说明 checkpoint restore、official evaluator、模型路径、统计文件和 init-state provenance 可以工作。但四组 success 均为 `0/5`，post-hoc action log 显示失败集中在 gripper 饱和：B0/B1 是尾段 positive saturation，B3 大多 negative-like，B4 大多 positive saturation。

因此后续 VLA 不能直接扩大到正式 benchmark。下一步先做 action-interface repair smoke：expert replay、checkpoint action replay、gripper sign flip、gripper clamp、unnormalization sanity 和 OFT constants 检查。P7-lite ranking/KL 只作为 offline mechanism repair 继续推进；在闭环 0/20 的情况下，不能用 semantic loss 结果声称 VLA 成功。

### 2026-10-13 P7-lite ranking dry-run

基于 P6 controls records 的 dry-run 显示：B1/B3/B4 对 role-swapped 与 area-matched random 已经很容易区分，但对 wrong-image 基本接近随机。B1 的 correct > wrong-image rate 为 49.28%，B3 为 49.93%，B4 为 52.15%；而 correct > role-swapped 和 correct > area-matched random 大多在 97% 以上。

这意味着下一轮 semantic repair 的重点不应是继续证明“比随机图好”，而应是提升实例/图像条件辨别力，尤其是 correct map vs wrong-image map。P7-lite 因此以 B1 为 base，优先使用 wrong-image-focused ranking/KL 目标做 100–500 step offline smoke。若闭环 action-interface smoke 仍为 0 success，则 P7-lite 不进入 rollout，只作为机制修复实验。

### 2026-10-13 gripper sign flip 诊断

在同一 P6 checkpoint、同一 LIBERO task0 init state 上，仅把 env.step 前的 gripper action 取反，B0/B1 仍为 `0/1`。这说明当前闭环失败不是“简单 gripper 符号翻转”能够立即修复的问题。后续 action-interface 诊断应进入更细的 replay：expert action replay、预测动作 unnormalization replay、force negative/positive gripper 时间窗、以及 OFT constants / bounds_q99 统计一致性检查。



## 2026-10-13 补充：P6 rollout 失败的 action-interface 边界

P6 B0/B1/B3/B4 已能进入官方 LIBERO episode loop，但 task0 smoke 全部失败。最新 action-interface audit 显示：训练数据侧已经对 gripper 执行 `1 - raw_gripper`，官方 OpenVLA/OFT rollout 侧又在 `env.step` 前执行 gripper normalization 与 OpenVLA sign inversion；同时 P6 checkpoint 目录没有官方格式 `dataset_statistics.json`，当前由 wrapper 手动注入 `project_train` stats。替换为官方 OFT stats 后，B0/B1 各 1 episode 仍为 `0/1`，post-hoc 仍是 `late_gripper_positive_saturation_possible`。

因此当前结论边界冻结为：P6 的闭环失败首先是 action-interface / evaluator / task execution 问题，而不是 semantic attribution 方法本身的有效性结论。VLA 后续必须先建立 action-interface 正控，包括 official checkpoint 正控、expert action replay、windowed gripper intervention、chunk timing 和 camera/task provenance；在 B0/B1 出现非零 smoke success 或失败类型被明确修复前，不进入正式 rollout benchmark。P7-lite 只作为 offline mechanism repair 继续准备，不能提前声称闭环收益。

### 2026-10-13 expert replay 正控对 VLA 结论边界的影响

为避免把闭环失败误判为语义归因方法失败，我们加入了 expert action replay 正控：不加载模型，直接重放 OpenVLA modified LIBERO RLDS 的 task0 expert action。结果显示 raw expert action 在相同 task0/init-state 设置下成功 `6/9`，按官方 train01→env 约定转换后同样成功 `6/9`，而 gripper flip 为 `0/9`。这给出一个明确边界：LIBERO task0、init state、raw action convention 和官方 env action convention 均可产生成功轨迹。

模型 rollout 与 expert replay 的动作日志对比进一步显示，B0/B1 的 translation L2 mean 明显小于 expert，B3/B4 出现 gripper 饱和。因此当前 VLA 负结果应被解释为 action-interface / action-head / phase 或 normalization 问题仍未解决，而不是直接证明 semantic attribution 对 VLA 无效。论文实验叙事应保留 P6 的负结果：B1/retention 是强 baseline，静态 semantic containment 尚未带来 action 收益；同时把 P7-lite 定位为 offline mechanism repair，而非已经可进入正式 rollout 的最终方法。

### 2026-10-13 raw-vs-processed trace 的实验含义

raw-vs-processed trace 显示，B0/B1 的 translation/rotation 在 official `process_action` 前后不变，动作幅度不足主要来自模型 raw prediction 或 action head，而不是 rollout 后处理。gripper 维度会被 official postprocess 改变，且当前 posthoc 的 positive/negative 命名可能存在符号歧义，因此论文中应避免用单一 gripper 语义解释失败，只报告 positive/negative saturation 和 expert replay 正控。

`num_open_loop_steps=1` 的最小 smoke 仍为 B0/B1 `0/1`，说明 chunk timing 不是单独充分修复项。当前 VLA 阶段的合理结论是：B1 是 offline action 最强 baseline，但闭环部署还需要 action head / raw prediction calibration；semantic ranking/KL 的 P7-lite 只能继续作为 offline mechanism repair，不能跳过 action-interface 问题进入正式 rollout benchmark。

### 2026-10-14 VLA failure analysis wording update

For P6 rollout smoke, gripper-related failure types should be reported as sign-neutral saturation patterns, i.e., positive or negative saturation, rather than open or close failures. The raw-vs-processed action trace shows that the official OpenVLA/OFT action post-processing changes the gripper dimension, so assigning semantic open/close meaning to a sign before final convention verification would overstate the evidence.

This wording change does not weaken or improve the result. The substantive conclusion remains: B1 is the strongest offline action baseline, P6 semantic containment does not yet improve action prediction, and current rollout failures require action-interface/action-head diagnosis before any formal LIBERO benchmark claim.

### 2026-10-14 VLA diagnostic result: scalar action scaling is insufficient

A diagnostic-only rollout test scaled B1's first six EEF action dimensions by 1.5× and 2.0× before `env.step`. Both runs completed the 230-step LIBERO task0 episode without environment error, but both remained at 0/1 success with late positive gripper saturation. This weakens the hypothesis that simple EEF magnitude under-scaling alone explains the closed-loop failure. The next VLA iteration should focus on action-head calibration, state drift under closed-loop rollout, proprio/observation mismatch, gripper phase, and stage-aware behavior rather than using scalar action scaling as a method.

### 2026-10-14 State-trace diagnosis for VLA rollout failure

We added an expert state trace by replaying the successful demo0/init0 raw expert trajectory and recording EEF pose and gripper state at each step. Compared with the B1 rollout trace over the first 50 aligned steps, the expert has higher action translation magnitude (0.7973 vs. 0.5298), higher rotation magnitude (0.0467 vs. 0.0179), and larger realized EEF displacement (0.00965 vs. 0.00680). The mean EEF position gap between the model rollout and the expert trajectory is 9.49 cm, with a maximum of 15.15 cm.

This supports a more precise failure interpretation: the B1 policy does not merely suffer from a scalar action magnitude mismatch; the closed-loop state trajectory diverges from the expert trajectory early. Since 1.5× and 2.0× action scaling do not recover success, the next VLA iteration should diagnose action-head calibration, state-conditioned rollout drift, proprioception/observation mismatch, gripper phase, and stage-aware behavior before any formal LIBERO benchmark claim.

### 2026-10-14 Policy-on-expert-state diagnosis

To separate closed-loop state drift from action-mapping error, we queried the B1 policy on states generated by the expert trajectory while executing only expert actions in the environment. B1 remains mismatched to expert actions even on these expert-driven states: expert translation magnitude is 0.7973 while B1 predicts 0.6054, and expert rotation magnitude is 0.0467 while B1 predicts only 0.0131. The processed-vs-expert translation difference is 0.5704 and the rotation difference is 0.0388.

This means the rollout failure is not only caused by state drift after the model executes its own actions. The action head, action target calibration, rotation/gripper target distribution, or task-specific action learning is already insufficient on expert states. Therefore the next VLA iteration should repair action supervision and calibration before using semantic attribution losses as a closed-loop improvement claim.

### 2026-10-14 Action calibration audit before semantic VLA claims

The action calibration audit shows that B1's expert-state mismatch is concentrated in rotation. The B1/expert translation magnitude ratio is 0.759, but the rotation ratio is only 0.281. Per-dimension ratios are especially low for rot_x (0.033) and rot_y (0.096), while rot_z is closer (0.848). This suggests that the native mean L1 action objective over the 8×7 action chunk may under-emphasize low-magnitude rotation dimensions in our short training setting.

Therefore, the next VLA step should not add semantic losses directly. We introduce a P6.1 action-calibration smoke that starts from B1 and changes only action supervision, using a per-dimension weighted L1 such as `[1,1,1,3,3,2,1]`. The required gate is improvement on held-out per-dimension metrics and policy-on-expert-state query before any rollout or semantic P7-lite claim.


### 2026-10-14 P6.1 action calibration negative result

We tested a minimal action-calibration repair before adding more semantic losses. Starting from the B1 retention checkpoint, P6.1 used a per-dimension weighted L1 objective with weights `[1,1,1,3,3,2,1]` for 100 steps. The training and checkpoint restore succeeded, but the held-out action prediction gate failed. Compared with B1, P6.1 improved gripper sign accuracy from 0.7866 to 0.8531, while worsening action MAE from 0.2937 to 0.3034, translation MAE from 0.3775 to 0.3963, and rotation MAE from 0.2621 to 0.2675.

This is a useful negative result. It shows that the VLA bottleneck is not solved by naively upweighting weak rotation dimensions. The paper should not claim a closed-loop VLA improvement from P6/P6.1. The defensible story is narrower: B1/retention remains the strongest offline baseline; static semantic containment does not yet help action prediction; rollout failures require action-head, state-drift, proprioception, gripper-phase, and task-stage diagnostics before formal LIBERO/LIBERO-Plus/RoboTwin benchmark claims.

### 2026-10-14 P6.1a action-calibration ablation

We then tested a milder calibration ablation, P6.1a, using weights `[1,1,1,1.5,1.5,1.2,1]` and a lower learning rate `3e-5` for 100 steps from the same B1 checkpoint. This ablation fixes the main failure mode of P6.1: it no longer worsens action, translation, and rotation MAE relative to B1. On the held-out action set, B1 gives action/translation/rotation MAE of `0.293686/0.377516/0.262067`, while P6.1a gives `0.293306/0.376033/0.261234`. However, the gains are very small, gripper MAE worsens from `0.137056` to `0.141341`, and only gripper sign accuracy improves modestly from `0.786573` to `0.802503`.

This result should be interpreted as a diagnostic partial recovery, not as a VLA method success. It suggests that a softer action calibration can avoid damaging the B1 retention baseline, but it does not yet solve the closed-loop bottleneck. The next admissible test is policy-on-expert-state query for P6.1a. If the expert-state rotation ratio and phase behavior do not improve, the VLA path should shift from semantic losses to action target normalization, rotation representation, action-head/proprio design, and task-stage supervision.

### 2026-10-14 P6.1a expert-state gate

We further evaluated P6.1a on policy-on-expert-state query, where the environment is advanced by expert actions and the policy is only queried on those expert-driven states. P6.1a slightly reduces the processed-vs-expert translation difference from `0.570370` to `0.563464` and the rotation difference from `0.038765` to `0.038335`. However, it does not improve the key rotation-magnitude bottleneck: B1's model rotation L2 mean is `0.013138`, P6.1a's is `0.013062`, while the expert is `0.046681`.

This closes the current action-calibration branch as a No-Go for rollout. The result says that mild weighted action calibration can avoid damaging the offline B1 baseline, but it still does not repair the rotation/action-head mismatch on expert states. For the paper, this is a useful boundary result: before making any closed-loop VLA claim, the next system iteration must audit action target normalization, rotation representation, action-head/proprio design, and stage supervision. Semantic ranking losses remain offline mechanism experiments until the action interface is fixed.


### P6 action target audit and its implication for VLA evidence

A read-only target audit was added after P6.1/P6.1a failed the rollout gate. The audit reads the exact RLDS episodes used by the P6 train/eval manifests, reproduces the gripper transform and bounds-q99 normalization, and computes per-dimension target distributions and H=8 chunk statistics.

Main evidence:

- Train split: 27 episodes, 2709 steps, 2520 H=8 chunks, 0 hash failures.
- Eval split: 9 episodes, 832 steps, 769 H=8 chunks, 0 hash failures.
- Train normalized translation mean_abs average is 0.4645.
- Train normalized rotation mean_abs average is 0.2839.
- Rotation target saturation is low, about 0.021 per rotation dimension.
- Rotation near-zero<=0.05 is only about 0.066–0.079.
- Eval rotation distribution is close to train.

This weakens the simple explanation that the model fails because rotation targets are numerically tiny after normalization. The more defensible interpretation is that the current VLA bottleneck lies in action-head learning, chunk execution, controller/action convention, or rollout interface. Therefore, semantic attribution losses should not be treated as the next direct fix for closed-loop VLA until the action interface gate is repaired.

For the paper narrative, this result is useful even though it is negative: it prevents overstating the semantic attribution method and gives a clean boundary between VLM/VLA grounding evidence and robotics execution evidence. VLA claims should require a repaired action-interface baseline before comparing B2/B3/B4-style semantic methods.


### 2026-10-14 Action-head and rollout-interface audit

We added a read-only audit of the OpenVLA/OFT action path after the action target audit. The audit confirms that the model predicts an 8-step, 7D action chunk, the native continuous objective is mean L1 over the full `B×8×7` chunk, and the official LIBERO rollout path executes chunked actions with gripper binarization and OpenVLA sign inversion. The source snapshot is OpenVLA-OFT commit `e4287e94541f459edc4feabc4e181f537cd569a8`.

The evidence sharpens the negative result. B1 has an expert-state translation magnitude ratio of `0.759` and a rotation magnitude ratio of only `0.281`. P6.1a slightly improves held-out action, translation, and rotation MAE, but its expert-state rotation ratio remains `0.280`. This means mild weighted action calibration changes the modules but does not repair the main rotation/action-head bottleneck. Since the target audit already shows non-trivial normalized rotation targets, the failure is more likely in action-head learning, chunk-level loss reduction, horizon-specific execution, or rollout action processing than in the absence of semantic attribution.

This affects the paper plan. P6/P6.1/P6.1a should be reported as a boundary and diagnostic result, not as a VLA success. The next VLA claim requires an action-interface gate: per-horizon error, first-action error, raw/processed/env-action consistency, and gripper convention verification. Semantic ranking or attribution losses can remain in the method pipeline, but they should not be used to claim closed-loop robotics improvement until B0/B1 action execution is repaired.


### 2026-10-14 Horizon-wise action diagnosis

A horizon × dimension diagnostic was added to avoid hiding rollout-relevant errors behind the full-chunk mean L1. The result again supports a No-Go decision for immediate rollout benchmark. P6.1a reduces first-horizon action MAE from `0.292031` to `0.291071`, but the improvement is only diagnostic-scale and the last-horizon MAE slightly worsens from `0.301750` to `0.302049`.

The more important evidence is dimensional under-production. At horizon 0, B1's predicted-to-target absolute magnitude ratios for `rot_x/rot_y/rot_z` are `0.477/0.181/0.338`; P6.1a only raises them to `0.512/0.206/0.358`. This is consistent with the expert-state query result: the action head still under-produces rotation even when the state is driven by the expert.

This gives the next VLA iteration a sharper target. The right repair is not another broad semantic loss, but an action-interface repair that explicitly handles horizon and action dimension behavior. Candidate repairs include first-action or horizon-aware weighting, action-head/proprio-projector-only smoke training, and rotation-specific calibration. These remain engineering gates before any formal LIBERO/LIBERO-Plus/RoboTwin benchmark claim.

### 2026-10-14 P6.2 action-head/proprio repair boundary

A narrower P6.2 repair was tested after P6.1/P6.1a and the action-interface audits. The experiment starts from the B1 checkpoint, freezes the policy LoRA, and updates only the action head and proprio projector with a horizon-weighted native `8×7` action loss. It passed the training and restore smoke, but the held-out offline gains were too small to unlock rollout claims: action MAE changed from `0.293686` to `0.293562`, first-horizon MAE from `0.292031` to `0.290828`, and rotation MAE from `0.262067` to `0.261015`, while translation and gripper MAE slightly worsened.

This sharpens the paper boundary. The current VLA evidence says that BlindVLA-style B1 remains the strongest robust baseline, simple semantic containment does not improve action prediction, and isolated action-head/proprio tuning produces only diagnostic-level gains. Formal LIBERO/LIBERO-Plus/RoboTwin benchmark claims require a stronger action-interface repair or a nonzero minimal rollout smoke; P7-lite ranking/KL remains an offline mechanism experiment until that gate is passed.

### 2026-10-14 P7-lite mechanism dry-run

We added a P7-lite dry-run that reuses the P6 control records and evaluates a discriminative semantic-evidence ranking objective without training. The strongest failure mode is correct-vs-wrong-image separation. For B1, correct evidence mass beats the wrong-image map in only `49.3%` of held-out records, and only `32.6%` with a `0.02` margin. In contrast, role-swapped and area-matched random negatives are already mostly separable.

This result gives a concrete next mechanism target: if semantic attribution is continued in VLA, the loss should prioritize image-instance-sensitive ranking rather than static containment. It does not establish a closed-loop VLA gain. The paper should state that P7-lite is an offline mechanism repair candidate and remains gated by action-head/interface repair before any formal LIBERO or RoboTwin benchmark claim.

### 2026-10-14 VLA positive-control checkpoint repair 与 evaluator preflight

当前 VLA 主线已切回 positive control：先确认官方/公开 OpenVLA-OFT checkpoint 能在 LIBERO `libero_spatial` task0 上完成最小 rollout，再决定是否继续 P6.3/P7-lite。101 SSH 已恢复。公开 checkpoint 原有三个 shard 不完整，其中 shard 1 已通过 mirror-first 续传完成并验证 SHA256；shard 2 正在续传，shard 3 尚未开始。完整 hash verified 前不运行 rollout。

官方 evaluator 必须使用 workspace 的 OFT 环境 `/vepfs-mlp2/c20250405/400040/transfer/vla_attention/vla_workspace/envs/oft/bin/python`，并设置 `LIBERO_CONFIG_PATH=/vepfs-mlp2/c20250405/400040/transfer/vla_attention/vla_workspace/libero_config`。`/root/starvla_cu124` 缺 `draccus`，不能作为官方 evaluator 环境。preflight 已确认 OpenVLA-OFT commit `e4287e94541f459edc4feabc4e181f537cd569a8`、LIBERO commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`，以及官方常量 `H=8, D=7, proprio=8, bounds_q99`。

首轮 task 固定为 `libero_spatial` task0：`pick up the black bowl between the plate and the ramekin and place it on the plate`，默认 init states `(50, 92)`，SHA256 `f061b67788d541001f3d6872be9a3a3b0b7de3490dac4683e181f9b0703f8e20`。positive control 至少 5 episodes；若仍为 0 success，按 evaluator wrapper、dataset statistics、action normalization、gripper convention、camera、task/init state、chunk execution 的顺序排查。

### 2026-10-14 官方 OpenVLA-OFT/LIBERO task0 positive control 通过

公开 `moojink/openvla-7b-oft-finetuned-libero-spatial` checkpoint 已修复完整，并用官方 OpenVLA-OFT evaluator 函数在 `libero_spatial` task0 上完成 5-episode positive control。结果为 `5/5` success，`success_rate=1.0`，远端目录为 `/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/positive_control_moojink_task0_20261014/`。

该结果改变 VLA 诊断边界：LIBERO 环境、官方 evaluator、默认 init state、agent/wrist camera、`H=8,D=7` chunk、`bounds_q99` unnormalization 和 gripper/action processing 在官方 checkpoint 上可成功。因此，P6 B0/B1/B3/B4 rollout smoke 的 0 success 不能再归因于基础管线整体不可用；后续应对比我们的 checkpoint restore、训练后 action head/proprio projector、normalization key、action magnitude、rotation under-production 和 rollout wrapper 差异。下一步仍按固定顺序做 expert action replay 与 tiny overfit，不直接跳到 P6.3/P7-lite。

### 2026-10-14 expert action replay positive control 复核

官方 checkpoint positive control 通过后，已复核 LIBERO task0 expert action replay。远端结果为 `/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_action_calibration/expert_replay_positive_control_20261014_after_moojink/`，3 条 demo × init 0/1/2 × `raw/train01_to_env` 共 18 trials，12 success，summary SHA256 `8680f9fd38d0cddf8624f8dfc0d4c759820f6bd128652165ff3b607340b35731`。

结论是：raw expert action 与 train01→env action 在可匹配 init 上可成功，gripper/sign 转换不是主要断点；但 expert action 不是任意 init 通用，跨 init 复用会失败。因此下一步 tiny overfit 必须使用 replay 成功的同 init demo，并把 rollout 回同一 init-state 作为 gate。

### 2026-10-14 tiny-overfit head+proprio 结果与同 init rollout gate

基于 expert replay 成功记录构建了 3 条 task0 episode 的 tiny manifest，SHA256 `9a2bd44eb0c3a1d51db2bf4c89be0ad05fb9c1d9265e0258f4ece1e08e2f4d1b`。从 P6 B1 出发，只训练 action head + proprio projector，300 steps，lr `1e-4`，训练通过，远端目录为 `/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_action_calibration/tiny_overfit_headproprio_seed17_300step_20261014_retry2/`。

offline tiny eval 有明确改善：action MAE `0.298864 -> 0.284692`，first-horizon MAE `0.288046 -> 0.282167`，rotation MAE `0.250380 -> 0.233718`，gripper sign acc `0.712862 -> 0.853714`。但 same-init rollout smoke 仍为 `0/1`，目录为 `/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_action_calibration/tiny_overfit_same_init_rollout_20261014/`。

该结果说明只调 action head/proprio 能改善 normalized offline prediction，但不足以修复 closed-loop state drift。VLA 主线仍应继续 action-interface repair，优先测试解冻 policy LoRA + action head + proprio 的 tiny-overfit，或对比官方 moojink checkpoint 的 action traces；semantic/P7 不应被恢复为闭环成功主张。

## 2026-10-14：tiny-overfit policy LoRA + head+proprio 启动

- [x] head+proprio-only tiny-overfit 的 `final.json` 已补充远端 `final_corrected.json`，修正原 `final.json` 中保留 `.tmp` checkpoint 路径的问题；实际 checkpoint 为 `/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_action_calibration/tiny_overfit_headproprio_seed17_300step_20261014_retry2/checkpoint_00300`。
- [x] 已确认标准 P1 provenance 文件真实路径为 `/vepfs-mlp2/c20250405/400040/transfer/vla_attention/repo/VLA_attention/experiment_workspace/results/P6_vla_b0_b4/p1_interface/p1_interface.json`，不是先前误写的 `results/P6_vla_b0_b4/p1_interface/report.json`。该文件记录 OpenVLA revision `47a0ec7fc4ec123775a391911046cf33cf9ed83f`、action shape `[1,8,7]`、normalization round-trip、module restore、visual grid 和 gradient finite/nonzero。
- [x] 已启动下一轮 tiny-overfit：从 B1 checkpoint 出发，解冻 policy LoRA + action head + proprio projector，300 steps，lr `5e-5`，仍使用同一个 3-episode tiny manifest，远端 job 为 `/vepfs-mlp2/c20250405/400040/transfer/vla_attention/jobs/tiny_overfit_policy_lora_20261014_retry3/`，PID `623231`。
- [x] 该作业已经通过 GPU gate、source-lock、P1 provenance 和模型加载，开始写 `train_metrics.jsonl`；截至首次记录，训练已达到 57/300 steps，loss finite，GPU0 约 23GB。
- [ ] 作业完成后必须立即做 offline action prediction 对比：B1 原始 checkpoint、head+proprio tiny checkpoint、policy-LoRA tiny checkpoint；若 policy-LoRA tiny 在同集 action MAE/first-horizon/gripper/rotation 上明显优于 head-only，再做 same-init 1-episode rollout。
- [ ] 若 policy-LoRA tiny offline 提升明显但 same-init rollout 仍失败，下一步不继续 semantic/P7，而是提取官方 moojink 成功 checkpoint 与我们 checkpoint 的 action trace，比较 action chunk、unnorm_key、proprio、translation/rotation scale 和 gripper trajectory。

## 2026-10-14：policy-LoRA tiny 评估、center-crop 复核与官方 action trace 对比

- [x] policy LoRA + head + proprio tiny-overfit 已完成 300/300 steps，远端目录为 `/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_action_calibration/tiny_overfit_policy_lora_headproprio_seed17_300step_20261014/`，checkpoint 为 `checkpoint_00300`，`final.json` 状态为 `passed`。
- [x] 三方 tiny offline action prediction 对比已生成：`tiny_eval_three_way_comparison.json`，SHA256 `092179040be1e6b63c3f2ff5fe5fb15ce4a46c105ab09b35e98fbe172d50ba83`。结果：policy-LoRA tiny 相比 B1 只有轻微改善，但比 head+proprio-only tiny 更差。

| metric | B1 | head+proprio tiny | policy-LoRA tiny | policy - B1 | policy - head |
|---|---:|---:|---:|---:|---:|
| action MAE | 0.298864 | 0.284692 | 0.296090 | -0.002774 | +0.011398 |
| first-horizon MAE | 0.288046 | 0.282167 | 0.292713 | +0.004666 | +0.010546 |
| translation MAE | 0.386396 | 0.372006 | 0.384227 | -0.002169 | +0.012221 |
| rotation MAE | 0.250380 | 0.233718 | 0.243450 | -0.006931 | +0.009732 |
| gripper MAE | 0.181721 | 0.175672 | 0.189600 | +0.007879 | +0.013928 |
| gripper sign acc | 0.712862 | 0.853714 | 0.827899 | +0.115036 | -0.025815 |

结论：解冻 policy LoRA 没有超过 head+proprio-only tiny，因此不满足“offline 明显优于 head-only 后再跑 same-init rollout”的 gate。当前不把 policy-LoRA tiny 当作闭环候选。

- [x] 为 rollout/trace 脚本新增 `--center-crop` 诊断开关，默认仍为 false，保持历史 P6 结果可复现。B1 使用 `center_crop=True` 的 1-episode rollout 结果仍为 `0/1`，目录为 `/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_action_calibration/b1_center_crop_one_episode_20261014/`。这说明 official positive control 与我们 wrapper 的 center-crop 差异不是单独充分原因。
- [x] 生成官方 `moojink/openvla-7b-oft-finetuned-libero-spatial` task0/init0 action trace，目录为 `/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_action_calibration/official_moojink_task0_init0_action_trace_20261014/`。该 trace 前 30 control steps 未必覆盖完整成功过程，只用于动作分布诊断，不是 benchmark。官方 loader 再次备份并改写了 moojink checkpoint 的 `config.json`，这是 OpenVLA-OFT 官方 loader 的既有行为。
- [x] 官方 trace 与 B1 center-crop rollout 前 30 步对比已生成：`/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_action_calibration/official_vs_b1_center_crop_action_compare_20261014/summary.json`。关键数值如下：

| metric | official moojink | our B1 center-crop | interpretation |
|---|---:|---:|---|
| processed translation L2 mean | 0.927672 | 0.561660 | B1 平移幅度明显偏弱 |
| processed rotation L2 mean | 0.041404 | 0.000821 | B1 rotation 几乎塌缩 |
| processed gripper mean | -1.000000 | -1.000000 | gripper sign 一致，不是主因 |
| action diff L2 mean | — | 0.670821 | 动作分布与成功 checkpoint 差距很大 |
| translation diff L2 mean | — | 0.666692 | 主要差距来自平移轨迹 |
| rotation diff L2 mean | — | 0.041017 | rotation under-production 仍成立 |

当前判断：P6 闭环失败的主断点更像是动作分布/动作幅度与 rotation 学习不足，而不是 evaluator、LIBERO 环境、init state、gripper sign 或 center-crop。下一步应优先做 action distribution repair：用官方成功 checkpoint 的 action trace/statistics 或 expert trajectory 约束 action magnitude、rotation 和 early-horizon，而不是继续扩大 semantic/P7。semantic/P7 仍不得声称闭环 VLA 有效。

### 2026-10-14 VLA action-interface gate update

当前 VLA 主线仍处在 action-interface repair 阶段。官方 moojink checkpoint 已在 LIBERO spatial task0 取得 5/5 positive control，expert replay 也能在匹配 init 下成功；因此基础 evaluator/LIBERO/action processing 可用。我们的 B1/P6 系列失败主要指向 action distribution，尤其 rotation under-production。

最新 distribution-repair smoke 从 B1 出发，只训练 action head + proprio projector，并加入 early-horizon、translation magnitude 与 rotation magnitude 约束。结果：held-out action/rotation 指标相对 B1 有弱改善，但 tiny gate 没有超过 head+proprio-only tiny-overfit。因此当前结论仍是 No-Go for rollout：在 action-interface gate 通过前，不扩大 semantic/P7，不启动正式 LIBERO/LIBERO-Plus/RoboTwin benchmark，不把 P7-lite 解释为闭环 VLA 成功。
