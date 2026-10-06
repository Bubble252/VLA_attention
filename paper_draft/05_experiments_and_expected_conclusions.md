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
