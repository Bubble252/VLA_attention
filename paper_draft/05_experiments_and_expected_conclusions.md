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
```

所有组固定 backbone/checkpoint、训练 episodes、train/val/test split、LoRA、action head、action chunk、图像增强、优化步数和随机种子；教师额外前向、离线 map 生成和二阶梯度成本单独报告。B1 的代码实现参照 BlindVLA，若未完整修复其 projector/optimizer/恢复链路，只写 `DB-style inspired baseline`。

判定顺序：先以 P4 的 VLM grounding 决定是否把 best-single `T_sem` 接入 VLA；再以动作输出归因、目标/背景干预和 ID/OOD 动作指标判断 B4 是否超过 B3。若 B4 只令图更集中、却不改善干预一致性或 OOD，则不能作为主贡献。若 B1 已经覆盖 B3/B4 的收益，论文主张退回 representation retention；若 B2/B3 有收益而 B1 无收益，才说明词级空间教师值得保留。

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

### 首轮训练监督与 Lavender 的可比性

VLM 主表的 V1--V4 使用图像 caption SFT，而不把 Entities box 或 referring-expression 作为训练标签：固定 caption prompt，监督原始 Flickr30k caption。Lavender 的基础监督也是 image-to-caption SFT，并将 Stable Diffusion 的 per-caption-token attention 作为额外 MSE 信号。我们的 V3/V4 在相同 caption token 上以语言条件空间归因 `A_lang` 对齐 `T_sem`；V2/V4 的 retention 项也不改变任务、数据或生成目标。

因此，Entities 的人工 phrase-box 只在独立 calibration/test 中选择、评价和反证教师图。若将来训练 “Where is the red ball?” 一类 referring prompt，必须单列为额外数据形式消融，不能与 V0--V4 主表混合，否则性能变化无法归因于归因对齐方法。

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
