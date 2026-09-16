# 论文预稿总纲：Structure-Native Language-Action Attribution for Spiking VLA

**状态**：论文规划稿，不代表实验结果已经完成。  
**论文类型**：Technique paper，带有跨模型评估设置。  
**主验证**：VLM grounding → LIBERO 机制验证 → DROID 泛化验证。  
**动作接口**：7D delta EEF。  
**主线骨干**：优先选择接口成熟的 OpenVLA/OFT 或 π0 系 VLA，并用 Qwen/LLaVA 做 VLM 归因普适性验证；SpikingBrain 只作为后续类脑扩展。

## 一句话主张

我们提出一种不依赖同构 cross-attention 的语言条件空间归因对齐框架：用 Lavender 的词级扩散图锚定 VLM 的语言归因，再约束 VLA 动作归因位于语言相关区域，并在脉冲模型的 full-attention / window / GLA 分层结构上进行选择性监督，从而改善 grounding、动作预测和跨场景泛化。

## Thinking template

| 阶段 | 本论文内容 |
|---|---|
| Research background | VLA 需要将语言目标落到视觉区域并转为连续机器人动作；现有热力图常被当作解释，但不同模型的 attention 不同且可能受 sink 污染。 |
| Limitation 1 | Lavender 主要监督同构 attention；Qwen/LLaVA/OpenVLA 等模型没有可直接比较的 cross-attention。 |
| Limitation 2 | VLM 的语言 grounding 与 VLA 的动作依赖通常分开评估，缺少 `A_lang → A_act` 的统一约束。 |
| Limitation 3 | attention 图与决策因果性可能不一致；仅凭热力图无法排除 state shortcut、sink 或后处理伪影。 |
| Key idea | 将 attention、gradient×activation 和 patch intervention 统一成图像坐标中的空间归因，并以结构感知方式施加语言到动作的一致性约束。 |
| Challenge 1 | 异构模型如何得到可比且坐标正确的空间归因？ |
| Challenge 2 | 如何让动作归因受语言目标约束，同时不把外部模型动作策略硬蒸馏进学生？ |
| Challenge 3 | 如何证明热力图对应真实决策，而非 attention sink 或 LIBERO 捷径？ |
| Module A | Spatial Attribution Bridge：token 索引恢复、层级适配、归一化和 `T_sem → A_lang`。 |
| Module B | Structure-Native D0/D1：`L_sem`、`L_contain` 以及后期 phase-conditioned attribution。 |
| Module C | Evidence Calibration：sink-free 双轨、patch occlusion、语言/state 反事实和 DROID 跨域评估。 |
| Contribution 1 | 提出跨 VLM/VLA 的语言条件空间归因接口和结构感知监督协议（Section 3）。 |
| Contribution 2 | 提出模型无关的 D0 containment 与 D1 phase transition，使语言证据约束动作证据，并在至少两个 VLA 接口上验证（Section 4）。 |
| Contribution 3 | 建立 attention/gradient/intervention 的证据分级和 DROID 泛化评估，验证方法不是特定骨干或仿真捷径（Section 5）。 |

## 逻辑一致性检查

- Limitations → Key idea：三项限制都由空间归因桥、结构内生动作约束和干预校准覆盖；**通过**。
- Key idea → Challenges：异构接口、动作约束、因果验证均是实现该 idea 的直接难点；**通过**。
- Challenges → Methodology：A/B/C 与三个挑战一一对应；**通过**。
- Methodology → Contributions：每个模块均有方法或实验贡献；**通过**。

## 论文主线与非主线

```text
主线：T_sem → A_lang → A_act → DROID 泛化
支撑：sink-free、occlusion、语言/state 反事实
扩展：D1 phase、B 路线 R_future、WAM/VAM、Next Forcing
```

B/WAM/VAM 只有在 `R_future` 能预测 DROID 未来成功并带来跨场景收益时，才在论文中升级为主结果；否则放在 ablation/future work。

SpikingBrain 不再承担主线风险：它保留为类脑扩展、异构层级消融和效率分析。若 OpenVLA/OFT 或 π0 系主线先完成，SpikingBrain 的加入只用于检验结构迁移，不影响论文主体交付。

## 论文级停止条件

如果 VLM grounding 中正确教师图不优于错图/随机图，或者 DROID 上 `A_act` 与 occlusion 图完全不一致，则停止扩展 WAM/VAM，先修正归因接口与坐标桥。
