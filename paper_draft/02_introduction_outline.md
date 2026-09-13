# 2. Introduction 结构稿

论文按六段组织，先写逻辑再写成稿。

## Paragraph 1：背景与运行例子

**目的**：从机器人“把碗放入盒子”说明语言、视觉和动作证据必须衔接。

- VLA 将图像、语言和 proprioception 映射为 delta EEF；成功动作不仅需要识别碗，还要在抓取、搬运和放置阶段选择不同证据。
- 常见 attention 热力图把高亮区域当作 grounding 证据，但“图不亮”可能来自 attention sink、窗口重排或模型依赖 state。
- 我们运行例子固定为 `pick bowl → place in container`，贯穿 VLM grounding、LIBERO 和 DROID。

## Paragraph 2：已有方法的三个限制

**限制一：attention 不可比。** Lavender 的扩散词图与 SpikingBrain full-attention、Qwen/LLaVA hidden states、OpenVLA action token 不在同一表示空间。

**限制二：语言与动作脱节。** 现有 VLM grounding 通常评估词—区域，VLA 通常评估 success rate 或 action error，缺少中间的动作归因约束。

**限制三：可视化证据不足。** attention 权重可能被 sink 污染，gradient 可能是代理，LIBERO 可能诱导 state shortcut；单张热力图不能支撑因果解释。

## Paragraph 3：问题本质和目标

本文研究：能否把不同模型的内部证据统一成图像坐标中的空间归因，并让动作归因位于语言目标支持区域？目标句：

> We seek a structure-native, model-agnostic spatial attribution interface that transfers diffusion-based language grounding into action-conditioned VLA decisions without distilling an external policy.

硬约束：训练时可使用 Lavender 教师，推理时移除教师；动作统一为 7D delta EEF；DROID 只做离线 action prediction 首轮；所有解释结论必须有 intervention 或反事实支持。

## Paragraph 4：关键挑战

1. **异构归因桥**：不同层的 token 顺序、分辨率和 attention 语义不同，直接 MSE 会对错位置。
2. **动作归因约束**：动作 head 没有统一 cross-attention，且外部动作 teacher 容易退化为策略蒸馏。
3. **证据可信度**：必须区分结构线索、可微代理和输入干预，并排除 sink 与 state shortcut。

## Paragraph 5：方案概览

- **Spatial Attribution Bridge**：恢复 patch 网格，统一 `T_sem`、`A_lang` 和 `A_act` 到原图坐标。
- **D0/D1 Structure-Native Alignment**：`L_sem` 锚定语言归因；`L_contain` 约束动作归因不偏离语言并集；D1 让 source→mixed→target 随阶段迁移。
- **Evidence Calibration**：raw/sink-free、gradient、patch occlusion、语言/state 反事实和 DROID 跨域测试。

运行例子中，碗和盒子先由 `T_sem` 定位；抓取阶段 `A_act` 应集中在碗/夹爪，放置阶段逐步转向盒子入口。

## Paragraph 6：贡献

1. 提出跨 VLM/VLA 的模型无关空间归因桥与结构感知层选择（Section 3）。
2. 提出 SpikingBrain-VLA 的语言—动作归因一致性 D0 及 phase 扩展 D1（Section 4）。
3. 提出 attention/gradient/intervention 证据分级协议，在 VLM、LIBERO 和 DROID 上验证 grounding、动作误差与泛化（Section 5）。

## Introduction 写作禁区

- 不把 attention map 等同于因果解释；
- 不声称 Lavender 本身理解动作动力学；
- 不把 WAM/VAM 写成首轮主方法；
- 不用 LIBERO 单独支撑真实机器人泛化；
- 不在没有结果前写“显著提升”，只写待验证假设。
