# AnySlot 阅读记录

**论文**：AnySlot: Goal-Conditioned Vision-Language-Action Policy for Zero-Shot Slot-Level Placement<br>
**arXiv**：`2604.10432v3`<br>
**本地文件**：`papers/anyslot_2604.10432.pdf`、`papers/anyslot_2604.10432.txt`<br>
**下载日期**：2026-11-07<br>
**官方 code**：`official_code_missing=true`。PDF 和 arXiv 页面未找到仓库 URL。<br>
**使用状态**：未进入 Goal18 代码；仅作为 G-P 失败后的 fallback 记录。

## 1. 方法

AnySlot 把“语言 grounding”和“低层控制”分开：

1. 语言指令先转成一个显式的空间视觉目标（slot marker）；
2. marker 投影到 head/wrist 相机视图；
3. goal-conditioned VLA policy 以该视觉目标执行放置。

它的核心不是回归一个单点坐标，而是把目标区域变成可被策略消费的视觉 goal。

## 2. 对本项目的可用部分

- 与 Goal18 的 `destination point` 问题直接相关：单点坐标接口可能丢失 slot 边界和局部几何；
- 如果 target-aligned point 仍不能把 G-P 压到 1.5 cm，可把 destination 从“点”升级为“区域/slot marker”；
- 可作为 `region/contact-point` 接口的上界对照。

## 3. 复现风险

- 依赖图像生成/渲染模块，项目目前没有同等的 slot marker 生成器；
- 原论文主要在 SlotBench/真实 UR10e 上评估，和当前 LIBERO `white_right` 任务不同；
- 没有官方 code，不能声称复现；
- 不能把 slot marker 本身当新颖性，它已经是公开方法。

## 4. 使用边界

当前只把它作为 region/visual-goal 接口的文献依据，不把 AnySlot 的数值当成本项目结果。
