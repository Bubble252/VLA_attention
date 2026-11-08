# ReGuide 阅读记录

**论文**：Correcting WHERE, Preserving HOW: Compositional Generalization for Vision-Language-Action Models via Referential Guidance<br>
**arXiv**：`2609.38616`<br>
**本地文件**：`papers/reguide_2609.38616.pdf`、`papers/reguide_2609.38616.txt`<br>
**代码链接**：截至 2026-11-08 未在正文中找到公开仓库 URL；未 clone 代码。

## 1. 方法

ReGuide 是 training-free wrapper，核心是把“在哪里交互”与“怎么交互”分开：

1. **Semantic rebinding**：用 grounding module 得到指令相关的 object/destination pose，并监控冻结策略的 action chunk 是否在向错误 referent 承诺。
2. **Geometric rebinding**：把 end-effector 沿 bounded flow 送到 demonstration-supported pre-contact / pre-place configuration。
3. **Hand-back**：进入 pre-contact set 后丢弃未执行 chunk，用当前真实观测重新查询冻结策略。
4. **Execution assistance**：hand-back 后仍用 lateral tolerance、height floor、completion certificate 和有限次数 re-arm 约束执行。

## 2. 输入与输出

- 输入：当前观测、instruction、grounding module 的 object/destination pose、训练 demonstrations 的统计量；
- 输出：对冻结 policy action 的 bounded override，或在 hand-back 后放行 policy action；
- 不训练 policy；不改变 backbone。

## 3. 对本项目的可用部分

- `R-GEO`：oracle destination + 3D pre-place + hand-back + release certificate；
- `R-NAIVE`：oracle destination，但只做 xy transport，不加 pre-place / semantic monitor；
- `R-PRED`：用现有 C1 point head 代替 oracle grounding；
- `R-SHUF`：把 destination 替换成 `plate_1` 做语义负控。

## 4. 复现风险

- 原论文依赖 demonstration statistics（Q50/Q95、pre-contact set、height profile）；本项目第一版只有单任务、固定桌面和少量 calibration，采用简化常量；
- 原论文的 grounding 是 GroundingDINO + depth；本项目使用已有 point head 的 2D 点 + 固定相机标定，属于“论文近似版”；
- 原论文在多个 backbone、16 个 composition cells 上评估；本项目只有 15 个 C1 init，不能做论文级显著性；
- 原论文代码未在正文中找到，不能声称“原实现复现”。

## 5. 使用边界

本实验只使用 ReGuide 的**方法结构**，不复制论文数值。任何结果都必须写成 `ReGuide-style pilot`，不能写成“复现 ReGuide”。
