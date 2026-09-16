# 52 份本地 PDF：摘要归纳与研究选项

日期：2026-09-16。范围是 references/papers 中的全部 52 个 PDF 文件；本报告不是全文翻译，也不是所有论文均已全文精读。依用户要求重点阅读摘要，结合少量正文说明模型、baseline、benchmark 的借鉴与重合。效果数字属于作者报告，未做代码复现。

新增专题：[三篇最近邻的原文方法分析](07_three_nearest_methods_deep_read.md)。Don't Blind、Anchor-Align、PosA-VLA 已进一步读方法、实施与相关消融；其中对 projector、方向词监督和 pose 双图的结论比本轮摘要索引更具体。涉及我们的公式调整仍为待验证建议。

## 阅读顺序

1. **先看本页**：总体结论与筛选优先级。
2. **04_selection_and_experiment_decisions.md**：可选模型组合、基线分组、benchmark 分工和待确认项。
3. **01_vla_alignment_and_shortcuts.md**：18 个 VLA/空间/捷径/benchmark 条目。
4. **02_world_models_and_dynamic_teachers.md**：16 个 WAM/VAM/动态教师条目。
5. **03_vlm_teachers_and_background.md**：18 个 VLM/教师/架构/旁支条目。
6. **05_inventory_and_reading_status.md**：逐文件阅读状态、页数、定位与重复说明。

前三份人工归纳合计覆盖 52 个文件；本文分类仅为组织方便，不代表所有条目同等相关。P43 的摘要在补读副本第 1–2 页，P52 以报告概览替代不存在的摘要。

本地 `inventory.json` 保存原文件名、SHA256、页数与自动提取证据；`source_evidence.md` 是页面提取文本（双栏顺序可能交错），不是摘要译文，也不上传飞书以免与人工归纳混淆。完整提取缓存位于 outputs/pdf_text。自动找到 Abstract 不代表边界正确；人工修正见阅读状态表。OpenVLA 使用单独补读副本，保留损坏原件与来源记录。

## 核心结论

当前方向值得继续，但相邻工作比原调研呈现得更密集。**“外部教师 + 空间对齐 + VLA”本身不足以构成新颖性**。更具体的候选问题是：同一图像坐标中，语言输出与动作输出所依赖的证据是否一致；这种输出条件归因约束，是否超过普通 feature anchoring、pose attention、历史记忆和数据增强。

| 关键论文 | 对原方案的影响 | 下一步需要的证据 |
|---|---|---|
| Lavender（P34） | 扩散词图与 attention alignment 已有 | 与原式对齐的同数据/同预算比较 |
| BridgeVLA（P05） | 先空间 heatmap 再 VLA 已有 | 显式空间预测头 vs 输出归因正则 |
| PosA-VLA（P18） | pose 锚定动作相关注意力已有 | pose-only、language-only、组合；不把 EEF 高斯当新贡献 |
| Anchor-Align（P10） | 语言—动作对齐与保持 VLM 表征已有 | 同观测下的标签对齐、feature anchoring 与归因对齐比较 |
| Don't Blind Your VLA / ReVLA（P06/P19） | 微调遗忘已有系统分析 | grounding 保留集、冻结/恢复视觉 baseline |
| AVA-VLA（P02） | 历史驱动动态视觉关注已有 | 相同历史长度下的 memory-only 与 phase/归因监督 |
| MotionEnhancer（P16） | 视频扩散运动 attention 蒸馏已有 | 动态教师与当前动作证据的额外连接价值 |
| Robust-WAM / MV-WAM（P20/P14） | future semantics、action/value 一致性已有 | 未来特征对齐 vs 未来目标对当前图像的归因 |
| LIT / Shortcut Learning（P04/P21） | shortcut 缓解与 OOD 已有方法/协议 | 通道约束、等预算增强和双向反事实 |

这不是“已被完全覆盖”的结论，也不是“没有重复”的保证。摘要只能定位高风险相邻工作；正式创新声明需读其方法、代码与消融。本轮不再延续此前未经充分检索的“很少有人连接语言与动作”的表述。

## 三条可筛选路径

- **A：保留 D0 主线（推荐先试）**。在一个成熟 VLA 上验证语义证据约束动作证据，配齐 anchoring、增强、pose 和错教师对照；结果成立后再加入第二结构。
- **B：强化时间条件（备选）**。同一物体在不同阶段有不同证据需求；与 AVA-VLA 的记忆作用、MotionEnhancer 的运动注意力分开。难点是弱 phase 标签和错误阶段不一定严格无效。
- **C：升级未来归因（后期探索）**。冻结 WAM/VAM，以明确未来任务分数回溯当前区域，检验超出直接 future-feature alignment 的价值。风险是预测误差与当前语义约束共同放大偏差。

三条不要求全部实现。主线骨干仍待用户确认；SpikingBrain 已确定后置，不再决定标题、核心创新或首轮停止条件。

## 数据/评估建议

建议把 **LIBERO ID + 可控 OOD 闭环 + DROID 离线** 放在不同表。LIBERO-Plus 是直接借鉴 LIT 的 OOD 候选；LIBERO-PRO 更偏任务/目标组合变化，CALVIN 更偏长程，SIMPLER、COLOSSEUM 等为替代方向，不全部强制加入。

DROID 必须用，但尚不能假设完整 object/operator/失败标签都存在。先读元数据，再确定场景或采集分组；保证 episode 不跨 split，归一化仅在训练集拟合。DROID 离线 MAE/RMSE 不等于控制成功率；真实未来仅限离线标签，不能泄漏到策略推理。

## 来源核对中的异常

- P17 `OREO.pdf` 实际为 OREOS LiDAR 定位，不能替代 object-aware robot regularization 文献。
- P13 Latent-WAM 为自动驾驶；P12 LaWAM 为机器人 latent subgoal，不是同一方法。
- P39/P41 SHA256 完全一致，52 文件中存在一对字节重复。
- P23/P24/P50 是 SpikingBrain 同研究家族的中文/英文/不同文件，不增加独立研究覆盖数。
- P43 原 OpenVLA PDF 损坏，已从 arXiv 获取可读 v3 副本补读摘要。
- P52 是 34 页个人研究幻灯片，无论文摘要；已读概览和相关机制部分，不能称全文论文精读。
- P26 WordCon 的概要在第 2 页，且 DOI 为占位符，不照抄为正式发表证据。

## 本轮完成标准

- [x] 52 文件均有编号、来源核对和单独条目，重复不遗漏。
- [x] 每条说明摘要/概要内容、借鉴、重合与适用边界。
- [x] 来源异常明确披露，未用记忆冒充摘要读取。
- [x] 模型和 benchmark 组合保留给用户筛选，SpikingBrain 后置。
- [ ] Git 提交与飞书内容回读校验：结果记录于 outputs/pdf_review_sync/，最终交付时核对。

该最后一项为操作记录，不通过编辑勾选本身证明上传成功。
