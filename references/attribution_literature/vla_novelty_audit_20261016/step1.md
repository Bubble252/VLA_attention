# Step 1 — 候选冻结与创新性拆解

时间：2026-10-16 Asia/Shanghai。技能：scoop-check，调用 paper-search 检索。研究审计，不修改训练配置、不启动 GPU 作业。

## 七步进度

- [x] 1. 拆解 novelty，锁定被审计候选
- [x] 2. 三路检索、去重、补充历史近邻（多源失败如实保留）
- [x] 3. 对检索记录逐项进行摘要筛选（24 检索记录 + 6 补充）
- [x] 4. 选择 7 篇最有威胁的候选
- [x] 5. 读取候选原文的方法、假设和实验范围（六篇新 PDF + Copper 本地 PDF）
- [x] 6. 四轴比较及重合等级（Level 2 — High Overlap）
- [x] 7. 给出最小差异、否定条件和总判定（原候选不宜冻结为主创新）

## 被审计的原始候选

来源：`../../../ideaspark_run/vla-language-dynamic-evidence/provisional_recommendation.json`。

- **Problem framing**：VLA 可能把同任务、相似观测但需要不同动作的状态混淆；静态语言名词图不足以描述接触与操作进展。目标是改善真实目标选择和闭环控制，不能用 latent 分离本身代表成功。
- **Core mechanism**：在原生动作通路实际使用且可训练的视觉 tokens 上，以冻结的 source/target 语义软图分别汇聚保留位置的特征；按照同任务/相机/可比 proprioception 下的示范 action chunk 距离构造正负对，以 triplet/ranking 辅助损失塑造表征；原生全图、proprioception 与动作头保持可访问，训练期辅助 readout 可移除。
- **Key insight**：语义定位相同的状态仍可能需要不同控制；用动作相似性组织任务相关视觉表征，而非强迫所有动作归因位于名词区域或令所有临近帧表征相似。
- **Application domain**：OpenVLA/OFT、LIBERO 操作任务；后续跨骨干/DROID 须独立验证。

## 不能偷换的候选边界

这是动作距离监督的关系/度量学习候选，不是已实现的 world model、未来预测方法或因果保证。不得在发现近邻后只改名称，然后把新候选当作原候选通过审计。任何改进建议单列为新提案。

## 已知证据与风险

VLA 当前 seed17 B1 MAE 0.293686、B3 0.326322、B4 0.384150，不能归因为语言必然有害。历史 VLM V4 裁剪/retention 路径缺陷也不能作为天然梯度冲突证据。相同动作不保证相同状态价值；不同动作可能由示范者多模态性、历史/接触或 proprioception 产生。需要审计是否已有 policy-similarity、bisimulation、action-aware contrastive learning 和 task-conditioned representation 方法。

提交规则：本次仅提交审计脚本、轻量结构化结果及 Markdown；PDF、原始检索长文本不默认入 Git。完成后按独立审计路径显式 `git add`、`git diff --cached --check`、`git commit`；不顺带提交工作区其他人的改动。云端同步使用既有文献目录规则，并以 API 返回和内容核验判断成功。
