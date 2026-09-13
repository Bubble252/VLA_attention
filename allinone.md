# 热力图与空间归因文献检索记录

## 检索问题

1. 哪些方法把 attention 或 hidden-state 梯度转换为视觉空间归因图？
2. 哪些工作证明 attention 图不等于决策因果证据？
3. VLA 中如何校准动作相关区域，并处理 sink/register？

## 检索结果与限制

本轮使用 paper-search、OpenAlex 和 Semantic Scholar 尝试检索 2023–2026 文献。Semantic Scholar 返回 429，部分 API 受代理/DNS 限制；因此没有把未确认的最新 VLA 论文写成已验证事实。高置信经典工作、已有本地归档和下载目标见 `references/attribution_literature/README.md`。

## 综合判断

文献形成四条互补路线：attention flow/rollout 提供结构线索；gradient/relevance propagation 提供输出条件归因；occlusion/perturbation 提供干预证据；sink/register 工作提醒注意力质量可能被固定 token 或位置污染。对我们的方案，最稳妥的实验协议是让 `T_sem` 只监督空间归因，不假设所有模型存在同构 cross-attention，并以 patch occlusion 作为 VLA 动作归因的校准。
