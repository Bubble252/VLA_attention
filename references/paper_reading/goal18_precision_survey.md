# Goal18 Precision Survey

**日期**：2026-11-08（失败后更新 2026-11-09）<br>
**用途**：记录 Goal18 相关检索、落盘状态、code 检查和失败后的 pivot 依据。不是已复现结论。

## 1. 检索

使用 `paper-search`，查询包括：

- `vision language action residual correction object grounding precision`
- `residual policy learning robot manipulation action correction`
- `visual grounding precise placement vision language action`
- `test-time residual correction robot policy`
- `target-agnostic guidance object-centric VLA`
- `open-vocabulary object grounding vision-language-action precise placement`

来源：arXiv、OpenReview；Semantic Scholar 返回 429，未把它当作完整覆盖。OpenReview forum/API 返回 challenge，TAG/R-VLA 的 OpenReview 页面未能直接下载。

## 2. 已有落盘

| 论文 | 本地文件 | reading note | 官方 code | 状态 |
|---|---|---|---|---|
| ReGuide `2609.38616` | `https://arxiv.org/abs/2609.38616`；`papers/reguide_2609.38616.pdf/.txt` | `references/paper_reading/reguide_2609.38616.md` | 正文未找到 | 已精读，Goal16/17 使用方法结构 |
| AnySlot `2604.10432` | `https://arxiv.org/abs/2604.10432`；`papers/anyslot_2604.10432.pdf/.txt` | `references/paper_reading/anyslot_2604.10432.md` | 未找到，`official_code_missing=true` | 已读摘要/方法；未进入代码 |
| TAG `2603.24584` | `https://arxiv.org/abs/2603.24584`；`papers/tag_2603.24584.pdf/.txt` | `references/paper_reading/tag_2603.24584.md` | 未找到，`official_code_missing=true` | 已读摘要/方法；未进入代码 |

## 3. 未落盘候选

| 论文 | 链接 | 下载 | 官方 code | 备注 |
|---|---|---|---|---|
| R-VLA: Residual Off-Policy Adaptation of a Frozen Vision-Language-Action Policy | `https://openreview.net/forum?id=jAz4zSLbmr` | 否：OpenReview challenge | 未查 | residual adaptation；非当前最小实验 |
| Test-Time Training as Residual Memory for Robot Policies | `https://arxiv.org/abs/2610.04701` | 否 | 未查 | 成本高；未进入代码 |
| When, Where, and What to Correct: Human-Guided Adaptive Residual Correction | `https://arxiv.org/abs/2610.37547` | 否 | 未查 | 决策问题；未进入代码 |
| Residual Off-Policy RL for Finetuning Behavior Cloning Policies | `https://arxiv.org/abs/2509.19301` | 否 | 未查 | 需要 RL/交互；不适合当前 15-init |
| TAG OpenReview 版本 | `https://openreview.net/forum?id=JniUQWZLiy` | 否：challenge；改用 arXiv | 未找到 | 使用 arXiv `2603.24584` |

## 4. 失败后的判断

Goal18 的 G-P 失败后，最便宜的证据不是再换一个 residual，而是做标签/评测对齐诊断：

- 训练标签：`visible_surface_centroid`
- 成功 oracle：receptacle object center
- visible mask centroid 到 oracle：mean `22.27 mm`，max `31.83 mm`
- projected object center 到 oracle：mean `10.64 mm`，max `13.26 mm`
- predicted point 到 projected object center：mean `(-11.26, -3.85) px`

因此当前优先级是 target-aligned point/region 监督；TAG/AnySlot 是标签修复后的 fallback。

## 5. 使用边界

- 本文件只记录检索和阅读状态，不声称复现上述论文；
- Goal18 主方法 `no_external_method_used=true`；
- 任何外部方法进入代码前，必须完成下载、code 检查和 reading note；
- 没有官方 code 的方法只能作为方法结构参考，不能写成“复现”。
