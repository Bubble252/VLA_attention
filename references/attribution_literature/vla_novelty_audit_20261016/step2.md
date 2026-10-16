# Step 2 — 检索与去重

2026-10-16T12:05:27.430883+00:00

查询：
- vision language action state aliasing
- robot policy representation metric learning
- language conditioned action similarity contrastive representation

年份 2020–2026，各源每查询上限 3。结果：arxiv=6, dblp=0, open_alex=9, openreview=0, semantic_scholar=0, crossref=9。24 原始记录，24 标题去重记录；补充 6 篇（3 篇 model-recall、2 篇上一轮近邻/用户文献、1 篇引用追溯），合计 30。

arXiv 第一查询 SSLError/SSLEOFError；DBLP 三查询 JSON 解析失败；OpenReview: openreview not installed；Semantic Scholar 三查询 HTTP 429。其余 API 有返回。失败和低召回不构成不存在近邻的证据。

TCN 2017 是历史机制近邻，显式超出 API 年份保留。原始摘要及来源保存在 search_raw.json / audit_records.json。