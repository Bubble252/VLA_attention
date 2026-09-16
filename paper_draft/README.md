# 论文预稿目录

本目录按论文写作顺序组织，不替代 `doc/` 中的工程执行细节：

1. `00_paper_blueprint.md`：论文类型、核心 idea、挑战—方法—贡献映射；
2. `01_idea_and_positioning.md`：类脑结合、创新边界和主线定位；
3. `02_introduction_outline.md`：六段式 Introduction 逻辑；
4. `03_related_work.md`：grounding、归因、VLA、phase、WAM/VAM 相关工作；
5. `04_system_and_method.md`：系统架构、D0/D1、B 路线和核心公式；
6. `05_experiments_and_expected_conclusions.md`：实验矩阵、指标、预期结果和结论门槛；
7. `06_conclusion_and_future_work.md`：结论模板、局限和后续工作；
8. `07_open_questions.md`：开始实验前仍需冻结的参数。

所有实验结果目前都标为“待测”或“预期”，不能当作已验证结论。

## 最新阅读入口（2026-09-16）

- [52 份 PDF 摘要归纳](../references/abstract_review/README.md)：逐篇来源、借鉴、重合与异常记录。
- [详细模型/基线/benchmark 筛选](../references/abstract_review/04_selection_and_experiment_decisions.md)：多种组合供研究者确认。

SpikingBrain 已明确后置，主线 VLA 组合仍待确认。此前预稿中“model-agnostic”“首次语言—动作连接”等只是研究目标；Anchor-Align、PosA-VLA、BridgeVLA、MotionEnhancer、Robust-WAM 等相邻工作需做进一步方法级对比后才可写创新结论。

## 公式预览

论文预稿中的核心公式统一使用 Markdown display math：`$$ ... $$`。在支持 MathJax/KaTeX 的 Markdown 预览器中会显示为居中公式；飞书转换器若不渲染 LaTeX，会保留公式源码，此时仍可依据上下文阅读。行内符号使用单个 `$...$` 或反引号，避免把长公式塞进行文。

## 飞书同步

此目录已配置到飞书文件夹 Token `RGgnwMcVUi4xd5klGX9cAThinBc`。在能访问飞书 API 的机器上执行：

```bash
cd /home/bubble/类脑计算/doc-sync-main
/usr/bin/python3.10 main.py sync --force
```

实时同步：

```bash
/usr/bin/python3.10 main.py live --config sync_config.json --poll-interval 3
```
