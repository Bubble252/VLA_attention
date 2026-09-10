# VLM 终局：基于层级注意力对齐的 SpikingBrain-VLA

本仓库记录最终研究方案、文献归档和后续实现入口。目标是在 LIBERO 仿真中验证：

> Lavender 的词级扩散空间先验，经过语言条件 patch 归因桥接后，是否能对 SpikingBrain 的 full-attention 层进行选择性监督，并改善 7D delta EEF 的 VLA 控制。

先读：

1. [`doc/01_project_background.md`](doc/01_project_background.md)
2. [`doc/02_technical_stack.md`](doc/02_technical_stack.md)
3. [`doc/03_execution_plan.md`](doc/03_execution_plan.md)
4. [`references/README.md`](references/README.md)

仓库不提交模型权重、LIBERO 数据集、运行日志和大规模输出。GPU 由服务器环境提供；本项目只固定软件、数据、模型版本和实验配置。

