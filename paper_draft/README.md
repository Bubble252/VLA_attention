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
