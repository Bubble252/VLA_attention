# 交付和飞书同步状态

2026-09-16：本地摘要归纳已形成，飞书同步尚未完成，不能以历史文件夹同步成功推断本批归纳已上传。

## 已完成

- [x] 原目录 52 个 PDF 的文件名、SHA256、页数/解析错误登记。
- [x] 三份人工归纳覆盖 P01–P52，逐项标明摘要/概要来源、借鉴与重合。
- [x] 处理 WordCon 第 2 页概要、无 Abstract 标题的 WAM 首页、SpikingBrain 同家族和 P39/P41 重复。
- [x] OpenVLA 原件损坏，补读 arXiv v3 副本，原件保留。
- [x] 总览、模型/基线/benchmark 选项与阅读状态表。
- [x] 清理主线中的 SpikingBrain 必选项，恢复主模型组合待确认状态。

## 当前交付状态

- [x] `08_blindvla_code_audit.md` 已上传并回读验证：`VQuTdFkvYoftBRxwJxlccSAmnMc`；
- [x] `07_three_nearest_methods_deep_read.md` 已上传并回读验证：`YF26dTe3eoqKSsxVKyrcBAuDnMd`；
- [x] `README.md` 已上传并回读验证：`ACUYdAKe8o9kdZxVZC8c4YfOn7f`；
- [x] `doc/03_execution_plan.md` 已上传并回读验证：`QvkMdlWOIo07hpxEFhBcFeZ6nQe`；
- [x] `paper_draft/05_experiments_and_expected_conclusions.md` 已上传并回读验证：`CJBYdsNuWoUrJUxufI0cMUmpn7f`。

## 待完成

- [ ] 归纳目录其余 3 份人工归纳上传至用户指定文献文件夹 DAB9w31AUiuZJAkN5B7cukoGn2g：`01_vla_alignment_and_shortcuts.md`、`02_world_models_and_dynamic_teachers.md`、`03_vlm_teachers_and_background.md`；
- [ ] `04_selection_and_experiment_decisions.md`、`05_inventory_and_reading_status.md` 上传至同一文件夹；
- [ ] 修正的核心计划与论文预稿同步到其原文件夹。
- [ ] 对每个云文档回读所有正文块并与本地转换结果核对。

完整批量同步在已经完成 5 个文档后遇到临时 DNS/Token 网络错误；通过代理重试时自动审批服务再次返回 502，命令被阻止。已成功的五份不受影响；不能把剩余文档判断为已上传。

## 恢复执行

已准备专用脚本：

```bash
cd /home/bubble/类脑计算/VLM终局
/usr/bin/python3.10 scripts/sync_pdf_review.py
```

脚本复用本地 DocSync 凭据，使用既有 app 身份，仅更新六份带“PDF摘要归纳-”前缀的新归纳及八份本轮改动的核心/论文文档；不递归同步整个 references，不上传原始 PDF 或原文摘录，不删除无关云文档。更新前保存原云端块，更新后核对嵌套/表格正文文本块；记录于 outputs/pdf_review_sync/verified.json。核验通过后才把六个 file→document 任务登记进同步配置，避免文件夹 token 被误当文档 token。

本文件是操作状态，未列入云端归纳正文，避免上传后产生自指的状态变更。
