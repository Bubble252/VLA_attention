# F0v2：最小可证伪 idea validation

目的：用最少新增计算判断 `T_sem → A_lang` 是否值得进入 F1-10k，不把 smoke loss 当作方法有效性。

## 固定输入

- 训练：已完成的 F0v2-256 对齐 manifest、SD1.5 phrase cache、DINOv2 retention teacher；
- 测试：从官方 Flickr30k Entities test 固定 64--128 个 image-phrase pair，不能与 F0 train image 重叠；
- 组别：V0、V1、V2、V3、V4；所有组共享 processor、LoRA、seed、评价代码和 held-out manifest；
- 负控：V3/V4 的 correct、wrong-word、wrong-image、random normalized map。

## 必须完成的检查

- [ ] V0 原始 checkpoint attribution evaluation；
- [ ] V1--V4 checkpoint 保存、恢复和 held-out evaluation；
- [ ] pointing、mass-in-box、IoU、map validity；
- [ ] `V3 > V1`；
- [ ] `V4 > V2`；
- [ ] correct teacher > wrong-word / wrong-image / random；
- [ ] 每项结果写入 `metrics.json`，失败原因写入 `analysis.md`。

## 判定

四项趋势全部成立：进入 F1-10k。任一项失败：停止扩大数据，先检查 token span、cache grid、负控和失败样本。

该目录只存小型 manifest、JSON、Markdown 和图表；模型、map、checkpoint 放 VEPFS。

当前状态：V1/V2 checkpoint 已保存；V3/V4 checkpoint 导出代码已提交（`339bbb3`），评估器和批量入口已提交（`cf7a7d4`、`9426f19`）。恢复 101 SSH 后先上传两个 runner，补跑 V3/V4 checkpoint，再执行 64 条 held-out 的 V0--V4 评估。
