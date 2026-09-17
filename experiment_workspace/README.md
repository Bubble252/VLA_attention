# 独立实验工作区

这里是本项目的**实验入口和轻量元数据目录**，与研究文档、参考仓库和模型权重分开。代码实现仍在 `src/vla_attention/`；本目录只保存一次实验所需的 manifest、冻结配置、协议链接和可回传的小型结果摘要。

真实运行目录固定为服务器 VEPFS：

```text
/vepfs-mlp2/c20250405/400040/transfer/vla_attention/
```

它应与本地目录保持相同的子目录结构；大图、teacher map、日志、checkpoint、数据、缓存只放 VEPFS，不进入 Git。VEPFS 路径和权限确认前，不创建或下载大文件。

## 目录职责

| 目录 | Git | 内容 |
|---|---|---|
| `manifests/` | 是 | 样本 ID、split、image SHA256、动作 schema；不含图像 |
| `configs/` | 是 | 每次运行冻结的 JSON/YAML 副本，记录 Git commit 与模型 revision |
| `results/` | 仅小型 JSON/CSV/Markdown | 指标、P1 report、失败记录；tensor 和图放 VEPFS |
| `logs/`、`checkpoints/`、`teacher_maps/` | 否 | 仅 VEPFS 的训练产物 |

## 数据使用顺序

1. **P1（20 个固定样本）**：只检查 token 坐标、梯度、target 和重复运行是否稳定；不优化参数，也不报告性能。
2. **Teacher calibration**：使用完整的独立 Flickr30k Entities calibration split，选择并冻结 best-single `T_sem`。
3. **VLM V0–V4**：使用完整的 Flickr30k Entities 训练集；在完整 Flickr30k test、RefCOCOg 和规定的图像扰动 split 上评测。
4. **VLA B0–B4**：LIBERO 的完整选定任务/episode；DROID 使用冻结的 episode-level offline split。DROID 不用于闭环成功率结论。

因此，20 样本不会削弱结果规模；它只是在完整训练前尽早发现“归因图位置错了但 loss 还能下降”的致命问题。

## 每次实验的提交顺序

```bash
# 1) 先提交协议与冻结配置
git add experiment_workspace/manifests experiment_workspace/configs experiments/<id>/protocol.md
git commit -m "research(<id>): freeze protocol and manifests"
git push origin main

# 2) 再在 VEPFS 运行；只回传摘要
git add experiment_workspace/results/<id>.json experiments/<id>/analysis.md
git commit -m "research(<id>): record audited results"
git push origin main
```

当前 GitHub 推送若继续报账户余额/403，保留本地 commit 并记录失败；不要用复制私钥或绕过审批的方式推送。
