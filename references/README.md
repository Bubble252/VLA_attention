# 参考文献与模型归档

本目录保存最终 VLA 方案的论文、官方代码快照、版本记录和校验信息。PDF 与代码目录因体积较大被 `.gitignore` 排除；Git 跟踪它们的 SHA256、来源、提交哈希和下载历史。

## 当前归档结果

- 必需论文：`16/16` 可用；另外归档了 4 份 SpikingBrain/LingBot 补充资料。
- 官方代码仓库：`14/14` 可用，固定提交见 `repo_versions.md` 与 `repositories.yaml`。
- PDF 完整性：`checksums.sha256` 已通过校验。
- 范围：只归档阅读和实现所需的论文/代码；未下载模型权重、LIBERO 数据集和训练数据。

## 论文登记

| 本地文件 | 角色 | 状态 |
|---|---|---|
| `papers/lavender_2502.06814.pdf` | 语言条件注意力教师 | 已归档 |
| `papers/spikingbrain_2509.05276.pdf` | SpikingBrain 主干 | 已归档 |
| `papers/qwen25_vl_2502.13923.pdf`、`qwen2_vl_2409.12191.pdf`、`qwen3_vl_2511.21631.pdf` | Qwen-VL 结构参照 | 已归档 |
| `papers/deepseek_vl2_2412.10302.pdf` | 高分辨率/token 路由参照 | 已归档 |
| `papers/openvla_2406.09246.pdf`、`pi0_2410.24164.pdf` | VLA 基线与连续动作参照 | 已归档 |
| `papers/libero_2306.03310.pdf`、`diffusion_policy_2303.04137.pdf` | LIBERO 与动作建模 | 已归档 |
| `papers/gla_2312.06635.pdf`、`attention_sinks_2309.17453.pdf` | 长上下文与注意力稳定性 | 已归档 |
| `papers/qwen_robotmanip_2606.17846.pdf` | 机器人操作 VLA | 已归档 |
| `papers/lingbot_*.pdf` | 视频/世界模型/VLA 路线参照 | 已归档 |
| `papers/SpikingBrain_Report_Eng.pdf`、`SpikingBrain_Report_Chi.pdf` | 官方技术报告 | 已归档 |

论文的 arXiv 编号、来源 URL 和项目角色在 `metadata.yaml` 中维护。

## 官方代码快照

`references/repos/` 当前包含以下 14 个仓库：

- `lavender`、`spikingbrain-7b`、`openvla-oft`
- `qwen2.5-vl`、`qwen3-vl`、`deepseek-vl2`
- `openvla`、`qwen-robotmanip`、`openpi`
- `libero`、`diffusion_policy`
- `lingbot-va`、`lingbot-vla`、`lingbot-vla-v2`

远端 URL、当前提交哈希和用途见 `repositories.yaml`；可复现实验时应从这里固定一个 commit，而不是依赖远端默认分支。

## 校验与补档

在项目根目录执行：

```bash
python3 scripts/verify_references.py
cd references && sha256sum -c checksums.sha256
```

需要补档或刷新代码时，脚本默认使用 `127.0.0.1:7897`：

```bash
cd /home/bubble/类脑计算/VLM终局
bash scripts/download_references.sh
```

`download_failures.log` 保留了历史失败和最后一次成功记录。下载脚本只根据本次运行产生的失败文件决定退出码，因此旧日志不会使后续成功运行误报失败。
