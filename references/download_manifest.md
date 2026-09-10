# 下载清单与优先级

本清单只下载论文和官方代码仓库，不下载模型权重、LIBERO 数据集或大规模训练数据。权重和数据应在 P4/P5 确定接口后，按固定 checkpoint 和数据版本单独下载。

## A 类：首轮实现必须具备

| 资料 | 类型 | 归档位置 | 当前状态 |
|---|---|---|---|
| Lavender / AstraZeneca-vlm | Git 仓库 | `references/repos/lavender` | 已归档，本地 commit `58fc71b` |
| SpikingBrain-7B | Git 仓库 | `references/repos/spikingbrain-7b` | 已归档，本地 commit `ef99987` |
| SpikingBrain 英文/中文报告 | PDF | `references/papers/SpikingBrain_Report_*.pdf` | 已归档并生成 SHA256 |
| OpenVLA-OFT | Git 仓库 | `references/repos/openvla-oft` | 已归档，本地 commit `e4287e9` |
| OpenVLA 论文 | PDF | `references/papers/openvla_2406.09246.pdf` | 已从本机资料归档 |
| LIBERO | Git 仓库 + 论文 | `references/repos/libero`、`references/papers/libero_2306.03310.pdf` | 待通过 7897 下载 |
| Qwen-RobotManip | 论文 + Git 仓库 | `references/papers/qwen_robotmanip_2606.17846.pdf`、`references/repos/qwen-robotmanip` | 论文已归档；仓库待通过 7897 下载 |

## B 类：模型结构和对照实验需要

| 资料 | 类型 | 归档位置 | 当前状态 |
|---|---|---|---|
| Qwen2.5-VL | 论文 + Git 仓库 | `papers/qwen25_vl_2502.13923.pdf`、`repos/qwen2.5-vl` | 待通过 7897 下载 |
| Qwen3-VL | 论文 + Git 仓库 | `papers/qwen3_vl_2511.21631.pdf`、`repos/qwen3-vl` | 待通过 7897 下载 |
| DeepSeek-VL2 | 论文 + Git 仓库 | `papers/deepseek_vl2_2412.10302.pdf`、`repos/deepseek-vl2` | 待通过 7897 下载 |
| Qwen2-VL | 论文 | `papers/qwen2_vl_2409.12191.pdf` | 待通过 7897 下载；仅作历史结构参照 |

## C 类：后置架构参照

| 资料 | 类型 | 归档位置 | 当前状态 |
|---|---|---|---|
| π0/openpi | 论文 + Git 仓库 | `papers/pi0_2410.24164.pdf`、`repos/openpi` | 待通过 7897 下载 |
| Diffusion Policy | 论文 + Git 仓库 | `papers/diffusion_policy_2303.04137.pdf`、`repos/diffusion_policy` | 待通过 7897 下载 |
| GLA | 论文 | `papers/gla_2312.06635.pdf` | 待通过 7897 下载 |
| Attention Sinks | 论文 | `papers/attention_sinks_2309.17453.pdf` | 待通过 7897 下载 |
| LingBot-Video | 论文 | `papers/lingbot_video_2607.07675.pdf` | 已从本机资料归档 |
| LingBot-VA 2.0 | 论文 | `papers/lingbot_va2_2607.08639.pdf` | 已从本机资料归档 |
| LingBot-VLA 2.0 | 论文 | `papers/lingbot_vla2_2607.06403.pdf` | 已从本机资料归档 |
| LingBot 代码仓库 | Git 仓库 | `repos/lingbot-*` | 待通过 7897 下载，且需先冻结具体 checkpoint |

## 执行命令

服务器上的 `127.0.0.1:7897` 可用时运行：

```bash
cd /home/bubble/类脑计算/VLM终局
bash scripts/download_references.sh
```

脚本默认只通过 `127.0.0.1:7897` 访问外网；如果 GitHub 在服务器仍不可达，可以额外设置 `GIT_MIRROR_PREFIX`，但论文下载仍使用同一 7897 代理。

执行后检查：

```bash
cat references/download_failures.log
cat references/checksums.sha256
cat references/repo_versions.md
git status --short
```

## 暂不下载的内容

- Qwen、DeepSeek、OpenVLA、π0、LingBot 的模型权重；
- LIBERO 数据集和 RLDS 数据；
- 大规模机器人预训练数据；
- 真机 SDK、驱动和控制器。

这些文件体积大且会绑定具体实验配置。先完成代码接口审计和 LIBERO 单任务 smoke，再根据选定 baseline 下载对应 checkpoint。
