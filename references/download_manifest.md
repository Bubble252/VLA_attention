# 下载清单与归档状态

本清单只涵盖论文与官方代码。模型权重、LIBERO 数据集和大规模训练数据在 P4/P5 固定基线与接口后单独下载。

## 已完成

| 类别 | 资料 | 本地位置 | 状态 |
|---|---|---|---|
| A | Lavender、SpikingBrain-7B、OpenVLA-OFT | `repos/lavender`、`repos/spikingbrain-7b`、`repos/openvla-oft` | 已归档并固定 commit |
| A | OpenVLA、LIBERO、Qwen-RobotManip | `papers/` 与对应 `repos/` | 论文和代码均已归档 |
| B | Qwen2-VL、Qwen2.5-VL、Qwen3-VL、DeepSeek-VL2 | `papers/` 与对应 `repos/` | 已归档；Qwen2-VL 仅作历史结构参照 |
| B | LLaVA-1.6 / LLaVA-NeXT、LLaVA-OneVision | 待归档到 `papers/` 与 `repos/llava-next` | 新增 VLM 普适性基线；下一次下载 goal 执行 |
| C | π0/openpi、Diffusion Policy、GLA、Attention Sinks | `papers/` 与对应 `repos/` | 已归档 |
| C | LingBot-VA、LingBot-VLA、LingBot-VLA 2.0、LingBot-Video | `papers/lingbot_*.pdf` 与 `repos/lingbot-*` | 已归档；用于视频/世界模型/VLA 架构参照 |
| C | Phase segmentation / progress methods | `references/phase_segmentation/` | 已建立方法索引；PDF 待 7897 代理可用后归档 |

完整的版本与文件名分别见 `repo_versions.md`、`repositories.yaml` 和 `metadata.yaml`。

## 校验命令

```bash
cd /home/bubble/类脑计算/VLM终局
python3 scripts/verify_references.py
cd references && sha256sum -c checksums.sha256
```

## 后续需要时的刷新命令

```bash
cd /home/bubble/类脑计算/VLM终局
bash scripts/download_references.sh
```

脚本默认通过 `http://127.0.0.1:7897` 下载；如 GitHub 不可达，可提供 `GIT_MIRROR_PREFIX`。历史网络失败保留在 `download_failures.log`，不会影响新一次成功运行的退出状态。

## 暂不下载

- Qwen、DeepSeek、OpenVLA、π0、LingBot 的模型权重；
- LIBERO 数据集和 RLDS 数据；
- 大规模机器人预训练数据；
- 真机 SDK、驱动和控制器。
