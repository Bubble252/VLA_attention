# P6 B0–B4 formal offline result summary（seed17, 2026-10-08）

本文件汇总 seed17 正式离线训练后的产物完整性和 held-out action prediction。它不是 rollout 结果，不报告 LIBERO success rate。

## 1. 产物完整性

- 远端结果根目录：`/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_b0_b4/formal_offline_20261007`
- 远端 repo：`/vepfs-mlp2/c20250405/400040/transfer/vla_attention/repo/VLA_attention`
- 远端 Git SHA：`unavailable:CalledProcessError(128, ['git', '-C', '/vepfs-mlp2/c20250405/400040/transfer/vla_attention/repo/VLA_attention', 'rev-parse', 'HEAD'])`
- 五组 B0–B4 均有 `train_metrics.jsonl`、`final.json`、`checkpoint_02520/05040/07560/10080`。
- 五组 `final.json.status` 均为 `passed`，五组 `train_metrics.jsonl` 均为 10080 行。
- 完整 fingerprint 见 `formal_offline_integrity_20261008.json` 与 `formal_offline_integrity_summary_20261008.md`。

## 2. Held-out offline action prediction

| group | n | action MSE ↓ | action MAE ↓ | translation MSE ↓ | rotation MSE ↓ | gripper MSE ↓ | gripper accuracy ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 native action baseline | 769 | 0.187634 | 0.308525 | 0.257857 | 0.141033 | 0.116765 | 0.769343 |
| B1 retention | 769 | 0.170122 | 0.293686 | 0.223477 | 0.136108 | 0.112097 | 0.786573 |
| B2 semantic attribution | 769 | 0.270470 | 0.383775 | 0.318018 | 0.149714 | 0.490093 | 1.000000 |
| B3 retention + semantic | 769 | 0.204097 | 0.326322 | 0.266259 | 0.142188 | 0.203334 | 0.815345 |
| B4 containment | 769 | 0.249770 | 0.384150 | 0.314967 | 0.149551 | 0.354832 | 1.000000 |

## 3. Paired interpretation currently allowed

- B1 vs B0：B1 action MSE 降低约 9.33%，当前 seed17 支持 BlindVLA-style retention 是强 baseline。
- B2 vs B0：B2 action MSE 升高约 44.15%，当前不支持 semantic attribution 单独提升 action prediction。
- B3 vs B1：B3 action MSE 升高约 19.97%，当前不支持在 retention 基础上直接加入 semantic attribution 有增量。
- B4 vs B3/B2：B4 相对 B3 更差；相对 B2 略好但仍弱于 B0/B1，不能作为 containment 有效证据。
- B4 最后一步 `gradient_norm_pre_clip=1776.0`，提示 containment/semantic 梯度尺度存在风险，后续需要优先排查 loss scale、warm-up 和 action evidence 定义。

## 4. Controls 状态

- held-out semantic cache 正在生成，controls 尚未完成。
- 已部署远端 guard：cache 完整且失败为 0 后自动运行 full controls evaluator。
- controls 输出目标：`/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_b0_b4/posttrain_eval_20261008/controls_eval_full_seed17`。
- raw-attention proxy 当前记录为 unavailable：pinned OpenVLA/OFT evaluator 没有暴露可比较视觉 attention map；机制比较使用 action-loss gradient×activation。

## 5. 当前 Go/No-Go 边界

在 controls 完成前，只能给出 action-only 临时判断：B1 可作为有效强 baseline 继续保留；B2/B3/B4 当前不能进入 rollout success 叙事。最终 Go/No-Go 必须等待 correct/wrong/random/area-matched controls。

