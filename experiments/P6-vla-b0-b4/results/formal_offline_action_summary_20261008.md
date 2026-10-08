# P6 B0--B4 held-out offline action prediction（seed17）

评测对象是 `formal_offline_20261007` 的 `checkpoint_10080`。本表只表示 held-out offline action prediction，不表示 LIBERO rollout success。

| group | action MSE ↓ | action MAE ↓ | translation MSE ↓ | rotation MSE ↓ | gripper MSE ↓ | gripper acc ↑ |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.187634 | 0.308525 | 0.257857 | 0.141033 | 0.116765 | 0.769343 |
| B1 | 0.170122 | 0.293686 | 0.223477 | 0.136108 | 0.112097 | 0.786573 |
| B2 | 0.270470 | 0.383775 | 0.318018 | 0.149714 | 0.490093 | 1.000000 |
| B3 | 0.204097 | 0.326322 | 0.266259 | 0.142188 | 0.203334 | 0.815345 |
| B4 | 0.249770 | 0.384150 | 0.314967 | 0.149551 | 0.354832 | 1.000000 |

## Paired comparison

| comparison | Δ action MSE ↓ | Δ action MAE ↓ | Δ translation MSE ↓ | Δ rotation MSE ↓ | Δ gripper acc ↑ | action MSE relative |
|---|---:|---:|---:|---:|---:|---:|
| B1-vs-B0 | -0.017512 | -0.014839 | -0.034380 | -0.004926 | 0.017230 | -9.33% |
| B2-vs-B0 | 0.082836 | 0.075250 | 0.060161 | 0.008680 | 0.230657 | 44.15% |
| B3-vs-B1 | 0.033975 | 0.032635 | 0.042782 | 0.006081 | 0.028771 | 19.97% |
| B4-vs-B3 | 0.045673 | 0.057828 | 0.048708 | 0.007363 | 0.184655 | 22.38% |
| B4-vs-B2 | -0.020700 | 0.000375 | -0.003051 | -0.000163 | 0.000000 | -7.65% |

## 当前结论边界

- B1（BlindVLA-style retention）相对 B0 在 action MSE/MAE、translation MSE、rotation MSE 和 gripper accuracy 上均有改善，是当前 seed17 offline action prediction 的最好组。
- B2（semantic attribution）相对 B0 明显变差；B3 相对 B1 也变差；B4 相对 B3 继续变差，但相对 B2 略改善。
- 因此当前 seed17 离线动作预测结果不支持“semantic attribution / containment 已经提升 VLA action prediction”的主张。
- 这不是最终 VLA 否定结论：还缺 held-out semantic controls、patch/action evidence 机制指标、multi-seed 和正式 rollout provenance。
- 在 controls 完成前，B2/B3/B4 的失败可能来自语义图质量、loss scale、action-evidence map 与 action loss 的耦合方式、或当前 offline 指标与机制目标不一致；不能直接写成方法完全无效。
