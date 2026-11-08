# Decision：Goal18 target-alignment pivot

**日期**：2026-11-09<br>
**Goal**：Goal18<br>
**Decision**：`pivot`<br>
**触发**：G-P 在 C0/C1/C2 上连续失败；连续三种修正指标不动；标签/评测审计发现目标不一致。

## 证据

- C0/C1/C2 都是 `6/15` 在 3 cm 内，mean error 约 `3.3 cm`；
- C1 offsets 约 `1e-4 m`，说明 train split 的平均 2D bias 不是主要矛盾；
- C2 train `0.46 mm`、val `8.07 mm`，eval 无改善，说明 residual head 过拟合；
- 训练标签是 `visible_surface_centroid`，评测 oracle 是 receptacle object center；
- mask centroid 到 oracle 的 label ceiling 是 `22.27 mm`，projected object center 到 oracle 是 `10.64 mm`；
- projected object center 目标在 15/15 上都进入 3 cm。

## 决策

停止当前 point-head 的 2D offset / C2 residual 精度线。

下一步不是直接堆 TAG/AnySlot，而是先做 target-aligned point/region 监督：用与成功 oracle 对齐的 projected object center 或经过审核的区域目标重建训练标签，再验证 point/region head。

## 边界

- 不使用 eval/validation 调标签；
- 12 个 confirmation init 在最终方法确定前冻结；
- target-aligned G-P 仍失败后，才进入 TAG/AnySlot/open-vocab detector；
- 任何外部方法进入代码前必须补齐 PDF/TXT、code 状态和 reading note；
- 不能把 projected object center 的离线 ceiling 写成模型结果。
