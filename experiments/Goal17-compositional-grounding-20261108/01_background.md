# Goal17：Compositional grounding repair

**日期**：2026-11-08<br>
**上游**：Goal16 `05_pilot_result.md`、doc 68

## 1. 问题

Goal16 的 15-init 结果显示：

- `B0`：0/15；
- `R-GEO`：2/15；
- `R-NAIVE`：4/15；
- `R-PRED`：0/15；
- `R-SHUF`：0/15。

动作底座有 headroom，但 deployable grounding 完全失败：point head 对 15 个 held-out `white_right` 全部预测到 `plate_1`，平均误差 `0.603 m`。

## 2. 为什么失败

训练任务只有：

```text
red_left
red_right
white_left
```

没有 `white_right`。模型把 `white` 绑定到左盘，没有学会 `white + right → plate_2`。这是 compositional binding failure，不是相机/反投影 bug。

## 3. 本实验要回答

> 用 2×2 counterfactual 训练 + counterfactual margin loss，能否让 point head 在 held-out `white_right` 上把目的地预测到 `plate_2`？如果能，简化 xy/release wrapper 能否把预测转成 official success？

## 4. 不做

- 不训练 residual/route/action head；
- 不改 15-init eval；
- 不用 oracle fallback；
- 不继续加 full 3D hand-back 模块。
