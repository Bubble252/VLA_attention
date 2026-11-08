# Goal18 Gate 记录

**日期**：2026-11-09<br>
**状态**：`FAIL`<br>
**Run Contract**：`PASS`（见 `evidence/run_contract.json`）<br>
**autoresearch**：已触发（同一 G-P 多个 arm 连续失败；连续三种修正指标不动）

## 1. G-P（grounding precision）

预注册阈值：

- mean destination error `<=0.015 m`
- max `<=0.030 m`
- 至少 `12/15` 在 `0.03 m` 内
- object in mask `>=12/15`
- 0 NaN

结果：

| arm | 3 cm 内 | mean | max | object in mask | 状态 |
|---|---:|---:|---:|---:|---|
| C0 | 6/15 | 0.03372 m | 0.07806 m | 15/15 | FAIL |
| C1-GLOBAL | 6/15 | 0.03379 m | 0.07819 m | 15/15 | FAIL |
| C1-PROMPT | 6/15 | 0.03394 m | 0.07836 m | 15/15 | FAIL |
| C1-RELATION | 6/15 | 0.03387 m | 0.07831 m | 15/15 | FAIL |
| C2-RESIDUAL | 6/15 | 0.03310 m | 0.07811 m | 15/15 | FAIL |

结论：`G-P FAIL`。

## 2. G-E（execution）

未运行。原因：G-P 未通过，按预注册规则不得进入 E1、C3 或完整动作 residual 实验；不能把“未运行”写成“执行方法失败”。

`G-E = NOT_RUN`。

## 3. G-S（稳定性）

未运行。没有 arm 通过 G-P，按规则不允许用 eval 选 seed 或扩大训练。

`G-S = NOT_RUN`。

## 4. G-C（组合泛化诊断）

Goal18 不包含 Track C 训练；没有新的 Track C 结果，因此不写“组合泛化已解决”。

`G-C = NOT_RUN`。

## 5. 失败判读

- C1 的 global/per-prompt offset 约 `1e-4 m`，没有修正空间；
- C2 train MAE `0.46 mm`、val `8.07 mm`、eval `33.1 mm`，是过拟合而不是可迁移修正；
- 15 个 `white_right` 的 predicted point 全部落在正确 mask 内，但只有 6 个进入 3 cm；
- 误差不是零均值：15/15 的 world dx、dy 都为正，mean 约 `+25.0 mm`、`+19.9 mm`。

## 6. 失败后的新证据

训练标签使用 `visible_surface_centroid`（`build_c1_point_manifest.py`），评测 oracle 使用 receptacle object center。两者不是同一个目标。

离线对齐诊断：

| 目标点 | mean error to oracle | max | 3 cm 内 |
|---|---:|---:|---:|
| 当前 predicted point | 33.72 mm | 78.06 mm | 6/15 |
| visible mask centroid | 22.27 mm | 31.83 mm | 0/15（按该诊断；均值仍超 15 mm） |
| projected object center | 10.64 mm | 13.26 mm | 15/15 |

另外：

- predicted point 到 visible mask centroid：mean `(-4.38, -1.50)` px；
- predicted point 到 projected object center：mean `(-11.26, -3.85)` px；
- mask centroid 到 projected object center：mean `(-6.88, -2.35)` px。

结论：当前 point head 的失败首先来自**训练目标与成功目标不对齐**；在标签不修的情况下继续做 2D offset 或 C2 residual 不可判读。

## 7. 最终状态

`Goal18 = FAIL`。<br>
`decision = pivot`：停止当前 point-head 精度线，先做 target-aligned point/region 监督；TAG/AnySlot 保留为标签修复后仍失败的 fallback。详见 `06_result.md` 和 `research/decisions/2026-11-09-goal18-target-alignment-pivot.md`。
