# Blocker：Goal18 destination precision failure

**日期**：2026-11-08（结果更新 2026-11-09）<br>
**Goal**：Goal18<br>
**状态**：`FAIL` + `pivot`<br>
**autoresearch**：已触发

## 1. 现象

- Goal17 `G4-MARGIN` destination mean error `0.0337 m`，max `0.0781 m`；
- `E1-ORACLE` 7/15，`E1-PRED` 2/15，`E2-PRED` 3/15；
- Goal18 五个 arm 全部 `FAIL`，3 cm 内都是 `6/15`；
- C1 global/per-prompt/per-relation offset 约 `1e-4 m`，没有改善；
- C2 train MAE `0.46 mm`、val `8.07 mm`、eval `33.1 mm`，是过拟合；
- 15/15 predicted point 在正确 mask 内，但只有 6/15 进入 3 cm。

## 2. 已排除

- binding 完全失败：`white_right` destination 已 15/15 在正确盘；
- 随机输出：预测点稳定落在目标区域，不是随机噪声；
- 执行 wrapper 完全无效：oracle destination 能到 7/15；
- release settle 太短：settle 32 已把 E1 从 0/15 提到 2/15；
- “oracle 与 mask centroid 一致”：不成立，mask centroid 与 object center 平均相差 `(-6.88, -2.35) px`。

## 3. Hypotheses

1. H1：predicted destination 的 2D bias 是主要误差；train-only offset / residual 能把 mean error 压到 `<=1.5 cm`，并把 E1 success 提高至少 2/15。
2. H2：如果 2D calibration 不够，剩余误差主要来自 release/execution offset；应先学 release XY residual，不应先上完整 6-DoF。
3. H3：E2-PRED 3/15 与 E1-PRED 2/15 的差异可能是 seed noise；必须 3 seeds 才能判断。
4. H4：训练标签 `visible_surface_centroid` 与成功评测目标 object center 不一致，是 G-P 失败的上游原因。

## 4. Falsification

- H1：先跑 global offset，再跑 per-prompt offset；如果 G-P 不过，停止 point-head 精度线；
- H2：如果 G-P 过、G-E 不过，比较 corrected destination 与 release 点 XY 误差；只做 release XY residual；
- H3：对最终 arm 跑 3 seeds；如果 1/15 差异不稳定，写成单 seed 机制信号；
- H4：不训练模型，离线比较 predicted point、visible mask centroid、projected object center 到 oracle 的误差。

## 5. 结果

### H1

**不成立。** C1 的 offset 接近零，C1 与 C0 的 mean error 差小于 `0.001 m`；C2 eval 也没有改善。

### H2

**未检验。** G-P 先失败，按预注册规则不能进入 C3 release residual。

### H3

**未检验。** G-E 未运行；单 seed 的 3/15 不能继续被当作稳定优势。

### H4

**成立。** 无训练诊断：

| 目标点 | mean error | max | 3 cm 内 |
|---|---:|---:|---:|
| predicted point | 33.72 mm | 78.06 mm | 6/15 |
| visible mask centroid | 22.27 mm | 31.83 mm | 0/15 |
| projected object center | 10.64 mm | 13.26 mm | 15/15 |

predicted point 到 visible mask centroid 平均 `(-4.38, -1.50) px`；到 projected object center 平均 `(-11.26, -3.85) px`。15/15 的 world dx、dy 都为正，符合“训练目标整体落在成功目标左侧/前侧”的系统性偏差。

## 6. 结论

`decision: pivot`。

停止当前 point-head 的 2D offset 精度线，先修训练目标与成功目标的对齐；TAG/AnySlot 作为标签修复后仍失败时的 fallback。详见 `research/decisions/2026-11-09-goal18-target-alignment-pivot.md`。
