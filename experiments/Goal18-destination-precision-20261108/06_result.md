# Goal18 结果：Destination precision + residual correction

**日期**：2026-11-09<br>
**状态**：`COMPLETED_FAIL`<br>
**Gate**：[05_gate.md](05_gate.md)<br>
**Run Contract**：`PASS`（`evidence/run_contract.json`）<br>
**autoresearch**：触发，decision = `pivot`

## 1. 结论

Goal18 的预注册 G-P 没有通过。五个 arm 都只有 `6/15` 个 `white_right` destination 在 3 cm 内，mean error 约 `3.3 cm`，max 约 `7.8 cm`。

C1 的 train-only 2D offset 接近零；C2 residual head 在 train 上拟合到 `0.46 mm`，但 val 是 `8.07 mm`，eval 仍是 `33.1 mm`，没有可迁移收益。

因此：

> **当前 failure 不是“2D offset 幅度不够”，而是训练标签（visible surface centroid）与成功评测目标（receptacle object center）不一致，加上 point head 的剩余定位误差。**

## 2. 预注册实验结果

| arm | correction | 3 cm 内 | mean | max | object in mask | 状态 |
|---|---|---:|---:|---:|---:|---|
| C0 | none | 6/15 | 0.0337207 m | 0.0780599 m | 15/15 | FAIL |
| C1-GLOBAL | global offset | 6/15 | 0.0337868 m | 0.0781930 m | 15/15 | FAIL |
| C1-PROMPT | per-prompt offset | 6/15 | 0.0339393 m | 0.0783585 m | 15/15 | FAIL |
| C1-RELATION | per-relation offset | 6/15 | 0.0338715 m | 0.0783053 m | 15/15 | FAIL |
| C2-RESIDUAL | MLP residual | 6/15 | 0.0330976 m | 0.0781082 m | 15/15 | FAIL |

C1 的 offset 量级：

- global：`(-0.000223, +0.000149) m`
- per-prompt：约 `1e-4 m`
- per-relation：约 `1e-4 m`

这些量级不可能把 3.3 cm 误差压到 1.5 cm。

C2 训练记录：

| step | train MAE | val MAE |
|---:|---:|---:|
| 250 | 0.101 cm | 0.815 cm |
| 750 | 0.059 cm | 0.812 cm |
| 1500 | 0.051 cm | 0.816 cm |
| 2500 | 0.046 cm | 0.807 cm |

结论：C2 学到了训练图像/训练场景的拟合，不是可迁移的 destination correction。

## 3. Gate 判读

- G-P：`FAIL`
- G-E：`NOT_RUN`（G-P 先失败，C3 被禁止）
- G-S：`NOT_RUN`
- G-C：`NOT_RUN`（无 Track C 训练）

没有把 `NOT_RUN` 写成 `FAIL`，也没有把 G-P 的 6/15 写成成功。

## 4. 新发现的标签/评测不一致

代码中训练点来自 `build_c1_point_manifest.py` 的 `point_record()`：

- `point_type = visible_surface_centroid`
- `destination_point_kind = visible_receptacle_surface`

但评测 `destination_error_m` 用的是 simulator 中的 receptacle object center：

```text
oracle_xyz = row["objects"][target_receptacle]
```

在 15 个 `white_right` 上做无训练的诊断：

| 目标 | mean error | max | 3 cm 内 |
|---|---:|---:|---:|
| predicted point | 33.72 mm | 78.06 mm | 6/15 |
| visible mask centroid | 22.27 mm | 31.83 mm | 0/15 |
| projected object center | 10.64 mm | 13.26 mm | 15/15 |

这说明：

1. 即使完美预测 visible mask centroid，也不能稳定达到 3 cm 成功目标；
2. 用 projected object center 作为目标，诊断上已经进入 1.5 cm mean / 3 cm max；
3. 当前 point head 预测点还比 mask centroid 再向左偏约 `4.38 px`，相对 projected object center 向左偏约 `11.26 px`；
4. 15/15 的 world dx、dy 都为正，符合“训练目标整体在成功目标左侧/前侧”的系统偏差，而不是随机噪声。

证据文件：

- `evidence/white_right_pixel_alignment.csv`
- `evidence/grounding_C0_rows.jsonl`
- `evidence/calibration.json`

## 5. 假设裁决

### H1：2D bias 是主要误差

**不成立。** train-only global/per-prompt/per-relation offset 全部接近零，C1 没有改善；C0 与 C1 的 mean error 差小于 `0.001 m`。

### H2：剩余误差来自 release/execution offset

**未检验。** G-P 先失败，按预注册规则不允许进入 C3 release residual。不能据此说 H2 不成立。

### H3：E2-PRED 的 3/15 优势是 seed noise

**未检验。** G-P 失败，G-E 未运行；不应跑 3 seeds 来掩盖上游 grounding 问题。

### H4：训练目标与成功目标不一致

**成立。** 训练标签是 visible surface centroid；成功 oracle 是 object center。诊断显示 projected object center 目标 mean `10.64 mm`、max `13.26 mm`，而 mask centroid 目标 mean `22.27 mm`。这解释了为什么 C1 在 train 上几乎没有偏移，却在 eval 上出现稳定正偏差。

## 6. Autoresearch 记录

本次触发了 autoresearch：

- 触发条件：同一个 G-P 在 C0/C1/C2 多个 arm 连续失败；连续三种修正后指标没有移动；
- blocker：`research/blockers/2026-11-08-goal18-destination-precision.md`；
- 证据：`evidence/` 下的 5 个 arm summary、per-init rows、calibration 和 pixel alignment；
- 排查顺序：数据 → 标签 → 评测 → baseline → 架构 → 文献；
- 最便宜 falsification：无训练的 label-target alignment 诊断和 C2 train/val 对照；
- decision：`pivot`。

本次没有把外部论文方法放进代码，因此 `no_external_method_used = true`。但完成了失败后的文献/code 状态审计：

| 方法 | 本地文件 | reading note | 官方 code | 状态 |
|---|---|---|---|---|
| ReGuide `2609.38616` | `papers/reguide_2609.38616.pdf/.txt` | 有 | 正文未找到 | 已在 Goal16/17 使用方法结构 |
| AnySlot `2604.10432` | `papers/anyslot_2604.10432.pdf/.txt` | 有 | 未找到 | 未进入代码；fallback |
| TAG `2603.24584` | `papers/tag_2603.24584.pdf/.txt` | 有 | 未找到 | 未进入代码；fallback |
| R-VLA OpenReview | 无 | 无 | 未查 | OpenReview challenge 阻止下载，不使用 |

## 7. 决策与下一步

`decision = pivot`。

不再继续当前 point-head 的 2D offset 线，也不立刻把 TAG/AnySlot 堆进代码。下一步先做 **target-aligned point/region 监督**：

1. 用 projected object center 或经审核的区域目标重建 point manifest；
2. 重新训练 point head，先在 train/validation 和 3 个 calibration init 上验证；
3. 如果 G-P 仍失败，再进入 TAG/AnySlot/open-vocab detector；
4. 12 个 confirmation init 在最终方法确定前冻结；
5. 任何外部方法进入代码前补齐 PDF/TXT、code 状态和 reading note。

## 8. 产物与复现

本地：

```text
experiments/Goal18-destination-precision-20261108/
  05_gate.md
  06_result.md
  evidence/run_contract.json
  evidence/calibration.json
  evidence/grounding_*_summary.json
  evidence/grounding_*_rows.jsonl
  evidence/white_right_pixel_alignment.csv
```

远端：

```text
$W/results/K4_lite/T4b/goal18_destination_precision_20261108/
  run_contract.json
  train/calibration.json
  train/residual_head.pt
  eval/grounding/{C0,C1-GLOBAL,C1-PROMPT,C1-RELATION,C2-RESIDUAL}/
```

关键命令：

```bash
python scripts/build_goal18_contract.py ...
python scripts/train_goal18_precision.py ...
python scripts/evaluate_goal18_grounding.py --arm C0 ...
python scripts/evaluate_goal18_grounding.py --arm C1-PROMPT ...
python scripts/evaluate_goal18_grounding.py --arm C2-RESIDUAL ...
```
