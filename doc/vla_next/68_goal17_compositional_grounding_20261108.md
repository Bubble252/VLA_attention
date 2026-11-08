# Goal17：Compositional grounding repair + simplified R-PRED

**日期**：2026-11-08<br>
**上游**：[Goal16](../experiments/Goal16-reguide-rebinding-20261108/05_pilot_result.md)、[67](67_dependency_repair_and_reguide_pivot_20261108.md)<br>
**状态**：`COMPLETED_PASS`（见 [06_result.md](../../experiments/Goal17-compositional-grounding-20261108/06_result.md)）

## Research Contract

```yaml
autoresearch:
  required: true
  triggers:
    - gate_fail_twice
    - no_metric_movement_after_3_fixes
    - heldout_collapse
    - seed_noise_ambiguous
  ledger: research/LEDGER.jsonl
  blockers_dir: research/blockers/
  reading_dir: references/paper_reading/
  max_outer_loops: 3
  check_command: python scripts/check_research_ledger.py --goal Goal17 --goal-doc doc/vla_next/68_goal17_compositional_grounding_20261108.md --strict
```

Goal17 是探索型 goal，`required: true`。触发 blocker 时必须走 autoresearch loop；如果连续失败，允许 outer loop 调研并重选 grounding 方案，但不得违反 eval 隔离、no oracle fallback、不得用 validation/eval 调参。

## 1. 为什么做 Goal17

Goal16 已经把问题拆开了：

| arm | success | released | target dest correct | wrong dest |
|---|---:|---:|---:|---:|
| `B0` | 0/15 | 0/15 | 0/15 | 14/15 |
| `R-GEO` | 2/15 | 2/15 | 10/15 | 3/15 |
| `R-NAIVE` | 4/15 | 1/15 | 10/15 | 3/15 |
| `R-PRED` | 0/15 | 0/15 | 0/15 | 14/15 |
| `R-SHUF` | 0/15 | 0/15 | 0/15 | 14/15 |

结论：

1. 动作底座有 headroom：把物体送到正确 `plate_2` 后能成功；
2. 当前 full 3D hand-back 不如简化 xy transport：`R-GEO 2/15 < R-NAIVE 4/15`；
3. deployable 瓶颈是 grounding：point head 对 15 个 held-out `white_right` 全部预测到 `plate_1`；
4. 这不是相机/反投影 bug，而是 **compositional binding failure**：训练只有 `red_left/red_right/white_left`，模型把 `white` 绑定到左盘，没有学会 `white + right → plate_2`。

Goal17 只做两件事：

> **A. 修 grounding：用 2×2 counterfactual 训练，让模型真正组合 object attribute 与 spatial relation。**<br>
> **B. 修 execution：用简化 xy transport + explicit release 重跑 R-PRED，不再默认 full 3D hand-back。**

## 2. 两个 track，不要混在一起

### Track D：Deployable

- 训练：4 个 prompt 组合（`red_left`、`red_right`、`white_left`、`white_right`）都用，但只来自 **train episodes**；
- 测试：Goal16 的 15 个 `white_right` eval init；
- 目的：让 `R-PRED` 在冻结 eval 上真正可用；
- 这是主 track。

### Track C：Compositional diagnostic

- 训练：仍然只给 3 个 prompt（去掉 `white_right`）；
- 测试：`white_right` held-out；
- 目的：测量模型是否真的学会了组合，而不是靠见过 `white_right`；
- 这是诊断 track，不用它调参，也不拿它刷主 Gate。

**必须同时报告 D 和 C。** 只报 D 会把“见过组合”误写成“学会组合”；只报 C 会继续得到 Goal16 的 0/15。

## 3. 数据怎么造

### 3.1 只允许 train episodes

- 训练帧必须来自 train split；
- 15 个 `white_right` eval init、validation、rollout 一律不得进入训练；
- 不得用 eval 图像、eval mask、eval success 做任何 teacher 或 pseudo-label。

### 3.2 2×2 counterfactual 四宫格

对每个训练帧，生成 4 条 language 条件：

| prompt | object label | destination label |
|---|---|---|
| `red_left` | `red_coffee_mug_1` | `plate_1` |
| `red_right` | `red_coffee_mug_1` | `plate_2` |
| `white_left` | `porcelain_mug_1` | `plate_1` |
| `white_right` | `porcelain_mug_1` | `plate_2` |

图像不变，只换 instruction 和标签。标签来自已有 mask / 几何，不使用 simulator 的 success 或 future frame。

### 3.3 特征提取

- 复用 `extract_c1_point_features.py` 的接口；
- 对每个训练帧跑 4 次 Qwen forward，缓存 visual/language hidden states；
- 训练集大小约为原来的 4 倍；
- 必须记录每条的 `image_id`、`prompt_id`、`object_point_px`、`destination_point_px`、`mask_path`、`split`。

## 4. 模型怎么改

### 4.1 保留 PointQueryHead 主体

先不引入大模型，先在现有 `PointQueryHead` 上验证：

- 输入：visual tokens + view ids + language tokens；
- 输出：object point + destination point；
- 原有 L1 / in-mask loss 保留。

### 4.2 增加 counterfactual margin loss

这是 Goal17 的核心，不是可选：

```text
同一张图：
  red_left   vs red_right   → destination 必须分开
  white_left vs white_right → destination 必须分开
  red_left   vs white_left  → object 必须分开
  red_right  vs white_right → object 必须分开
```

建议实现：

- `destination_margin_loss`：同图左右 prompt 的 destination 点距离至少 `m_d`，否则惩罚；
- `object_margin_loss`：同图红/白 prompt 的 object 点距离至少 `m_o`；
- 可选 `relation_consistency`：左右 destination 应该分别靠近 `plate_1` / `plate_2` 的 mask 质心。

没有 margin loss，模型仍然可以靠“white → 左盘”的捷径把训练 loss 降下去。

### 4.3 如果 PointQueryHead 仍不 work

按顺序尝试，不要直接跳大模型：

1. token-level relation binding：不要把 language tokens 全 mean-pool，保留 `left/right` 附近的 token；
2. 加一个小的 relation classifier / spatial token；
3. 用 VLM open-vocabulary 选择 object/receptacle，再接几何反投影；
4. 最后才考虑 Grounding DINO / SAM 等外部 detector。

## 5. 执行 wrapper：简化是控制变量，不是判 3D 死刑

Goal16 的结果不是“xy-only 一定更好”，而是：

- `R-GEO`：2/15，但 2/15 都是 released；
- `R-NAIVE`：4/15，但只有 1/15 released，另外 3/15 是 hold-only；
- `R-GEO` re-arm 总数 26，`R-NAIVE` 只有 4。

所以正确结论是：**full 3D hand-back 没有证明出优势，而且引入了更多 re-arm 复杂度；但它的 released success 并不差。** 不能因为总 success 4 > 2 就把 3D 直接丢掉。

Goal17 的执行端要保留两个 arm：

| arm | 执行方式 | 作用 |
|---|---|---|
| `E1` | 简化 xy transport + explicit release；z/rotation 保持 base policy；不 re-arm 或最多 1 次 | 主 readout，低方差，先确认 grounding 是否真的被用上 |
| `E2` | Goal16 的 full 3D approach + hand-back + re-arm | 对照，防止把有用的 3D 机制提前丢掉 |

执行顺序：

1. 先用 oracle destination 跑 `E1`，确认 release/settle 逻辑；
2. 接 predicted destination 跑 `E1`；
3. 只有 grounding Gate 通过后，再用同一批 predicted points 跑 `E2`；
4. 如果 `E2 > E1`，说明 3D hand-back 值得保留；如果 `E1 >= E2`，才把 `E1` 定为默认。

**简化只是为了先隔离 grounding 变量，不是最终方法结论。**

## 6. Gate

### G-A：Grounding（Track D）

- 15 个 eval init 全部有预测；
- destination error `<=0.15 m` 至少 `12/15`；
- object point in mask 至少 `12/15`；
- 左右 counterfactual margin 方向正确至少 `12/15`；
- 红白 object margin 方向正确至少 `12/15`；
- 不允许 oracle fallback。

### G-B：Execution（R-PRED with simplified wrapper）

- `R-PRED` official success `>=1/15`；
- `R-PRED - B0 >= +1/15`；
- `R-PRED` released success `>=1/15`；
- `R-PRED` wrong-destination 不高于 `B0`；
- 0 error / 0 NaN / 0 flew-out。

### G-C：Compositional diagnostic（Track C）

- 只报告，不参与主 Gate；
- 如果 Track C 仍然 `0/15`，但 Track D 通过，结论只能是“deployable 接口修好了”，不能写“组合泛化学会了”。

## 7. 失败裁决

| 观察 | 判读 | 下一步 |
|---|---|---|
| Track D grounding 仍 `<12/15` | 现有 point head 路线不够 | 转 VLM open-vocab / detector，不再调 margin |
| Grounding 过、R-PRED 仍 0/15 | 执行 wrapper 或 release 是瓶颈 | 修 release/settle，不回去训练 grounding |
| Track D 过、Track C 仍 0/15 | 只是见过组合 | 论文只能写 deployable，不能写 compositional |
| R-PRED 过、R-NAIVE+ 不过 | 预测点与执行不匹配 | 检查 predicted point 的坐标/视角/时序 |

## 8. 一句话

**Goal17 不是继续调 ReGuide wrapper，而是用 2×2 counterfactual + margin loss 修 grounding；执行端先用简化 xy + release 做主 readout，同时保留 full 3D hand-back 作为 E2 对照，避免把有用的 3D 机制提前丢掉。**

---

## 9. Goal17 提示词

```text
你现在执行 Goal17：Compositional grounding repair + simplified R-PRED。

autoresearch: required

先读：
- doc/vla_next/68_goal17_compositional_grounding_20261108.md
- doc/vla_next/67_dependency_repair_and_reguide_pivot_20261108.md
- experiments/Goal16-reguide-rebinding-20261108/05_pilot_result.md
- experiments/Goal16-reguide-rebinding-20261108/grounding_summary.json
- experiments/Goal16-reguide-rebinding-20261108/predicted_points.jsonl

硬约束：
1. 不训练 residual/route/action head；只修 grounding 和简化 execution wrapper。
2. 15 个 white_right eval init、validation、rollout 不得进入训练；不得用 oracle fallback。
3. 训练帧只能来自 train split；不得用 eval 图像/mask/success 做 teacher 或 pseudo-label。
4. 必须实现 2×2 counterfactual 四宫格：red/white × left/right；图像不变，只换 instruction 和标签。
5. 必须实现 counterfactual margin loss：左右 destination 必须分开，红白 object 必须分开；没有 margin 的 baseline 也要跑。
6. 同时跑 Track D（4 prompt 训练，episode-held-out）和 Track C（3 prompt 训练，white_right held-out）；主 Gate 只看 Track D，Track C 只做诊断。
7. 先跑 oracle destination + 简化 xy/release wrapper（E1），确认 release 逻辑；再跑 predicted destination（E1）。
8. 保留 E2：predicted destination + full 3D hand-back（Goal16 R-GEO 对照）；不要因为 E1 更简单就提前丢掉 3D。
9. 如果 Track D grounding 仍 <12/15，停止 point-head 线，转 VLM open-vocab/detector；不要继续加 hand-back 模块。
10. 所有结果必须报告 destination error、object in-mask、counterfactual margin、official success、released success、wrong-destination、动作幅度。
11. 必须使用 autoresearch skill：
    - inner loop：每个 hypothesis 记录 experiment、metric、result、decision；
    - outer loop：连续失败时允许调研论文并重选 grounding 方案；
    - 每次触发 blocker 都要写 research/blockers/ 和 research/LEDGER.jsonl；
    - decision 只能是 continue / pivot / stop。
12. 完成前必须运行：
    python scripts/check_research_ledger.py --goal Goal17 --goal-doc doc/vla_next/68_goal17_compositional_grounding_20261108.md --strict
    检查不通过不能 commit。
13. 最后更新 result md、提交并 push；禁止 git add -A。

执行顺序：
- 先写 Run Contract；
- 构造 2×2 训练特征；
- 跑 no-margin baseline；
- 跑 margin 版；
- 先 oracle destination + E1 简化 wrapper；
- 再 predicted destination + E1；
- grounding Gate 通过后，跑 predicted destination + E2 full 3D 对照；
- 按 G-A/G-B/G-C 判读；
- 失败就按裁决表停线或转 VLM open-vocab；
- 写 result md、ledger、运行 checker、commit/push。
```
