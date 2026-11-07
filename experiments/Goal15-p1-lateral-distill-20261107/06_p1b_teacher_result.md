# P1-B：A10 teacher label cache

**日期**：2026-11-07
**状态**：`P1-B PASS`（离线教师标签缓存）
**裁决**：允许进入 P1-C offline student；但这不是 base-policy closed-loop 教师，也不说明 residual 方法已经成立。

## 1. 结论

P1-B 在 `cache_train_v2` 的 96 个 training episodes 上生成了 A10 教师标签：

- 96 episodes、11,630 transitions 全部对齐；
- `white_right` 计数为 0；
- release 正样本 `88 transitions / 88 episodes`，达到 `>=50` 的协议阈值；
- plateau 桶 `37` 条、大修正桶 `262` 条，均非空；
- 所有数组 finite，生成错误为 0；
- `source_index` 为 `0..11629`，唯一且与 `meta.jsonl` 一致。

该 PASS 只证明教师标签可以生成并与训练 cache 对齐，不证明学生能学会，也不证明闭环会提升。

## 2. Gate 结果

| 检查 | 实际值 | 结果 |
|---|---:|---|
| 96 个 training episodes 全部有标签 | 96 / 96 | PASS |
| transition 数与 cache 一致 | 11,630 / 11,630 | PASS |
| `white_right` 计数 | 0 | PASS |
| release 正样本 transitions | 88（要求 `>=50`） | PASS |
| release 正样本 episodes | 88（要求 `>=50`） | PASS |
| plateau 桶 `0.027–0.035m` | 37 | PASS |
| 大修正桶 `0.05–0.10m` | 262 | PASS |
| delta 分量绝对值最大值 | `0.10000000149`（容差内） | PASS |
| 所有数组 finite | true | PASS |
| 生成错误 | 0 | PASS |

完整 `summary.json`：

```text
experiments/Goal15-p1-lateral-distill-20261107/p1b_teacher_summary.json
```

`summary.json` SHA256：

```text
3896865af49909c157c76a2c19b78959ba9cfc4c1c512bf8e0329d284394a77c
```

## 3. 分布审计

教师标签不是平滑连续分布，而是明显的三档结构：

| 统计 | 数值 |
|---|---:|
| 精确零修正 | 6,590 / 11,630（`56.66%`） |
| 至少一个轴达到 `0.10` cap | 4,547 / 11,630（`39.10%`） |
| 两个轴都达到 `0.10` cap | 3,131 / 11,630（`26.92%`） |
| delta norm `p50` | 0 |
| delta norm `p75` | `0.14142136` |
| release active transitions | 1,442 |
| 有 release 的 episodes | 88 / 96 |

最常见的教师输出是：

| delta `(dx, dy)` | transitions |
|---|---:|
| `(0, 0)` | 6,590 |
| `(-0.10, +0.10)` | 1,426 |
| `(-0.10, -0.10)` | 1,194 |
| `(+0.10, +0.10)` | 274 |
| `(+0.10, -0.10)` | 237 |

这意味着 A10 教师大量输出“零修正”或“单轴/双轴打满”。P1-C 必须同时报告 cosine、dominant-axis sign、magnitude ratio 和 cap 桶指标；只看整体 MAE 会把三档开关误判成连续横向控制。

分语言统计：

| language | transitions | episodes | zero | any-axis cap | release |
|---|---:|---:|---:|---:|---:|
| red mug / left plate | 3,657 | 27 | `58.82%` | `37.33%` | 27 |
| red mug / right plate | 4,521 | 34 | `57.93%` | `37.54%` | 33 |
| white mug / left plate | 3,452 | 35 | `52.72%` | `43.02%` | 28 |

三种训练语言都有标签，但每种语言的零修正和 cap 比例不同；P1-C 需要报告分语言指标，不能只看总体平均。

## 4. 协议偏差与限制

### 4.1 state source

本次 P1-B 使用的是：

```text
HDF5 expert states + cache timesteps
```

不是字面意义上的“用 base policy 闭环产生状态”。这样做是为了让教师标签与 `cache_train_v2` 的离线训练分布严格对齐。它不满足“base-policy closed-loop teacher”这个更强的说法：

- P1-B 可以称为 **offline teacher-label cache**；
- 不能称为 **closed-loop teacher rollout**；
- base-policy closed-loop 教师数据留到 P1-D 或其他独立实验验证；
- 任何后续结果都必须带上 `state_source=hdf5_expert_replay` 的边界。

### 4.2 release gate

协议正文只写了 release 正样本数 `>=50`。脚本同时检查 transitions 和 episodes，本次两项都为 88，因此阈值歧义没有影响本次裁决。后续若出现 transitions 达标但 episodes 不足，必须先修订协议，不能静默放宽。

### 4.3 `teacher_delta_chunk`

当前 `teacher_delta_chunk` 是把同一 A10 单步修正复制到 8 步，不是执行 8 步真实教师 rollout 的结果。P1-C 默认只使用 `teacher_delta_xy`；如果要使用 chunk，必须在报告中标记 `approximate_chunk=true`，不能把它当成真实教师轨迹。

## 5. 复现

远端输出：

```text
$W/results/K4_lite/T4b/p1_lateral_distill_20261107/p1b_teacher_full
```

运行命令：

```bash
$W/envs/oft/bin/python $R/scripts/build_p1_teacher_cache.py \
  --cache "$W/results/K4_lite/T4_residual/cache_train_v2" \
  --inventory "$W/results/K4_lite/T1_router/inventory_v1/inventory.json" \
  --manifest "$W/data/jgate_c1/counterfactual_20261027/manifest.jsonl" \
  --config "$R/configs/vla/k4_lite_t4b_v12_p0_authority_release.json" \
  --output "$OUT"
```

冻结输入 SHA256：

| 输入 | SHA256 |
|---|---|
| inventory | `23fca7dcdcb432f5d414fa076a43e1242ae03ffc9166383112bd61594dee0f0b` |
| P0 manifest | `e9564ca9aef800e618e3fdb5d144892eabe9471815bbd24bae3140583946f45e` |
| P0 config | `ecb501f86b4b795d94429cc17ce21bb2461014b64a54b3b006f2c9d344cef52d` |

## 6. 下一步

P1-C 已按协议完成并失败，结果见 `07_p1c_result.md`：

1. `R2-ROUTE` 没有相对 `R1-CAP` 产生增益，且失败 magnitude Gate；
2. P1-D 15-init 闭环不启动，不扩 30/50；
3. 若要继续 P1 线，只能先做新的、预注册的离线诊断（train-only magnitude calibration + 3 seeds），不能把本次 FAIL 改写成 PASS；
4. 任何论文近似组件仍不能升级。
