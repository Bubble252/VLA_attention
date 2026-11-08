# Goal17 结果：Compositional grounding repair + simplified R-PRED

**日期**：2026-11-08<br>
**状态**：`PASS`（G-A PASS；G-B PASS；Track C 仍为诊断失败）<br>
**上游**：`05_e1_release_settle_amendment.md`、Goal16 `05_pilot_result.md`<br>
**远端结果**：`$W/results/K4_lite/T4b/goal17_compositional_grounding_20261108/`

## 1. 一句话结论

**2×2 同图 counterfactual 数据把 held-out `white_right` 的 destination grounding 从 0/15 修到 15/15；margin loss 不是必要条件。执行端原先 0/15 的主因是 release/settle 太短：settle 从 4 步加到 32 步后，E1-PRED 从 0/15 提升到 2/15，正式通过 G-B。E2-PRED（full 3D hand-back）为 3/15，仍是数值最好的 arm，但只比 E1 多 1/15，不能单 seed 下宣称 3D 显著更优。**

## 2. Run Contract

| 项 | 结果 |
|---|---|
| contract status | `PASS` |
| train manifest | 764 images × 4 prompts = 3,056 rows |
| val manifest | 118 images × 4 prompts = 472 rows |
| eval init | 15（3 calibration + 12 confirmation） |
| eval / validation in train | 0 |
| `white_right` train | 0（episode-held-out） |
| GPU | 只使用 GPU1 |
| training scope | 只训练 `PointQueryHead`；无 residual/route/action head |
| oracle fallback | disabled |

完整 SHA 见 `evidence/run_contract.json`。

## 3. G-A：Grounding 修复

| arm | track | destination <=0.15 m | object in mask | destination margin | object margin | mean destination error |
|---|---:|---:|---:|---:|---:|---:|
| `G3-BASE` | C | 0/15 | 0/15 | 0/15 | 0/15 | 0.590 m |
| `G4-NO-MARGIN` | D | **15/15** | **15/15** | **15/15** | 12/15 | 0.0320 m |
| `G4-MARGIN` | D | **15/15** | **15/15** | 14/15 | 12/15 | 0.0337 m |
| `G3-MARGIN` | C | 0/15 | 5/15 | 0/15 | 4/15 | 0.588 m |

**G-A PASS。**

### 3.1 关键发现：真正起作用的是 4-prompt 数据，不是 margin loss

- `G4-NO-MARGIN` 和 `G4-MARGIN` 都是 15/15 destination、15/15 object-in-mask；
- `G4-NO-MARGIN` 的 mean error 还略低（0.0320 vs 0.0337 m）；
- `G4-MARGIN` 的 destination margin 还少 1 个 init（14/15 vs 15/15）。

因此本轮没有证据支持“margin loss 是修复 binding 的必要条件”。更准确的因果表述是：

> **同图 2×2 prompt 覆盖（`red/white × left/right`）让 point head 见到了 held-out 组合所需的监督；margin loss 没有带来可检测的额外收益。**

### 3.2 Track C 仍然失败

`G3-MARGIN` 只训练 3 个 prompt（没有 `white_right`），destination 仍为 0/15。它说明：

- 没有 `white_right` 数据时，单靠 margin loss 不能教会组合；
- Track D 的 15/15 是 **deployable 接口修复**，不能写成“组合泛化已经解决”。

## 4. G-B：执行端

### 4.1 settle 长度是第一个执行瓶颈

| arm | settle steps | success | released success | hold-only | released placement | target dest correct | wrong dest |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B0` | - | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 | 14/15 |
| `E1-ORACLE` | 4 | 4/15 | 1/15 | 3/15 | 6/15 | 10/15 | 3/15 |
| `E1-PRED` | 4 | 0/15 | 0/15 | 0/15 | 12/15 | 10/15 | 3/15 |
| `E1-ORACLE` | 32 | 7/15 | 4/15 | 3/15 | 3/15 | 10/15 | 3/15 |
| `E1-PRED` | 32 | **2/15** | **2/15** | 0/15 | 10/15 | 10/15 | 3/15 |

settle=4 时，`E1-PRED` 已经有 12/15 released placement，但 0/15 official success。轨迹显示释放后物体 z 仍以每步约 0.002–0.005 m 下降，夹爪宽度也只从约 0.006 m 开到 0.021 m；4 步不足以让 simulator 的 success predicate 触发。

settle 校准只用 3 个 calibration init：

| settle steps | E1-PRED calibration success | released success |
|---:|---:|---:|
| 4 | 0/3 | 0/3 |
| 8 | 0/3 | 0/3 |
| 16 | 0/3 | 0/3 |
| 32 | 1/3 | 1/3 |

冻结 settle=32 后，15-init 结果为 2/15；其中 confirmation 12 个 init 为 1/12。**G-B PASS**，但 confirmation 增益很小，必须按单 seed 小样本结果报告。

### 4.2 E2：full 3D hand-back 仍然值得保留

| arm | success | released success | hold-only | released placement | rearm total | wrong dest |
|---|---:|---:|---:|---:|---:|---:|
| `E2-GEO` | 2/15 | 2/15 | 0/15 | 5/15 | 26 | 3/15 |
| `E2-PRED` | **3/15** | 2/15 | 1/15 | 5/15 | 26 | 3/15 |

按 Goal17 第 5 节规则，`E2-PRED (3/15) > E1-PRED-SETTLE32 (2/15)`，因此 **full 3D hand-back 保留为 best arm**；`E1-SETTLE32` 是更简单的 passing fallback。

但 3/15 与 2/15 只差 1 个 init，单 seed 下不能写成“3D 显著优于 E1”。正确表述是：

> **3D hand-back 仍是一个有效 arm，且数值最好；E1 + settle32 已经足以通过 Gate。两者差距在单 seed 噪声范围内。**

### 4.3 为什么 E1 这么依赖 settle

`G4-MARGIN` 的 destination mean error 是 0.0337 m，max 0.0781 m。E1 的释放门是 0.03 m，而且门是相对 **predicted destination** 而不是真实 plate 中心。预测误差 3 cm 叠加释放门 3 cm，物体可能离真实 plate 中心 6 cm；这仍满足 0.15 m placement 阈值，但不满足官方 success 的 0.03 m 精确条件。E2 的 policy hand-back 可以在释放前修正这部分 residual error，所以更容易把 predicted destination 转成成功。

## 5. 最终 Gate

| check | 结果 |
|---|---|
| G-A destination 12/15 | PASS（15/15） |
| G-A object in mask 12/15 | PASS（15/15） |
| G-A destination margin 12/15 | PASS（14/15） |
| G-A object margin 12/15 | PASS（12/15） |
| G-B `E1-PRED` success >=1 | PASS（2/15） |
| G-B `E1-PRED - B0 >=1` | PASS（+2） |
| G-B `E1-PRED` released success >=1 | PASS（2/15） |
| G-B wrong destination <= B0 | PASS（3 vs 14） |
| 0 error / 0 NaN / 0 flew-out | PASS |
| `E2-PRED` 同阈值审计 | PASS（3/15；3 vs 0；2/15；wrong 3） |

机器可读选择结果：`evidence/selection.json`。

## 6. 可以写与不能写

### 可以写

- 2×2 counterfactual 数据修复了 held-out `white_right` 的 deployable grounding；
- margin loss 不是本轮修复的必要条件；
- release settle 从 4 到 32 使 E1-PRED 从 0/15 到 2/15；
- E1-SETTLE32 和 E2-PRED 都通过预注册 G-B；
- E2-PRED 数值最好（3/15），但优势只有 1/15。

### 不能写

- 不能写“组合泛化已经解决”：Track C 仍是 0/15；
- 不能写“margin loss 已被证明有效”：no-margin baseline 一样 15/15；
- 不能写“3D hand-back 显著优于简化 E1”：3 vs 2，单 seed；
- 不能写“settle=32 是稳健超参数”：它是在 3 个 calibration init 上选的，confirmation 只有 1/12；
- 不能把 2/15 或 3/15 写成高成功率；它是一个正的、可复现的机制信号，不是可用产品。

## 7. 产物

本地：

```text
experiments/Goal17-compositional-grounding-20261108/05_e1_release_settle_amendment.md
experiments/Goal17-compositional-grounding-20261108/06_result.md
experiments/Goal17-compositional-grounding-20261108/evidence/
```

远端：

```text
$W/results/K4_lite/T4b/goal17_compositional_grounding_20261108/
  run_contract.json
  train/{G4-NO-MARGIN,G4-MARGIN,G3-MARGIN}/
  eval/grounding/{G3-BASE,G4-NO-MARGIN,G4-MARGIN,G3-MARGIN}/
  eval/e1/                         # settle=4
  eval/e1_settle_32/               # settle=32, G-B PASS
  eval/e2/                         # full 3D diagnostic
  eval/calib/*                     # calibration sweep
  eval/selection/                  # machine-readable selection
```

## 8. 复现命令

```bash
# 远端 GPU1
cd $R
SETTLE_STEPS=32 bash server/run_goal17_grounding.sh eval-e1-settle
bash server/run_goal17_grounding.sh eval-e2

python3 scripts/select_goal17_execution.py \
  --e1-baseline-summary "$OUT/eval/e1/summary.json" \
  --e1-settle-summary "$OUT/eval/e1_settle_32/summary.json" \
  --e2-summary "$OUT/eval/e2/summary.json" \
  --goal-config configs/vla/goal17_grounding.json \
  --output-dir "$OUT/eval/selection"
```

## 9. 下一步

1. **提高 destination 精度**：当前 mean error 3.4 cm，执行端需要约 3 cm 以内；先把 grounding 误差压到 `<0.015 m`，再重跑 E1。
2. **学习 residual 修正**：E2 的 hand-back 说明 policy 能吸收约 3 cm 的 grounding 误差；下一版可以让模型显式预测这个 residual，而不是依赖固定的 3D wrapper。
3. **多 seed**：至少 3 seeds 验证 `E1-SETTLE32` 与 `E2-PRED` 的 1/15 差异是否稳定。
4. **不要继续在 14/15 上找突破口**：当前任务的白盘 right 样本已经 15/15；应转向误差更大的 held-out 组合或新的低分任务。
