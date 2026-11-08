# Goal18 技术栈

**日期**：2026-11-08<br>
**状态**：`DRAFT`

## 1. 复用资产

| 资产 | 用途 |
|---|---|
| Goal17 `G4-MARGIN` point head | 冻结的 destination/object 2D 预测 |
| Goal17 4-prompt feature index | train/val 特征与点标签 |
| `build_goal16_grounding_sidecar.py` | 相机反投影、point head 加载 |
| `evaluate_goal17_grounding.py` | 现有 15-init grounding 评分口径 |
| `audit_goal17_rpred.py` | 现有 E1/B0 执行评测 |
| Goal17 `grounding_rows.jsonl` | C0 的 base predictions |

## 2. 新增

```text
scripts/train_goal18_precision.py
scripts/evaluate_goal18_grounding.py
scripts/audit_goal18_execution.py
configs/vla/goal18_precision.json
server/run_goal18_precision.sh
tests/test_goal18_precision.py
```

## 3. C1：offset calibration

- 输入：frozen point head 在 train split 上的 destination prediction；
- 标签：train split 的 destination mask 点标签；
- 输出：global / per-prompt / per-relation 的 2D offset；
- 只在 train split 拟合，eval 只做最终测量。

## 4. C2：residual head

- 输入：visual mean、language mean、object/destination point、prompt id；
- 输出：destination 的桌面平面 `dx, dy`；
- 目标：`true_xy - predicted_xy`；
- 小 MLP，固定步数，保存 checkpoint 和 val 误差。

## 5. 评测

- G-P：mean error、max error、`<=0.03 m` 数量、object in-mask；
- G-E：E1 success、released success、correct-destination subset success、wrong destination；
- G-S：最终 arm 3 seeds；
- G-C：Track C 诊断，不参与主 Gate。
