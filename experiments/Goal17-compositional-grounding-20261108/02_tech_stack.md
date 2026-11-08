# Goal17 技术栈

**日期**：2026-11-08<br>
**状态**：`DRAFT`

## 1. 复用资产

| 资产 | 用途 |
|---|---|
| `PointQueryHead` | object/destination 2D 点预测 |
| `extract_c1_point_features.py` | 冻结 Qwen 特征提取 |
| `train_c1_point_head.py` | point head 训练与评估 |
| `reguide_rebinding.py` | 简化 xy/release wrapper |
| Goal16 manifest / mask | 训练标签与 eval 对照 |

## 2. 新增

```text
scripts/build_goal17_counterfactual_features.py
scripts/train_goal17_grounding.py
scripts/audit_goal17_rpred.py
configs/vla/goal17_grounding.json
tests/test_goal17_grounding.py
server/run_goal17_grounding.sh
```

## 3. 数据

- 训练帧只来自 train split；
- 每帧生成 4 个 prompt：`red_left`、`red_right`、`white_left`、`white_right`；
- object / destination 标签来自已有 mask 或几何；
- 15 个 `white_right` eval init 不得进入训练。

## 4. 损失

```text
L = L_point_l1
  + λ_mask * L_in_mask
  + λ_dest * L_destination_margin
  + λ_obj  * L_object_margin
```

- `L_destination_margin`：同图左右 prompt 的 destination 必须分开；
- `L_object_margin`：同图红/白 prompt 的 object 必须分开；
- 必须跑 `λ=0` 的 no-margin baseline。

## 5. 评测

- Track D：4 prompt 训练，15 个 eval init episode-held-out；
- Track C：3 prompt 训练，`white_right` prompt-held-out（诊断）；
- 指标：destination error、object in-mask、counterfactual margin、official success、released success、wrong-destination。
