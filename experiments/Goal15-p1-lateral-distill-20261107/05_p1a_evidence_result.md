# P1-A：route-centroid evidence qualification

**日期**：2026-11-07
**状态**：`P1-A PASS`
**裁决**：允许使用冻结 T1 route centroid 作为 P1 的可部署 evidence，但学生 arm 必须命名为 `R2-ROUTE`，不得写成 `R2-PRED`。P1-A 通过只解锁 P1-B，不解锁训练或闭环。

## 1. 结论

`cache_train_v2` 和 `cache_validation_v2` 中的 6D `geometry` 满足 P1-A 的最低证据要求：

- 只来自冻结 T1 route 权重和 J2 visual tokens 的二维 token grid；
- 不读取 GT mask、object id、task id、success 或仿真坐标；
- train/validation/rollout 的 episode ID 与 init-state SHA 均零重叠；
- `white_right` 在训练和 validation 中的计数均为 0；
- 训练与 validation 的 geometry 全部 finite，非零修正覆盖率为 `1.0`。

P0 的 15 个 `white_right` init 只与 rollout 来源重合，与 train/validation 的重合数均为 0。因此 P0 评估集不会污染 P1 训练或 validation。

## 2. 数据

| split | transitions | episodes | `white_right` | coverage | source/target in grid |
|---|---:|---:|---:|---:|---:|
| train | 11,630 | 96 | 0 | 1.000 | 1.000 |
| validation | 1,824 | 15 | 0 | 1.000 | 1.000 |

Geometry 是 `[source_x, source_y, target_x, target_y, dx, dy]`，坐标范围为归一化二维图像网格 `[-1, 1]`。`dx/dy` 可以超出 `[-1, 1]`，这是两端点差值，不是坐标越界。

## 3. 隔离审计

| 检查 | 结果 |
|---|---|
| train cache 与 inventory train 的 episode 集合一致 | PASS |
| validation cache 与 inventory validation 的 episode 集合一致 | PASS |
| train/validation/rollout episode 两两零重叠 | PASS |
| train/validation/rollout init-state SHA 两两零重叠 | PASS |
| P0 `white_right` init 与 train 零重叠 | PASS（0） |
| P0 `white_right` init 与 validation 零重叠 | PASS（0） |
| P0 `white_right` init 与 rollout 重合 | PASS（15；符合预期） |

## 4. 证据边界

- P1-A 选择的是 `route_centroid`，不是 J2 point head，因此没有 `point_in_mask` 指标；不能声称 `R2-PRED` 已就绪。
- 旧 Qwen/starVLA point head 没有接入 J2，仍标记为 `not_compatible`；本结果不改变这个边界。
- geometry 是二维 token-grid 证据，不是完整三维物体位姿；它只能支持横向 `xy` 修正和 release 的输入，不应被写成三维空间 grounding。
- 该 PASS 只说明 evidence adapter 可用，不说明 student 能学会教师方向，也不说明闭环会成功。

## 5. 复现

本地审计脚本：

```text
scripts/audit_p1_evidence_adapter.py
```

远端输出：

```text
$W/results/K4_lite/T4b/p1_lateral_distill_20261107/p1a_evidence/audit_p1_evidence_adapter.json
```

本地归档：

```text
experiments/Goal15-p1-lateral-distill-20261107/p1a_evidence_audit.json
```

归档 SHA256：

```text
113edaa05745f3a95741a226259013357e1f3a29b57a0aa6ac790fc268594206
```

冻结输入 SHA256：

| 输入 | SHA256 |
|---|---|
| train cache summary | `4aaf53f620ab1166b1dd3b3e816834e07a89edfd7092b508caeab3f83d694b8f` |
| validation cache summary | `a887bebd64cb28023e52b38a6cff411524ed30ece8181ea686f463a2d0d1d91b` |
| inventory | `23fca7dcdcb432f5d414fa076a43e1242ae03ffc9166383112bd61594dee0f0b` |
| P0 manifest | `e9564ca9aef800e618e3fdb5d144892eabe9471815bbd24bae3140583946f45e` |
| router | `d425a0e7a2297633b1bb9d2c79f706989c16705daabb7e99a6e9013fe52e8c06` |
| J2 checkpoint fingerprint | `747a28690383ca45e795518121e9aed3f4610265c14618a69f203922b4238e20` |

## 6. 下一步

P1-A 已通过，按协议进入 **P1-B teacher label cache**：

1. 用 `T1_router/inventory_v1/inventory.json` 的 96 个 training episodes 重放到 LIBERO；
2. 用 `A10` 生成 `teacher_delta_xy` 和 release 标签；
3. 验证 release 正样本 `>=50`、plateau/大修正桶非空、`white_right=0`；
4. P1-B 通过后才允许写 P1-C offline student。
