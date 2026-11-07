# 04 Stage 0 结果：数据审计与 D0 Run Contract

## 1. 结论

Stage 0 的数据契约已经建立：

- `goal13_audit.json`：`audit_status=PASS`；
- `D0_contract.json`：`contract_status=PASS`，`arm_status=PENDING_GATE`；
- `white_right` 训练计数为 0；
- train / validation / rollout 交集为 0；
- D0 实际 loader 长度为 `11630`，采样权重为 `1.0`，trajectory 数为 `96`。

这证明 D0 的数据、split、sampling weight、config SHA 和 remote code commit 可以对齐。它不代表 D0 方法有效，也不代表 G0 已全部完成。

## 2. 远端数据审计

审计脚本：`scripts/audit_goal13_rlds.py`

审计范围：

- 256 个 RLDS shard；
- 3,959 条原始 record；
- 218 条 Goal13 目标 episode；
- 使用 raw record SHA256 与 `c1_split_manifest_v1.json` 交叉核验。

语言计数：

| prompt | source episodes | train episodes |
|---|---:|---:|
| `red_left` | 37 | 27 |
| `red_right` | 44 | 34 |
| `white_left` | 45 | 35 |
| `white_right` | 0 | 0 |
| `yellow_and_white_right` | 48 | 0（D0 训练未加入） |
| `white_bowl_right` | 44 | 0（D0 训练未加入） |

split：

| split | episodes |
|---|---:|
| train | 96 |
| validation | 15 |
| rollout | 15 |

交集：

| pair | overlap |
|---|---:|
| train / validation | 0 |
| train / rollout | 0 |
| validation / rollout | 0 |

## 3. D0 实际 loader 契约

使用 `configs/vla/goal13_d0.yaml` 在远端的 `star105_dump_dataset_contract.py` 上实测：

| 项 | 值 |
|---|---:|
| data mix | `libero_c1_gate` |
| dataset | `libero_90_gate_lerobot` |
| raw steps | 12,302 |
| drop incomplete chunks 后 length | 11,630 |
| trajectories | 96 |
| sampling weight | 1.0 |
| effective batch | 4（4 × 1 × 1） |
| seed | 17 |
| remote code commit | `4507931a625c844404c4a76128fb116536d8ca7c` |
| config SHA256 | `61cbe071a8ca21d5e1d603b03c6a17e552a55ea00ca72aadc32133f12c23b252` |

## 4. Sidecar / HDF5 状态

远端检查结果：

- HDF5 只有 C1 三套：red-left、red-right、white-left；
- 没有找到 carrier / yellow-and-white / white-bowl 相关 HDF5 或 GT mask；
- 因此 carrier 侧只能标记 `weak_sidecar=true`；
- 后续 `I2` 若使用 carrier，只能走 P1 pseudo point，不能声称使用 GT。

原始检查输出见 [goal13_sidecar_check.txt](results/goal13_sidecar_check.txt)。

## 5. G0 状态

| 检查 | 状态 |
|---|---|
| C1 三 prompt 计数 37 / 44 / 45 | PASS |
| `white_right` 训练计数 0 | PASS |
| carrier 至少 40 episodes | PASS（48） |
| train/validation/rollout 交集 0 | PASS |
| 实际 sampling weights 1.0 | PASS |
| HDF5 / sidecar 状态明确 | PASS（carrier 缺失，标记 weak sidecar） |
| 100 帧语言与画面人工抽检 | PASS |

因此 G0 的数据部分已经 PASS。carrier 侧没有 GT mask，仍保留 `weak_sidecar=true`。

## 6. 100 帧视觉抽检

抽检方式：C1 三任务各取 17 个 demo，每个 demo 取首帧和末帧，共 100 帧。

证据：

- contact sheet：[goal13_g0_contact_sheet.png](results/goal13_g0_contact_sheet.png)；
- 采样清单：[goal13_g0_visual_sample.json](results/goal13_g0_visual_sample.json)；
- 放大对照：[goal13_g0_task_panels.png](results/goal13_g0_task_panels.png)；
- 自动一致性检查：[goal13_g0_placement_check.json](results/goal13_g0_placement_check.json)。

最终帧红色杯像素质心 x 中位数：

| task | red centroid x median | 解释 |
|---|---:|---|
| `red_left` | 0.319 | 红杯最终在画面左侧 |
| `red_right` | 0.885 | 红杯最终在画面右侧 |
| `white_left` | 0.636 | 红杯不是被移动对象，位置与红杯任务不同 |

`red_left` 与 `red_right` 的最终红杯位置方向相反，和语言一致。100 帧抽检 PASS。

## 7. 下一步

1. D0 2-step training smoke 已跑通并通过 postrun contract，见 [05_d0_smoke_result.md](05_d0_smoke_result.md)；
2. D0 offline `limit=2` eval smoke 已通过，见 [06_d0_offline_smoke_result.md](06_d0_offline_smoke_result.md)；
3. 跑 D0 matched 2.5K training 和 15 条闭环 pilot；
4. 为 D1 carrier 转换 LeRobot 数据并生成 observed dataset contract；
5. 只有 D0 复现或解释 `white_right=0/15` 后，才比较 D1；
6. 不启动 A2，不合并 F0。
