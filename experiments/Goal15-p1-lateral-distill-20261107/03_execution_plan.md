# P1 执行计划

**日期**：2026-11-07
**状态**：`P1_A_PASS_P1_B_PASS_P1_C_FAIL_3SEED_CONFIRMED`；P1-C v4 offline Gate 失败，3-seed 复现确认 `R2-ROUTE` 稳定劣于 `R1-CAP`，P1-D 不启动，不跑闭环

## 1. 总体顺序

```text
P1-A evidence adapter qualification
  -> P1-B teacher label cache
  -> P1-C offline student smoke/pilot
  -> P1-D 15-init closed-loop pilot
  -> 只有正信号才扩 30/50 + 3 seeds
```

任何一步 FAIL/INVALID 都停止后续步骤，不用“再调一点容量”跳过。

## 2. P1-A：evidence adapter qualification

### 2.1 目的

确认 `R2-PRED` 的部署输入真的可得，且不是把 GT 或旧 Qwen 资产偷渡进 J2。

### 2.2 选项

| 选项 | 条件 | arm 名 |
|---|---|---|
| J2 point head | point-in-mask、L1、覆盖率和 split 全过 | `R2-PRED` |
| route centroid evidence | 只使用冻结 route 权重；不读 GT | `R2-ROUTE` |
| 两者都不过 | 标 `INVALID` | 不训练 |

### 2.3 Gate

- [x] evidence 只使用部署时输入，不读取 GT object/plate/mask；
- [x] train/validation/rollout split 零重叠；
- [x] `white_right` 训练计数为 0；
- [x] point/region 覆盖率 `>=0.90`（route-centroid coverage = 1.000）；
- [x] 若使用 point head：point-in-mask `>=0.80`，destination L1 报告且不劣于旧 Qwen 资产的同口径基线；
- [x] 输出 SHA、config SHA、checkpoint SHA 全部记录。

**P1-A 裁决**：`PASS`。证据类型为 `route_centroid`，学生 arm 为 `R2-ROUTE`；未使用 J2 point head，不能命名为 `R2-PRED`。结果见 `05_p1a_evidence_result.md`。

### 2.4 失败处理

- 没有 J2 point head 且 route evidence 不可用：`INVALID`，先修 adapter；
- 旧 Qwen point head 不能直接接 J2：记录为 `not_compatible`，不算 P1 方法失败；
- 不允许用 GT mask/point 作为 `R2-PRED` 的 fallback。

## 3. P1-B：teacher label cache

### 3.1 目的

把 `A10` 的特权横向修正转成监督标签。

### 3.2 标签

```text
teacher_delta_xy = clip(A10_action_xy - base_action_xy, -0.10, 0.10)
teacher_release  = A10 release trigger at this step
```

- z/rotation 标签不产生；
- release 标签必须记录 `release_step`、`release_z`、`release_xy`；
- 同一状态要保留正确的 destination 和错误 destination 的 route pair；
- 标签分布必须覆盖 plateau（`0.027–0.035m`）和大修正桶（`0.05–0.10m`）；
- `white_right` 为 0。

P1-B 已有可用的 episode→HDF5 demo→84D init-state 映射：

```text
$W/results/K4_lite/T1_router/inventory_v1/inventory.json
```

该 inventory 覆盖 96 个 train episodes，init-state 与 validation/rollout 零重叠。P1-B 必须使用它重放，不得用 `target_action` 或 route centroid 冒充 A10 教师。

### 3.3 数据生成

初始只使用 `cache_train_v2` 的 96 个 training episodes。对每个 episode：

- 用 base policy 产生状态；
- 在可执行阶段应用 A10 教师；
- 记录 base action、teacher delta、release、route/point evidence、proprio；
- 保存 episode-level split。

**P1-B 实现记录（2026-11-07）**：本阶段实际使用 `HDF5 expert states + cache timesteps` 重放，以确保教师标签与 `cache_train_v2` 的离线训练分布严格对齐。它属于 `offline teacher-label cache`，不是 base-policy closed-loop teacher rollout；后者留到 P1-D 或独立实验验证。该偏差已写入 `06_p1b_teacher_result.md`，后续结果必须带 `state_source=hdf5_expert_replay`。

`teacher_delta_chunk` 是把同一单步教师修正复制到 8 步，不能当作真实 8-step teacher rollout；P1-C 默认只使用 `teacher_delta_xy`。

### 3.4 Gate

- [x] 96 个 training episodes 全部有标签，或缺失原因逐条记录；
- [x] `white_right` 计数为 0；
- [x] train/validation/rollout init overlap 为 0；
- [x] plateau 桶和大修正桶都非空；
- [x] release 正样本数 `>=50`，否则标 `INVALID`，不能判 release 学习失败；
- [x] 所有数组 finite，NaN 为 0；
- [x] cache manifest、行数、shape、SHA 全部记录。

**P1-B 裁决**：`PASS`（离线教师缓存）。96 episodes、11,630 transitions、release `88 transitions / 88 episodes`、`white_right=0`、plateau `37`、大修正桶 `262`、无生成错误。结果见 `06_p1b_teacher_result.md`。该 PASS 只解锁 P1-C，不解锁 P1-D 或方法论文。

**P1-C 裁决（2026-11-07）**：`FAIL`。commit `4763365`、config v3、2000-step pilot 下，`R2-ROUTE` 的 validation cosine `0.7036`、sign `0.7952`、magnitude `0.6734`，失败 magnitude Gate；`R1-CAP` 仍在 cosine/sign/MAE 上优于 `R2-ROUTE`，没有 evidence 增益。seed `27/37` 复现后，`R2-ROUTE` 在 `3/3` seeds 上 cosine/sign 均更低、MAE 更高，`H-DISTILL-NOT-SUPPORTED`。按停止条件，P1-D 不启动。结果见 `07_p1c_result.md` 和 `09_p1c_seed_robustness_result.md`。

## 4. P1-C：offline student training

### 4.1 Arms

| arm | evidence | 训练 |
|---|---|---|
| `R1-CAP` | 无图像/语言 evidence，只有 base action/proprio；参数量与 `R2` 对齐 | 500 smoke + 2000 pilot |
| `R2-ROUTE` | predicted route centroid + proprio + base action | 500 smoke + 2000 pilot |
| `R3-GEO` | oracle geometry | 诊断，同预算 |

`R0` 不训练，直接复用 P0 `B0` 的闭环基线。

### 4.2 训练约束

- 所有 arms 同初始化、同数据、同 batch、同 step、同 seed；
- `R1-CAP` 与 `R2` 参数数量差 `<=0.1%`；
- z/rotation 输出严格为 0（abs max `<=1e-8`）；
- residual xy abs max `<=0.10`；
- release head 只在训练/评测中学习，部署阈值冻结在 validation；
- 不用 `white_right` 调参；
- 训练日志必须打印 `R1-CAP` / `R2` 的实际参数量和采样权重。

### 4.3 Offline Gate

| 指标 | 通过条件 |
|---|---|
| correction cosine | `>=0.50` |
| dominant-axis sign accuracy | `>=0.75` |
| magnitude ratio median | `0.70–1.30` |
| plateau bucket | 不劣于 validation overall |
| large-correction bucket | 不劣于 validation overall |
| release precision | `>=0.70`，且报告 recall/F1 |
| release recall | `>=0.60` |
| z/rotation abs max | `<=1e-8` |
| residual xy abs max | `<=0.10` |
| NaN/Inf | `0` |

若 release 正样本 `<50`，release Gate 标 `INVALID`，不判方法失败。

### 4.4 停止条件

- smoke 不是 exact no-op at gate=0：停止；
- z/rotation 非零或 xy 越界：停止；
- loss/gradient 非 finite：停止；
- offline Gate FAIL：停止闭环，写失败 verdict；
- `R1-CAP` 与 `R2` 参数量不匹配：`INVALID`，不判 evidence 失败。

## 5. P1-D：15-init 闭环 pilot

### 5.1 设置

- 任务：C1 `LIVING_ROOM_SCENE5`；
- prompt：`put the white mug on the right plate`；
- 15-init manifest 与 P0 完全一致；
- 3 calibration + 12 confirmation 分开报告；
- seed 7；
- arms：`R0`、`R1-CAP`、`R2-ROUTE`、`R3-GEO`；
- release/settle 与 P0 一致：`xy=0.03`、`z=0.05`、streak `3`、settle `4`；
- `R0` 复用 `B0` 基线，不重新训练。

### 5.2 闭环 Gate

| 检查 | 通过条件 |
|---|---|
| `R2` official success | `>=3/15` |
| `R2` confirmation | `>=2/12` |
| `R2 - R1-CAP` | `>=2/15` |
| `R2` released success | `>=2/15` |
| seen group-any | `>=10/15` |
| wrong-destination | 不超过 `B0` |
| 稳定性 | 0 error、0 NaN、0 飞出、动作幅度不塌缩 |
| 契约 | manifest/checkpoint/config/evaluator SHA 一致 |

`R3-GEO` 只作诊断：若 `R3` 明显优于 `R2`，说明 predicted evidence 仍是瓶颈；不能把 `R3` 写成部署方法。

### 5.3 失败分支

| 观察 | 解释 | 下一步 |
|---|---|---|
| offline 好、闭环无提升 | 学生学会标签，执行/release/authority 不稳 | 查 release 和 authority，不扩网络 |
| offline 差 | evidence->action 映射不可学 | 改 evidence 或 subgoal，不调容量 |
| `R2` 不如 `R1-CAP` | 增益来自容量 | 判 `H-DISTILL` 不成立 |
| `R3` 明显优于 `R2` | predicted evidence 是瓶颈 | 回到 point/路由接口 |
| `R2 >=3/15` | 出现真实正信号 | 扩 30/50 + 3 seeds，再考虑 caddy |

## 6. 预算与资源

- 只使用 GPU1；
- P1-B 标签生成 96 episodes，预计与 P0 同量级或略短；
- P1-C smoke `500` steps，pilot `2000` steps；
- P1-D 15-init × 3 个训练 arm + `R0`，预计 1–2 小时；
- 任何阶段超过预估 2 倍仍无进展，先停下审计，不盲目加步数。

## 7. Git 流程

协议提交：

```bash
cd /home/bubble/类脑计算/VLM终局
git add experiments/Goal15-p1-lateral-distill-20261107/01_background.md \
        experiments/Goal15-p1-lateral-distill-20261107/02_tech_stack.md \
        experiments/Goal15-p1-lateral-distill-20261107/03_execution_plan.md \
        experiments/Goal15-p1-lateral-distill-20261107/04_run_contract.md
git diff --cached --check
git commit -m "docs(vla): freeze p1 lateral distillation protocol"
```

实现提交：

```bash
git add scripts/ configs/vla/ tests/ server/
git diff --cached --check
git commit -m "feat(vla): add p1 lateral distillation pipeline"
```

结果提交：

```bash
git add experiments/Goal15-p1-lateral-distill-20261107/
git diff --cached --check
git commit -m "results(vla): report p1 lateral distillation pilot"
```

每次只 add 本阶段相关文件；不使用 `git add -A`；不自动 push。
