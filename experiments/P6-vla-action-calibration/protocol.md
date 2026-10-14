# P6.1 action calibration smoke protocol

P6 的 B1 是当前最强 offline action baseline，但后续 rollout 与 expert-state query 暴露出 action 基础问题：B1 在 expert-driven states 上仍与 expert action 不匹配，尤其 rotation 维度明显偏弱。因此 P6.1 的目标不是引入 semantic loss，而是先修复 action supervision / calibration。

## 状态

- [x] P6 B1 held-out offline action prediction 完成。
- [x] LIBERO rollout smoke 完成但 success=0。
- [x] expert replay 正控成功。
- [x] scalar EEF action scaling 1.5x/2.0x 无效。
- [x] expert-state policy query 完成，确认 B1 在 expert states 上也存在 action mismatch。
- [x] action calibration audit 完成，rotation 是最弱点。
- [x] 实现 P6.1 weighted-action smoke。
- [x] 完成 100-step smoke。
- [x] held-out eval 未通过 gate；不进入 expert-state query / one-episode rollout smoke。

## 依据

当前 action calibration audit：

- Translation magnitude ratio B1/expert: `0.759`。
- Rotation magnitude ratio B1/expert: `0.281`。
- rot_x pred/expert: `0.033`。
- rot_y pred/expert: `0.096`。
- rot_z pred/expert: `0.848`。

这说明 rotation 并不是均匀偏小，主要问题集中在 rot_x/rot_y。训练统计也显示 rotation 维度 std 远小于 translation，而 OFT 原生 loss 是全维 mean L1，容易让小幅 rotation 维度在总 loss 中被 translation/gripper 掩盖。

## P6.1 首轮实验设计

base：P6 `B1_seed17/checkpoint_10080`。

只改 action supervision，不加入 semantic loss：

```text
L = mean( w_d * |a_pred[d] - a_target[d]| ) + lambda_retention * L_retention
```

首轮权重建议：

```text
w = [1, 1, 1, 3, 3, 2, 1]
```

解释：rot_x/rot_y 最弱，先给 3x；rot_z 较接近，但仍给 2x；translation/gripper 保持 1x，避免破坏 B1 已有 offline baseline。

训练规模：

- 先 100--500 optimizer steps；
- batch 仍按 P6 设置，避免显存变化；
- seed=17；
- 不覆盖 P6 checkpoint；
- 输出目录独立为 `results/P6_vla_action_calibration/`。

## 必须通过的 gate

1. action loss finite；
2. checkpoint restore OK；
3. held-out per-dim action eval 不显著恶化 translation/gripper；
4. expert-state query 的 rotation ratio 明显改善；
5. 若 1--4 通过，再跑 B1/P6.1 one-episode rollout smoke。

## 不允许的解释

- P6.1 不是我们的语义归因主方法；
- P6.1 只是 action 基础校准；
- 若 P6.1 有改善，只能说明原 B1 action head/calibration 是 bottleneck；
- P7-lite semantic repair 必须在 action calibration gate 后再解释。

## 下一步命令边界

101 恢复且 GPU0 空闲时：

1. weighted-action loss smoke 已实现；
2. 100-step smoke 已完成；
3. held-out action per-dim 已评估；
4. 本轮 held-out gate 未通过，不运行 policy-on-expert-state query；
5. 不运行 one-episode rollout smoke，除非后续 ablation 先通过 held-out gate。


## 2026-10-14 P6.1 首轮结果与 gate

远端训练与评估路径：

- 训练：`/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_action_calibration/p61_weighted_l1_seed17_100step_20261014_042923`
- held-out eval：`/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_action_calibration/heldout_eval_20261014_043503`
- 本地汇总：`experiment_workspace/vla_remote_results/20261014_p61_action_calibration/summary.md`

P6.1 从 B1 checkpoint 出发，使用权重 `[1,1,1,3,3,2,1]` 做 100-step weighted L1 calibration。训练本身通过，checkpoint restore 文件完整；但 held-out offline action eval 没有通过 pre-rollout gate。

| metric | B1 reference | P6.1 weighted L1 | delta | better? |
|---|---:|---:|---:|---|
| action_mae | 0.293686 | 0.303364 | +0.009677 | no |
| translation_mae | 0.377516 | 0.396345 | +0.018829 | no |
| rotation_mae | 0.262067 | 0.267450 | +0.005384 | no |
| gripper_mae | 0.137056 | 0.132161 | -0.004895 | yes |
| gripper_accuracy_sign_normalized | 0.786573 | 0.853056 | +0.066482 | yes |

结论：P6.1 提升了 gripper sign accuracy，但损伤了 action/translation/rotation 主误差；这不满足“held-out per-dim action eval 不显著恶化 translation/gripper”的 gate，也不满足 rotation 改善目标。因此本轮不进入 expert-state query，不进入 one-episode rollout，更不能进入正式 benchmark。

下一步只允许做更小的 action calibration ablation，例如降低 rotation 权重、只调 gripper/rotation head、冻结更多模块或缩短/降低学习率；每个 ablation 仍必须先过 held-out offline gate，再考虑 expert-state query。

## 2026-10-14 P6.1a 更温和 ablation 与 gate

P6.1 首轮 `[1,1,1,3,3,2,1]` 说明“直接强拉 rotation 权重”会损伤主动作误差。P6.1a 因此只测试一个更温和版本：从同一个 B1 checkpoint 出发，使用 weights=`[1,1,1,1.5,1.5,1.2,1]`、learning rate=`3e-5`、100 steps。该实验仍然只属于 action calibration 诊断，不加入 semantic loss，不覆盖 P6/P6.1 结果。

执行记录：首次 launch 错误引用 `offline_train.jsonl` 失败；正确 manifest 为 `train.jsonl`，SHA256=`695681afb03012bd18c34d09c9a4fbf15ca489605d4463cbf891cd7213bca921`，修正后训练与 checkpoint restore 通过。

| metric | B1 reference | P6.1 weighted | Δ P6.1 | P6.1a lowrot/lowlr | Δ P6.1a |
|---|---:|---:|---:|---:|---:|
| action_mae | 0.293686 | 0.303364 | +0.009677 | 0.293306 | -0.000380 |
| translation_mae | 0.377516 | 0.396345 | +0.018829 | 0.376033 | -0.001483 |
| rotation_mae | 0.262067 | 0.267450 | +0.005384 | 0.261234 | -0.000833 |
| gripper_mae | 0.137056 | 0.132161 | -0.004895 | 0.141341 | +0.004285 |
| gripper_accuracy_sign_normalized | 0.786573 | 0.853056 | +0.066482 | 0.802503 | +0.015930 |

结论：P6.1a 不是强结果。它只说明“降低 rotation 权重 + 降低学习率”避免了 P6.1 对 action/translation/rotation 的明显破坏，并带来极小主误差改善；但 gripper MAE 变差，gripper sign accuracy 的改善也远弱于 P6.1。因此 P6.1a 不能直接作为 rollout candidate 或论文方法主张。

下一 gate：只允许先跑 P6.1a 的 policy-on-expert-state query，比较 expert-state 上的 translation/rotation magnitude ratio、per-dim error 和 gripper phase。只有 expert-state 也改善，才允许 one-episode rollout smoke；否则停止该 calibration 分支，改查 action normalization、rotation representation、action head / proprio projector 结构。

本地汇总：`experiment_workspace/vla_remote_results/20261014_p61a_lowrot_lowlr/summary.md`。

## 2026-10-14 P6.1a expert-state query gate

按照 P6.1a held-out gate，继续做了 policy-on-expert-state query：环境由 expert action 推进，模型只在 expert-driven states 上被查询，模型动作不执行，因此该结果不是 rollout benchmark。

| metric | B1 | P6.1a | Δ |
|---|---:|---:|---:|
| model processed trans L2 mean | 0.605427 | 0.615595 | +0.010168 |
| model processed rot L2 mean | 0.013138 | 0.013062 | -0.000076 |
| processed-vs-expert trans diff L2 mean | 0.570370 | 0.563464 | -0.006906 |
| processed-vs-expert rot diff L2 mean | 0.038765 | 0.038335 | -0.000429 |
| processed-vs-expert gripper abs diff mean | 0.120000 | 0.120000 | +0.000000 |

Expert trans L2 mean 为 `0.797271`，expert rot L2 mean 为 `0.046681`。P6.1a 虽然让 translation diff 和 rotation diff 略降，但没有解决 rotation magnitude bottleneck：模型 rotation L2 仍约 `0.013`，远低于 expert `0.0467`，且比 B1 略低。因此 P6.1a **No-Go for rollout**。

行动边界：不运行 P6.1a one-episode rollout，不进入正式 benchmark；停止当前 weighted-loss calibration 分支。下一步转向 action target normalization、rotation representation、action head/proprio projector 和 task-stage supervision 审计。


## 2026-10-14 action target / normalization audit

After P6.1 and P6.1a failed the rollout gate, we ran a read-only audit before any additional training.

Artifacts:

- Script: `scripts/audit_oft_action_targets.py`
- Remote result: `/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_action_calibration/action_target_audit_20261014_054039/`
- Local mirror: `experiment_workspace/vla_remote_results/20261014_action_target_audit/`

Coverage:

- Train: 27 episodes, 2709 steps, 2520 H=8 chunks, 0 failures.
- Eval: 9 episodes, 832 steps, 769 H=8 chunks, 0 failures.
- The script reproduces the P6 target transform: `target_gripper = 1 - clip(raw_gripper,0,1)` and bounds-q99 normalization.

Result:

- Normalized translation target mean_abs avg: 0.4645.
- Normalized rotation target mean_abs avg: 0.2839.
- Rotation normalized saturation is only about 0.021 per rotation dimension.
- Rotation near-zero<=0.05 is about 0.066–0.079.
- Eval target distribution is close to train.

Interpretation:

The action target distribution does not support the hypothesis that rollout failure is mainly caused by rotation targets being too small after normalization. Rotation has sufficient normalized amplitude. The stronger hypothesis is now action-head/chunk/interface mismatch: the model under-produces rotation in expert-state query even though the supervised target distribution contains non-trivial rotation.

Protocol update:

- Keep P6.1/P6.1a as diagnostic No-Go.
- Do not start formal rollout benchmark from these checkpoints.
- Next repair should audit or minimally modify the action head/chunk/interface path before semantic P7 training is interpreted as a VLA method improvement.

## 2026-10-14 action-head / rollout-interface audit

After target normalization was ruled out as the main explanation, we ran a read-only source-and-result audit of the OpenVLA/OFT action path.

Artifacts:

- Script: `scripts/audit_oft_action_head_interface.py`
- Remote result: `/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_action_calibration/action_head_interface_audit_20261014_055847/`
- Local mirror: `experiment_workspace/vla_remote_results/20261014_action_head_interface_audit/`
- OpenVLA-OFT commit: `e4287e94541f459edc4feabc4e181f537cd569a8`

Verified contracts:

- The native action head predicts an `H=8`, `D=7` action chunk.
- The native continuous objective is mean L1 over the full `B×8×7` chunk.
- Official rollout executes chunked actions by default.
- Official LIBERO postprocess binarizes gripper from `[0,1]` to `[-1,+1]` and flips sign for OpenVLA.

Result summary:

| metric | B1 | P6.1a | delta |
|---|---:|---:|---:|
| action_mae | 0.293686 | 0.293306 | -0.000380 |
| translation_mae | 0.377516 | 0.376033 | -0.001483 |
| rotation_mae | 0.262067 | 0.261234 | -0.000833 |
| gripper_mae | 0.137056 | 0.141341 | +0.004285 |
| expert-state trans_ratio | 0.759374 | 0.772128 | +0.012754 |
| expert-state rot_ratio | 0.281449 | 0.279822 | -0.001627 |

Interpretation:

The audit makes the current failure mode more specific. Rotation targets are non-trivial after normalization, but the policy still under-produces rotation on expert-driven states. P6.1a changes the modules and slightly improves held-out MAE, but it does not repair expert-state rotation magnitude. The full-chunk mean L1 objective may hide first-step or phase-specific execution errors when the rollout controller consumes chunked actions.

Protocol consequence:

Do not run formal LIBERO/LIBERO-Plus/RoboTwin benchmark from P6.1/P6.1a. The next experiment must be an action-head/interface gate: per-horizon loss, first-action loss, horizon-wise output distribution, and raw/processed/env-action consistency. Semantic P7 loss remains offline-only until B0/B1 action-interface smoke is repaired.

## 2026-10-14 horizon × dimension diagnostic

We added a read-only diagnostic to decompose offline action error by action horizon and dimension.

Artifacts:

- Script: `scripts/audit_oft_horizon_dim_errors.py`
- Remote result: `/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_action_calibration/horizon_dim_audit_20261014_060808/`
- Local mirror: `experiment_workspace/vla_remote_results/20261014_horizon_dim_audit/`

Result:

| group | action MAE | first-horizon MAE | last-horizon MAE | translation MAE | rotation MAE | gripper MAE |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 0.293686 | 0.292031 | 0.301750 | 0.377516 | 0.262067 | 0.137056 |
| P6.1a | 0.293306 | 0.291071 | 0.302049 | 0.376033 | 0.261234 | 0.141341 |

The first-horizon improvement from P6.1a is only `-0.000960`, while the last horizon slightly worsens. Horizon × dimension statistics show persistent under-production in rotation, especially `rot_y` and `rot_z`. At horizon 0, B1's pred_abs_to_target_abs ratios for `rot_x/rot_y/rot_z` are `0.477/0.181/0.338`; P6.1a raises them only to `0.512/0.206/0.358`.

Consequence:

P6.1a remains No-Go for rollout. The next repair should explicitly target horizon/dimension behavior, for example first-action/horizon-aware loss or a narrower action-head/proprio-projector smoke. Any repair must pass held-out offline metrics and expert-state query before rollout.

## 2026-10-14：P6.2 horizon-weighted action-head/proprio repair smoke

本轮在 P6.1/P6.1a、target normalization audit、action-head/interface audit 和 horizon×dimension audit 之后，执行了一个更窄的 action-head/interface 修复实验。P6.2 从 P6 B1 `checkpoint_10080` 出发，冻结 policy LoRA，只训练 action head 与 proprio projector，使用 native `H=8, D=7` chunk 的 horizon-weighted action L1。

远程与本地产物：

- 远程训练目录：`/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_action_calibration/p62_horizon_headproprio_seed17_150step_20261014_063738/`
- 远程 eval：`/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_action_calibration/p62_horizon_headproprio_seed17_150step_20261014_063738/heldout_eval_20261014_064305/`
- 本地镜像：`experiment_workspace/vla_remote_results/20261014_p62_horizon_headproprio/`
- 本地 summary：`experiment_workspace/vla_remote_results/20261014_p62_horizon_headproprio/summary.md`

配置：steps=`150`，lr=`3e-5`，action weights=`[1,1,1,2,3,2,1]`，horizon weights=`[2,1.5,1.25,1,1,0.8,0.8,0.8]`。训练通过 `P62_HORIZON_ACTION_REPAIR_SMOKE_PASSED`，checkpoint restore 文件已生成；该实验不是 rollout benchmark。

同口径 held-out offline action prediction：

| group | action MAE | first-horizon MAE | last-horizon MAE | translation MAE | rotation MAE | gripper MAE | gripper sign acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 reference | 0.293686 | 0.292031 | 0.301750 | 0.377516 | 0.262067 | 0.137056 | 0.786573 |
| P6.1 weighted L1 | 0.303364 | 0.303968 | 0.311687 | 0.396345 | 0.267450 | 0.132161 | 0.853056 |
| P6.1a lowrot/lowlr | 0.293306 | 0.291071 | 0.302048 | 0.376033 | 0.261234 | 0.141341 | 0.802503 |
| P6.2 head+proprio | 0.293562 | 0.290828 | 0.302845 | 0.378003 | 0.261015 | 0.137880 | 0.844603 |

P6.2 相对 B1：action MAE `-0.000124`，first-horizon MAE `-0.001203`，rotation MAE `-0.001052`，但 translation MAE `+0.000487`，gripper MAE `+0.000824`。因此 P6.2 是弱改善，不是 action-interface 修复完成的证据。

Gate 判断：**No-Go for formal rollout / benchmark**。P6.2 可以作为“只调 action head/proprio 有微弱可动性”的诊断证据，但不能解锁 LIBERO/LIBERO-Plus/RoboTwin 正式 benchmark。下一步仍应做 P6.3 action-head/interface repair 或 expert-state query 复核；P7-lite ranking/KL 只能作为 offline mechanism repair 继续推进，不能替代 action 基础问题。

### 2026-10-14 VLA positive-control checkpoint repair 与 evaluator preflight

当前 VLA 主线已切回 positive control：先确认官方/公开 OpenVLA-OFT checkpoint 能在 LIBERO `libero_spatial` task0 上完成最小 rollout，再决定是否继续 P6.3/P7-lite。101 SSH 已恢复。公开 checkpoint 原有三个 shard 不完整，其中 shard 1 已通过 mirror-first 续传完成并验证 SHA256；shard 2 正在续传，shard 3 尚未开始。完整 hash verified 前不运行 rollout。

官方 evaluator 必须使用 workspace 的 OFT 环境 `/vepfs-mlp2/c20250405/400040/transfer/vla_attention/vla_workspace/envs/oft/bin/python`，并设置 `LIBERO_CONFIG_PATH=/vepfs-mlp2/c20250405/400040/transfer/vla_attention/vla_workspace/libero_config`。`/root/starvla_cu124` 缺 `draccus`，不能作为官方 evaluator 环境。preflight 已确认 OpenVLA-OFT commit `e4287e94541f459edc4feabc4e181f537cd569a8`、LIBERO commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`，以及官方常量 `H=8, D=7, proprio=8, bounds_q99`。

首轮 task 固定为 `libero_spatial` task0：`pick up the black bowl between the plate and the ramekin and place it on the plate`，默认 init states `(50, 92)`，SHA256 `f061b67788d541001f3d6872be9a3a3b0b7de3490dac4683e181f9b0703f8e20`。positive control 至少 5 episodes；若仍为 0 success，按 evaluator wrapper、dataset statistics、action normalization、gripper convention、camera、task/init state、chunk execution 的顺序排查。

### 2026-10-14 官方 OpenVLA-OFT/LIBERO task0 positive control 通过

公开 `moojink/openvla-7b-oft-finetuned-libero-spatial` checkpoint 已修复完整，并用官方 OpenVLA-OFT evaluator 函数在 `libero_spatial` task0 上完成 5-episode positive control。结果为 `5/5` success，`success_rate=1.0`，远端目录为 `/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/positive_control_moojink_task0_20261014/`。

该结果改变 VLA 诊断边界：LIBERO 环境、官方 evaluator、默认 init state、agent/wrist camera、`H=8,D=7` chunk、`bounds_q99` unnormalization 和 gripper/action processing 在官方 checkpoint 上可成功。因此，P6 B0/B1/B3/B4 rollout smoke 的 0 success 不能再归因于基础管线整体不可用；后续应对比我们的 checkpoint restore、训练后 action head/proprio projector、normalization key、action magnitude、rotation under-production 和 rollout wrapper 差异。下一步仍按固定顺序做 expert action replay 与 tiny overfit，不直接跳到 P6.3/P7-lite。

### 2026-10-14 expert action replay positive control 复核

官方 checkpoint positive control 通过后，已复核 LIBERO task0 expert action replay。远端结果为 `/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_action_calibration/expert_replay_positive_control_20261014_after_moojink/`，3 条 demo × init 0/1/2 × `raw/train01_to_env` 共 18 trials，12 success，summary SHA256 `8680f9fd38d0cddf8624f8dfc0d4c759820f6bd128652165ff3b607340b35731`。

结论是：raw expert action 与 train01→env action 在可匹配 init 上可成功，gripper/sign 转换不是主要断点；但 expert action 不是任意 init 通用，跨 init 复用会失败。因此下一步 tiny overfit 必须使用 replay 成功的同 init demo，并把 rollout 回同一 init-state 作为 gate。

### 2026-10-14 tiny-overfit head+proprio 结果与同 init rollout gate

基于 expert replay 成功记录构建了 3 条 task0 episode 的 tiny manifest，SHA256 `9a2bd44eb0c3a1d51db2bf4c89be0ad05fb9c1d9265e0258f4ece1e08e2f4d1b`。从 P6 B1 出发，只训练 action head + proprio projector，300 steps，lr `1e-4`，训练通过，远端目录为 `/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_action_calibration/tiny_overfit_headproprio_seed17_300step_20261014_retry2/`。

offline tiny eval 有明确改善：action MAE `0.298864 -> 0.284692`，first-horizon MAE `0.288046 -> 0.282167`，rotation MAE `0.250380 -> 0.233718`，gripper sign acc `0.712862 -> 0.853714`。但 same-init rollout smoke 仍为 `0/1`，目录为 `/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_action_calibration/tiny_overfit_same_init_rollout_20261014/`。

该结果说明只调 action head/proprio 能改善 normalized offline prediction，但不足以修复 closed-loop state drift。VLA 主线仍应继续 action-interface repair，优先测试解冻 policy LoRA + action head + proprio 的 tiny-overfit，或对比官方 moojink checkpoint 的 action traces；semantic/P7 不应被恢复为闭环成功主张。

## 2026-10-14：tiny-overfit policy LoRA + head+proprio 启动

- [x] head+proprio-only tiny-overfit 的 `final.json` 已补充远端 `final_corrected.json`，修正原 `final.json` 中保留 `.tmp` checkpoint 路径的问题；实际 checkpoint 为 `/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_action_calibration/tiny_overfit_headproprio_seed17_300step_20261014_retry2/checkpoint_00300`。
- [x] 已确认标准 P1 provenance 文件真实路径为 `/vepfs-mlp2/c20250405/400040/transfer/vla_attention/repo/VLA_attention/experiment_workspace/results/P6_vla_b0_b4/p1_interface/p1_interface.json`，不是先前误写的 `results/P6_vla_b0_b4/p1_interface/report.json`。该文件记录 OpenVLA revision `47a0ec7fc4ec123775a391911046cf33cf9ed83f`、action shape `[1,8,7]`、normalization round-trip、module restore、visual grid 和 gradient finite/nonzero。
- [x] 已启动下一轮 tiny-overfit：从 B1 checkpoint 出发，解冻 policy LoRA + action head + proprio projector，300 steps，lr `5e-5`，仍使用同一个 3-episode tiny manifest，远端 job 为 `/vepfs-mlp2/c20250405/400040/transfer/vla_attention/jobs/tiny_overfit_policy_lora_20261014_retry3/`，PID `623231`。
- [x] 该作业已经通过 GPU gate、source-lock、P1 provenance 和模型加载，开始写 `train_metrics.jsonl`；截至首次记录，训练已达到 57/300 steps，loss finite，GPU0 约 23GB。
- [x] policy-LoRA tiny 作业完成后的三方 offline action prediction 对比已由下一节完成；结论是 policy-LoRA tiny 没有超过 head+proprio-only，因此不触发 same-init rollout gate。
- [x] official moojink trace、B1 center-crop trace 与 official-vs-B1 对比已由下一节完成；后续不继续 policy-LoRA 分支，转向 action distribution / rotation repair。

## 2026-10-14：policy-LoRA tiny 评估、center-crop 复核与官方 action trace 对比

- [x] policy LoRA + head + proprio tiny-overfit 已完成 300/300 steps，远端目录为 `/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_action_calibration/tiny_overfit_policy_lora_headproprio_seed17_300step_20261014/`，checkpoint 为 `checkpoint_00300`，`final.json` 状态为 `passed`。
- [x] 三方 tiny offline action prediction 对比已生成：`tiny_eval_three_way_comparison.json`，SHA256 `092179040be1e6b63c3f2ff5fe5fb15ce4a46c105ab09b35e98fbe172d50ba83`。结果：policy-LoRA tiny 相比 B1 只有轻微改善，但比 head+proprio-only tiny 更差。

| metric | B1 | head+proprio tiny | policy-LoRA tiny | policy - B1 | policy - head |
|---|---:|---:|---:|---:|---:|
| action MAE | 0.298864 | 0.284692 | 0.296090 | -0.002774 | +0.011398 |
| first-horizon MAE | 0.288046 | 0.282167 | 0.292713 | +0.004666 | +0.010546 |
| translation MAE | 0.386396 | 0.372006 | 0.384227 | -0.002169 | +0.012221 |
| rotation MAE | 0.250380 | 0.233718 | 0.243450 | -0.006931 | +0.009732 |
| gripper MAE | 0.181721 | 0.175672 | 0.189600 | +0.007879 | +0.013928 |
| gripper sign acc | 0.712862 | 0.853714 | 0.827899 | +0.115036 | -0.025815 |

结论：解冻 policy LoRA 没有超过 head+proprio-only tiny，因此不满足“offline 明显优于 head-only 后再跑 same-init rollout”的 gate。当前不把 policy-LoRA tiny 当作闭环候选。

- [x] 为 rollout/trace 脚本新增 `--center-crop` 诊断开关，默认仍为 false，保持历史 P6 结果可复现。B1 使用 `center_crop=True` 的 1-episode rollout 结果仍为 `0/1`，目录为 `/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_action_calibration/b1_center_crop_one_episode_20261014/`。这说明 official positive control 与我们 wrapper 的 center-crop 差异不是单独充分原因。
- [x] 生成官方 `moojink/openvla-7b-oft-finetuned-libero-spatial` task0/init0 action trace，目录为 `/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_action_calibration/official_moojink_task0_init0_action_trace_20261014/`。该 trace 前 30 control steps 未必覆盖完整成功过程，只用于动作分布诊断，不是 benchmark。官方 loader 再次备份并改写了 moojink checkpoint 的 `config.json`，这是 OpenVLA-OFT 官方 loader 的既有行为。
- [x] 官方 trace 与 B1 center-crop rollout 前 30 步对比已生成：`/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_action_calibration/official_vs_b1_center_crop_action_compare_20261014/summary.json`。关键数值如下：

| metric | official moojink | our B1 center-crop | interpretation |
|---|---:|---:|---|
| processed translation L2 mean | 0.927672 | 0.561660 | B1 平移幅度明显偏弱 |
| processed rotation L2 mean | 0.041404 | 0.000821 | B1 rotation 几乎塌缩 |
| processed gripper mean | -1.000000 | -1.000000 | gripper sign 一致，不是主因 |
| action diff L2 mean | — | 0.670821 | 动作分布与成功 checkpoint 差距很大 |
| translation diff L2 mean | — | 0.666692 | 主要差距来自平移轨迹 |
| rotation diff L2 mean | — | 0.041017 | rotation under-production 仍成立 |

当前判断：P6 闭环失败的主断点更像是动作分布/动作幅度与 rotation 学习不足，而不是 evaluator、LIBERO 环境、init state、gripper sign 或 center-crop。下一步应优先做 action distribution repair：用官方成功 checkpoint 的 action trace/statistics 或 expert trajectory 约束 action magnitude、rotation 和 early-horizon，而不是继续扩大 semantic/P7。semantic/P7 仍不得声称闭环 VLA 有效。

## 2026-10-14：expert / official / B1 action distribution audit

- [x] 生成只读动作分布审计，不启动模型、不占 GPU。远端结果：`/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_action_calibration/expert_official_b1_action_distribution_compare_20261014/summary.json`；本地副本：`experiment_workspace/vla_remote_results/P6_vla_action_calibration/expert_official_b1_action_distribution_compare_20261014/summary.json`；summary SHA256=`266b8224a75ae5627043aabe822c6f52d5bb1b7cb6841de6e7e91b931473cb83`。
- [x] 审计口径已区分 official trace 的 `10 wait + 30 control`、expert replay 的 `wait/expert` phase，以及 B1 trace 无 phase 的限制。因此报告两种 B1 口径：旧对齐的 `first30`，以及跳过前 10 个 wait-like step 后的 `skip_wait_then30`。

| comparison | translation ratio | rotation ratio | interpretation |
|---|---:|---:|---|
| B1 first30 / official control30 | 0.605451 | 0.019825 | 与旧 official-vs-B1 对比一致，B1 rotation 几乎塌缩 |
| B1 skip-wait30 / official control30 | 0.806219 | 0.096544 | 即使跳过初始 wait-like zeros，B1 rotation 仍只有 official 的约 10% |
| B1 skip-wait30 / expert-success control | 0.895273 | 0.044216 | 相对成功 expert action，B1 translation 接近但 rotation 仍极低 |
| official control30 / expert-success control | 1.110459 | 0.457994 | official learned policy 与 expert action 分布不完全一致，因此 official 只作为成功策略参照，不直接当训练 target |

当前结论：gripper sign、center-crop、evaluator 和 init state 已不是主疑点；最稳的断点是 rotation under-production，其次是 early-horizon action magnitude。下一步不进入 semantic/P7 训练，也不进入正式 LIBERO/LIBERO-Plus/RoboTwin benchmark；先做一个更窄的 action repair smoke：从 B1 出发，只训练 action head + proprio projector，加入 first-horizon/early-horizon、rotation-aware 和 action-magnitude 约束，150--300 step 后先过 offline tiny/held-out gate。

## 2026-10-14：distribution-repair head+proprio smoke 与 gate

- [x] 基于 action distribution audit 启动了独立 repair smoke，没有引入 semantic/P7 loss，也没有覆盖历史 P6/P7 结果。远端训练目录：`/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_action_calibration/tiny_overfit_distrepair_headproprio_seed17_150step_20261014/`；checkpoint：`checkpoint_00150`；训练状态：`P6_TINY_OVERFIT_SMOKE_PASSED`。
- [x] 训练配置：从 B1 `checkpoint_10080` 出发，只训练 action head + proprio projector；steps=`150`，lr=`5e-5`，action weights=`[1,1,1,2,4,3,1]`，horizon weights=`[3,2,1.5,1.25,1,0.8,0.8,0.8]`，translation magnitude weight=`0.05`，rotation magnitude weight=`0.25`。该配置只用于 action-interface repair 诊断，不是论文主方法。
- [x] gate summary 已生成：远端 `results/P6_vla_action_calibration/distrepair_headproprio_150step_gate_summary_20261014.json`；本地副本 `experiment_workspace/vla_remote_results/P6_vla_action_calibration/distrepair_headproprio_150step_gate_summary_20261014.json`；SHA256=`2c76b074a17fbce03b19d0eb169f80fe6cf6d5c6d53fc0fd4de22aa904e96cc0`。

| split / metric | B1 | head+proprio tiny | policy-LoRA tiny | dist-repair 150 | gate |
|---|---:|---:|---:|---:|---|
| tiny action MAE | 0.298864 | **0.284692** | 0.296090 | 0.296642 | dist-repair 未超过 head+proprio |
| tiny rotation MAE | 0.250380 | **0.233718** | 0.243450 | 0.248656 | dist-repair 未超过 head+proprio |
| tiny gripper sign acc | 0.712862 | **0.853714** | 0.827899 | 0.791667 | dist-repair 未超过 head+proprio |

| held-out metric | B1 | P6.2 head+proprio | dist-repair 150 | interpretation |
|---|---:|---:|---:|---|
| action MAE | 0.293686 | 0.293562 | **0.293096** | 比 B1/P6.2 略好 |
| translation MAE | 0.377516 | 0.378003 | 0.378456 | 略差，不足以证明修复 |
| rotation MAE | 0.262067 | 0.261015 | **0.259969** | 有弱改善 |
| gripper MAE | 0.137056 | 0.137880 | **0.136398** | 有弱改善 |
| gripper sign acc | 0.786573 | **0.844603** | 0.832087 | 好于 B1，低于 P6.2 |

Gate 判断：**No-Go for rollout / No-Go for semantic-P7 closed-loop claim**。虽然 held-out 上 action/rotation 有弱改善，但 tiny gate 没有超过 head+proprio-only，说明这次 magnitude/rotation repair 没有解决同 init tiny-overfit 闭环前置问题。因此不运行 same-init rollout，不进入正式 LIBERO/LIBERO-Plus/RoboTwin benchmark，也不把 P7-lite 加回闭环训练。下一步应转向更底层的 action interface / controller 诊断：action chunk 执行时序、rotation representation、action head 输出尺度、proprio 输入、以及 official successful checkpoint 与我们 checkpoint 的 step-wise state/action 对齐。
