# P6 正式训练后评测边界（2026-10-08）

本文件约束当前 goal 在 B0–B4 seed17 正式 offline 训练完成后的工作范围。它只负责恢复 checkpoint、held-out offline action prediction、归因控制和结果汇总，不重新训练已有组，不覆盖已有产物，不把 offline 指标写成 LIBERO rollout success。

## 交互文档

执行前必须读取并以这些文件为准：

1. `experiments/P6-vla-b0-b4/protocol.md`：模型、动作头、归因、controls、指标和结论边界；
2. `experiments/P6-vla-b0-b4/freeze_decision_20261007.md`：已冻结的 checkpoint、manifest、步数、lambda、temperature 和未决项；
3. `configs/vla/p6_b0_b4_formal_preflight_20261006.json`：运行时路径和 P1 约束；
4. `experiment_workspace/PROGRESS.md`：阶段状态、远端 PID、日志和 SHA；
5. `paper_draft/05_experiments_and_expected_conclusions.md`：只在真实结果生成后更新预期结论和结论边界。

代码与产物只允许交互：

- 训练脚本：`scripts/run_oft_b0_b4_formal_offline.py`（只读，不重新启动）；
- held-out action evaluator：`scripts/eval_oft_formal_offline.py`；
- held-out semantic input/cache：`scripts/export_libero_semantic_inputs.py`、SD1.5 cache；
- post-train controls：`scripts/eval_oft_posttrain_controls.py`；
- 远端训练结果：`results/P6_vla_b0_b4/formal_offline_20261007/`；
- 新评测结果：`results/P6_vla_b0_b4/posttrain_eval_20261008/`。

## 当前允许的顺序

- [x] 核验 B0–B4 每组 10080/10080、`final.json`、四个 checkpoint 和训练曲线；
- [x] 启动 held-out offline action prediction；
- [x] 导出 offline_eval semantic inputs；
- [ ] 完成 held-out SD1.5 semantic cache；
- [ ] 运行 correct、same-image role-swapped（操作性 wrong-word）、wrong-image、random、area-matched random controls；
- [ ] 记录 OpenVLA/OFT 没有可比 raw-attention proxy 时的不可用原因；
- [ ] 生成 action metrics、control metrics、B0–B4 paired comparison 和 Go/No-Go gate；
- [ ] 将真实结果写入 `PROGRESS.md`、P6 result summary 和论文实验章节；
- [ ] 只提交小脚本、JSON summary、Markdown 和配置，不提交 checkpoint、大 cache、大数据。

## 明确禁止的动作

1. 不重新训练 B0–B4，不更改 checkpoint、seed、manifest、lambda、temperature、action schema 或 teacher revision；
2. 不使用 train semantic cache 代替 held-out semantic cache；
3. 不在 held-out 结果出来前选择性删除失败样本或重跑某一组；
4. 不把 action MSE、归因 support mass 或 checkpoint restore 说成 rollout success；
5. rollout provenance 尚未通过时不启动正式闭环 rollout；
6. raw attention 不可用时不伪造 proxy；只记录原因并使用 gradient×activation 作为跨模型比较接口。

## 结果解释

- action prediction 是离线行为指标，主报告 action MSE/MAE、translation/rotation/gripper 分项、per-dim、per-horizon 和 per-episode；
- controls 是机制指标，报告 action evidence 在 semantic support 上的 mass、KL 和控制差值；
- seed17 只能作为单 seed 正式证据，不能宣称统计显著或三 seed 泛化；
- B0–B4 结果只支持“离线动作预测和归因机制是否出现方向性证据”，不能单独证明 LIBERO 闭环成功；
- 如果 controls 或 evaluator 被阻塞，保留已有训练结果，记录阻塞原因，继续不依赖该步骤的汇总和文档工作。

