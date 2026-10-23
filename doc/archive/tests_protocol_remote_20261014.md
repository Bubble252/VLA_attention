# P7-lite：B1 上的判别式 semantic repair 预实验

P6 的结论是：B1/BlindVLA-style retention 是当前最强动作基线；B2/B3/B4 的静态 semantic containment 没有带来 held-out action prediction 收益。因此 P7-lite 不继续扩大 P6 的静态 containment，而是在 B1 上只测试一个更小、更有诊断意义的修复：让动作证据图对正确语义图的得分高于多个负控语义图。

## 状态

- [ ] 本地 loss 与配置准备完成。
- [ ] 101 上完成最小 offline smoke。
- [ ] 与 P6 controls 共用 held-out semantic cache/evaluator，确认不覆盖 P6 artifacts。
- [ ] 若 smoke 有正向机制信号，再决定是否进入正式 P7 训练。

## 核心假设

P6 的 containment 只要求动作证据落在 `source ∪ target` 支撑区域内，约束偏弱，也容易被面积、平滑图、错误图解释。P7-lite 改为判别式约束：同一个 action evidence map 必须更支持正确语义图，而不是 wrong-image、role-swapped 或 area-matched random。

给定动作证据图 `A_act`、正确语义支撑 `S+` 和负控集合 `{S-_k}`，先定义支撑得分：

```text
score(A, S) = sum_i norm(A)_i * norm(S)_i
```

ranking 损失：

```text
L_rank = mean_k ReLU(margin + score(A_act, S-_k) - score(A_act, S+))
```

可选 KL 版本保留为后续扩展：把 `[score(correct), score(negatives)]` 变成 logits，最小化目标分布 `[1,0,...]` 的 cross entropy。首轮先使用 ranking，因为它对 score 标度不敏感，适合 smoke。

## 与 P6 的差异

| 项目 | P6 B3/B4 | P7-lite |
|---|---|---|
| 基线 | B0/B1/B2/B3/B4 全矩阵 | 只从 B1 出发 |
| semantic 目标 | containment：少看支撑外区域 | ranking：正确图得分必须高于负控图 |
| 负控 | 训练后评估阶段使用 | 进入训练/修复目标本身 |
| 目标 | 看 semantic containment 是否直接提升 action | 看 semantic evidence 是否能变得判别化 |
| 结论边界 | 已显示不能直接作为最终主方法 | 只做修复 smoke，不能直接 claim benchmark |

## 首轮配置建议

- base checkpoint：P6 `B1_seed17/checkpoint_10080`。
- held-out evaluator：复用 P6 controls 的 correct / wrong-image / role-swapped / area-matched random maps。
- batch：1，保持与 P6 VLA smoke 一致，先避免显存和 attribution 图复杂度。
- steps：先 100–500 step smoke；有效后再上 2k–10k。
- margin：0.02 起步；若 loss 很快为 0，升到 0.05。
- lambda_rank：0.01 起步；若 action loss 明显恶化，降到 0.003；若机制指标无变化，升到 0.03。
- 成功条件：correct score 相对三类负控至少两类稳定提高，同时 action MSE 不显著劣于 B1。

## 记录规则

每次运行必须记录：命令、Git SHA、配置 SHA、输入 cache SHA、checkpoint 路径、日志路径、summary 路径、失败样本路径。所有记录写入 `experiment_workspace/PROGRESS.md`。不得覆盖 P6 的 checkpoint/cache/results。
