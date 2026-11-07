# 07 D0 Matched 2.5K 训练与完整离线评测

## 1. 结论

D0 matched 2.5K 已经完成，postrun contract 和完整离线评测都通过：

```text
postrun_status          = PASS
arm_status              = RUN_PASS
offline contract_status = PASS
num_groups              = 118
destination pairs       = 78
object pairs            = 78
```

这证明当前 D0 checkpoint 可以作为一个可复核的 matched 基线候选，并且离线反事实选择指标为正。它**不证明** `white_right` 闭环会成功，也不证明 D1 或任何方法改进有效。`white_right` 仍然只能由 15-init 闭环 pilot 判断。

离线层面同时出现两个方向不同的信号：

- destination / object 的 correct-vs-wrong margin 都为正；
- 但相对历史 C1-only checkpoint，overall action MAE 和 gripper MAE 明显变差。

因此不能只报“destination margin 转正”就宣布 D0 变好。当前更合理的判断是：**这次 matched run 的动作选择性有正信号，但动作保真度下降，必须等闭环 pilot 解释。**

## 2. Run Contract

| 项 | 值 |
|---|---:|
| run | `star13_d0_matched_2500` |
| initialization | A3 12.5K |
| optimization steps | 2,500 |
| per-device batch | 4 |
| grad accumulation | 1 |
| effective batch | 4 |
| effective samples | 10,000 |
| raw dataset steps | 12,302 |
| effective dataset length | 11,630 |
| trajectories | 96 |
| seed | 17 |
| config snapshot SHA256 | `953e5e1658105eb3a40f5703b7bdc5e9d75d5906e3a8204b18f633a1a944ef56` |
| final checkpoint SHA256 | `6df931aeaed2f870affc775b6728a3eb99e5e59cebc21b7c66ef6906f3e7493e` |
| checkpoint size | 696,254,492 bytes |
| float tensors | 248 |
| non-finite tensors | 0 |

配置与历史 `star11_c1_adapt_2500` 的 `config.full.yaml` 对齐：同一 A3 初始化、同一 D0 数据、同一 `11630` effective length、同一 seed、同一 2.5K steps、同一 effective batch 4。历史 YAML 虽然写过 accumulation 4，但其训练日志明确是 `accumulation steps = 1`、`Total batch size = 4`，因此本轮使用 1 是正确的 matched 设置。

需要保留的限制：历史 checkpoint 的远端 code commit 没有被冻结成可比的 Run Contract 字段，所以历史与当前只算“初始化、数据、预算 matched”，不能把数值差异严格归因于代码版本或方法。

## 3. 完整离线评测

评测使用 `scripts/star12_counterfactual_eval.py`，同一图像、state 和 expert action chunk，只替换语言；validation 15 episodes，sidecar 118 groups，seed 17。

| relation | pairs | correct MAE | wrong MAE | margin | 95% bootstrap CI | positive fraction |
|---|---:|---:|---:|---:|---|---:|
| destination | 78 | 0.145168 | 0.160273 | +0.015105 | [0.007568, 0.023540] | 0.679 |
| object | 78 | 0.146349 | 0.160694 | +0.014345 | [0.006304, 0.021798] | 0.615 |

动作幅度：

| relation | overall | translation | rotation | gripper |
|---|---:|---:|---:|---:|
| destination | 0.987 | 1.058 | 0.863 | 0.912 |
| object | 1.006 | 1.075 | 0.927 | 0.860 |

解释：

1. 两个 relation 的 margin bootstrap CI 下界都大于 0，说明 correct instruction 的动作整体比 wrong instruction 更接近 expert chunk；
2. 这不是 `white_right` 结果，因为 held-out prompt 没有 expert action label；
3. rotation 和 gripper amplitude ratio 小于 1，动作幅度没有爆炸，但 gripper 有收缩迹象；
4. offline 的 `region_results=null` 是预期行为，D0 没有 region head。

## 4. 与历史 C1-only 的同口径参考

历史文件：`star105_goal12_20261106/evals/offline_base_star11_c1_adapt_2500.json`。

| 指标 | 历史 C1-only | D0 matched | 变化 |
|---|---:|---:|---:|
| destination margin | -0.001149 | +0.015105 | +0.016254 |
| destination CI | [-0.004982, 0.002565] | [0.007568, 0.023540] | 下界转正 |
| object margin | +0.075410 | +0.014345 | -0.061064 |
| destination correct MAE | 0.121510 | 0.145168 | +0.023657 |
| destination gripper MAE | 0.021567 | 0.119135 | +0.097568 |
| object gripper MAE | 0.024924 | 0.129992 | +0.105068 |
| destination amplitude ratio | 1.177 | 0.987 | -0.190 |
| object amplitude ratio | 1.071 | 1.006 | -0.065 |

结论边界：

- D0 的 destination margin 比历史 checkpoint 更“会区分语言”，但 correct MAE、gripper MAE 和 object margin 都更差；
- 两端都是同一个方法、同一数据配置、不同训练产物，因此这更像单次训练的方差或未冻结的代码/环境差异，不能写成 D0 方法改进；
- 这也说明只看 destination margin 会误导。后续 D0 vs D1 必须同时报告 correct MAE、gripper、amplitude 和闭环 strict destination。

## 5. 为什么还没有 `white_right` 结论

离线 evaluator 不评 `white_right`：held-out prompt 没有 expert action label。`white_right` 的唯一有效证据是冻结的 15-init × 4-prompt 闭环：

- seen group-any；
- seen group-all；
- `white_right` official success；
- `white_right` strict destination；
- wrong-destination / target-not-reached 失败分类。

因此本文件不写 `white_right=0/15` 或 `>=1/15`。15-init pilot 正在单独运行，结果将写入 `08_d0_c1_pilot_result.md`。

## 6. 证据文件

- [D0_matched_postrun.json](results/D0_matched_postrun.json)
- [D0_matched_checkpoint_health.json](results/D0_matched_checkpoint_health.json)
- [D0_matched_contract.json](results/D0_matched_contract.json)
- [star13_d0_matched_2500_offline_full.json](results/star13_d0_matched_2500_offline_full.json)
- [star13_d0_matched_2500_offline_full_contract.json](results/star13_d0_matched_2500_offline_full_contract.json)

远端大文件不提交：

```text
$G13/checkpoints/star13_d0_matched_2500/final_model/model.safetensors
$G13/logs/star13_d0_matched_2500.log
$G13/evals/star13_d0_matched_2500_offline_full.log
```

## 7. 下一步

1. 完成 D0 15-init C1 pilot，检查是否复现 `white_right=0/15`；
2. 若复现，记录 seen group-any、group-all、wrong-destination 和 strict destination，并把它固定为 D1 的唯一比较基线；
3. 若不复制，先解释训练方差、checkpoint 代码版本和 evaluator 差异，不能直接开始 D1；
4. D0 pilot 通过后，再构建 D1 carrier LeRobot 数据、observed contract 和 matched pilot；
5. 在 D0/D1 Gate 完成前，不启动 I 层，不合并 A2/F0。
