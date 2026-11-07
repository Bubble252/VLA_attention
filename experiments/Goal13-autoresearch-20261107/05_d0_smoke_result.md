# 05 D0 2-step Smoke Postrun 结果

## 1. 结论

`D0` 的 2-step training smoke 已通过 postrun contract：

```text
postrun_status = PASS
arm_status     = SMOKE_PASS
gate_status    = NOT_EVALUATED
```

这只证明“D0 的启动配置、数据长度、batch 和 checkpoint 链路可复核”，**不证明 D0 复现了 0/15，也不证明任何方法有效**。正式 Gate 仍要等 D0 matched pilot 和离线/闭环评测。

## 2. 运行事实

| 项 | 值 |
|---|---:|
| run | `star13_d0_smoke_2` |
| optimization steps | 2 |
| per-device batch | 4 |
| grad accumulation | 1 |
| effective batch | 4 |
| raw dataset steps | 12,302 |
| effective dataset length | 11,630 |
| trajectories | 96 |
| seed | 17 |
| remote code commit | `4507931a625c844404c4a76128fb116536d8ca7c` |
| source config SHA256 | `61cbe071a8ca21d5e1d603b03c6a17e552a55ea00ca72aadc32133f12c23b252` |
| full config SHA256 | `b04140c69b84ec7da2ccd60f0425797e83c52da35aff90dd42da3a61eb4a56ce` |
| final checkpoint SHA256 | `a20e983c1833a1233675927827a5632ec230def93549063cfbb4e486645bd41e` |
| checkpoint size | 696,254,492 bytes |

## 3. 权重有限性

日志没有打印 `loss`，因为 `logging_frequency=10` 而 smoke 只有 2 steps。这里不把“没有报错”当作 loss 正常；改用 checkpoint 权重有限性作为更直接的数值证据：

| 项 | 值 |
|---|---:|
| float tensors | 248 |
| elements | 174,055,687 |
| non-finite tensors | 0 |
| all_finite | true |

## 4. 已核验的 contract

- 训练日志中的 `optimization steps=2`、`per-device batch=4`、`grad_accum=1`、`total batch=4` 与 config 一致；
- 训练日志中的 raw/effective dataset length 都是 `11630`；
- `summary.jsonl` 的 steps 为 2；
- `dataset_statistics.json` 为 12,302 transitions / 96 trajectories；
- `config.full.yaml` 的 `max_train_steps`、batch、accumulation、seed、output dir 与契约一致；
- `config.full.yaml` 指向的 snapshot SHA 与契约 config SHA 一致；
- 远端 repo HEAD 与契约 code commit 一致；
- `steps_2_model.safetensors` 与 `final_model/model.safetensors` 字节级 SHA 相同；
- checkpoint health 覆盖 248 个 float tensors，全部 finite。

## 5. 审计过程修正

第一次 postrun 判为 `INVALID`，唯一失败项是 `log_effective_batch`：rich logger 把 `Total batch size` 和 `= 4` 折成多行，解析器没有匹配到。该问题已通过折行回归测试修复并重新运行，第二次 `PASS`；这不是训练失败，也不是方法失败。

## 6. 证据文件

- [D0_smoke_postrun.json](results/D0_smoke_postrun.json)
- [D0_smoke_checkpoint_health.json](results/D0_smoke_checkpoint_health.json)
- [D0_contract.json](results/D0_contract.json)
- [observed_d0.json](results/observed_d0.json)

## 7. 下一步

1. 用这个 checkpoint 跑 D0 offline `limit=2` eval smoke，确认 evaluator、checkpoint loader、动作反归一化和输出文件链路；
2. 再跑 D0 matched pilot，检查 `white_right=0/15` 是否复现，或解释差异；
3. 只有 D0 契约和复现都稳定后，才允许构建 D1 carrier LeRobot 数据和比较 D0/D1；
4. 不启动 A2，不合并 F0。
