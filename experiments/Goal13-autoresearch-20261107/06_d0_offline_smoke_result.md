# 06 D0 Offline `limit=2` Smoke 结果

## 1. 结论

D0 的 offline evaluator 链路通过 smoke 契约：

```text
postrun_status = SMOKE_PASS
gate_status    = NOT_EVALUATED
num_groups     = 2
```

这只证明 validation split、sidecar、checkpoint loader、动作归一化和输出 JSON 可以串起来。**它不是 action quality 结论，也不是 `white_right` 闭环结果。**

## 2. 运行配置

| 项 | 值 |
|---|---|
| arm | D0 |
| checkpoint | `star13_d0_smoke_2/final_model/model.safetensors` |
| checkpoint SHA256 | `a20e983c1833a1233675927827a5632ec230def93549063cfbb4e486645bd41e` |
| config snapshot SHA256 | `61cbe071a8ca21d5e1d603b03c6a17e552a55ea00ca72aadc32133f12c23b252` |
| validation | 15 episodes，C1 三 seen prompts 各 5 |
| sidecar groups | 118 个 `(episode_id, state_index)` |
| smoke limit | 2 groups |
| seed | 17 |
| evaluator | `scripts/star12_counterfactual_eval.py` |

## 3. 结果

| relation | pairs | correct MAE | wrong MAE | margin | 95% CI |
|---|---:|---:|---:|---:|---|
| destination | 2 | 0.893800 | 0.893661 | -0.000139 | [-0.001673, 0.001396] |
| object | 2 | 0.893800 | 0.892585 | -0.001215 | [-0.002349, -0.000081] |

动作幅度：

| 指标 | 值 |
|---|---:|
| overall amplitude ratio | 2.287 |
| translation amplitude ratio | 3.290 |
| rotation amplitude ratio | 8.131 |
| gripper amplitude ratio | 0.733 |

## 4. 为什么不能解读这些数字

1. 这个 checkpoint 只训练了 **2 steps**，action head 仍接近随机初始化；大幅动作和接近 0.89 的 MAE 是预期现象。
2. 每个 relation 只有 **2 个 pair**，bootstrap CI 不具统计意义。
3. `region_results=null` 是正确行为：D0 没有 region head。
4. offline evaluator 不评 `white_right`，因为 held-out 没有 expert action label；`white_right` 只能走闭环。

因此当前结果不能说明 D0 方法失败、不能说明 D0 复现了 0/15，也不能用于比较 D1。

## 5. 快速 failure diagnosis

唯一需要修的是训练预算，而不是 evaluator：

- evaluator 成功构造 train/validation dataset；
- checkpoint 中的 248 个 float tensors 全部 finite；
- config/checkpoint SHA 与 postrun contract 一致；
- destination/object pair 和 amplitude 指标都成功写出；
- `region_results` 按预期为空。

## 6. 证据文件

- [D0_offline_smoke_limit2.json](results/D0_offline_smoke_limit2.json)
- [D0_offline_smoke_contract.json](results/D0_offline_smoke_contract.json)
- [D0_offline_smoke_limit2.log](results/D0_offline_smoke_limit2.log)

## 7. 下一步

1. 在 GPU1 上跑 D0 matched 2.5K training；
2. 对 2.5K checkpoint 重新做 postrun contract 和 offline evaluation；
3. 跑 D0 15 条闭环 pilot，目标是复现 `white_right=0/15`，或给出初始化/训练 budget 差异解释；
4. D0 通过后再构建 D1 carrier 数据。
