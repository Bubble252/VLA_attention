# V0--V4 共享代码架构

## 不变的运行骨架

```text
Flickr manifest + image/phrase/box
        │
        ├── Qwen adapter ───────────────► SFT / phrase-score attribution
        ├── frozen retention encoder ───► L_retention（仅 V2/V4）
        └── frozen best-single diffusion ► T_sem map cache ─► L_sem（仅 V3/V4）
                                                              │
                                              evaluator: grounding / OOD / retention
```

所有组共用同一 Qwen revision、同一 train manifest、同一 batch schedule、同一 seed 集、同一 evaluation manifest；差异只能来自下表启用的 loss，不能偷偷改变数据或预算。

| 组 | 可训练参数 | loss | 禁止项 |
|---|---|---|---|
| V0 | 无 | 无，仅 checkpoint eval | 训练、teacher 图 |
| V1 | Qwen SFT adapter | `L_sft` | retention、semantic teacher |
| V2 | 与 V1 相同 | `L_sft + λ_ret L_retention` | diffusion teacher |
| V3 | 与 V1 相同 | `L_sft + λ_sem L_sem` | retention teacher |
| V4 | 与 V1 相同 | `L_sft + λ_ret L_retention + λ_sem L_sem` | 更换数据或预算 |

## 将实现的模块

| 模块 | 职责 | 不承担的职责 |
|---|---|---|
| `data/flickr_entities.py` | 读取冻结 JSONL、加载图片/phrase/boxes、验证 SHA256 | 不重划 split |
| `adapters/qwen25_runtime.py` | processor、visual token positions、phrase/answer score、hidden-state hook | 不猜测网格或重排 token |
| `teachers/retention.py` | 冻结 feature encoder、student/teacher patch coordinate bridge | 不作为 `T_sem` |
| `teachers/diffusion_runtime.py` | 冻结 best-single diffusion，离线生成/缓存词图 | 不在 test set 选超参 |
| `losses/vlm.py` | `L_sft`、`L_retention`、`L_sem`，逐项记录数值 | 不隐藏 loss 开关 |
| `train/vlm_runner.py` | 固定 manifest、resume、20--50 step smoke、训练 | 不修改 teacher 或 split |
| `evaluation/grounding.py` | pointing、mass-in-box、IoU、wrong-word/image/random controls | 不把训练图当测试指标 |

## 依赖顺序

1. Qwen P1：运行时 adapter 和 token grid 已验证；
2. teacher calibration：选择并冻结一个 `T_sem`，离线 cache 只覆盖 train/calibration；
3. retention teacher、层、projector seed 固定；
4. 生成 V0--V4 五份不可变 manifest；
5. 每组先 20--50 step smoke，检查数据、loss、checkpoint restore；
6. smoke 均通过后才提交正式训练。

`V3/V4` 不能在 best-single 未冻结时实现为“任意扩散模型都能替换”的运行时开关；候选比较发生在 teacher calibration，不发生在 test 或主训练中。
