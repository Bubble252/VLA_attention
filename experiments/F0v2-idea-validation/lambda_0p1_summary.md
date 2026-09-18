# lambda_sem = 0.1 总结

所有结果使用同一 phrase-score gradient×activation 归因定义。20-step、100-step 和 500-step 使用相同的 F0v2 训练 manifest；held-out 固定为 64 条 image-disjoint Flickr30k Entities test records。`lambda=0.1` 的 sweep-100 与原先的 100-step V3/V4 结果相同，因为 seed、数据、配置完全一致。

## V3：semantic attribution

| 训练阶段 | Pointing | Mass-in-box | Top-20% box IoU | 对照 | Δ Pointing | Δ Mass | Δ IoU |
|---|---:|---:|---:|---|---:|---:|---:|
| 20-step λ=0.1 | 0.2969 | 0.3197 | 0.2683 | V1-20 | -0.0625 | +0.0014 | -0.0009 |
| 100-step λ=0.1 | 0.3906 | 0.3365 | 0.2679 | V1-100 | +0.0313 | -0.0013 | -0.0001 |
| 500-step λ=0.1 | 0.3750 | 0.3312 | 0.2685 | V1-500 | +0.0781 | +0.0028 | -0.0013 |

V3 的 λ=0.1 在 100 和 500 steps 都提高 pointing，但 mass/IoU 没有同步、稳定提升。500-step 的 pointing 增量较大，但这是 64 条样本上的方向性结果，不能单独作为方法成立证据。

## V4：retention + semantic attribution

| 训练阶段 | Pointing | Mass-in-box | Top-20% box IoU | 对照 | Δ Pointing | Δ Mass | Δ IoU |
|---|---:|---:|---:|---|---:|---:|---:|
| 20-step λ=0.1 | 0.3594 | 0.3210 | 0.2682 | V2-20 | 0.0000 | -0.0104 | -0.0014 |
| 100-step λ=0.1 | 0.3438 | 0.3333 | 0.2696 | V2-100 | 0.0000 | +0.0049 | +0.0015 |
| 500-step λ=0.1 | 0.2344 | 0.2678 | 0.2692 | V2-500 | -0.1563 | -0.0604 | +0.0002 |

V4 的 λ=0.1 在 100 steps 出现小幅 mass/IoU 增量，但 20 和 500 steps 没有复现；说明 semantic loss 与 retention 的交互仍不稳定。

## Teacher 对照

| Map | Pointing | Mass-in-box | Top-20% box IoU |
|---|---:|---:|---:|
| Correct SD phrase map | 0.6563 | 0.4088 | 0.3253 |
| Same-image wrong-word | 0.2969 | 0.3272 | 0.2781 |
| Wrong-image | 0.2969 | 0.3234 | 0.2701 |
| Random | 0.2969 | 0.3033 | 0.2676 |

这说明 teacher map 本身包含有效的 phrase-region 信息；当前瓶颈是把它稳定转移到学生归因图，而不是 teacher 没有空间语义。

## 总结判断

`lambda_sem=0.1` 是目前 V3 的最佳候选，尤其在 100-step 的 pointing 和 500-step 的 pointing 增量上，但它没有让三个空间指标一致提高，也没有在 V4 上稳定复现。当前不能把 λ=0.1 定为最终超参，也不能据此启动 F1-10k。下一步应固定 V3 λ=0.1，单独测试 teacher map temperature、normalization 和 loss warm-up；V4 作为交互稳定性对照保留。
