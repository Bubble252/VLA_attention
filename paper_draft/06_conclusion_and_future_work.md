# 6. Summary 与 Future Work

## 6.1 论文结论模板

最终结论必须按证据填写：

1. **空间归因桥**：说明哪些模型、哪些归因接口成功映射到同一图像网格，以及坐标/干预校准结果。
2. **VLM grounding**：报告正确 Lavender 教师相对无对齐、错词和错图的变化；若跨模型不一致，明确限定范围。
3. **VLA D0**：报告 `L_contain` 是否减少语言无关动作证据，并说明 delta EEF 误差和成功率是否同步改善。
4. **DROID 泛化**：报告真实视觉变化下的离线动作预测和反事实结果；不能用 LIBERO 替代 DROID。
5. **D1/B**：只有阶段迁移或未来归因通过负控，才写入正向结论；否则作为失败但有价值的边界。

## 6.2 局限

- Lavender 教师来自扩散模型，可能在遮挡、细粒度接触和真实相机分布下定位不稳；
- gradient×activation 依赖目标标量和梯度噪声，不等价于因果解释；
- patch occlusion 计算昂贵且可能产生非自然图像；
- DROID 首轮以离线 action prediction 为主，不能替代闭环真实机器人实验；
- SpikingBrain full/window/GLA 的归因语义仍需接口审计，不能把所有状态矩阵 reshape 成注意力图；
- D1 phase 和 B future attribution 依赖弱标签/目标评分器，存在标签噪声。

## 6.3 Future work

1. 用更强的 intervention 或 causal mediation 替代单 patch occlusion；
2. 在 DROID 之外加入更大跨操作者、相机和 embodiment 的数据；
3. 将 WAM/VAM 的 `R_future` 与 D1 phase 联合，研究当前动作对未来任务进展的可预测性；
4. 对 event-driven/spiking backbone 做能耗、延迟和稀疏率评估；
5. 探索无需 Lavender 推理时教师的在线归因自蒸馏；
6. 在真机上验证 `A_act` 是否能用于异常检测和安全回退。

## 6.4 论文最终故事的最小版本

如果资源或结果有限，保留：

```text
Lavender T_sem
→ SpikingBrain/Qwen/LLaVA A_lang
→ D0 A_act containment
→ LIBERO 机制
→ DROID 离线泛化
```

D1、B/WAM/VAM、Next Forcing 和真机全部作为后续扩展，不让论文失去单一主线。
