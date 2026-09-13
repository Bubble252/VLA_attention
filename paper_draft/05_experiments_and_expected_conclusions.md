# 5. Experiments 与预期结论

## 5.1 实验总表

| 实验 | 数据/模型 | 主要问题 | 预期结论 |
|---|---|---|---|
| E0 接口与坐标桥 | 合成棋盘格、Flickr30k | token/grid/window 恢复是否正确 | 峰值位置恢复误差低于预设阈值，否则停止后续训练 |
| E1 Sink 与归因诊断 | SpikingBrain、Qwen、LLaVA | 热力图是否被 sink 或固定热点污染 | raw/sink-free 差异按模型报告，不把图像美观当收益 |
| E2 VLM grounding | Flickr30k/Entities；SpikingBrain、Qwen、LLaVA | `T_sem → A_lang` 是否改善 grounding | 正确教师优于错词/错图/随机教师；至少两个结构不同模型成立 |
| E3 VLM 证据校准 | 同 E2 | attention/gradient 与干预是否一致 | intervention agreement 高于 attention-only；若不成立，降低解释性主张 |
| E4 LIBERO 离线 VLA | 一个 pick-place | D0 是否改善单步动作和归因 containment | action MAE 不升高，`A_act` 越界比例下降 |
| E5 LIBERO rollout | 同 E4 | D0 是否改善闭环控制 | 成功率和接近/抓取/放置子阶段至少一项改善 |
| E6 DROID 离线泛化 | pick-place 子集；7D delta EEF | 是否跨真实视觉分布有效 | D0 相对 baseline 的收益不只出现在 LIBERO |
| E7 D1 phase | LIBERO/DROID 轨迹 | `source→mixed→target` 是否存在且有益 | 正确 phase 优于反向/随机/固定百分比；否则保留 D0 |
| E8 B WAM/VAM | DROID 离线 future attribution | 未来归因是否比静态图更接近动作成败 | 只有通过 horizon/动作负控才进入主结果 |
| E9 跨骨干 | OpenVLA/OFT、Qwen/LLaVA | 方法是否依赖 SpikingBrain | 统一归因接口在至少两种结构上有效 |

## 5.2 基线与消融

### VLM

- no alignment；
- Lavender 原式 attention-to-attention；
- teacher-to-attribution；
- `[7,15,23,31]`、`[23,31]`、`[31]`、随机层；
- raw/sink-free；
- 正确教师、错词、错图、随机图；
- attention、gradient×activation、occlusion。

### VLA

- behavior cloning baseline；
- `L_sem only`；
- `L_sem + L_contain`（D0）；
- D0 + D1 phase；
- D0 + C state-rule diagnostic；
- D0 + B future refiner；
- B/C without D；
- action expert 蒸馏负控；
- wrong phase、wrong horizon、wrong action、random teacher。

## 5.3 指标

### Grounding

- pointing accuracy；
- phrase-region IoU；
- pointing game top-1；
- attribution entropy；
- cross-augmentation consistency；
- intervention agreement。

### 动作

- 7D delta EEF MAE/RMSE；
- gripper accuracy；
- trajectory smoothness；
- `A_act` 落在 `A_lang` 并集外的质量比例；
- patch occlusion 与 `A_act` 的 rank correlation；
- LIBERO success rate；
- DROID 场景/物体/相机划分下的离线误差。

## 5.4 每一步结论门槛

1. **E0 失败**：坐标桥或窗口逆变换错误，不能继续。
2. **E1 显示 sink 很强**：保留双轨并启用 mask；如果 sink 只影响可视化，不改变主损失。
3. **E2 正确教师无优势**：检查教师图质量和词 span，暂停 VLA；不能靠调大 `λ` 掩盖。
4. **E3 attention 与干预不一致**：论文只保留 gradient/intervention 结论，不宣称 attention 解释。
5. **E4 D0 降低动作误差且 containment 改善**：进入 E5；若 grounding 改善但动作无效，检查 action head 归因接口。
6. **E5 成功率下降**：降低 `λ_contain`、检查过强约束和 state 输入；D0 不能作为成功方法。
7. **E6 DROID 无收益**：主结论限定为 LIBERO 机制，不声称真实泛化，并优先排查数据转换和分布差异。
8. **E7 phase 负控无差异**：D1 退回诊断，D0 保持主线。
9. **E8 future attribution 无额外信息**：B 放入 future work，不增加 world-model loss。
10. **E9 只在 SpikingBrain 有效**：论文将“普适性”改为“结构感知 SpikingBrain 方法”，不夸大跨模型结论。

## 5.5 预期主表结构

| 模型/数据 | Baseline | `L_sem` | D0 | D1 | D0 + B | 归因干预一致性 |
|---|---:|---:|---:|---:|---:|---:|
| SpikingBrain / LIBERO | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| SpikingBrain / DROID | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| OpenVLA / DROID | 待测 | 待测 | 待测 | 后置 | 后置 | 待测 |

所有“预期”在实际结果前都保持为假设，不能写成已验证结论。
