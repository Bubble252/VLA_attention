# 执行计划与 Git 操作：VLM 优先、VLA 后验的最终方案

**版本**：Final execution plan v1.0  
**日期**：2026-09-10  
**执行原则**：先完成文档和可验证接口，再运行代码；每个阶段都要有验收条件、commit 和可回滚点。

## 1. Git 仓库初始化

仓库路径：

`/home/bubble/类脑计算/VLM终局`

初始化命令：

```bash
cd /home/bubble/类脑计算/VLM终局
git init -b main
git add README.md .gitignore doc references
git commit -m "docs: define final spikingbrain vla plan"
```

如果后续配置远端：

```bash
git remote add origin <remote-url>
git push -u origin main
```

当前没有预设远端地址，因此只做本地初始化和 commit，不伪造 push 结果。

### 1.1 新电脑恢复与继续开发

线上仓库：`https://github.com/Bubble252/VLA_attention.git`。新电脑先恢复文档、计划和代码：

```bash
git clone https://github.com/Bubble252/VLA_attention.git
cd VLA_attention
git status
git pull --ff-only
```

Git 不含 PDF、`references/repos/`、权重、数据、实验日志、`outputs/` 或 `doc-sync-main/sync_config.json`。恢复顺序必须是：

1. 读取 `README.md`、`references/abstract_review/README.md` 与 `paper_draft/08_model_baseline_benchmark_options.md`；
2. 运行 `bash scripts/download_references.sh` 恢复脚本已登记资料；
3. 查看 `references/repositories.yaml`，补 clone 新候选并记录实际 commit；
4. 根据冻结的模型/benchmark 下载权重、LIBERO 或 DROID；
5. 新建本机飞书凭据配置，再选择是否同步。

恢复后不得因 PDF、权重或数据缺失而修改 Git 历史或把大文件提交到线上仓库。

## 2. 阶段总览

| 阶段 | 目标 | 产物 | 完成标记 |
|---|---|---|---|
| P0 | 文档、仓库、文献目录冻结 | 三份 doc、README、references | [ ] |
| P1 | 本地代码和模型接口审计 | 入口表、张量规格、版本记录 | [ ] |
| P2 | 教师图和坐标桥诊断 | 可视化、恢复单元测试 | [ ] |
| P3 | 学生归因接口诊断 | attention/gradient/action attribution 报告 | [ ] |
| P4 | VLM grounding 证明 | Prismatic/Qwen/InternVL/Ovis 候选中筛选两个；SpikingBrain 后置 | [ ] |
| P5 | VLA delta EEF smoke | LIBERO 单任务闭环 | [ ] |
| P6 | VLA 主方法训练 | D：Structure-Native Action Attribution Consistency | [ ] |
| P7 | 跨模型、跨 VLA 骨干和动作相关性细化 | π0/π0.5、MolmoAct2、OpenVLA/OFT；SpikingBrain 后置 | [ ] |
| P8 | 消融和反证 | 随机、错图、错词、错层、无对齐结果 | [ ] |
| P9 | 最终报告 | VLM + LIBERO 指标表、曲线、结论、真机计划 | [ ] |

GPU 型号、数量和显存不作为 P0 的阻塞项；服务器由用户提供，实验只记录运行环境摘要。

## 3. P0：文档、仓库和文献归档

### 任务

- [x] 写清项目背景和原始想法来源；
- [x] 明确 Lavender 词图与不同 VLM/VLA 空间归因之间需要桥接；
- [x] 固定 delta EEF 7D 输出；
- [x] 固定 LIBERO 优先、真机后置；
- [x] 初始化 Git；
- [x] 创建 `references/papers`、`references/bib`；
- [x] 下载首批核心论文并记录 SHA256；
- [x] 登记官方仓库 commit 和模型/论文版本；
- [x] 记录教师选择原则：首轮使用扩散语义教师，VLA 阶段纳入 Action-Relevance Refiner，而不是直接蒸馏动作 teacher。

### 验收

- 三份文档互相引用且没有 GPU 硬编码；
- `references/README.md` 能说明每个文件的用途和来源；
- `git status` 只有预期的新文件。

### Git

```bash
git add README.md .gitignore doc references
git commit -m "docs: freeze final vla scope and references"
git push -u origin main  # 配置远端后执行
```

## 4. P1：本地代码与接口审计

### 任务

- [ ] 记录 Lavender 当前 commit `58fc71b`；
- [ ] 【后期可选】记录 SpikingBrain 当前 commit `ef99987`；
- [ ] 导出实际 `window_size`、`fullatt_block_indexes` 和视觉层数；
- [ ] 确认 `grid_thw`、window reorder 和 patch index 的变换；
- [ ] 确认语言 hidden state 与动作 query 的获取位置；
- [ ] 检查每个 checkpoint 是否真的支持 `output_attentions`；
- [ ] 生成 `outputs/interface_audit.md`。

### 重点检查

不要把 `SWA` 返回的 `attn_output+1e-17` 当作概率矩阵；不要把 GLA 状态矩阵直接 reshape 成二维空间图；不要绕过窗口索引恢复。

### Git

```bash
git add outputs/interface_audit.md configs
git commit -m "audit: record model tensor interfaces"
git push
```

## 5. P2：Lavender 教师图和坐标桥

### 任务

- [ ] 用固定图像和固定指令生成词级教师图；
- [ ] 保存词名、token span、原图尺寸、输出尺寸；
- [ ] 实现 patch 网格到教师图的 resize；
- [ ] 实现 window reorder 的逆变换；
- [ ] 在合成棋盘格图上验证峰值位置；
- [ ] 输出原图、patch map、教师图和叠加图。

### 验收

- 同一对象在原图、patch 图和教师图中位置一致；
- 坐标变换有单元测试；
- 无效词或不存在对象会被 mask，而不是产生零损失假样本。

### Git

```bash
git add src/attention_bridge scripts configs
git commit -m "feat: add teacher map and patch coordinate bridge"
git push
```

## 6. P3：学生归因接口诊断

### 6.0 Attention Sink / 归因稳定性前置诊断

- [ ] 在正式热力图训练前，对每个模型计算视觉 token 的总归因质量、top-1 token mass、空间熵和跨层/跨增强一致性；
- [ ] 检查 BOS/CLS、padding、分隔符以及固定边缘 patch 是否长期吸收高权重；
- [ ] 对 full-attention、window/SWA、GLA 和 gradient×activation 分别做 sink mask 对照；
- [ ] 不默认把 sink-free 图直接用于主损失；先以 raw/sink-free 双轨结果和干预归因作决定，避免把预处理收益误当成对齐收益；
- [ ] 若 sink mask 改变目标 IoU 超过预设阈值（建议 5 个百分点），将该模型标记为“需 sink 校正”，并把校正作为 P3 的必需接口；
- [ ] 增加 token-drop、输入裁剪和指令改写三种稳定性测试，确认峰值来自目标区域而非固定 token；
- [ ] 对每个候选层至少抽取若干 head，报告 head-wise 与 head-mean 结果；
- [ ] 增加 patch occlusion/intervention 归因：遮挡单个或小块视觉 patch，测量动作变化 `Δa`，作为 attention/gradient 热力图的外部校准；

这里的目标不是提出新的 Sink-free Attention 方法，而是确认热力图具有空间语义。Sink 诊断只作为归因质量控制和负控；除非它显著影响 VLM 结果，否则不单独训练 sink-removal 模块。

### 6.1 归因证据等级

实验报告按证据强度分层：patch occlusion/intervention 是决策相关性的校准证据；gradient×activation 是可微代理；attention 权重只是结构线索。任何“模型看到了目标”的结论至少要由干预结果或语言/状态反事实实验支持，不能只凭漂亮热力图。

### 任务

- [ ] 【后期可选】导出 SpikingBrain `[7,15,23,31]` full-attention 层的 q/k/v 或可用权重；
- [ ] 【后期可选】导出 SpikingBrain window/SWA 的局部表征；
- [ ] 标记 GLA 层并记录无显式 attention 的事实；
- [ ] 【后期可选】为 SpikingBrain 实现 V0 相似度归因和 V1 gradient×input；
- [ ] 为一个 Qwen 系模型实现 answer score 到视觉 hidden states 的 gradient×input；
- [ ] 为 LLaVA-1.6/OneVision 实现同样的 answer score 或 grounding score 到视觉 hidden states 的 gradient×input；
- [ ] DeepSeek-VL2 后置，只在 Qwen 与 LLaVA 至少一个接口跑通后再接入；
- [ ] 为 OpenVLA/OFT 记录 action token 或 delta EEF adapter 的动作条件归因入口；
- [ ] 比较早层、中层、后层的空间峰值和熵；
- [ ] 评估随机层和错配词图作为负控。

### 验收

- 有一份按模型、按层、按归因类型的可视化；
- 能解释实际主线模型的归因目标与层选择；SpikingBrain 层号仅供后置参考；
- 如果 full-attention 层无法稳定导出，必须在报告中转为 V1 gradient×input，而不是偷偷使用伪权重。

### Git

```bash
git add src/attention_bridge src/evaluation outputs/attribution_interface_audit.md
git commit -m "feat: diagnose spatial attribution interfaces"
git push
```

## 7. P4：VLM grounding 证明

### VLM 阶段的独立目标

P4 不以 BlindVLA 的 VLA 结果为前置 gate，但应将其抽象成 **DB-style VLM feature-retention baseline**。P4 独立回答：词级扩散空间教师是否让 VLM 的 `A_lang` 更准确地对应语言短语，而不是只让通用视觉 feature 更稳定。

- [ ] 建立 phrase grounding 校准/验证集和独立 VLM 保留集；前者测 region，后者测目标/属性/位置概念；
- [ ] 分开保存 raw query attention、general-prompt ratio、answer/phrase-score gradient 与输出扰动图；它们不是同一个归因量；
- [ ] 使用真实 processor/grid 元数据恢复 patch，不用平方根猜网格；
- [ ] 分别校准 Stable Diffusion、PixArt-α、PixArt-Σ、Playground-v2.5，冻结 best-single 后做 `T_sem → A_lang`；
- [ ] 对照无对齐、DB-style VLM patch feature alignment、扩散教师对齐、错词/错图/随机图、单教师/ensemble；
- [ ] VLM 阶段干预测 answer/phrase score，不在本阶段声称动作归因或闭环 OOD。

BlindVLA 的 ratio 可视化、token/grid 审计和教师缓存思想可作为实现参考。P4 的 DB-style 只是将其 feature-retention 思想迁移到 VLM SFT，不是完整 BlindVLA 复现；P6 再以同样思想作为 VLA 强 baseline。SpikingBrain 保持后置，具体 VLM/VLA 版本待筛选。

### P4 的 VLM 对照矩阵

| 组 | VLM 训练项 | 要回答的问题 |
|---|---|---|
| V0 | 原始 checkpoint，仅评估 | 未适配视觉/语言/空间能力 |
| V1 | 相同数据普通 VLM SFT | SFT 对 grounding 与保留能力的影响 |
| V2 | V1 + DB-style 中层 patch feature alignment | 通用视觉表征保持是否已足以改善 grounding/OOD |
| V3 | V1 + best-single `T_sem → A_lang` | 词级空间监督是否有独立作用 |
| V4 | V2 + V3 | 词级空间监督是否在 feature retention 之上仍有增量 |

V2 的 feature teacher、teacher preprocess、层位置、projector seed、额外前向成本必须固定；若使用 BlindVLA 上游脚本，先修正可训练 projector 参数组和 checkpoint 恢复问题。V3 的扩散教师在校准集冻结。VLM 阶段不计算 `A_act`、不训练 D0，也不使用 VLA action success 作为主指标。

**VLM 继续条件**：V3 或 V4 必须在 phrase grounding 上优于 V1/V2，且正确教师优于错词/错图/随机教师；若 V2 已经覆盖 V3/V4 的收益，空间教师主张退回为 feature retention 的替代实现，不直接进入 D0 的强创新叙事。

### 任务

- [ ] 选择首轮 VLM grounding 数据：Flickr30k / Flickr30k Entities，优先复用 Lavender 公开的 Flickr1k Stable Diffusion attention maps；
- [ ] 对每个样本生成 Stable Diffusion、PixArt-α、PixArt-Σ、Playground-v2.5 的可用词级教师图；无法导出语言条件二维图的候选退出 T_sem 比较；
- [ ] 在独立校准集审计每个教师的 token span、层/步/CFG、pointing/IoU、无效词率和跨 seed 一致性；
- [ ] 冻结 best-single 教师后再训练主实验；多教师 ensemble 仅作为校准权重冻结的消融，不能先平均后称主教师；
- [ ] 在选定 VLM 上比较无对齐、DB-style feature alignment、各个单扩散教师的 attention/attribution 对齐、teacher-to-attribution 对齐；
- [ ] 在一个 Qwen 系模型上使用 gradient×input 归因接口做同一教师对齐；
- [ ] 在 LLaVA-1.6/OneVision 上使用同一 teacher-to-attribution 接口做 VLM 普适性对照；
- [ ] 记录正确教师图、错词教师图、错图教师图、随机教师图；
- [ ] 报告 pointing accuracy、目标区域 IoU、VQA/grounding accuracy、归因图熵。
- [ ] 对 VLA 热力图增加 patch occlusion 的 `Δa` IoU/pointing 对照，检查 attention 与真实动作敏感区域是否一致；
- [ ] 增加语言反事实（相关/不相关指令）和 state 遮挡反事实，区分视觉 grounding、语言条件和 state shortcut；
- [ ] 同时报告 raw attribution 与 sink-free attribution，避免仅凭未经校正的热力图宣称方法有效；
- [ ] 增加 sink-only、边缘 patch-only 和随机空间图负控，检验指标是否会被固定热点投机获得；

### 验收

- 至少 Qwen 系与 LLaVA-1.6/OneVision 两个结构不同模型上，正确教师图优于错图/随机教师图；
- teacher-to-attribution 对齐不劣于 Lavender 原式 attention 对齐；
- 主线模型的结构感知层选择有同层数、同预算消融支持；
- 若仅一个主线模型有收益，应限定适用范围，不能声称普适。

### Git

```bash
git add src/attention_bridge src/models configs outputs/vlm_grounding
git commit -m "exp: validate attribution alignment on vlm grounding"
git push
```

## 8. P5：LIBERO delta EEF smoke

### 任务

- [ ] 接入 LIBERO 环境；
- [ ] 选一个最小 pick-place / put-object-into-container 任务；
- [ ] 将 observation、instruction、proprioception 转为统一输入；
- [ ] 接入 7D delta EEF action head；
- [ ] 实现动作限幅、归一化和反归一化；
- [ ] 保存完整 rollout 和失败原因；
- [ ] 保存 patch occlusion、语言反事实和 state 遮挡的 rollout/动作差异，作为 LIBERO shortcut 诊断；
- [ ] 先用随机或冻结主干验证环境接口。

### 验收

- 环境能连续执行至少一个 episode；
- action shape 始终是 `[7]` 或 `[H,7]`；
- 轨迹不会因为坐标系、旋转或夹爪符号错误而立即崩溃；
- 结果可由固定 seed 重放。

### Git

```bash
git add src/policies src/libero scripts configs
git commit -m "feat: add libero delta eef rollout interface"
git push
```

## 9. P6：VLA 主方法训练

### P6-0：与 BlindVLA 的 VLA 强 baseline 比较（P6 首要 gate）

BlindVLA 是 VLA 微调阶段的 feature-retention 方法，不是 P4 VLM grounding 的前置方法。本阶段要检验：即使视觉表征已被 DB-style 方法保持，语言条件空间归因与动作证据约束是否仍有独立价值。

- [ ] 固定一个 VLA 学生、起始 checkpoint、训练 episodes/划分、LoRA、动作头、action chunk、图像增强、优化步数和随机 seed；
- [ ] 固定 DB-style 视觉教师、层位置、projector seed/恢复状态和 bridge 配置，记录额外前向、显存与 wall-clock；
- [ ] 使用 P4 已冻结的 best-single `T_sem`，不在此阶段重新选择扩散教师或混入 ensemble；
- [ ] 按下表进行 paired comparison：

| 组 | VLA 训练项 | 目的 |
|---|---|---|
| B0 | 原生 VLA BC/SFT | 原始动作能力 |
| B1 | DB-style 中层 patch feature alignment | 视觉表征保持是否已足以解释收益 |
| B2 | `L_sem`：冻结 `T_sem → A_lang` | VLA 内语言空间 grounding 是否有独立作用 |
| B3 | B1 + B2 | 表征保持与语言空间语义是否互补 |
| B4 | B3 + D0 soft action-evidence leakage constraint | 在特征已保持后，动作空间证据约束是否仍有增量 |

- [ ] 对 B4 加入 object-only、object+EEF、面积匹配随机 mask 对照，不能把夹爪/接触/障碍区域一律视为泄漏；
- [ ] 采用同源教师控制或明确记录 C-RADIO feature 与扩散 map 同时改变 teacher/监督形式的混杂；
- [ ] 报告 VLM 保留能力、phrase grounding、动作离线/闭环、raw/ratio/gradient/occlusion 一致性和训练成本；

**继续条件**：P4 已确认正确扩散教师改善语言空间 grounding；B4 再在 B3 之上改善动作相关指标或 task-preserving OOD，且正确教师优于错词/错图/随机教师。若 B4 无独立增量，停止把 D0 写作核心创新，转为 feature retention 或空间监督的诊断结果；若 B1 已解释全部动作收益，暂不进入 D1/B 路线。

### 任务

- [ ] 训练无对齐 OpenVLA/OFT 或 π0 系 baseline；
- [ ] 加入所选主线模型的 similarity bridge 作为对照；层号由接口审计确定；
- [ ] 加入 VLM 阶段验证过的 gradient×input 归因对照；
- [ ] 实现 D 路线：`A_lang` 由 Lavender 锚定，`A_act` 默认由 action loss 或 action output 对 visual tokens 的 gradient×activation 得到；
- [ ] 若模型天然提供 action query attention，则作为可解释分支记录，不作为默认前提；
- [ ] 实现 D0 containment：

  $$
  L_{\mathrm{contain}}=\sum_i A_{\mathrm{act}}(i)\left(1-A_{\mathrm{lang}}^{\cup}(i)\right).
  $$

- [ ] 暂不把 `L_phase` 作为首版依赖；只记录 source/target attribution mass，检查是否自然出现 source→mixed→target 迁移；
- [ ] 后期实现 D1a/D1b：用 LIBERO 状态规则或成功 demonstration 事件边界构造 soft phase target `A_phase(t)`；
- [ ] D1 默认使用三阶段 `source / mixed / target`，五阶段 `approach_source / grasp / move / approach_target / release` 只在事件检测稳定后启用；
- [ ] 使用归一化 MSE，初始 `lambda_align=0.05`；
- [ ] 保存 `L_action`、`L_align`、成功率和归因 IoU；
- [ ] 首版采用 `H=1`，不同时引入 action chunk；
- [ ] 记录训练/验证/测试拆分和随机种子。
- [ ] 记录微调前后一个轻量 VLM/VQA 保留集结果，检查 action loss 是否造成视觉语言能力退化；

### 验收

- 对齐损失可下降且没有 NaN；
- 动作损失没有因对齐项而发散；
- 至少完成一个任务组的 baseline 与主方法配对实验；
- VLM 阶段有效的语言教师图在 VLA 中仍能约束 `A_lang`；
- `L_contain` 能减少 `A_act` 落到语言无关区域的比例；
- D0 报告 `mass_source(t)`、`mass_target(t)` 随时间的变化曲线，作为 D1 是否值得启用的依据；
- 训练时使用教师图，推理时不依赖 Lavender。

### Git

```bash
git add src/models src/policies configs scripts
git commit -m "feat: train spikingbrain vla with spatial attribution alignment"
git push
```

## 10. P7：参考模型、VLA 骨干和动作相关性细化

### 任务

- [ ] 一个 Qwen 系模型：完成 VLM grounding 和归因接口对照；
- [ ] LLaVA-1.6/OneVision：完成 VLM grounding 和归因接口对照，作为非 Qwen 系普适性证据；
- [ ] DeepSeek-VL2：作为后置扩展，不阻塞首轮；
- [ ] Qwen-RobotManip：优先做论文/接口对照，确认其表示、运动和行为对齐设计；
- [ ] OpenVLA/OFT：作为第二个 VLA 骨干做普适性验证，跑官方 LIBERO 设置或可比设置，并实现 attribution adapter；
- [ ] B 路线：评估 LingBot-VA / LingBot-Video 是否能从未来 latent、目标状态进展或接触变化中生成 `R_act`；
- [ ] B-next 路线：评估 Next Forcing 的多 chunk future prediction 是否能产生 short/mid/long `R_future`；
- [ ] B-next 只做离线 future-attribution 分析，不在 D0/D1 首轮训练中引入额外 world-model loss；
- [ ] B 路线：评估 π0、LingBot-VLA、Qwen-RobotManip 是否能从候选动作分数或动作归因中生成 `R_act`；
- [ ] C 路线：用 LIBERO demonstration 派生 weak Action-Relevance Refiner：阶段标签、接触点、目标进展、source/target 区域切换；
- [ ] 完成 `D only`、`D + C diagnostic`、`D + B refiner`、`B/C without D` 对照；
- [ ] 统一记录输入图像、指令、动作接口、训练数据和参数量。

### 约束

参考模型的动作接口若不同，必须分开报告原生结果和统一 7D delta EEF adapter 结果，不能混成一张不具可比性的表。VLM 阶段的普适性用 grounding 指标证明，VLA 阶段的普适性优先需要 OpenVLA/OFT 与 π0 系两类接口不同的骨干；SpikingBrain-VLA 后置。B/C 路线必须作为 D 的扩展或诊断出现，不能反过来变成主结论。

### Git

```bash
git add references outputs/model_comparison.md configs
git commit -m "exp: compare vlm and vla attribution adapters"
git push
```

## 11. P8：消融和反证实验

必须完成：

- [ ] 无对齐；
- [ ] 所有层对齐；
- [ ] 【SpikingBrain 后期可选】`[7,15,23,31]`；
- [ ] 【SpikingBrain 后期可选】`[23,31]`；
- [ ] 【SpikingBrain 后期可选】仅 `[31]`；
- [ ] 随机层；
- [ ] 打乱词图；
- [ ] 错词教师图；
- [ ] 错配图片教师图；
- [ ] `L_sem only`、`L_sem + L_contain`、`L_sem + L_contain + L_phase`；
- [ ] `L_phase` 负控：反向 phase、随机 phase、固定百分比分 phase、正确 phase 但错 source/target 词图、只加 `L_phase` 不加 `L_contain`；
- [ ] `T_sem only`、`T_sem + random R_act`、`T_sem + wrong-stage R_act`、`T_AR correct`；
- [ ] `D only`、`D + C diagnostic`、`D + B refiner`、`B/C without D`；
- [ ] 直接蒸馏 action expert 的策略动作作为负面对照，证明本方法不是策略蒸馏；
- [ ] world/video-action refiner 与扩散教师分开比较，不能混入首轮 VLM 主结论；
- [ ] 增加 head-wise、layer-wise 与 head-mean 对照，禁止只展示单层平均 attention；
- [ ] 增加 LIBERO 与一个分布更真实或视觉变化更大的保留集/数据增强对照；若暂时没有 DROID 数据，至少做背景、机械臂位姿、物体颜色和相机裁剪扰动；
- [ ] Next Forcing 的 `R_future` 与 LIBERO event phase、D0/D1 `A_act` 分开报告；
- [ ] window/SWA/GLA 伪 attention 直接 MSE；
- [ ] V0 similarity 与 V1 gradient×input。
- [ ] raw map vs sink-free map；
- [ ] sink mask 类型：特殊 token、固定边缘 patch、数据驱动高频 patch；
- [ ] sink 诊断关闭时的结果，确认收益不是由预处理本身造成。

每个消融至少固定数据划分、seed、训练预算和动作头。若资源有限，VLM 阶段优先保证正确教师、错词、错图、随机教师四组；VLA 阶段优先保证无对齐、L_sem only、D0、错图，并与简单 feature anchoring/增强比较。

### Git

```bash
git add configs outputs/ablation
git commit -m "exp: run hierarchical alignment ablations"
git push
```

## 12. P9：最终报告

### 报告内容

- [ ] 总体和分任务 success rate；
- [ ] VLM grounding accuracy、pointing accuracy、目标区域 IoU；
- [ ] delta EEF MAE/RMSE；
- [ ] 轨迹平滑度；
- [ ] 目标区域 IoU/pointing accuracy；
- [ ] 训练损失曲线；
- [ ] 层选择和负控结果；
- [ ] 失败案例可视化；
- [ ] 对 Qwen/LLaVA/OpenVLA 的归因接口和可比性说明；DeepSeek-VL2 作为后置扩展说明；
- [ ] 教师选择说明：为什么首轮使用扩散语义教师，VLA 阶段如何纳入 Action-Relevance Refiner；
- [ ] 细化器分析：`R_act` 是否把物体级语义图改造成阶段条件动作相关图；
- [ ] 真机阶段新增风险与所需接口。

### 最终判定

主方法只有在以下条件同时满足时才可进入真机准备：

1. VLM 阶段在至少两个结构不同的模型上优于无对齐和错图/错词教师；
2. 相比无对齐 baseline，LIBERO 成功率提升；
3. 正确教师图优于打乱/错配教师图；
4. 选择性 full-attention 或有效归因接口优于所有层统一对齐；
5. 结果在至少两个随机种子或等价重复实验中保持；
6. 推理时可以完全移除 Lavender。

### Git

```bash
git add outputs reports doc
git commit -m "report: summarize vlm and libero validation"
git push
```

## 12. 时间安排与停止条件

建议按“文档与接口 → VLM grounding → LIBERO 单任务 → 小规模对照 → 完整消融”的顺序推进。任何阶段若发现以下问题，应暂停扩大实验：

- 坐标桥无法通过合成图测试；
- action shape、旋转定义或夹爪符号不一致；
- full-attention 层输出无法复现；
- Qwen/LLaVA/OpenVLA 的视觉 hidden state 无法稳定映射回空间；
- VLM grounding 阶段正确教师图不优于错图/错词教师；
- 对齐损失下降但动作成功率持续下降；
- 负控与正确教师图效果没有区别。

这些问题应先写入 issue 和 commit，而不是通过增加训练轮数掩盖。

## 核心主线与数据集定位补充

## 主线模型待筛选：SpikingBrain 已确定后置

从本版本开始，SpikingBrain 不再是主实验骨干。主线采用“模型无关归因桥 + 成熟 VLA paired baseline”的组织方式：

| 优先级 | 候选 | 角色 | 进入主线条件 |
|---|---|---|---|
| P0 | OpenVLA/OFT | 首选开放 VLA，保留原生 action token/adapter | checkpoint、LIBERO 设置和视觉 hidden state 接口跑通 |
| P0 | π0/π0.5 系 | 连续 flow/action expert 对照 | 能稳定抽取 action-conditioned visual attribution，许可证和权重可用 |
| P1 | Qwen-VL、LLaVA-OneVision | VLM grounding 与归因桥普适性 | 同一 `T_sem` 下正确教师优于错图/随机图 |
| P2 | LingBot-VLA | 多 embodiment/视频动作扩展 | 主线两个 VLA 完成后再加入 |
| P3 | SpikingBrain | 类脑、局部—全局层级和稀疏效率扩展 | 不阻塞主线，只做结构迁移/附加分析 |

模型选择遵循 *Breaking the Vision–Action Shortcut* 的结构覆盖原则：优先选接口不同的 VLA，而不是堆叠同一系列模型。每个主线骨干都必须拥有 architecture-matched paired baseline：相同数据、训练预算、动作表示、native action objective 和评估协议。

### 可选择的论文主线组合

| 组合 | 主模型 | 优点 | 风险 | 建议 |
|---|---|---|---|---|
| A | OpenVLA/OFT + π0 系 | 覆盖 action token 与连续 flow expert，最能证明归因接口跨动作头 | π0 接口和权重工程复杂 | 首选论文主线 |
| B | OpenVLA/OFT + LingBot-VLA | 覆盖开放 VLA 与视频/多 embodiment 路线 | 数据和视频接口更重 | 资源允许时的扩展主线 |
| C | OpenVLA/OFT 单骨干 + Qwen/LLaVA VLM | 工程最稳，适合先快速形成结果 | 跨 VLA 普适性较弱 | 首轮最小可交付 |
| D | SpikingBrain + OpenVLA | 可讲类脑异构结构 | 会把论文叙事拉回 SpikingBrain，主线复杂度高 | 只作为后续扩展 |

A 仅为建议，未由用户冻结；不设置未经确认的两周期限。先完成候选接口审计，再由用户筛选模型组合。C 可先做最小试验，但单个 VLA 不足以证明跨 VLA 普适性。

### 从论文借鉴的 benchmark 设计

- LIBERO 负责 ID 与闭环机制；DROID 负责真实视觉分布下的离线泛化；
- 参考 LIBERO-Plus 的 task-preserving OOD 思路，在 DROID 设计 camera、background、object、scene、operator 划分；
- 每个主线 VLA 使用 paired baseline，不直接比较不同模型的原生绝对分数；
- attention/gradient 图必须和 patch occlusion、语言反事实、state 遮挡一起报告；
- 结果分为性能表、OOD 表、机制诊断表和组件消融表，不把所有指标塞进单一总表。

### OOD 专项计划：检验视觉/语义捷径，而非泛化万能论

本方法的 OOD 假设是：当任务目标和动作可行性保持不变，而背景、光照、干扰物或语言表述变化时，经过语义空间与动作证据约束的策略应更稳定；当目标词或目标位置变化时，策略应相应改变。该假设不自动覆盖新运动学、跨 embodiment、重接触动力学和未校准相机几何。

| OOD 类别 | 具体变化 | 首选评估 | 预期可验证结论 | 不可据此声称 |
|---|---|---|---|---|
| 视觉无关变化 | background、texture、lighting、distractor、sensor noise | LIBERO-Plus / SimplerEnv / DROID 分组 | 动作对无关视觉变化更稳定，且 `A_act` 对无关区域泄漏减少 | 已解决所有视觉 OOD |
| 语言/指代变化 | rephrase、object swap、多实例颜色/属性、position swap | LIBERO-PRO / VL-Think 风格保留集 / DROID 指令分组 | 模型能重新将语言落到正确对象或位置 | 已具备任意开放世界语言能力 |
| 状态变化 | robot initial state、有限范围 proprioception 变化 | LIBERO-Plus / paired state intervention | 正常 state 使用与错误 shortcut 可区分 | state 依赖本身就是捷径 |
| 相机与几何 | viewpoint、crop、有限相机变化 | LIBERO-Plus / SimplerEnv / DROID camera split | 对已校准或可处理的视角变化更稳健 | 已解决三维空间和相机外参问题 |
| 动作/embodiment/动力学 | 新 action space、接触、长时序、双臂、移动底盘 | RoboTwin、CALVIN、真机、后期 LingBot | 只在单独实现 D1/B/动作适配后评估 | D0 自动解决跨 embodiment |

每类 OOD 至少报告：ID 对照、每个扰动维度的结果、正确教师与错教师/随机教师、以及语言/视觉/state 反事实。总平均只能作摘要，不能掩盖特定维度退化。若 ID 提升但 task-preserving OOD 无提升，主张限定为 ID 训练效果；若 DROID 离线误差改善但没有闭环测试，主张限定为真实数据离线泛化。

### OOD 验收门槛

- [ ] 对无关视觉扰动：动作变化、success 或离线误差不劣于 baseline，并且目标区域干预比背景干预更能改变预测；
- [ ] 对目标/指令反事实：替换目标词、颜色或位置时，预测应朝新目标改变；不变化不是“鲁棒”；
- [ ] 对 state 反事实：在物理有效范围内成对改变 state，区分必要控制信息与背景/状态捷径；
- [ ] 对 DROID：按 episode 与可靠 metadata 分组，禁止随机切帧；没有 object/operator 字段时不虚构该 split；
- [ ] 对结论：只有正确教师优于错词/错图/随机教师，且 OOD 趋势在至少一个闭环 benchmark 与一个真实数据离线分组中成立，才使用“缓解视觉/语义 OOD”表述。

## 文档同步规则

- [x] `doc/` 下的三个 Markdown 已成功同步到飞书目标文件夹；
- [ ] 此后任何一个文档发生保存修改，都必须实时同步到对应飞书云文档；
- [ ] 使用 `doc-sync-main` 的文件夹实时同步模式，保持进程持续运行：

```bash
cd /home/bubble/类脑计算/doc-sync-main
/usr/bin/python3.10 main.py live --config sync_config.json --poll-interval 3
```

该模式监听 `/home/bubble/类脑计算/VLM终局/doc` 下的 Markdown 文件，并将本地修改推送到飞书；同时会轮询云端变化并回写本地。运行前必须确认 User Access Token 有效、飞书应用权限未被撤销。若进程停止，实时同步规则暂时失效，应重新启动上述命令。

核心 idea 是用 Lavender 的语言条件空间教师约束空间归因，并让 VLA 动作归因与语言相关区域一致。P1–P4 是主线必需，负责接口、坐标桥和 VLM 证据；P5–P6/P8–P9 是主线验证，负责 LIBERO 机制验证、DROID 泛化验证和反事实排除；P7 的 B/C 与 D1 是后期扩展，不能改变 D0 核心假设。任何新增模型必须说明其对归因可信度、动作相关性或跨模型迁移的贡献。

WAM/VAM 先验证记录轨迹上的未来目标与 wrong-action/wrong-future 负控，再检查泛化收益；DROID 离线误差不能证明闭环成功率，后者须在 LIBERO/真机验证。视频生成质量、attention 图美观或单步 action loss 改善本身不构成升级依据。

LIBERO 只承担可控的接口与机制 smoke test；DROID 是最终数据泛化验证集。DROID 阶段必须固定官方版本和 commit，选择与 pick-place 对应的子集，统一 RGB、语言、proprioception 和 7D delta EEF，按场景或任务划分数据，并重跑 occlusion、语言反事实、state 遮挡和 sink-free 对照。主结果至少包含 baseline、D0、错误教师和随机教师。

当前仍需确认：DROID 首轮子集与 episode 数、视频帧率、动作是否直接采用官方末端增量表示、首轮做离线预测还是 rollout，以及 LIBERO/DROID 的机制与泛化分工。当前建议 LIBERO 做机制、DROID 做泛化，具体字段待下载完成后冻结。
