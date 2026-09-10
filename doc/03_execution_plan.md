# 执行计划与 Git 操作：LIBERO 优先的最终 VLA 方案

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

## 2. 阶段总览

| 阶段 | 目标 | 产物 | 完成标记 |
|---|---|---|---|
| P0 | 文档、仓库、文献目录冻结 | 三份 doc、README、references | [ ] |
| P1 | 本地代码和模型接口审计 | 入口表、张量规格、版本记录 | [ ] |
| P2 | attention 教师图和坐标桥诊断 | 可视化、恢复单元测试 | [ ] |
| P3 | SpikingBrain 层级 attention 诊断 | full/window/GLA 报告 | [ ] |
| P4 | VLA delta EEF smoke | LIBERO 单任务闭环 | [ ] |
| P5 | 层级对齐训练 | V0 bridge + action head | [ ] |
| P6 | 参考模型和基线比较 | Qwen/DeepSeek/OpenVLA/π0/LingBot 对照表 | [ ] |
| P7 | 消融和反证 | 随机、错图、错层、无对齐结果 | [ ] |
| P8 | LIBERO 最终报告 | 指标表、曲线、结论、下一步真机计划 | [ ] |

GPU 型号、数量和显存不作为 P0 的阻塞项；服务器由用户提供，实验只记录运行环境摘要。

## 3. P0：文档、仓库和文献归档

### 任务

- [x] 写清项目背景和原始想法来源；
- [x] 明确 Lavender 词图与 SpikingBrain patch 归因之间需要桥接；
- [x] 固定 delta EEF 7D 输出；
- [x] 固定 LIBERO 优先、真机后置；
- [x] 初始化 Git；
- [x] 创建 `references/papers`、`references/bib`；
- [ ] 下载首批核心论文并记录 SHA256；
- [ ] 在文档中登记无法确认的模型版本。

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
- [ ] 记录 SpikingBrain 当前 commit `ef99987`；
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

## 6. P3：SpikingBrain 层级诊断

### 任务

- [ ] 导出 `[7,15,23,31]` full-attention 层的 q/k/v 或可用权重；
- [ ] 导出 window/SWA 的局部表征；
- [ ] 标记 GLA 层并记录无显式 attention 的事实；
- [ ] 用 V0 相似度归因生成 `A_l(w)`；
- [ ] 比较早层、中层、后层的空间峰值和熵；
- [ ] 评估随机层和错配词图作为负控。

### 验收

- 有一份按层的 attention/归因可视化；
- 能解释为何主实验选择 `[23,31]`；
- 如果 full-attention 层无法稳定导出，必须在报告中转为 V1 gradient×input，而不是偷偷使用伪权重。

### Git

```bash
git add src/attention_bridge src/evaluation outputs/layer_diagnostic.md
git commit -m "feat: diagnose hierarchical attention roles"
git push
```

## 7. P4：LIBERO delta EEF smoke

### 任务

- [ ] 接入 LIBERO 环境；
- [ ] 选一个最小抓取/放置任务；
- [ ] 将 observation、instruction、proprioception 转为统一输入；
- [ ] 接入 7D delta EEF action head；
- [ ] 实现动作限幅、归一化和反归一化；
- [ ] 保存完整 rollout 和失败原因；
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

## 8. P5：主方法训练

### 任务

- [ ] 训练无对齐 SpikingBrain-VLA baseline；
- [ ] 加入 `[23,31]` 的 V0 similarity bridge；
- [ ] 使用归一化 MSE，初始 `lambda_align=0.05`；
- [ ] 保存 `L_action`、`L_align`、成功率和归因 IoU；
- [ ] 首版采用 `H=1`，不同时引入 action chunk；
- [ ] 记录训练/验证/测试拆分和随机种子。

### 验收

- 对齐损失可下降且没有 NaN；
- 动作损失没有因对齐项而发散；
- 至少完成一个任务组的 baseline 与主方法配对实验；
- 训练时使用教师图，推理时不依赖 Lavender。

### Git

```bash
git add src/models src/policies configs scripts
git commit -m "feat: train spikingbrain vla with selective alignment"
git push
```

## 9. P6：参考模型和基线比较

### 任务

- [ ] Qwen2.5-VL：抽取视觉/语言表征，接统一动作头；
- [ ] Qwen3-VL：按官方 checkpoint 能力做表征对照；
- [ ] DeepSeek-VL2：记录高分辨率 token/专家路由差异；
- [ ] OpenVLA：跑官方 LIBERO 设置或可比设置；
- [ ] π0、LingBot：完成结构级比较，只有依赖和数据条件满足时才运行；
- [ ] 统一记录输入图像、指令、动作接口、训练数据和参数量。

### 约束

参考模型的动作接口若不同，必须分开报告原生结果和统一 7D delta EEF adapter 结果，不能混成一张不具可比性的表。

### Git

```bash
git add references outputs/model_comparison.md configs
git commit -m "exp: compare reference vlm and vla backbones"
git push
```

## 10. P7：消融和反证实验

必须完成：

- [ ] 无对齐；
- [ ] 所有层对齐；
- [ ] `[7,15,23,31]`；
- [ ] `[23,31]`；
- [ ] 仅 `[31]`；
- [ ] 随机层；
- [ ] 打乱词图；
- [ ] 错配图片教师图；
- [ ] window/SWA/GLA 伪 attention 直接 MSE；
- [ ] V0 similarity 与 V1 gradient×input。

每个消融至少固定数据划分、seed、训练预算和动作头。若资源有限，优先保证 `[23,31]`、无对齐、随机层、错图四组。

### Git

```bash
git add configs outputs/ablation
git commit -m "exp: run hierarchical alignment ablations"
git push
```

## 11. P8：LIBERO 最终报告

### 报告内容

- [ ] 总体和分任务 success rate；
- [ ] delta EEF MAE/RMSE；
- [ ] 轨迹平滑度；
- [ ] 目标区域 IoU/pointing accuracy；
- [ ] 训练损失曲线；
- [ ] 层选择和负控结果；
- [ ] 失败案例可视化；
- [ ] 对 Qwen/DeepSeek/OpenVLA 等参考模型的可比性说明；
- [ ] 真机阶段新增风险与所需接口。

### 最终判定

主方法只有在以下条件同时满足时才可进入真机准备：

1. 相比无对齐 baseline，LIBERO 成功率提升；
2. 正确教师图优于打乱/错配教师图；
3. 选择性 full-attention 对齐优于所有层统一对齐；
4. 结果在至少两个随机种子或等价重复实验中保持；
5. 推理时可以完全移除 Lavender。

### Git

```bash
git add outputs reports doc
git commit -m "report: summarize libero vla validation"
git push
```

## 12. 时间安排与停止条件

建议按“文档与接口 → 单任务 → 小规模对照 → 完整消融”的顺序推进。任何阶段若发现以下问题，应暂停扩大实验：

- 坐标桥无法通过合成图测试；
- action shape、旋转定义或夹爪符号不一致；
- full-attention 层输出无法复现；
- 对齐损失下降但动作成功率持续下降；
- 负控与正确教师图效果没有区别。

这些问题应先写入 issue 和 commit，而不是通过增加训练轮数掩盖。
