# BlindVLA 官方代码审计：VLM 阶段的重点参考

日期：2026-09-16。来源 https://github.com/CognitiveAISystems/BlindVLA.git；本地 `references/repos/blindvla`；固定 commit `06855fcb91d65c88ca351a2d45af0be860a91987`，shallow clone 已完成，工作区干净。根 LICENSE 为 MIT，子项目和模型/数据许可证另行核验。此次为静态代码审计，未安装训练环境、未下载权重/数据、未运行 GPU 训练，不宣称复现了论文成绩。

## 1. 对用户新要求的落实

VLM 阶段以 Don't Blind 的**表征保持诊断、监督层选择和投影器对照**为重点参考；不能只跑 Lavender 图对齐再看好看的热力图。训练前/后的视觉语言能力应在同一测试集上测量，同时比较 feature alignment 与空间图监督。

主线模型仍待筛选，SpikingBrain 后置。参考 BlindVLA 不等于首轮必须迁移整套 ManiSkill/SimplerEnv；VL-Think 的静态图像诊断可与 Flickr30k Entities/RefCOCO 的空间标注分工。

## 2. 仓库入口与准确行号

下面路径均相对 `references/repos/blindvla`，行号对应固定 commit。

| 入口 | 定位 | 内容 |
|---|---|---|
| README.md | Installation / Alignment / VL-Think | 140 episode、2k steps 的 warm-up 说明，1.4k 训练数据外链与评估环境 |
| openvla/vla-scripts/finetune_align.py | 118–127 | mode=alig、λ=0.2、layer=16、C-RADIOv3-L、冻结 projector |
| 同上 | 130–188 | 反归一化、projector 定义、冻结教师 |
| 同上 | 263–295 | 随机建 projector、冻结与 optimizer 参数组 |
| 同上 | 374–440 | hidden states、教师在线前向、逐 patch cosine、反传 |
| 同上 | 493–568 | eval 与 checkpoint 保存 |
| openvla/vla-scripts/finetune.py | 独立入口 | 普通 SFT，无相同 alignment 配置 |
| openvla/vizualization/vizualize_attention.py | 260–305 | 双 prompt attention ratio 与平方网格可视化 |
| openvla/vizualization/vizualize_tsne.py | 22–58 | 文本对象 token/视觉 pooling 等可选模式，固定 seed |
| openvla/prismatic/extern/hf/modeling_prismatic.py | 400–404 | BOS 后插入视觉 token，再接文本 |
| ManiSkill/mani_skill/envs/tasks/digital_twins/bridge_dataset_eval/put_on_in_scene_multi.py | 3082–3117 等 | 指令、目标名和位置真值接口 |
| SimplerEnv/simpler_env/openvla_eval_batched.py | 40、93–96 等 | seed、并行环境、评估入口 |

README 的命令仍写 `vla-scripts/finetune.py`；实际对齐入口是 `openvla/vla-scripts/finetune_align.py`。根 README 的简化示例还有不完整 `pos, pos_end = 1,`，不能直接复制运行。README 的 π0.5 真机代码为未完成 TODO；不能把它当已支持后端。

## 3. 实际实现的监督对象

学生前向开启 output_hidden_states，取 `output.projector_features.shape[1]` 作为视觉 token 数 N，然后在选定 hidden_states 索引切出 `[1:1+N]`。这与本地模型 BOS 后插入视觉块的实现对应。

```text
student = hidden_states[l][:, 1:1+N]
u = frozen_projector(student)
z = frozen_visual_teacher(image)
L_feature = Σ_l mean_batch,patch[-cos(normalize(u), normalize(z))]
L_total = L_action + 0.2 L_feature
```

它监督的是**内部高维 patch features**，不是词级图，不计算输出梯度归因，也没有动作 containment。多层 loss 是求和而非层数平均，因此层数消融会改变有效损失强度；我们应固定层平均或明确重调权重。

默认 teacher 为 `c-radio_v3-l`。脚本在每个 batch 的 no_grad 分支运行 teacher，没有特征缓存实现；README 的 precomputed 描述不能当作此入口已实现缓存。增加 crop/jitter 时缓存要绑定增强后的图像，否则空间对应不正确。

## 4. 论文没有写清的 projector，代码给出部分答案

AlignmentProjector 是 LayerNorm → Linear → SiLU → Dropout(0.1) → Linear → SiLU → Dropout(0.1) → Linear，内部宽度 2048；输出维度从实际教师结构读取，不能按论文表格把 2048 当 teacher 输出维数。

本入口创建新随机 projector，默认 requires_grad=False 且 eval；没有发现加载已有 alignment_projector checkpoint 的路径。这里的 frozen projector 仍处在 student 的 autograd 路径上，梯度可穿过固定映射更新学生，教师目标没有变成随机标签。**随机冻结本身不是 bug**；需验证不同 projector seed 的稳健性和是否需初始化适配。

比随机初始化更明确的实施问题：

- optimizer 只接收 vla.parameters()，alignment_projector 是独立 module。因此设置 freeze_alignment_projector=False 后，它虽产生梯度却不被 AdamW 更新；不能直接用该开关复现可训练 projector 消融。
- projector 在 DDP 外构造，入口没有显式 seed/broadcast；多卡需保证相同初始化和更新同步，当前静态阅读不能证明已保证。
- checkpoint 只保存 projectors_list[0]，多层 projector 的其他权重没有同样保存；恢复复现需补齐状态。
- mode=orig 不构造 projectors_list，但保存分支仍引用它；baseline 必须使用正确入口或加条件保护。

这些是对已下载 commit 的代码判断，不代表作者所有实验分支都使用同样脚本。归档目录保持原样；如后续复现，修补应放独立工作分支，并同时保留 upstream 与 corrected 结果标识。

## 5. 热力图原来是 query/general attention ratio

可视化脚本以两个 prompt 分别前向：对象问题（默认 Do you see a can?）与通用描述（Describe the scene briefly.），使用 eager attention。每层取最后一个 prompt token 对视觉块的 attention，并对所有 heads 取均值：

```text
q_i = mean_heads attention(object_prompt)[last_query, visual_i]
g_i = mean_heads attention(general_prompt)[last_query, visual_i]
r_i = q_i / (g_i + 1e-6)
```

因此图展示的是相对通用 prompt 的响应增强，不是原始 attention，更不是 action-output gradient attribution。它可能抑制公共热点，也可能放大小分母的噪声；并不等于模型内部 sink 被消除。

源码还用视觉 token 数的平方根猜网格；非平方 token fallback 会截断/填充。对于动态分辨率、多图和 window reorder，必须使用实际 processor/grid 元数据，不能照搬。hidden_states 与 attentions 索引也不是天然同一个层编号：脚本图标题额外加 1，需明确 embedding slot 与 block 编号。

我们的 VLM 图应至少保存五种独立量：raw query attention、general attention、ratio map、answer/phrase-score gradient attribution、扰动后输出分数变化。共享颜色尺度和定量评估，不能只挑 ratio 更好看的图。

## 6. 图像预处理与合法梯度路径

训练脚本把 pixel_values 沿通道分为两个三通道张量。这是 OpenVLA 两个视觉编码器的处理输入，不应误认作 head/wrist 两个相机。RADIO 分支 resize 到 256 并使用硬编码均值/标准差反归一化；跨模型迁移必须改为实际 processor 元数据，并核验裁剪、范围、patch grid 和反变换。

默认 RADIO forward 获取真实输出维度；DINO/Theia 分支存在各自处理方式，不能按同一 normalization 假设。非默认 e-radio 分支使用未定义的 x.shape，运行前应修复或拒绝该配置。

学生视觉 token 出现在 instruction 之前。在严格 causal decoder 下，这些位置不能看见后续语言；其 feature retention 与语言条件图是不同对象。要形成 A_lang，应该使用能读取视觉的语言 query/目标输出分数，而不能把任何视觉 token 相似度命名为词级归因。

冻结 teacher 的 no_grad 是正确的；冻结 projector 不能连输入一起放入 no_grad，否则对学生的辅助梯度也会丢失。我们需用小样本实际检查 alignment loss 对 backbone/action branch 的梯度。

## 7. 训练验证不能原样照抄

除了上面的 optimizer/checkpoint 问题，脚本 eval 段只设 no_grad，没有显式 vla.eval()，而训练前调用了 train()；eval dataset 还传入 image_aug=cfg.image_aug。验证随机性和 dropout 状态需核查，不能仅把每次 eval 波动当方法效果。

没有本地 PyTorch 训练环境，因此本轮只做 AST/源码检查，不声称这些路径已在真实模型上触发或修复。权重、数据、teacher hub 版本仍须锁定；torch.hub 默认获取上游内容，不自动与本仓库 commit 一起固定。

## 8. VL-Think 如何服务我们的 VLM 实验

仓库提供 shape/color、图标、奇偶、箭头等八类板选择任务，控制技能保持简单。get_language_instruction、get_target_name、where_target 可提供场景语义和位置标签。

我们可优先借鉴一个**同场景视觉语言保留集**：固定机器人场景，用不同目标/属性指令测是否识别正确对象和位置；评估 VLM 原 checkpoint、普通 SFT、feature alignment、语义归因对齐。先用静态图片不必马上做完整 rollout，但数据来源、负例、位置平衡要固定。

- presence QA 必须同时有存在/不存在的对象，防止模型总答 yes；
- target location 测 left/center/right 时使用真实标签，不从热力图自身生成真值；
- 用相同控制难度配对任务，区分识别错误与动作执行错误；
- 用 grounding 标注检查位置，用 VQA 检查概念，再用 occlusion 检查输出依赖，三者不能互代；
- t-SNE 的 per-class prompt 带有类别词，聚类可能受语言提示影响；补同 prompt 或纯视觉 pooled features 对照，并报告 linear probe/grounding，不能单凭散点图声称恢复视觉语义。

## 9. 新的 VLM 首轮建议（待用户确认）

| 组 | 配置 | 目的 |
|---|---|---|
| V0 | 原始 VLM checkpoint，仅评估 | 记录未适配的能力 |
| V1 | 相同数据普通 SFT | 标准基线 |
| V2 | V1 + DB-style 中层 patch feature cosine | Don't Blind 重点参考，检验表征保持 |
| V3 | V1 + Lavender-style 词图对齐 | 外部空间先验基线 |
| V4 | V1 + teacher-to-output-attribution 对齐 | 我们的归因监督候选 |
| V5 | V2 + 空间归因约束 | 表征已保持后，空间证据监督是否仍有增量 |

V2 移植到 Qwen/LLaVA 是 adaptation baseline，不称完整 BlindVLA 复现。先单层、固定教师和同数据；随后比较层位置、冻结/可训练 projector、多随机初始化、教师类型。C-RADIO feature 与 SD map 同时改变教师与监督形态，必须另做同源教师控制或承认组合效应。

评估四栏：任务准确率/保留集、phrase-region grounding、raw/ratio/gradient/perturbation 一致性、训练/教师缓存成本。V5 不是一定优于 V2 的预设结论。

## 10. 最小实施验收

- [x] 官方 clone 完成、commit 与许可证登记。
- [x] 核对 teacher、layer、loss、projector 和热力图的源码路径。
- [x] 识别 README 命令与实际入口差异。
- [ ] 后续开始复现时建立独立环境、下载权重与数据。
- [ ] 固定 projector seed/恢复状态；可训练分支加入 optimizer/DDP。
- [ ] 校验 grid/token 与 teacher augment 一致，拒绝截断凑方形。
- [ ] 分清 raw/ratio/gradient 图，存元数据并做负控。
- [ ] 保留原始 checkpoint 及固定验证集，审计 eval mode 与增强。

当前完成“下载与继续分析”；实验运行不在本次已完成范围，主线模型未冻结，SpikingBrain 仍后置。
