# 执行计划与 Git 操作：VLM 优先、VLA 后验的最终方案

> 2026-09-28 当前执行补充：用户授权保留 VLM 全量目标，同时在独立 `vla_workspace` 完成非 GPU OpenVLA/OFT + LIBERO 准备，cache 验收后错峰做 B0 20–50 step/restore/rollout。不得停止当前 cache 或新增 GPU job。见 `experiments/P6-vla-b0-b4/non_gpu_preparation.md`。
>
> 正式训练门禁纠错：SD batch cache 的 1000 条 calibration metadata 均为 1400 attention tensors，混入 inversion/null-text 优化；原 reconstruction-only 路径为 100。旧校准不能认证声明的配方，frozen config 已标为 invalidated，F1 启动器拒绝启动。保留已有产物和当前任务；修复代码先做 CPU 检查，GPU 空闲后在新目录做单样本等价验证。wrong-word 指标存在无同图候选时回退 wrong-image 的标签错误，不得称同图反事实已通过。完整 train 与 10k 子集的范围、正式 trainer 断点/曲线，以及 V2/V4 retention 到策略参数的梯度路径仍须核验；不能凭脚本存在宣布就绪。

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

P1 在本机或 101 的空闲 GPU 做小样本 audit；正式训练只在 101 提交火山 8 卡任务。服务器连接、VEPFS 目录、环境隔离、提交模板和资源约束见 `server/01_vla_attention_training.md`。

### 任务

- [x] 2026-09-17：101 直连与 VEPFS 可写性验证；项目根固定为 `/vepfs-mlp2/c20250405/400040/transfer/vla_attention/`；不得使用 `/root` 放置项目文件或缓存；
- [x] 2026-09-17：创建 `repo/`、`envs/`、`hf_cache/`、`models/`、`data/`、`teacher_maps/`、`runs/`、`checkpoints/`、`results/`、`jobs/` 目录；VEPFS 当时约余 620 TB；
- [x] 2026-09-17：确认既有 `/root/starvla_cu124` 可导入 Torch、Transformers、Hugging Face Hub、Datasets、Diffusers、Accelerate、PEFT；该环境只作为依赖基线，后续不修改它；
- [ ] 将本项目已提交代码传入 `repo/VLA_attention/`，并在 `envs/p1/` 建立项目专属环境；记录 Python/torch/transformers/diffusers 版本；
- [ ] 用服务器网络检查 Hugging Face、Flickr30k/Entities 官方来源和 7897 代理可用性；下载失败时记录 URL、时间、HTTP 状态和镜像替代，不静默换源；
- [ ] 冻结 Qwen2.5-VL-7B 的官方 checkpoint revision、许可证、SHA256 和实际磁盘路径；第二 VLM 不作为 Qwen 全量验证的启动门槛；
- [ ] 下载 Flickr30k Images 与 Entities annotations 的完整原始资料，保存来源和校验值；取得许可/访问限制时停在该 gate，不用不明镜像替代；
- [ ] 从完整数据建立可复现 train/calibration/test manifest；P1 从 calibration manifest 固定抽取 20 个样本，不替代后续全量训练；
- [ ] Qwen2.5-VL-7B 完成并冻结 P1 `report.json`，用 `scripts/validate_p1_report.py` 校验后进入正式 V0–V4；第二 VLM 的 P1 在 Qwen 主线启动后并行准备；
- [ ] Prismatic 曾列入首轮双模型候选，但其 Llama-2 衍生权重需要合法 Hugging Face 登录与条款访问；这不再阻塞 Qwen 首轮，且不得以镜像或未授权副本绕过；
- [x] Qwen checkpoint 配置已审计：vision depth 32、full-attention blocks `[7,15,23,31]`、patch size 14、spatial merge 2、window size 112；P1 坐标桥须使用 post-merge grid；
- [ ] 记录 Lavender 当前 commit `58fc71b`；
- [ ] 【后期可选】记录 SpikingBrain 当前 commit `ef99987`；
- [ ] 导出实际 `window_size`、`fullatt_block_indexes` 和视觉层数；
- [ ] 确认 `grid_thw`、window reorder 和 patch index 的变换；
- [ ] 确认语言 hidden state 与动作 query 的获取位置；
- [ ] 检查每个 checkpoint 是否真的支持 `output_attentions`；
- [ ] 生成 `outputs/interface_audit.md`。

### 2026-09-17 训练前准备进度与严格边界

**已完成**：服务器直连、VEPFS 可写性、独立路径与依赖基线检查。101 的系统盘仅剩约 207 MB，所有本项目的 repo、虚拟环境、HF cache、模型、数据、日志与 checkpoint 都只允许放在上述 VEPFS 根；101 GPU 只用于 P1 调试，正式训练仍须在火山 8 卡队列提交。

**当前未完成**：代码传输、项目环境、模型/数据下载、全量 split、任何一个模型的 P1 结果。当前不能声称“开始训练”，也不能把 20 个 P1 样本称为全量实验。

**分阶段下载**：先保障 Flickr30k Entities、Qwen2.5-VL-7B 和已通过 smoke 的 SD1.5 fallback。Qwen P1 通过后，可以并行准备四候选 teacher calibration，以及 OpenVLA/OFT + 单任务 LIBERO 的环境与动作接口 smoke；无需等待完整 VLM V0–V4 结论才开始 VLA 工程准备。PixArt、Playground、Prismatic、π0.5、MolmoAct2、DROID 等扩展模型/数据仍按校准结果、接口和存储情况分阶段加入，避免一次性占用带宽和磁盘。

**全量数据的使用**：P1 固定抽 20 个校准样本，目标是发现动态分辨率、patch token、phrase score 和梯度路径错误。teacher calibration 使用完整独立 calibration split；V0--V4 使用完整训练集，完整 Flickr30k Entities test 和 RefCOCOg 评估；不因 P1 已通过就跳过全量数据训练与评价。

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

### P4 benchmark 冻结：词级空间 grounding 与 VLM OOD

VLM 阶段不直接使用 SimplerEnv 作为主 benchmark，因为它是控制环境，不能替代 phrase-region 真值。首轮按以下四层组织：

| 层级 | 数据/设置 | 角色 | 主指标 |
|---|---|---|---|
| Teacher calibration | Flickr30k Entities 训练集以外的独立 calibration split | 选择 `T_sem` 的 best-single、token span、层/步/CFG | phrase pointing、region IoU、无效词率、跨 seed 一致性 |
| ID grounding | Flickr30k Entities 官方冻结 test split | 主 VLM 空间 benchmark | phrase pointing、mass-in-box、IoU、answer/phrase attribution agreement |
| Semantic transfer | RefCOCOg 官方冻结 split | 更长 referring expression、多实例与关系指代的外部迁移 | pointing、IoU、指代正确率 |
| Visual task-preserving OOD | 在 Flickr30k Entities test 图上生成固定 brightness/contrast/color/noise/blur/JPEG/resize 扰动，box 不变 | 不依赖无关视觉因素的空间 grounding | 每种扰动相对 ID 的下降、正确/错教师差距 |

VL-Think/SimplerEnv 风格的静态截图 QA 仅做 VLM 保留诊断，检验目标、属性、位置和 yes/no，不作为 phrase-region 主榜；完整 SimplerEnv control 放在 VLA OOD 阶段。不得把 RefCOCOg 的外部表达迁移、Flickr 的图像扰动和 VLA 闭环 OOD 混成一个平均分。

### P4.0 2026-09-24 最终冻结条款（主矩阵、训练、指标、复现）

本节是运行前的最终 protocol。它冻结的是比较协议和选择规则，不把尚未完成的 pilot 结果写成已验证结论。任何改动都必须新建 protocol 版本，不覆盖已有结果。

#### 3. V0--V4 主矩阵与机制对照

| 组 | 训练内容 | 必须回答的问题 |
|---|---|---|
| V0 | 原始 Qwen2.5-VL-7B checkpoint，仅统一评测 | 原始模型在同一评测器上的起点是什么 |
| V1 | 完整 Flickr30k caption SFT，只有 caption CE | 提升是否只是普通 SFT 的结果 |
| V2 | V1 + 冻结 DINOv2 ViT-L/14 的中层 patch feature retention | 通用视觉表征保持是否已经足够 |
| V3 | V1 + 单一冻结 `T_sem → A_lang` | 词级语义空间归因是否有独立价值 |
| V4 | V2 + 与 V3 完全相同的 `T_sem → A_lang` | 语义归因是否在 retention 之上仍有增量 |

V3 与 V1、V4 与 V2 是论文主比较。每个主比较都配套正确教师、错词、错图、随机图和 raw-attention/rollout proxy；负控使用相同图像、短语、训练预算和随机种子。DVD-style feature distillation、完整 BlindVLA policy 和多教师 ensemble 是独立对照或扩展，不得混入 V0--V4 主矩阵。VLM 阶段不加入 `A_act`、D0/D1 或 action success。

#### 4. 训练与 teacher 选择规则

**为什么首轮用 1 epoch。** 1 epoch 不是理论最优值，而是首轮正式实验的固定计算预算：完整训练集恰好遍历一次，能让 V1--V4 在相同样本顺序、相同更新次数和相同 teacher-map 覆盖率下比较，避免某个组因为训练更久而获得不公平优势。它也便于将当前 20--50 step smoke、F0 小样本和正式全量结果分开。若所有组在 1 epoch 结束时的 validation caption loss 仍共同明显下降，预先登记一个 **所有组一起延长到 2 epoch** 的敏感性实验；不得按单个模型的 test 结果决定是否续训。

正式首轮冻结为：

- 完整有效 Flickr30k caption train manifest，1 epoch（等价的 `max_steps` 由冻结 manifest 计算）；
- V1--V4 使用完全相同的 BF16、LoRA target/rank、effective batch、optimizer、warm-up、checkpoint cadence 和数据顺序；
- 三个正式训练 seed：17、29、41；V0 只需一次原始 checkpoint 评测。若主效应接近判定边界，再增加预先登记的 seed 53、67，不能只为挑出有利方向；
- 无 test-based early stopping；validation 只用于监控和预先声明的配置选择；
- `gradient_clip_norm` 只作用于 optimizer 将更新的参数，attribution 反传使用独立的梯度生命周期；
- semantic loss 的 `lambda_sem=0.10`、temperature=1.25、warm-up=50、16×16 map bridge 和 q=.95 先作为 **provisional candidate**，待正式 calibration 重新确认后写入不可变 config。

**Teacher 何时确定。** teacher 在正式 V1--V4 训练之前确定，不能训练完再挑 teacher。先在与 train/test 按 image ID 隔离的约 1000 张 validation calibration split 上，用同一 phrase span、同一 null-text/attention 抽取规则、同一归一化、同一 16×16 bridge 比较 SD1.5、PixArt-α、PixArt-Σ 和 Playground-v2.5。选择规则只看 validation：

1. 主排序指标为 phrase pointing；
2. tie-break 为 mass-in-box/soft-IoU，再看无效词率、跨 seed 一致性和缓存失败率；
3. 不使用官方 test、RefCOCOg、OOD 或下游 capability 分数选 teacher；
4. 只选一个 best-single，不把 ensemble 当作主方法；
5. 若新 teacher 在 calibration 期间合法可用且按相同 protocol 完成校准，可以加入候选；否则不为“新”而改变主线；
6. 若没有候选通过最低有效性检查，使用 SD1.5 作为工程 fallback，并如实报告“teacher 选择未建立”，不得宣称 SD1.5 最优。

teacher 选定后冻结模型 revision、attention block、diffusion step、CFG/null-text、token span、normalization、map grid、cache manifest 和 SHA256；随后才生成 train/test cache。生成 test cache 不改变任何选择。

#### 5. 指标、官方评测与多元成功标准

所有 benchmark 使用其官方 split、官方 prompt/输入格式、官方 evaluator 和固定版本；内部 pilot scorer 只用于调试，不进入论文主表。主次关系如下：

- **主指标（空间机制）**：Flickr30k Entities phrase pointing。V3/V1 与 V4/V2 的 paired difference 按 image cluster bootstrap 报告；
- **空间次指标**：mass-in-box、soft-IoU、calibrated box IoU、top-k IoU、attribution entropy、跨增强一致性、patch occlusion/intervention agreement；
- **能力次指标**：COCO Captions、VQAv2、TextVQA、POPE、MME、WorldMedQA-V；分别使用官方 evaluator，逐项报告绝对分数、相对 paired baseline 的增量和 95% 区间，不用未经完成的项目计算总平均；
- **外部迁移/OOD**：RefCOCOg 官方 split、Flickr task-preserving visual perturbation；它们不参与 teacher/threshold 选择；
- **效率指标**：达到相同 validation caption loss 或 grounding 分数所需的 optimizer steps、wall-clock、峰值显存、离线 teacher-cache 时间和总成本。若只减少在线步数但 cache 成本大幅增加，不能直接称作整体更快。

成功标准分层，避免“只有 IoU 上升才算成功”：

1. **强成功**：V3>V1 且 V4>V2；主指标在三 seed 同方向，image-cluster paired 95% CI 下界大于 0，并达到预先登记的实用增量（默认 pointing 绝对 +2 个百分点）。
2. **效率成功**：主指标不劣于对应 baseline（允许的非劣界为 -1 个百分点），同时在相同目标分数下减少至少 20% optimizer steps 或 wall-clock；必须同时报告离线 teacher 成本。
3. **机制成功**：主指标提升不足，但正确教师稳定优于错词/错图/随机图，`V3 > V3-attn-proxy`，intervention/OOD 一致性改善，且官方能力 benchmark 无显著退化。此时结论限定为“机制证据成立”，不写成全面性能提升。
4. **能力成功**：几何主指标保持不退化，并在至少一个官方能力类别上取得稳定 paired 改善；其他指标即使没有提升也完整记录，作为可能的附加发现。
5. **失败**：主指标、机制负控和能力/效率都没有稳定证据，或出现明显语言能力退化。此时停止扩大 teacher/loss 组合，先审计数据、归因接口和训练动态。

单项意外提升必须保留原始结果，但不能改写主结论。统计显著、实用增量、效率、OOD 和 capability 的证据分别报告。

**VLM 到 VLA 的论文叙事。** VLM 阶段证明“语言短语对应的空间归因可以作为跨模型监督接口，并且不是任意空间正则”；VLA 阶段把相同接口接到 action-loss gradient×activation，构造 `A_act`，检验 `A_act` 是否落在 `A_lang` 的目标区域并随动作阶段变化。LIBERO 用于机制和闭环 success，LIBERO-Plus/LIBERO-PRO 用于可控 OOD，DROID 用于按 episode/scene/camera 分组的离线真实数据泛化，SimplerEnv 用于视觉/语义保留诊断，RoboTwin 后置为 embodiment 扩展。首轮不再增加额外 benchmark；只有当主线通过后，才考虑 CALVIN/ManiSkill 等时序扩展。

#### 6. 可复现性与模型普适性

每个正式运行必须同时保存并提交：Git commit SHA、运行 config SHA、数据 manifest/SHA256、模型与 teacher revision、processor/tokenizer 版本、环境 lockfile、seed、命令行、checkpoint SHA、cache manifest、失败样本、训练/评测日志和官方 evaluator commit。VEPFS 运行目录按 `runs/<experiment>/<model>/<seed>/` 隔离，不覆盖历史结果；每完成 10/25/50/75/100% 处理量都更新 `experiment_workspace/PROGRESS.md`。

冻结协议和代码后，必须单独提交一个可审计 commit；不把 checkpoint、teacher map 或大数据文件加入 Git：

```bash
git add doc/03_execution_plan.md \
  experiments/P4-vlm-v0-v4/protocol.md \
  paper_draft/05_experiments_and_expected_conclusions.md \
  experiment_workspace/PROGRESS.md \
  configs/experiments/P4_vlm_v0_v4_template.json
git commit -m "docs: freeze VLM V0-V4 protocol and evaluation gates"
git push origin main
```

若当前工作树含有其他实验改动，先用 `git diff` 审计并拆成独立 commit；不能为了冻结文档把未审计的训练脚本或历史结果一并提交。

普适性采用分层声明：

- Qwen2.5-VL-7B 是首轮主骨干；
- Qwen 完成后，使用同一 protocol 在一个非 Qwen 骨干（优先 LLaVA-OneVision；若 P1/权重合法性不通过则选择 Prismatic-7B）至少复现 V1/V3；
- 只有两个骨干都通过接口审计并呈现同方向机制证据，才能写“跨 VLM 骨干可迁移”；只跑 Qwen 时只能写“Qwen 实例验证”；
- InternVL3.5、Ovis2.5、Qwen3-VL、DeepSeek-VL2 作为后续扩展，不影响首轮主结果；
- 任何模型没有标准 cross-attention 时，统一优先使用 output-conditioned gradient×activation，只有在可审计的 attention 存在时才加入 attention proxy；不能为获得一致性而伪造 attention。

### P4.0 冻结状态

- [x] 主矩阵、主比较、负控和非主线对照边界已冻结；
- [x] 1 epoch 的实验理由、无 early stopping 规则和三 seed 预算已冻结；
- [x] teacher 的 validation-only 选择时点和 best-single 规则已冻结；
- [x] 主/次指标、官方 evaluator、效率/OOD/机制多元成功标准已冻结；
- [x] 可复现性字段和跨模型普适性声明边界已冻结；
- [ ] 1000 图 calibration、最终 teacher、manifest SHA、正式 V0--V4 训练结果仍待执行，不能提前写成结论。

### P4.1 Lavender-style 下游能力评测轨道

P4 不只评估热力图的几何质量，还要复刻 Lavender 的能力验证逻辑。几何轨道回答“归因是否落在目标区域”，能力轨道回答“这种对齐是否改善真实 VLM 能力并保持原有能力”。两条轨道独立报告，不能用一条轨道的提升替代另一条。

**2026-09-24 最新状态：** teacher/student map 的共同网格、top-k/soft-IoU 口径和阈值已做联合校准，当前为 16×16/q=.95；但选择只基于 20 张独立 calibration 图，仍是 provisional。修正版 V3 的 KL、KL+rank、KL+moment、JS 和 V4 KL+rank 均有 3 个训练 seed。按图像聚类的几何 bootstrap 显示四个 V3 loss 的 common soft-IoU 增量为正，但 calibrated box-IoU 区间均跨零；V4 对 V2 也未建立确定的 box-IoU 增益，故当前不选唯一最佳 loss。修正版下游 pilot 已完成：19 个 adapter × 六项任务共 114 份报告，90 个匹配训练 seed 的比较均完成 10,000 次配对 bootstrap，逐项结果见 `experiment_workspace/results/F0v2_corrected_capability_20260923/paired_capability_sweep_20260924.md` 和 JSON。V3 的 COCO CIDEr 增量三个 seed 均为正；VQA/TextVQA 增量受 seed 显著影响。V4 对 V2 的 CIDEr、WorldMedQA-V 均值方向为正，但其他任务不一致。该 pilot 使用小样本，VQA/TextVQA 是内部 scorer，METEOR 缺失；其区间只表达固定 checkpoint 下的样本变化，不能视为训练 seed 总体不确定性或完整官方 benchmark 结论。详见 `experiment_workspace/PROGRESS.md` 和 `experiments/F0v2-idea-validation/geometry_loss_sweep_results/calibrated16_20260923/capability_sweep_interpretation_20260924.md`。

**当前判断与继续 gate：** 已看到值得继续检验的空间归因与 caption 信号，但还没有足够证据证明该方法稳定改善 grounding 或整体 VLM 能力，也没有锁定胜出的 loss。下一阶段应先扩充独立 student calibration（目标 1000 张并冻结 ID），完成正确教师/错词/错图/随机图控制，再以官方 evaluator 和完整冻结 split 复核；至少需要在多个训练 seed 上同时观察到几何主指标与下游任务不退化/改善。未通过这些 gate 前不进入“有效性已证明”的表述，也不把旧 global-clipping pilot 混回主结果。

**已完成的历史运行快照（不再代表当前进度）：** 此处以下方 2026-09-23 的 26/114、61/114 等记录保留审计时间线；修正版下游套件现已完成，状态由本段及 `experiment_workspace/PROGRESS.md` 顶部记录覆盖。

**此前运行快照（26/114，已由下文后续状态覆盖）：** 当时 V1/V2 seed29 与 seed41 均完成六项任务，V3 KL seed17 已完成 COCO 与 VQAv2；V2 seed41 的 COCO CIDEr .772、VQAv2 pilot consensus .763、TextVQA pilot consensus .673、POPE accuracy .849、MME pair accuracy .719、WorldMedQA-V accuracy .426。WorldMedQA-V 的 `B .` 等格式解析缺陷已修复，并从 seed29 原始预测重算：V1 accuracy .484、V2 .461，V2−V1 的样本配对 95% CI 为 [-.066,+.016]。VQA/TextVQA 使用内部 pilot scorer。

**远端续跑核验：** 报告数已增至 **61/114**，同一评测进程仍在运行 V3 KL+moment seed17 的 VQAv2。V3 KL seeds17/29/41 的六项任务均已完成并回收到本地；其他 V3 loss 与 V4 仍待执行。V3 KL 中期、逐 seed 的下游结果见 `experiments/F0v2-idea-validation/geometry_loss_sweep_results/calibrated16_20260923/capability_interim_seed_analysis.md`：CIDEr 和内部 TextVQA pilot 三 seed 上升，但 VQAv2 方向不一致、POPE 与 WorldMedQA-V 改变量小、MME 只有一个 seed 上升，暂不支持广泛能力改善。

**统计审计补充：** POPE pilot 有 384 条问题但只有 128 张图片，配对 bootstrap 已改为按图片聚类；MME 保持以 64 个完整正反 pair 为重采样单位。V3 KL 的 per-seed 结果和区间只反映固定 checkpoint 下的样本不确定性，训练 seed 的离散度另行报告。新实现 `scripts/paired_capability_bootstrap.py` 的 3 项回归测试通过。

**几何统计校正（2026-09-23）：** held-out 64 条 phrase 实际只有 62 张独立图片，其中两张图各有两条 phrase。几何 paired bootstrap 已改为先按 `image_id` 聚合，再按图片重采样；不能把 64 行当作 64 个独立图像样本。四种 V3 loss 的 common soft-IoU 增量均为正，但 calibrated box IoU 的 95% 区间全部跨零；V4 对 V2 也未建立稳定 box-IoU 增益。结果和 sampling counts 见 `experiment_workspace/results/F0v2_geometry16_fixclip_heldout_20260923/cluster_bootstrap/`。校准脚本同时改为联合选择共同分辨率与阈值，重新计算仍为 16×16/q=.95；由于当前 student calibration 只有 20 张图，该选择仍是 provisional。

**能力套件完整性检查：** `scripts/summarize_capability_sweep.py` 检查 19 个 adapter × 6 项任务是否全部有报告，并验证统一 greedy 解码、`max_new_tokens=64` 和 `max_image_pixels=1,003,520`。远端报告数以服务器 `find` 为准；本地只拉回部分报告时不能生成最终能力总表。

**首轮最小能力套件**固定为：

| 能力类别 | Benchmark | 指标 | 作用 |
|---|---|---|---|
| Caption | COCO Captions、Flickr30k captions | CIDEr、BLEU-4、METEOR、ROUGE-L | 对齐是否改善描述生成 |
| General VQA | VQAv2 | 官方 accuracy | 一般视觉问答是否保持 |
| Fine-grained/OCR | TextVQA | 官方 accuracy | 细粒度文字和局部区域能力 |
| Hallucination | POPE | 官方 accuracy / hallucination rate | 是否减少视觉幻觉 |
| Broad perception | MME | 官方默认分数 | 综合感知是否退化 |
| OOD | WorldMedQA-V | 官方多语言 VQA accuracy | 未见领域和语言的迁移 |

F1-10k 方向性结果通过后，扩展加入 `OK-VQA、DocVQA、OCRBench、InfoVQA、MMBench、MMStar、MMMU、ScienceQA、HatefulMemes`。这借鉴 Lavender 的 benchmark 分组和 OOD 设计，但不宣称完整复现其 20 项结果；没有冻结版本或可靠 evaluator 的项目不进入平均分。

能力套件使用与 V0--V4 完全配对的 checkpoint、prompt、解码参数、图像分辨率、few-shot 设置、评测脚本和数据重叠审计。每个项目报告绝对分数、相对 V1/V2 的增量、V3 对 V1 以及 V4 对 V2 的 paired difference；只有一个类别的项目全部完成时才计算 macro average。记录训练样本数、wall-clock、教师 cache 成本、显存和 semantic-map loss，绘制 loss/grounding 与下游分数的关系，但不把相关性当作因果证据。

首轮能力套件只在 `V0/V1/V3` 上先跑通，确认评测脚本与输出格式；随后补齐 `V2/V4`。正确教师、错词、错图和随机图优先在 `VQAv2、TextVQA、POPE` 小子集执行，以判断能力收益是否依赖正确语义图，而不是任意空间正则。

**2026-09-23 pilot 实际进度与临时结论：**

> 2026-09-23 随后的优化器审计发现：当时 V3/V4 使用的全局梯度裁剪把仅用于 attribution 反传、但不参与 optimizer 更新的视觉参数纳入了裁剪范数。故下列 pilot 数字现在只作为历史故障诊断，不能作为当前方法的定量结论；修正版训练与重评结果完成后替换。

- [x] 已在同一固定 pilot ID 和解码/图像预算下完成 Qwen V0、V1、V2、V3-KL+rank、V4-KL+rank 的 COCO Captions、VQAv2、TextVQA、POPE、MME、WorldMedQA-V 六项试点；V3/V4 是 seed17、100-step 单 checkpoint，不能代表 seed 方差。
- [x] 所有模型的 MME 已统一用 128 question rows / 64 complete pairs 的 `mme_pair` schema 补跑；首次 V0/V2 binary-schema 报告弃用，不混入比较。
- [x] 已生成 V3-vs-V1、V4-vs-V1 及 V4-vs-V2 的 paired bootstrap；结果和解释见 `experiments/F0v2-idea-validation/geometry_loss_sweep_results/calibrated16_20260923/capability_pilot_summary.md` 与同目录 `paired_capability_*.json`。V4 相对 V2 的 VQAv2、TextVQA、POPE、MME、WorldMedQA-V 区间均未建立增益。
- [x] Pilot 发现 V3 的 VQA/TextVQA 表现风险：V3 VQAv2 consensus 为 0.159（V1 0.702），TextVQA 为 0.367（V1 0.660）；TextVQA 原始输出有长串重复标点现象。不得把这归为几何监督成功；先审计解码输出、prompt/template、loss 训练动态和 adapter。
- [ ] Pilot 尚非正式 Lavender benchmark 结论：VQAv2/TextVQA 使用内部 leave-one-annotator-out 近似评分；WorldMedQA-V 为 256 样本 pilot；COCO caption 缺 METEOR（101 未安装 Java）；需用官方 evaluator 和冻结完整 split 复核。
- [ ] V3/V4 至少扩展 3 个 seed 的下游能力评估；完成正确/错词/错图/随机 teacher controls；按 V3>V1 与 V4>V2 分别报告配对置信区间及样本外一致性。

**能力—几何联合判定：**

| Flickr30k Entities 几何结果 | Lavender-style 能力结果 | 解释 |
|---|---|---|
| 提升 | 提升或保持 | 支持语义空间对齐同时改善定位和任务能力 |
| 提升 | 下降 | 对齐约束过强或破坏语言能力，不能称成功 |
| 不变/下降 | 提升 | 只能归因于 SFT/一般正则，不能声称 grounding 改善 |
| 不变/下降 | 不变/下降 | 暂停扩大教师和损失组合，先查接口、数据和评测 |

WorldMedQA-V 必须保持完全未参与训练、teacher calibration 和阈值选择。OOD 结果按语言、问题类型和错误案例分组；视觉扰动结果按 corruption 类型报告。定性图同时保存输入、教师图、`A_lang`、模型答案、正确答案和错误类型，但定性图不能替代定量证据。

**P4.1 完成条件：**

- [ ] Flickr30k Entities 上完成 pointing、mass-in-box、IoU 和 intervention agreement；
- [x] 首轮六项能力 pilot 在 V0--V4 上使用统一推理/eval runner 完成（官方 full-split evaluator 复核仍未完成）；
- [ ] 正确教师、错词、错图、随机图在至少三个能力 benchmark 小子集上完成；
- [ ] 至少一个几何指标和一个下游能力类别在 3 个 seed 上保持方向一致；
- [ ] 几何提升但能力下降时，停止将该配置推进到 VLA，并记录为过强约束失败。

### 已冻结的首轮模型与 OOD 角色

- [x] VLM 全量验证启动骨干：Qwen2.5-VL-7B；
- [ ] VLM 跨架构复核骨干：候选为 LLaVA-OneVision 或 Prismatic-7B，须按权重合法可用性与归因接口 P1 选择一个；InternVL3.5、Ovis2.5、Qwen3-VL、DeepSeek-VL2 为后续扩展；
- [x] VLA 首轮：OpenVLA/OFT + π0.5；MolmoAct2 后置；
- [x] OOD 分工：LIBERO-Plus 做可控闭环 OOD，SimplerEnv 做 VLA 视觉/语义保留诊断，DROID 做真实数据离线泛化；RoboTwin 后置到 WAM/VLA 扩展；
- [x] SpikingBrain 后置，不作为上述任一首轮门槛。

### VLM 阶段的独立目标

P4 不以 BlindVLA 的 VLA 结果为前置 gate，但应将其抽象成 **DB-style VLM feature-retention baseline**。P4 独立回答：词级扩散空间教师是否让 VLM 的 `A_lang` 更准确地对应语言短语，而不是只让通用视觉 feature 更稳定。

- [ ] 建立 phrase grounding 校准/验证集和独立 VLM 保留集；前者测 region，后者测目标/属性/位置概念；
- [ ] 分开保存 raw query attention、general-prompt ratio、answer/phrase-score gradient 与输出扰动图；它们不是同一个归因量；
- [ ] 使用真实 processor/grid 元数据恢复 patch，不用平方根猜网格；
- [x] 登记 T_sem 候选的实际 checkpoint revision、许可与 gated 状态：见 `configs/teachers/semantic_teacher_candidates.json`；选择仍须等待独立 calibration split；
- [ ] 首轮先完成 SD1.5 在独立 calibration split 上的正式校准并冻结；PixArt-α、PixArt-Σ、Playground-v2.5 的横向教师比较属于扩展，不阻塞 Qwen V0–V4；
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

### 首轮监督形式：与 Lavender 可比的 caption SFT

首轮 V1--V4 固定为 image-to-caption SFT：输入是图像加固定 caption prompt（例如 *Describe the image in a single sentence as a caption.*），标签是 Flickr30k 的原始 caption。caption CE 监督完整 caption；语义归因损失只用于 Entities 标注中能够精确映射到该 caption token span、且具有有效空间教师图的实体名词短语。因此训练样本清单需要包含图像 ID、caption ID、原始短语字符/token span、教师图 ID 和过滤原因。

\[
L_{V1}=L_{\mathrm{caption\ CE}},\quad
L_{V3}=L_{\mathrm{caption\ CE}}+\lambda_{\mathrm{sem}}L_{\mathrm{sem}},
\]

V2/V4 只在相同样本、prompt、caption token、训练步数和 seed 上增加 retention loss。这样 V3/V4 相对 V1/V2 的提升才可归因于 `T_sem → A_lang`，并可与 Lavender 的 caption-SFT + diffusion attention-alignment 公平比较。

**首轮数据与监督划分（已确认）**：使用 Flickr30k 官方训练图像及其全部有效 caption 做 SFT；训练时，Entities 的短语标注仅用于选择可定位、可对齐的实体名词短语，并建立其与 caption token 的精确桥接。语义损失把这些短语对应的学生归因图对齐到离线语义教师图。**标注框坐标不进入训练损失，也不用于构造教师图或学生图的目标区域**；因此这是以实体短语为条件、以教师伪图为目标的归因蒸馏，不是 box-supervised detector/grounder。完整 caption 的 CE 仍覆盖所有 caption token，未匹配短语和非实体 token 只参与 caption CE，不参与语义图损失。

官方 train/val/test 按 image ID 隔离，并冻结图像、caption、Entities XML、过滤结果和 manifest SHA256。官方 validation 只用于事先声明范围内的教师/归因接口校准和超参选择；不得反复扩展搜索后再称其为独立验证。官方 test 不参与训练、teacher/层/阈值/超参选择，只在协议冻结后做最终评估。主 grounding 评测可使用 test 中的实体框计算 pointing、mass-in-box 与 IoU，但这些框只作为评测真值。任何 image + referring expression 的显式定位训练都另设实验组，不并入 V0--V4。

### 已确认的 Qwen 首轮实现冻结

- [x] 学生：`Qwen2.5-VL-7B-Instruct`；视觉 encoder 冻结，BF16 LoRA 训练语言层/跨模态投影相关的显式 target modules；V1--V4 使用完全相同的 LoRA rank、target modules、optimizer、LR、batch schedule、seed 与 checkpoint cadence；
- [x] `T_retention`：冻结 `DINOv2 ViT-L/14`，仅提供 patch feature target 给 V2/V4；其 revision、图像 preprocess、patch grid bridge、抽取层、projector seed 和 `λ_ret` 必须在 calibration 后写入 manifest；
- [ ] `T_sem`：正式训练前对 SD1.5、PixArt-alpha、PixArt-Sigma、Playground-v2.5 按 P4.0 的 validation-only 规则完成校准并选择唯一 best-single；SD1.5 仅作为已通过工程 smoke 的 fallback。冻结模型、revision、token span、attention blocks、diffusion step、CFG、normalization 后离线缓存 train 所需 map；V3/V4 训练期间不在线运行 diffusion；
- [x] 范围：先完成 Qwen V0--V4；Prismatic 的合法授权、下载和同类 P1 是第二 VLM 骨干扩展，不阻塞 Qwen 主结果。

### 历史运行状态快照说明

本节早期的 2026-09-17 状态表已经被 `experiment_workspace/PROGRESS.md` 中 2026-09-24 的纠错几何与能力 pilot 记录取代，不再作为当前运行状态依据。当前权威状态是：Qwen P1、F0 smoke、SD1.5 单图/小样本/64 条 map cache 和部分 V0–V4 pilot 已通过；独立 1000 图校准、最终 manifest/hash、正式三 seed 全量训练与官方 evaluator 仍未冻结或完成。数据和模型资料已归档，但最终 archive/hash 与有效样本交集仍须复核。

### 调整后的执行顺序

1. 完成并验证 Flickr30k Entities archive，建立官方 `train/val/test` manifest；
2. Qwen P1：20 个固定 val 图，验证 phrase-score gradient、post-merge grid、可重复 map 和 `report.json`；已完成单样本真实 Qwen 前向 smoke（`image_grid_thw=[[1,34,36]]`、29 hidden-state layers），后续 processor 固定 `use_fast=False`；
   - [x] 2026-09-17：完整 Qwen P1 已通过，20/20 真实 val 样本的 `report.json` 通过 `validate_p1_report.py`；
3. 下载/冻结 DINOv2 ViT-L/14，检查 DINO patch 与 Qwen post-merge grid 的坐标桥；
4. 在独立 val calibration split 对四个合法可用的 teacher 候选执行同一词图校准；按 pointing→mass/soft-IoU→有效率/稳定性选择 best-single。若候选均未通过最低有效性检查，才冻结已通过 smoke 的 SD1.5 作为 fallback；
5. 实现共享 Qwen LoRA runner 和 V0--V4 五份不可变 manifest；
6. 对 V1--V4 各跑 20--50 step smoke，确认 caption CE、retention、semantic loss、cache 命中、checkpoint resume 和 GPU memory；V0 仅运行 evaluation；
7. smoke 全部通过后，依次运行 V0 evaluation、V1、V2、V3、V4 正式训练与统一 Flickr30k Entities test/OOD evaluation；
8. 只有 Qwen 主结论及错误教师/随机图反证成立后，才扩展 Prismatic 和 referring-prompt `R*` 消融。

### 实验检查点记录规则（运行中强制执行）

每次出现可恢复、可比较或会改变后续决策的节点，必须同时更新：本节的完成勾选、`experiment_workspace/PROGRESS.md`、对应 `experiments/<id>/protocol.md`，并提交小型 Markdown/JSON 摘要。大 tensor、map、checkpoint 仍只放 VEPFS。

必须记录的节点包括：

- 数据或 teacher cache 处理达到 **10%、25%、50%、75%、100%**，或任务异常退出；
- 固定 manifest、有效样本交集、失败样本集合或 SHA256 发生变化；
- 20/50-step smoke 完成、出现 NaN/OOM、loss 异常、checkpoint 保存或恢复；
- teacher calibration 的单模型 pilot 完成、候选淘汰或 best-single 冻结；
- 每个 V0--V4 训练完成、evaluation/负控完成、配置或解释边界改变。

每条检查点至少写：时间、experiment ID、Git commit、服务器 job PID/日志路径、数据 manifest SHA256、处理总数/成功数/失败数、当前配置（model/teacher/revision/seed/steps）、关键数值、结果/日志路径、下一步和是否允许扩大运行。中间指标必须标记为 **partial / non-final**，不得把 10 图 pilot、20-step smoke 或部分 cache 当作主表结果。

### 快速全流程 gate：F0-256，然后 F1-10k

为避免 10k 条 `20×10` null-text map cache（单 GPU 约数天）阻塞工程验证，先运行 `F0-QWEN-SD-DINO-V0V4-256`：从已冻结 10k caption pair manifest 确定性抽取 256 条；V0--V4 全部使用这同一个 256 集合，SD1.5 map 在训练前离线生成；各训练组完成 20-step smoke 和 100-step flow run，验证 cache、loss、checkpoint、evaluation 和负控路径。F0 不用于论文主表或显著性结论。

F0 全部通过后，完全复用代码和配置语义扩展为 F1-10k；唯一改变是固定 pair 数量和对应的离线 map cache，不能在组间改变数据或训练预算。

### 最小可证伪路径：尽快判断 idea 是否 work

如果当前目标只是判断核心 idea 是否值得继续投入，不等待 F1-10k、PixArt、Playground 或 Prismatic。使用已经完成的 F0v2-256 对齐 manifest、SD1.5 cache 和 DINO teacher，增加一个**未参与 F0 训练的 held-out 小测试集**（建议从官方 Entities test 固定 64--128 个 image-phrase pair，记录 SHA256），按以下顺序执行：

1. **V0 evaluation**：原始 Qwen checkpoint，只生成 phrase-score attribution，不训练；保存 pointing、mass-in-box、IoU 和 map validity；
2. **V1--V4 checkpoint gate**：每组保存 LoRA/projector checkpoint，立即在同一 held-out 集合恢复并评测；不能只用 smoke loss 判断；
3. **核心增量表**：比较 `V3-V1` 和 `V4-V2`，同时报告绝对值、相对变化和每个样本/phrase 的 bootstrap 置信区间；
4. **语义负控**：V3/V4 分别替换为 wrong-word、wrong-image、random normalized map，其他配置完全不变；正确 map 必须优于三类负控；
5. **快速判定**：只有当 `V3 > V1`、`V4 > V2`、correct > wrong/random 且 held-out 指标方向一致时，才进入 F1-10k。任一条件失败，先分析失败样本和归因图，不扩大训练。

该路径的产物是 `experiments/F0v2-idea-validation/` 下的 protocol、checkpoint manifest、metrics.json、negative_controls.json 和 analysis.md。它是方向性证据，不替代 F1-10k 主实验；F0 smoke loss 下降本身不算 idea 验证。

### 近邻方法比较的解释边界

- V2 是 BlindVLA-inspired DB-style retention，目的在于排除“仅保持视觉表征即可”的解释；它不是完整 BlindVLA VLA 复现，BlindVLA 的 action/policy 比较留给 OpenVLA/OFT VLA 阶段；
- V3 是 Qwen 上的 diffusion teacher → language-conditioned attribution alignment；因 Qwen 缺少 Lavender 原文的标准 cross-attention，不能称为 Lavender exact reproduction；
- F1 增加同 teacher/数据/预算的 raw attention/rollout proxy baseline，用来直接比较 attention 对齐与 attribution 对齐；
- 随后在可导出显式 cross-attention 的 VLM 上运行小规模 `LAV-exact`，才比较我们与 Lavender 原式的差异。

### F1-10k：判定核心 idea 是否 work 的预设验收矩阵

F0-256 只证明工程链路；以下五条在 F1-10k 的冻结 Flickr30k Entities test 与固定 visual-OOD split 上共同决定能否声称核心 idea work。每条均须报告 mean/std 或多 seed，并保留逐扰动结果；不得只选有利指标。

| 判据 | 必要比较 | 预期支持性结果 | 若不成立，结论边界 |
|---|---|---|---|
| C1：独立语义增量 | `V3 > V1` | pointing、mass-in-box、IoU 至少主要指标提升 | 不称 semantic attribution 在普通 SFT 之上有效 |
| C2：超出 retention | `V4 > V2` | 加 `L_sem` 后仍有增量 | 不称超出 DB-style visual retention；方法可能只是 retention 替代物 |
| C3：教师语义因果性 | correct map > wrong-word / wrong-image / random | 正确教师最优，错误或随机显著退化 | 不称模型使用语言条件 teacher；可能是任意空间正则 |
| C4：attribution 必要性 | `V3 > V3-attn-proxy` | phrase-score gradient attribution 优于 raw attention/rollout proxy | 不称 attribution 相比直接 attention 对齐有必要性 |
| C5：独立泛化 | ID + 固定 OOD 同趋势 | test 与每类 visual-OOD 不劣于对应 baseline | 限定为 ID 训练效果，不称改善视觉 OOD |

`V3-attn-proxy` 与 V3 共用 SD1.5 map、数据、LoRA、预算、seed、评价和负控；唯一差异是学生图从 `A_lang` 替换为有 provenance 的 raw attention/rollout proxy。它不是 Lavender exact 复现，标签必须明确。

### BlindVLA / Lavender 源码的执行时机

**现在（F0/F1 前）只吸收源码机制，不做完整迁移训练。** 已从 Lavender 固定：真实图像条件、caption token、SD cross-attention、DDIM/null-text inversion、离线 map cache；已从 BlindVLA 固定：冻结视觉 patch teacher、中间视觉 token、patch bridge、cosine retention、paired data/budget baseline。它们分别对应当前 V3/V4 与 V2 的实现约束。

**不应现在完整复现的原因：** Lavender 原式假设学生有标准 cross-attention，Qwen 的 student map 是 phrase-score attribution；把源码直接 patch 进 Qwen 会制造不公平且不可靠的“exact baseline”。BlindVLA 是 VLA policy/action 论文，完整复现需要其 OpenVLA/ManiSkill/SimplerEnv action pipeline，在 VLM F0/F1 上运行不能回答主问题。

**后置补强顺序：** F0 打通 → F1 的 C1--C5 成立或明确失败 → `V3-attn-proxy`（Qwen）→ 有可靠显式 cross-attention 的 VLM `LAV-exact` 小规模对照 → OpenVLA/OFT VLA 阶段完整 BlindVLA-style paired policy comparison。若 F1 C1/C2 已失败，不投入完整近邻复现来掩盖核心机制无效。

**VLM 继续条件**：V3 或 V4 必须在 phrase grounding 上优于 V1/V2，且正确教师优于错词/错图/随机教师；若 V2 已经覆盖 V3/V4 的收益，空间教师主张退回为 feature retention 的替代实现，不直接进入 D0 的强创新叙事。

### 任务

- [ ] 准备 Flickr30k Entities 的 train/calibration/test manifest，冻结 calibration 与 test ID；RefCOCOg split、扰动生成 seed/强度和 VL-Think screenshot set 单独记录；
- [ ] 对每个样本生成 Stable Diffusion、PixArt-α、PixArt-Σ、Playground-v2.5 的可用词级教师图；无法导出语言条件二维图的候选退出 T_sem 比较；
- [ ] 在独立校准集审计每个教师的 token span、层/步/CFG、pointing/IoU、无效词率和跨 seed 一致性；
- [ ] 冻结 best-single 教师后再训练主实验；多教师 ensemble 仅作为校准权重冻结的消融，不能先平均后称主教师；
- [ ] 在选定 VLM 上比较无对齐、DB-style feature alignment、各个单扩散教师的 attention/attribution 对齐、teacher-to-attribution 对齐；
- [ ] 在一个 Qwen 系模型上使用 gradient×input 归因接口做同一教师对齐；
- [ ] 在 LLaVA-1.6/OneVision 上使用同一 teacher-to-attribution 接口做 VLM 普适性对照；
- [ ] 记录正确教师图、错词教师图、错图教师图、随机教师图；
- [ ] 报告 pointing accuracy、目标区域 IoU、VQA/grounding accuracy、归因图熵。
- [ ] 对 VLM 增加 phrase/answer-score 的 patch occlusion、目标词替换和错词反事实，检查图是否对应语言条件输出；
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
| B5（候选，过 gate 后才做） | B3 + D0-CF：冻结 B0 的 patch-occlusion action-loss map 作为动作条件反事实证据 | DVD 启发能否补足语言名词区域未覆盖、但对动作预测必需的视觉线索 |

- [ ] 对 B4 加入 object-only、object+EEF、面积匹配随机 mask 对照，不能把夹爪/接触/障碍区域一律视为泄漏；
- [ ] B5 先做离线 gate：用冻结 B0 和专家动作计算 patch removal 对 action loss 的影响；比较正确/错误动作、随机 patch、背景 patch 和至少两种 mask 算子。只有 action 相关区域的干预响应稳定且负控可区分，才缓存反事实图并训练 B5；
- [ ] 采用同源教师控制或明确记录 C-RADIO feature 与扩散 map 同时改变 teacher/监督形式的混杂；
- [ ] 报告 VLM 保留能力、phrase grounding、动作离线/闭环、raw/ratio/gradient/occlusion 一致性和训练成本；

**继续条件**：P4 已确认正确扩散教师改善语言空间 grounding；B4 再在 B3 之上改善动作相关指标或 task-preserving OOD，且正确教师优于错词/错图/随机教师。D0-CF 不属于必需主线：只有离线反事实图通过 mask 稳定性、动作负控和目标/背景干预检查后，才增加 B5；若 B5 只改善热图集中度、没有改善动作预测/闭环或 OOD，则不保留为主方法。若 B4 无独立增量，停止把 D0 写作核心创新，转为 feature retention 或空间监督的诊断结果；若 B1 已解释全部动作收益，暂不进入 D1/B 路线。

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
- [ ] 对 VLA 动作图增加 patch occlusion 的 `Δa` IoU/pointing 对照，检查 `A_act` 与动作敏感区域是否一致；
- [ ] 评估 DVD 启发的 D0-CF 候选：先冻结 B0，离线测量 patch occlusion 对专家动作损失的影响；通过动作/背景/错动作/mask 算子负控后，才训练 B5；
- [ ] 增加语言反事实和有限范围 state 反事实，区分视觉 grounding、语言条件、合理 proprioception 与 state shortcut；

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

### VLA benchmark 纳入范围与阶段

LIBERO-Plus、LIBERO-PRO 和 RoboTwin 均纳入路线图，但承担不同问题，不要求在首轮同时跑完：

| Benchmark | 计划阶段 | 在本项目中的角色 | 首轮是否阻塞 |
|---|---|---|---|
| LIBERO 标准任务 | VLA 接口 smoke 与主机制实验 | 固定 pick-place 任务，验证 7D delta EEF、动作归因和闭环成功率 | 是；先完成最小任务闭环 |
| LIBERO-Plus | LIBERO 主实验之后的 task-preserving OOD | 测背景、纹理、光照、干扰物、初始状态/相机等变化下，目标和动作语义保持时的稳健性 | 否；主线接口与 ID 机制通过后加入 |
| LIBERO-PRO | 语言与指令泛化扩展 | 测语言改写、目标/属性/关系指代变化；需单独核对其任务定义与可用 split，不与视觉扰动结果混成一个 OOD 总分 | 否；在 LIBERO 标准任务稳定后加入 |
| DROID offline | VLA 主线的真实数据泛化阶段 | 在真实机器人演示上做离线动作预测；按 episode/场景分组，不能用离线误差代替闭环成功率 | 否；LIBERO 机制验证后启动 |
| RoboTwin | 后期模拟器扩展 | 评估更复杂操纵、双臂/多任务或 embodiment 变化；用于检验方法边界和扩展性，不替代 LIBERO 首轮因果对照 | 否；D0 主线结果和动作接口稳定后加入 |

因此，当前建议的推进次序是：VLM 收尾期间并行做 LIBERO 环境与动作接口 smoke、固定最小任务并跑 B0；冻结 VLM 侧可用的 `T_sem → A_lang` 接口后做 LIBERO B1–B4；标准 LIBERO 机制实验稳定后，再按资源顺序加入 LIBERO-Plus、LIBERO-PRO、DROID offline；RoboTwin 放在更后面的操纵/embodiment 扩展。benchmark 纳入计划不等于已经下载、配置或完成实验，相关状态须在 `PROGRESS.md` 单独打勾并记录 commit、数据版本和 split。

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

## 13. 进入正式 VLM/VLA 实验前的最短路径（2026-09-24）

这一节覆盖旧计划中“pilot 已完成但正式训练尚未冻结”的状态。目标是尽快验证核心 idea，同时保留可以发表的对照和审计证据。

### 13.1 VLM 首轮只冻结 Qwen 主线

首轮不等待第二 VLM 骨干的完整训练，也不等待 SpikingBrain；但四个已登记的 semantic teacher 候选必须在正式长训练前完成 validation-only calibration，因为 teacher 身份会直接影响 V3/V4 的公平性。Qwen 的正式 VLM 主线必须按以下顺序执行：

1. 验证 Flickr30k Entities 完整 archive、图片/XML/Sentences/caption 行数和 SHA256；
2. 用独立 validation split 完成约 1000 张 teacher/student calibration；
3. 先把 SD1.5 FP32 null-text 作为工程 fallback，配置候选为 `lambda_sem=0.10`、`teacher_temperature=1.25`、warm-up 50；正式 `T_sem` 必须由四候选 validation calibration 的 best-single 规则确定，同时固定 semantic-off 匹配组。该配置只是 sweep 得到的首轮候选，不是已证明最优；
4. 并行生成 full train cache 和 held-out cache，但不允许用 held-out 调参；
5. 在同一 manifest、同一图像预算和同一 evaluator 下训练 V0–V4，至少 3 个 seed；
6. 用 geometry、caption、VQA/OCR、hallucination、OOD 五类结果共同判定，不以单个 IoU 或单个 CIDEr 提升作为成功标准；
7. 只有当 `V3>V1`、`V4>V2`、correct teacher>wrong/random 且能力不退化在独立数据上重复，才将主张升级为“idea 有效”。否则保留为 teacher-map 可用、student transfer 尚未稳定。

### 13.1.1 其他 VLM 的规划状态与普适性范围

项目不是只规划 Qwen，但“列为候选”不等于已有可运行配置或已经完成适配：

| 模型 | 当前角色 | 开始 Qwen 全量验证的前置条件？ | 当前状态/建议 |
|---|---|---:|---|
| Qwen2.5-VL-7B-Instruct | VLM 首轮主骨干 | 是 | 已有 P1、V0–V4 配置与实验结果；先完成正式全量验证 |
| LLaVA-OneVision | 第二架构候选 | 否 | 计划用于验证非 Qwen 结构的归因桥；尚需冻结 checkpoint、许可、processor/grid 和 P1 |
| Prismatic-7B | 第二架构候选 | 否 | 有 P1 配置，但权重涉及合法 HF 登录/条款访问；不作为当前启动门槛 |
| InternVL3.5、Ovis2.5、Qwen3-VL、DeepSeek-VL2 | 后续扩展池 | 否 | 仅进入模型候选/普适性规划，尚未冻结训练配置或完成 P1 |

建议的最快且可解释路线是：先让 Qwen 独立完成全量 V0–V4；同时只做一个第二骨干的权重/接口 P1，不同时铺开多个完整训练。若要把论文主张写成“跨 VLM 普适”，至少要在一个非 Qwen、空间接口不同的骨干上复现关键比较（建议 V1/V3，并视资源补 V2/V4）；Qwen 单模型只能证明该机制在 Qwen 上可行。

### 13.1.2 正式开跑前需要冻结的实验细节

下列项目中前五项是提交正式长训练前的硬门槛；第二模型选择和额外能力 benchmark 不阻塞 Qwen 首轮：

| 事项 | 当前建议默认值 | 仍需冻结的内容 |
|---|---|---|
| 训练数据 | Flickr30k 官方 train 的完整有效图像/caption；caption CE 覆盖完整 caption，语义损失只覆盖可映射 Entities 实体短语 | archive/标注完整性、去重、有效样本数、短语 token span、过滤规则、固定 manifest 与 SHA256 |
| split 使用 | 官方 train 训练；官方 val 做预注册范围内校准；官方 test 仅最终评估，按 image ID 隔离 | split 文件、图像交集审计、manifest 与 SHA256 |
| 主评测 | Flickr30k Entities 官方 held-out phrase boxes；GT box 只作评测真值 | phrase-span 到 caption token 的映射；多框实体的 IoU/pointing/mass-in-box 聚合规则 |
| 归因对象 | Entities 标注中可映射到 caption 的实体名词短语；仅其语义损失使用实体 span，caption loss 仍监督全部 caption token | 精确 token span、特殊 token 排除、multi-token phrase 聚合方式 |
| 教师条件 | 四候选 teacher 统一使用固定 caption prompt，并抽取对应实体 phrase token map；SD1.5 仅为 fallback | prompt 模板、tokenizer span 对齐、无效/不可见实体的过滤规则；需确认与现有 F0 cache 完全一致 |
| 学生归因 | 目标 phrase token score 对 post-merge visual tokens 的 gradient×activation | 归因层、符号/正负归因处理、归一化、Qwen grid/crop bridge 的 frozen 版本 |
| 训练对照 | V0 原始评测；V1 caption SFT；V2 retention；V3 semantic；V4 retention+semantic | LoRA target/rank、学习率、batch/累积、总步数、保存点、三 seed |
| 语义损失 | `lambda_sem=0.10`、温度 1.25、warm-up 50 作为候选 | 选择规则和正式值；不得用 test split 调参 |
| 负控 | wrong-word、wrong-image、random map | 负控采样规则、是否每个 seed 都跑、与正确教师相同面积/质量预算 |
| 能力保持 | 建议用 COCO Captions 官方 validation 做 caption 能力复核 | 官方 evaluator 版本、解码参数、预设非劣界；现有 pilot scorer 不直接进入主表 |
| 外部泛化 | RefCOCOg 作为后续跨数据集 phrase-grounding 验证 | 是否纳入同一论文首轮主表；首轮可先不以它阻塞 Qwen 训练 |
| 统计 | V1–V4 至少 3 个预注册 seed；V0 原始权重可一次评测 | 固定 seed、主指标、image-level bootstrap/多重比较处理和停止规则 |

对“尽快开始”的建议：现在先锁定 **Qwen + Flickr30k Entities 主指标 + 四候选 validation calibration + 三个 seed**；SD1.5 只用于工程 fallback 和 smoke。先把数据 manifest、caption/entity span bridge、归因层/归一化、训练超参和 evaluator 变成不可变 config。VLM 正式长训练前，不需要先等第二骨干，但必须完成已登记 teacher 候选的 validation-only 选择。

### 13.2 VLA 首轮采用 OpenVLA/OFT + LIBERO

VLA 不直接把当前 VLM 的 V2 称为 BlindVLA。必须复现其 VLA 阶段的 feature-retention 思路作为强 baseline，然后在同一 action pipeline 上增加我们的空间归因约束：

**StarVLA 不是首轮模型依赖。** P6 的模型定义冻结为 OpenVLA/OFT；StarVLA 仅可作为服务器上的已有运行环境或参考工具，且必须通过只读版本、processor、动作单位、LIBERO 观测和 checkpoint restore 审计。`/root/code/Starvla` 中的 SpikingBrain/Robotwin 专属训练脚本不能直接运行、修改或复制。若要把 StarVLA 自身作为骨干，需另建 adapter 和独立实验矩阵，不能把其结果混入 OpenVLA/OFT 的 B0–B4 主比较。

**原生 VLA 优先规则。** 如果一个模型系列已经有公开、可复现的原生 VLA 版本，就直接使用其原生 VLA backbone/action head，不再执行“该 VLM + StarVLA action head”的二次拼接。例如 OpenVLA/OFT、π0/π0.5、MolmoAct2 和 LingBot-VLA 应优先走各自官方 VLA 接口。只有在该系列没有可用 VLA 形态，或我们明确要做跨动作头的独立 VLM-to-VLA 消融时，才使用 StarVLA；这类结果必须作为独立扩展组报告。

| 顺序 | 组 | 目的 |
|---|---|---|
| B0 | OpenVLA/OFT 原生 BC/SFT | 建立动作能力和动作头基线 |
| B1 | B0 + BlindVLA-style frozen visual feature retention | 排除“收益只是视觉表征保持” |
| B2 | B0 + `T_sem → A_lang` | 检验语言条件空间归因是否独立改善动作学习 |
| B3 | B1 + `T_sem → A_lang` | 检验 retention 与语义归因是否互补 |
| B4 | B3 + D0 `A_act` containment | 检验动作相关区域约束是否有额外增量 |

首版 B4 的 `A_act` 默认由 action loss 或 action output 对 visual tokens 的 gradient×activation 得到。模型若有 action query attention，可作为额外解释分支，但不能作为跨模型必备接口。`R_future`、LingBot-VLA、Next Forcing 和 action/world-model refiner 在 B0–B4 通过后再加入，不能让动态教师替代首轮的因果对照。

### 13.3 首轮必须先完成的代码和接口

- `adapters/openvla_oft.py`：visual hidden states、action loss、7D delta EEF、动作归一化和 checkpoint restore；
- `attribution/action_gradient.py`：固定 action target、gradient×activation、patch-grid bridge、归因有限性和重复性；
- `losses/retention.py`：BlindVLA-style projector/feature retention，明确 optimizer 参数组；
- `losses/semantic.py`：冻结 `T_sem → A_lang`，不在训练中重新选择 teacher；
- `losses/action_containment.py`：D0 soft containment，首版不引入 `L_phase`；
- `trainers/vla_bc.py`：B0–B4 配置化训练，所有组共享 data/seed/budget/action head；
- `evaluation/libero.py`：成功率、动作轨迹、patch occlusion、语言反事实和 state 反事实；
- `configs/vla/`：每组独立配置、manifest SHA、model revision、teacher revision、seed；
- `results/`：VLM 与 VLA 分开保存，禁止覆盖 F0 pilot。

### 13.4 不能跳过的正式实验门槛

VLM 门槛：独立 calibration、冻结 cache、三 seed、官方或可复核 evaluator、正确/错词/错图/随机控制。

VLA 门槛：B0 单任务闭环、B1 BlindVLA-style baseline、B2/B3 语义约束、B4 D0 action attribution、同一 action head 与同一预算的 paired comparison。

不阻塞首轮的扩展：第二 VLM 骨干、PixArt/Playground teacher、π0.5、MolmoAct2、LingBot-VLA、LIBERO-Plus/LIBERO-PRO、DROID、RoboTwin、`R_future` 和真机。它们在首轮成功后用于普适性、OOD 和动态扩展，而不是用来补救主线接口不稳定。

### 13.5 依据 BlindVLA、LIT 与 Next Forcing 的颗粒度冻结

三篇参考工作的职责必须分开：

- BlindVLA：作为 `B1` visual feature retention baseline，并提供 VL-Think、feature collapse、attention sink 和 VLM→VLA 保留诊断；
- Breaking the Vision-Action Shortcut / LIT：作为 paired baseline、异构 action interface、task-preserving OOD 和 counterfactual 的实验设计参考；LIT 的 spatial-goal action prior 与 latent interface 作为独立 `E-LIT` baseline，不并入 B0–B4；
- Next Forcing：作为 WAM 的 multi-chunk future teacher，后置为 `F0` future-attribution diagnosis 和 `F1` `R_future` refiner，不改变首轮动作归因主问题。

VLA 实验按四层推进：

| 层级 | 最小范围 | 允许的结论 |
|---|---|---|
| G0 工程 gate | OpenVLA/OFT、一个 LIBERO pick-place、20–50 steps、restore | 接口可运行；不能声称 idea 有效 |
| G1 最小 idea gate | B0–B4、固定单任务 manifest、seed 17/29/41、offline + 短 rollout + controls | `T_sem → A_lang → A_act` 是否值得继续 |
| G2 论文级证据 | LIBERO 四 suite、LIBERO-Plus 分维度 OOD、VL-Think/反事实、DROID offline | 机制、task-preserving OOD 和真实数据迁移 |
| G3 扩展 | π0.5、MolmoAct2、WAM/LingBot-VLA/Next Forcing、RoboTwin、真机 | 跨结构和未来动态教师泛化 |

G1 的必要条件是：正确 teacher 的归因质量优于 wrong-word/wrong-image/random；B2/B3 相对 matched baseline 至少在一个动作指标和一个 patch-occlusion 机制指标上有增量；B4 相对 B3 的收益不能被随机 mask 解释；三 seed 方向满足预注册一致性；ID success 不超过预注册容忍下降。G1 未通过时不扩大到 G2/G3，也不通过增加 world model 或动作头来掩盖主线不稳定。

正式比较时不得把不同模型的原生 success 直接横向排序。每个 backbone 必须保留其 native action objective、action representation 和 conditioning mechanism；跨模型只比较同一方法相对各自 paired baseline 的增量。DROID 的 offline MAE/RMSE 只能作为真实数据迁移证据，不能替代 LIBERO 或真机闭环 success。

后续动态教师的最小验证顺序固定为：

```text
F0：冻结 B0，比较 A_act 与 short/mid/long R_future 的一致性和 wrong-future 负控
F1：D only vs D + R_future vs D + T_sem ⊙ R_future
F2：必要时复现 Next Forcing 的 single/multi-chunk、timestep shift、
    multi-layer fusion、causal chain 消融
```

详细比较和实现边界见 `references/paper_reading/vla_granularity_review.md`。

### 13.6 101 暂不可用时的并行本地队列

2026-09-25 的只读 SSH 复核显示：`115.190.90.101:27219` TCP 可达，但 SSH banner/kex 阶段超时或被远端关闭。因此当前把工作明确拆成可本地完成和必须远端完成两类，避免将网络阻塞误判为研究停滞。

**本地立即完成：**

- P6 配置准入和 B0–B4 矩阵合同测试；
- VLA adapter 的输入/输出、`A_act` shape/finite、restore artifact schema；
- LIBERO episode-level manifest builder 的 synthetic fixture 测试；
- Flickr30k/Entities split、caption/entity span、teacher calibration manifest 和 cache schema；
- VLM formal runner 的 launch guard、JSONL 曲线字段和 checkpoint 命名；
- BlindVLA projector optimizer、multi-layer checkpoint、loss normalization 的代码审计；
- 整理 Git 可提交清单和服务器恢复后的上传/校验脚本。

**必须等 101 或同等 GPU 环境：**

- OpenVLA/OFT/π0.5 真实 checkpoint/processor P1；
- LIBERO 官方环境枚举、动作 schema、episode manifest；
- B0 20–50 step GPU smoke、restore 和短 rollout；
- teacher 大 cache、V0–V4 三 seed 全量训练、官方 evaluator；
- B1–B4、LIBERO-Plus、DROID offline 和跨模型扩展。

101 恢复后固定顺序：

```text
SSH_OK 只读探针
→ repo/Git SHA、磁盘、GPU、进程检查
→ 上传并校验本地代码
→ P1 adapter + LIBERO manifest audit
→ B0 smoke + restore
→ B1
→ B2/B3/B4
→ LIBERO-Plus/DROID
```

在远端恢复前，不把本地 synthetic fixture、配置检查或代码测试写成 VLA 结果。

### 2026-10-06 实际状态补充：cache gate 与 B0

- [x] 三类 cache 已有权威审计：LIBERO semantic 10080/10080，Flickr SD1.5 overlay 9952/9952（5 条失败样本由独立 retry overlay 补齐，原 cache 未覆盖），C-RADIO 5040/5040。
- [x] 修复 B0 driver 对旧 deterministic parity 和原始 Flickr 分片 audit 的错误依赖；新 gate 验证 semantic/radio audit、Flickr overlay audit 和 5 条 retry audit。
- [x] 独立版本化 B0 driver 完成 cache audit、OpenVLA/OFT P1、30-step action smoke、fresh-process restore 和官方 LIBERO episode-loop engineering rollout。
- [x] P1 action shape 为 `[1,8,7]`，双相机 patch grid 为 `16×16`，归一化 round-trip 与重复运行检查通过；B0 峰值显存约 20.2 GiB；restore prediction exact，continuation 的 loss/grad norm 在 1e-4 内且 LR/scheduler exact。
- [x] 两个未充分训练 B0 rollout episode 完整运行 230 steps、无环境错误，但成功 0/2；仅作为工程链路证据，不作为性能结论。
- [ ] 严格 same-image role-swapped phrase 和 area-matched random controls 尚未完成；旧 cross-episode phrase/random 结果不能替代它们。
- [ ] B0–B4 正式长训练仍需冻结 effective batch、gradient accumulation、正式 rollout manifest/evaluator revision 和三 seed controls；在这些字段冻结前不启动 10k-step 长训练。

### 2026-10-06 严格 teacher controls 实际状态

- [x] 建立并审计独立 control GPU window，满足 cache audit、exclusive GPU、fresh timestamp 和 UUID 校验。
- [x] `run_oft_controls_smoke.py` 已支持同一 episode/timestep/camera 的 source-target role swap，以及保持 top-20% 支持面积的随机位置 control。
- [x] `role_swapped_same_image` 与 `area_matched_random` 均完成 20-step finite interface smoke；结果写入独立版本化目录，未覆盖旧结果、B0 或任何 cache。
- [ ] control smoke 不是性能或机制显著性结论；正式 gate 仍要在固定 validation/control 集上做 paired metrics，并与 correct teacher、cross-episode phrase、wrong-image 一起报告。
