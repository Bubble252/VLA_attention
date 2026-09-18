# Lavender 与 BlindVLA：匹配对象、训练技巧和可复刻性审计

本文基于本地归档源码，而不是仅依据论文摘要。

- Lavender：`references/repos/lavender`，主要入口为 `llama_finetune/src/llama_recipes/mllama/modeling_mllama.py`、`diffag/diffag_xattn_manager.py`、`configs/diffag_config.py`。
- BlindVLA：`references/repos/blindvla`，固定 commit `06855fcb91d65c88ca351a2d45af0be860a91987`，主要入口为 `openvla/vla-scripts/finetune_align.py`。

## 一句话差异

Lavender 让 **同一个词的学生空间 attention map** 逼近 Stable Diffusion 的词—区域 map；BlindVLA 让 **学生视觉 patch feature** 逼近冻结视觉教师 feature。前者是语言条件空间监督，后者是语言无关的视觉表征保持。

| 项目 | Lavender | BlindVLA | 我们当前 V3/V4 |
|---|---|---|---|
| 学生监督量 | 词级 image attention map | 中间视觉 patch embedding | phrase-score `gradient×activation` map |
| 教师 | 离线 Stable Diffusion per-word map | 在线冻结 C-RADIOv3-L / DINO / Theia feature | 离线 SD null-text phrase map；V2/V4 加 DINO feature |
| 目标损失 | 归一化图的 MSE | patch cosine distance | normalized mass 的 KL(teacher || student) |
| 语言条件性 | 是，逐词 | 否 | 是，输出 phrase-score 条件 |
| 直接解决的问题 | text-region binding | fine-tuning 遗忘视觉表征 | 输出依赖的语言条件空间归因 |
| 与我们关系 | 强基线 / 外部 semantic map 机制 | 强表征保持基线 | 候选核心机制 |

## Lavender 实际如何匹配

### 1. 教师图不是训练时在线 SD

`attention-map-generation/run_seg_batch_multip.py` 先从 SD 提取每个词的 attention map；数据集 loader 再把它们以 `sd_attn: Dict[word, map]` 读入 batch。因此 VLM 微调阶段不反传 SD，也不需要每步运行 diffusion。

这个工程选择应该复刻：教师图必须离线缓存、记录 caption/word、tokenization、resolution 和 map 版本。我们已经采用离线 SD null-text cache，这一点与 Lavender 一致且更严格，因为输入图通过 inversion 条件化。

### 2. 学生图来自可微 attention，不是 gradient attribution

Lavender 改写 MLLama cross-attention，保存 word-to-vision attention；跨层/头聚合后送入轻量卷积投影器。`diffag_xattn_manager.py` 的 `DiffagProjection` 是若干 2D conv、norm、ReLU 和最后 1×1 conv，不是直接拿 raw attention 做 loss。

它随后按词聚合子词，单词图做 min-max normalization，resize 到 32×32。核心实现位于 `compute_perword_attention_dict`：重复子词先平均；图减最小值、除以范围；再 resize。

### 3. 实际 loss

在 `modeling_mllama.py`：只对同时出现在学生图和 SD 字典中的词计算

$$
L_{Lav} = \frac{1}{|\mathcal W|}\sum_{w\in\mathcal W}
\operatorname{MSE}(M^{SD}_w,\;P(M^{stu}_w)).
$$

默认 `sd_xattn_loss_scale=10.0`。配置还提供：只对 noun、adjective、NAV 词或 subject/object 对齐；多层 attention 的 average/max；attention map 的 instance/batch/group normalization；MSE decay / inverse decay；以及 projection pretraining。

### 4. 值得借鉴的 tricks

- **词筛选**：对 noun 或 subject/object，而不是对全部 token 做 spatial loss。功能词和标点造成的 sparse/noisy map 不应主导训练。
- **一词一图、一词一损失**：先以词为单位平均多子词，再平均有效词的 loss，避免长 caption 获得更多监督权重。
- **空间投影器**：卷积投影器能把多头、多层 attention 转成可与 SD 图比较的单通道 spatial map；这比直接将 raw attention 或 raw gradient map 丢进 KL 更有表达力。
- **range normalization**：对每词图 min-max，再 MSE；它关注形状而不是总质量。我们目前 mass-normalized KL 对峰值浓度和微小噪声更敏感。
- **loss schedule**：源码具有 `mse_decay`、`inverse_mse_decay`、`loss_linear_alpha` 和 projector pretrain，说明作者没有把固定 loss scale 当成唯一方案。
- **调试可视化**：每 `plot_step` 同时保存 RGB、SD 图、投影后学生图和可选 raw 图。这对发现 token/resize 错配比只看最终数值更有效。

### 5. 不能照搬的点

Lavender 的 MLLama 有标准 text cross-attention；Qwen2.5-VL 的视觉 token 与文本 token 在 causal decoder 中通过 self-attention 交互，没有同样的 cross-attention 模块。因此不能逐行移植其 hook。

在 Qwen 的机制等价实现中，学生图应定义为：回答/teacher-forced caption 中的目标 phrase token，对视觉 token 的 self-attention slice；选定语言层和 head 聚合；再通过轻量空间投影器产生词图。由于 phrase token 位于图像之后，它可以读到视觉 token，语义上与 Lavender 的词→图区域监督相近。

这属于 **Lavender-adapted baseline**，必须这样命名，不能称为严格 Lavender 复现。

## BlindVLA 实际如何匹配

### 1. 学生与教师量

`finetune_align.py` 用 `output_hidden_states=True`，根据 `output.projector_features.shape[1]` 获取视觉 token 数 N，然后从选择的 language hidden layer 截出 BOS 后 `[1:1+N]` 的 N 个视觉 token。冻结视觉 teacher 对同一图像输出 patch feature。

$$
L_{BV}=
\sum_{l\in\mathcal L}
\operatorname{mean}_{b,i}\left[-\cos\left(
\operatorname{norm}(P_l(h^{stu}_{l,i})),
\operatorname{norm}(z^{teach}_i)
\right)\right].
$$

默认 `align_coeff=0.2`、`align_layers="16,"`、teacher=`c-radio_v3-l`。这是语言无关的视觉 feature retention，不回答“哪个词应该看哪里”。

### 2. Teacher 和 projector tricks

- 教师 eval + `requires_grad=False` + `torch.no_grad()`：这是应直接保留的正确做法。
- C-RADIO 分支把 OpenVLA image tensor 反归一化、resize 到 256，再送教师；视觉 teacher 的预处理必须精确锁定。
- 每个对齐层有一个 `LayerNorm → Linear → SiLU → Dropout → Linear → SiLU → Dropout → Linear` projector，内部宽 2048。
- 默认冻结 projector；虽然它是随机初始化，梯度仍可穿过它更新学生。

### 3. 代码层面的不能直接复刻点

BlindVLA 代码有几个需要修正后再作为基线的工程问题：

- optimizer 只接收 `vla.parameters()`；若解冻独立 projector，projector 不在 optimizer 中，实际上不会更新；
- 多层 projector 只保存第一个，恢复不完整；
- 多层 alignment loss 求和而非平均，层数改变会同时改变有效 loss strength；
- projector 在 DDP 外构造，需确保多卡 seed/broadcast 一致；
- README 的命令指向普通 `finetune.py`，而非实际对齐入口 `finetune_align.py`。

因此可以复刻 **方法机制**，不应原样复制该脚本的 checkpoint/optimizer 行为。

## 与我们当前 V0–V4 的关系

我们的 V2 是 BlindVLA 的 DINO-adapted feature retention baseline：冻结 DINO、将 teacher patch feature bridge 到 Qwen post-merge grid、进行 feature cosine loss。它保留了 BlindVLA 的问题定义，但与官方的 C-RADIO、层16和 projector 细节不同，应在论文中称为 *BlindVLA-style visual feature retention*。

我们的 V3 当前是：

$$
A_{stu}=|\nabla_{e_{vis}}s_{phrase}\odot e_{vis}|,
\quad
L_{sem}=D_{KL}(\operatorname{Norm}(T_{SD})\|\operatorname{Norm}(A_{stu})).
$$

它和 Lavender 共享 SD external semantic teacher，但学生对象和 loss 都不同：我们对齐输出 phrase-score 的 gradient attribution mass，Lavender 对齐 explicit attention map 的 MSE。它和 BlindVLA 共享“防止 VLM/VLA 在微调中丢失视觉能力”的动机，但不共享监督对象。

## 能否直接复刻：建议的三条可比路线

| 路线 | 是否应做 | 复刻程度 | 在论文中的身份 |
|---|---|---|---|
| B0：BlindVLA-style DINO/C-RADIO feature cosine | 应做 | 机制可直接复刻；修复 projector/optimizer/checkpoint | 强表征保持 baseline |
| L0：Lavender-adapted Qwen attention-map MSE | 应做 | Qwen 上只能机制等价复刻 | 强 external semantic-map baseline |
| Ours：phrase-score gradient attribution + semantic map | 保留 | 当前实现 | 主方法候选 |

最小的公平表应是 `V1`、`B0`、`L0`、`Ours`、`B0+Ours`。其中 L0 与 Ours 使用**相同 SD cache、同样的词筛选、同一 token-grid bridge 和同一数据**；否则结果不能说明是 student map 定义更好，还是教师/数据不同。

## 对当前结果的直接启发

当前 lambda/temperature/warm-up sweep 反复显示：gradient attribution 版 V3 的 pointing/mass 能改善，但 IoU 不稳定。Lavender 的源码提示一个具体、可证伪的改变，而不是继续调标量：

1. 构造 phrase token → visual token 的 direct attention map；
2. 对 phrase 子词、head、选定 layer 做固定聚合；
3. 用轻量 2D spatial projection、per-word min-max normalization、32×32 MSE 对齐同一 SD map；
4. 仅监督 noun / subject-object phrase；
5. 采用 inverse warm-up，使 attention map projector 和 caption SFT 先稳定；
6. 将 L0 视为 Lavender-adapted baseline；若我们的 gradient attribution 仍优于 L0，才支持“归因而非 attention”是必要的。

BlindVLA 的 lesson 是：不要用 L0 的成功或失败解释所有现象。必须保持 B0 feature retention，并测试 `B0+Ours` 是否仍有额外收益。若 semantic map 只在没有 retention 时有用，它更可能只是缓解视觉遗忘；若在 B0 后仍有增益，才更接近语言条件空间归因的独立贡献。

## 推荐下一步

暂停无边界 scalar sweep。先实现 L0 和经修正的 B0，使用固定 64 条 held-out 和至少三 seed。只有在 `Ours > L0` 且 `B0+Ours > B0` 时，才扩大到 F1-10k 和 VLA。

## 目前哪些“涨点”可以有理有据地主张

这里的“可主张”严格限于已经完成的 F0v2 受控实验：固定 256 条训练 pair、固定 64 条 image-disjoint held-out records、同一归因定义和同一 metric 实现。它不等于跨模型、跨数据集或论文最终结论。

### 证据等级

| 等级 | 含义 | 允许的表述 |
|---|---|---|
| A：直接且稳定 | 同一比较在多 seed / 多训练长度中同方向，且覆盖目标指标 | “在本设定下稳定提升” |
| B：方向性支持 | 单次或部分 seed 提升；其他指标或 bootstrap 不稳定 | “显示方向性收益 / 值得作为候选” |
| C：机制有效但未转移 | teacher 或中间量通过控制，但学生最终指标未稳定改善 | “teacher signal 有效，transfer 尚未证实” |
| D：尚未实测 | 只有源码分析或设计，未运行公平实验 | “应作为 baseline，不能声称提升” |

### 已运行方法与可说、不可说的结论

| 方法 | 最公平的当前比较 | 观测增量 | 证据等级 | 目前可以写什么 | 目前不能写什么 |
|---|---|---|---|---|---|
| V1 caption SFT | 100-step V1 相对 V0 | Pointing `+0.0781`；Mass `+0.0283`；IoU `-0.0013` | B | caption SFT 改善本 held-out 的 target pointing 和 attribution mass | SFT 提升全部 grounding 指标，或提升来自 semantic supervision |
| SD teacher map | correct 相对 same-image wrong-word | Pointing `+0.3594`；Mass `+0.0816`；IoU `+0.0472` | A/C | SD phrase map 含有可识别的词—区域空间信息，且不是任意空间先验 | SD teacher 被学生成功利用，或 SD teacher 必然优于所有视觉教师 |
| BlindVLA-style V2 | 100-step V2 相对 V1 | Pointing `-0.0156`；Mass `-0.0093`；IoU `+0.0001` | C | DINO feature retention 已作为独立对照接入；当前配置未显示 grounding 增益 | BlindVLA-style retention 在 Qwen/Flickr 上有效，或其能改善语言条件 spatial grounding |
| Ours V3, λ=.10,T=1 | 100-step 相对 semantic-off V3 λ=0 | Pointing `+0.0469`；Mass `+0.0110`；IoU `-0.0043` | B | semantic loss 存在改善 pointing/mass 的候选 scale | 当前 KL map matching 稳定提升三项 grounding 指标 |
| Ours V3, λ=.10,T=1.25 | seed17 相对 λ=0,T=1；seed29 matched baseline | seed17 `+0.0781/+0.0068/+0.0001`；seed29 `+0.0469/+0.0047/-0.0011` | B | teacher softening 到 T=1.25 在两个 seed 上都提高 pointing/mass；是最值得后续保留的 candidate | T=1.25 已确定提升 IoU，或已证明方法整体有效；两 seed paired bootstrap 区间仍跨 0 |
| Ours V3 warm-up 50 | seed29 相对同 seed no-warm-up candidate | Pointing `+0.0156`；Mass `+0.0076`；IoU `-0.0009` | B | semantic loss 的介入时机影响优化，warm-up 可改善 pointing/mass | warm-up 解决了几何定位问题，或对所有指标有收益 |
| Lavender-adapted L0 | 尚未运行 | — | D | 这是最必要的 external semantic-map attention baseline | Lavender 式 MSE 必然优于当前 KL，或我们已经超过 Lavender |

增量的顺序为 `Pointing / Mass-in-box / Top-20% box IoU`。A/C 表示 teacher 的空间质量已得到强控制支持，但该结论不自动转移到 student fine-tuning 后的 grounding 改善。

### 对论文结果表的直接建议

在当前阶段，唯一能明确写成“已验证”的正面结果应是 **SD teacher map 的词—区域质量优于 wrong-word、wrong-image 和 random controls**。对于学生模型，只能写：

> 在小规模固定设置下，semantic attribution supervision 在两个 seed 上呈现 pointing 与 mass-in-box 的方向性收益；然而 top-20% box IoU 未稳定改善，因此尚不能称为稳健 grounding gain。

若 L0 跑完后 `Ours > L0`，并且在至少三 seed 上同时满足 `V3 > V1` 的 pointing、mass 和 IoU，才可以升级为“语言条件空间归因匹配带来 grounding 提升”。若 `B0+Ours > B0` 同时成立，才能进一步声称该收益并非普通视觉表征保持造成。

## Lavender-adapted 的跨模型适用性合同

不能保证原版 Lavender 在所有 VLM 上无改动运行。原版依赖 MLLama 的标准 cross-attention API；大量 VLM 用的是视觉 token prefix 加 causal self-attention、window attention 或 fused attention kernel。我们能保证的是一个**有能力前提的适配接口**，而不是声称所有模型共享同一 hook。

### 统一数学接口

对样本 `(x, c)`，每个模型 adapter 必须提供一个由目标词/phrase `w` 条件化的、非负的视觉 token 图：

$$
M^{stu}_{w}=\operatorname{MapAdapter}(f,x,c,w)\in\mathbb R_+^{N},
\qquad
G:\{1,\ldots,N\}\rightarrow[0,1]^2.
$$

`G` 是 visual token 到归一化二维坐标的桥。将学生图 reshape/resample 到共同网格，经轻量 projector 和每词 normalization 后，才计算：

$$
L_{L0}=\frac{1}{|\mathcal W|}\sum_{w\in\mathcal W}
\operatorname{MSE}\left(\operatorname{Norm}(P(M^{stu}_w)),\operatorname{Norm}(M^{SD}_w)\right).
$$

SD map、词筛选、空间网格和 MSE 定义跨模型固定；唯一模型相关的部分是 `MapAdapter` 与坐标桥 `G`。这才是可比较的通用性。

### 四类模型与对应 adapter

| 模型结构 | 可用学生图 | 是否称 Lavender-adapted | 例子/说明 |
|---|---|---|---|
| 显式 cross-attention | phrase query → visual key 的 cross-attention | 是，最接近原版 | MLLama 类结构 |
| 视觉 prefix + decoder self-attention | answer/caption phrase token → visual token 的 self-attention slice | 是，机制等价 | Qwen2.5-VL、LLaVA/Prismatic 类、许多 InternVL 类结构 |
| 动作 token / action query decoder | action token 或 action query → visual token attention | 只能称 Lavender-inspired VLA baseline | 监督对象从词变成动作，不等于原论文词图监督 |
| 无可读 attention 或 fused kernel 不返回权重 | phrase/action score 的 gradient attribution 或 occlusion | 否；这是我们的 attribution adapter | 不能把 fallback 伪装成 Lavender |

对 prefix decoder，目标 query 必须位于 visual token **之后**，否则 causal mask 不允许该 token 读取图像。对多图、动态分辨率或 window reorder，`G` 必须来自模型返回的 grid metadata，绝不能用 `sqrt(N)` 猜二维网格。

### 每个新模型的准入测试

模型只有同时通过以下测试，才能进入 L0 表：

1. **视觉块定位**：精确记录 visual token index、view ID、grid height/width、merge/window reorder；
2. **目标词定位**：目标 phrase 在 teacher-forced sequence 中的 token span 可找到，且 query 位于可见视觉 token 的 causal 位置；
3. **attention 可读且可微**：关闭 flash/fused attention 或使用 eager/SDPA 路径后，能返回目标 query 对 visual keys 的权重，并验证 loss 对该图有梯度；
4. **坐标桥可逆审计**：将一个人工单热点从 token grid 映回 image coordinate，确认与 patch center 一致；
5. **同图反事实**：同图错误词、错误图、random map 不能得到与正确词图相同的 teacher/student agreement；
6. **地图独立评估**：同时报告 pointing、mass-in-box、IoU 和 perturbation，不用模型自身 attention 充当 ground truth。

失败时的处理也必须固定：如果 1--4 中任何一项失败，该模型只能进入 `AttributionAdapter` 或 feature-retention baseline，不进入 Lavender-adapted aggregate。这样“适用于多模型”是可证伪的工程合同，而不是泛化宣传。

### 能在论文中怎样表述

在至少两种不同融合结构模型通过上述准入测试前，应该写“architecture-adapted attention-map baseline”，而不是“universal Lavender”。当 explicit cross-attention 和 prefix-self-attention 各有一个模型通过相同 teacher、相同 token-grid protocol 和相同 evaluation 后，才能写：

> Our adapter instantiates the same word-conditioned spatial-map matching objective across heterogeneous VLM fusion mechanisms.

这仍不意味着所有黑箱、无 attention 输出的模型都可使用；对这类模型，我们的 gradient attribution adapter 才是更普适的路线。

## 我们真正的多模型主线：统一接口，不是统一 attention

用户的目标是多模型适配。正确的抽象不是“所有模型都对齐 Lavender attention”，而是所有模型均输出同一语义的**语言条件空间证据图**：

$$
\mathcal A_f(x,c,q)=(M, G, \text{metadata}),
$$

其中 `f` 是任意 VLM/VLA，`x` 是图像，`c` 是语言条件，`q` 是回答 phrase 或 action target，`M` 是视觉 token 的非负空间证据，`G` 是 token-to-image coordinate bridge。教师 SD map、空间 loss、反事实和 grounding 指标统一；模型间变化只局限于如何得到 `M`。

| 组件 | 作用 | 适用范围 | 是否是我们的核心贡献 |
|---|---|---|---|
| `AttentionAdapter` / L0 | 直接从 phrase query 到 visual token 的 attention 得到 `M` | 有可读 cross-attention 或 self-attention 的模型 | 否；Lavender-adapted 强 baseline |
| `GradientAdapter` / Ours | 从 phrase/action score 对视觉 token 的 gradient×activation 得到 `M` | 有白盒梯度的 VLM/VLA；不要求标准 cross-attention | 是；多模型主线候选 |
| `OcclusionAdapter` | 遮挡视觉区域后测 phrase/action score 变化 | 黑箱或 attention 不可读模型 | 否；最通用的审计/评价 fallback，训练成本高 |
| `FeatureRetentionAdapter` / B0 | DINO/C-RADIO teacher cosine 保持 patch feature | 有视觉 patch feature 的模型 | 否；BlindVLA-style 正交基线 |

### 它们的区别不是“谁替代谁”

`AttentionAdapter` 回答“模型显式 attention 指向哪里”；它训练便宜、一阶可导，但依赖结构，且 attention 不必等于输出因果证据。

`GradientAdapter` 回答“为了提高当前 phrase/action score，哪些视觉 token 的激活最重要”；它天然绑定输出目标，适配 Qwen、prefix VLM、action-token VLA 等不同融合形式，但要计算二阶梯度，训练更贵且可能不稳定。

`OcclusionAdapter` 回答“遮掉哪里会让输出分数下降”；它最接近因果检验、最模型无关，但不适合逐 batch 训练，更适合验证 heatmap 的真实性。

`FeatureRetentionAdapter` 不回答“语言词看哪里”。它仅维持视觉 feature，因此可与前三者叠加，用于排除“空间图变好只是因为视觉编码器没有遗忘”的解释。

### 跨模型执行规则

对每个模型先进行 capability audit，再选择最强可用 adapter：

1. 有稳定可读 attention：同时运行 `AttentionAdapter` 和 `GradientAdapter`；前者是 Lavender L0，后者是 Ours，直接比较 `Ours > L0`。
2. 没有标准 cross-attention、但可以对 score 反传：运行 `GradientAdapter`；不能为了凑 baseline 把不可靠 attention 命名为 Lavender。
3. 无梯度或无 attention：仅用 `OcclusionAdapter` 做评价；该模型不进入 gradient-training claim。
4. 有视觉 patch feature：额外运行 `FeatureRetentionAdapter`，并检查 `B0+Ours > B0`。

VLM 阶段的 `q` 是 teacher-forced phrase score；VLA 阶段的 `q` 改为 delta-EEF action token log-probability、action head scalar 或 action loss。接口和 teacher map 不变，只有 output target 变了。这样“从 VLM 到 VLA”的扩展不是另起炉灶，而是把同一个语言条件空间证据定义从 answer target 换到 action target。
