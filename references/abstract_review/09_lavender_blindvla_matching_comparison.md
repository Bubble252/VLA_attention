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
