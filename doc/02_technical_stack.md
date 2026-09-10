# 技术栈与实现规格：语言条件空间归因对齐

**版本**：Final design draft v1.0  
**日期**：2026-09-10  
**原则**：先把张量、层级、动作接口和验证标准写清楚，再开始代码复现。

## 1. 目录和版本管理

目标仓库：

`/home/bubble/类脑计算/VLM终局`

建议结构：

```text
VLM终局/
├── README.md
├── .gitignore
├── doc/
│   ├── 01_project_background.md
│   ├── 02_technical_stack.md
│   └── 03_execution_plan.md
├── references/
│   ├── README.md
│   ├── papers/                 # PDF 或官方导出的论文
│   ├── bib/                    # BibTeX/RIS
│   └── checksums.sha256
├── src/
│   ├── attention_bridge/
│   ├── models/
│   ├── policies/
│   ├── libero/
│   └── evaluation/
├── configs/
├── scripts/
├── logs/
└── outputs/
```

权重、数据集、运行日志和实验输出不直接提交 Git；只提交配置、代码、指标摘要和复现说明。

## 2. 已有本地代码与可复用入口

### 2.1 Lavender

路径：

`/home/bubble/类脑计算/参考/vlm`

当前提交：

`58fc71b attention map generation pipeline`

优先复用：

- `llama_finetune/src/llama_recipes/diffag/diffag_xattn_manager.py`
- `llama_finetune/src/llama_recipes/mllama/modeling_mllama.py`

需要保留的行为：

1. 词级 attention map 的生成和归一化；
2. 选定层、最后若干层、全层显示三种模式；
3. 教师图按共享对象词对齐；
4. 保存每个样本的词名、原图尺寸、教师图尺寸和归一化方式。

### 2.2 SpikingBrain

路径：

`/home/bubble/类脑计算/参考/spikingbrain仓库/SpikingBrain-7B`

当前提交：

`ef99987 Update Technical Report`

优先阅读：

- `hf_7B_VLM/configuration_spikingbrain_vl.py`
- `hf_7B_VLM/modeling_spikingbrain_vl.py`
- `hf_7B_VLM/gla.py`

关键配置：

```python
window_size = 112
fullatt_block_indexes = [7, 15, 23, 31]
```

实现时必须记录实际 checkpoint 的层数和配置，不能只依赖默认值。

## 3. 环境与依赖边界

GPU 型号、数量和显存由服务器提供，本项目不把硬件写死。需要记录的是运行时软件和数据版本：

- Python 3.10 或项目官方要求的版本；
- PyTorch、Transformers、Accelerate；
- Flash-Attention（若官方实现依赖）；
- OpenCV/Pillow；
- Gymnasium、LIBERO；
- NumPy、SciPy、scikit-learn；
- TensorBoard 或 WandB（二选一，先用 TensorBoard 也可）；
- Git LFS 仅在确实需要版本化小型权重时启用。

下载失败时按以下顺序处理：

1. 使用 `http://127.0.0.1:7897`；
2. 使用国内镜像或 ModelScope；
3. 记录实际 URL、commit、文件大小和 SHA256；
4. 不把未验证的自动下载脚本当作文献来源。

## 4. 两阶段数据流

```text
VLM image + instruction/question
            │
            ├── Lavender（训练前/离线）→ teacher map T(word)
            └── VLM student
                  ├── attention / hidden states / gradients
                  └── language-conditioned spatial attribution G(word, image, target)
                         ↓
                  L_task + λ_align L_align

LIBERO RGB + instruction + proprioception
            │
            ├── 同一套 teacher map T(word)
            └── VLA student
                  ├── SpikingBrain-VLA 或 OpenVLA/OFT
                  ├── action query / action token / delta EEF loss
                  └── action-conditioned spatial attribution G(word, image, action)
                         ↓
                  L_deltaEEF + λ_align L_align + λ_smooth L_smooth
```

部署时移除 Lavender 和 `L_align`。VLM 部署只保留学生模型；VLA 部署只保留学生 VLA、动作头和归一化器。

## 5. 教师与学生归因接口

### 5.1 教师 registry

首轮只启用扩散教师：

```yaml
teacher:
  type: lavender_stable_diffusion
  signal: word_cross_attention
  output: T_word_image_region
  use_in_inference: false
```

备选教师只登记，不进入首轮主实验：

| 教师 | 首轮状态 | 进入条件 |
|---|---|---|
| Lavender / Stable Diffusion | 启用 | 默认教师，证明 VLM grounding |
| Qwen/DeepSeek 离线归因 ensemble | 关闭 | 扩散教师定位失败或需要 sanity check |
| LingBot-VA / π0 / video-action world model | 关闭 | VLA smoke 成立后，用于时序和动作可达性教师 |

教师输出必须保存：词/短语、token span、原图尺寸、教师图尺寸、归一化方式、生成模型版本、随机种子和文件 SHA256。

### 5.2 学生归因标准接口

所有模型都实现同一个抽象接口：

```python
def compute_spatial_attribution(model, image, text, target, word_spans, adapter_cfg):
    """
    Returns:
        maps: FloatTensor[B, W_word, H, W]
        meta: token grid, resize rule, source layer, attribution type
    """
```

其中 `target` 在 VLM 阶段是答案 token、类别或文本判断目标；在 VLA 阶段是 action token、动作维度或 delta EEF loss。输出图必须已经回到原图坐标或可逆 patch 网格。

## 6. 空间归因桥接层的实现规格

### 6.1 输入输出

输入：

- 视觉层输出 `V_l: [B,N,D]`；
- 指令或动作查询 `q_w: [B,D]`；
- patch 网格信息 `grid_thw`；
- 可选的 window reorder/index；
- 教师词图 `T_w: [B,H_s,W_s]`。

输出：

- `A_l_w: [B,H_s,W_s]`；
- `valid_mask`，标记对象词是否在当前样本中出现；
- `layer_type`，取 `full/window/gla/hidden_gradient/action_gradient/proxy`；
- `grid_transform_id`，保证离线图和训练图使用同一还原过程。

### 6.2 Patch 归因方式

按实现难度分三个版本：

**V0：相似度归因（必须先实现）**

```text
score_i = cosine(project_q(q_w), project_v(V_l[i]))
A = softmax(score / temperature)
```

优点是稳定、可微、无需改动 attention kernel；缺点是它是归因图而非原生 attention。

**V1：VLM target gradient×input**

以答案 token log-prob、类别分数或文本判断 score 为目标，计算视觉 patch hidden state 的 `gradient × activation`。这是 Qwen、DeepSeek 等没有可比 cross-attention 时的首选接口。

**V2：动作 logit / delta EEF gradient×input**

以某一维 delta EEF 或动作总 logit 为目标，计算 patch 对动作的梯度，得到语言条件的动作归因。用于验证 V0 是否只是相似度假象。

**V3：显式 cross-modal routing**

增加轻量 query-to-patch 模块，使 `q_w` 对 patch 产生显式权重。只有 V0/V1 通过诊断后才实现，避免一开始改动主干过多。

### 6.3 坐标恢复

SpikingBrain 的 token 可能经过窗口重排。必须保存：

```text
original_patch_index
window_id
local_patch_index
grid_t, grid_h, grid_w
```

恢复顺序：

1. 从窗口序列恢复原始 patch 索引；
2. 将时间维合并或选择当前帧；
3. reshape 到 `[grid_h, grid_w]`；
4. 双线性 resize 到 Lavender 教师图大小；
5. 对有效区域重新归一化。

如果恢复前后同一个合成图的峰值位置不一致，禁止进入对齐训练。

## 7. 层级策略与实现分支

| 分支 | 张量是否存在 | 默认用途 | 禁止事项 |
|---|---:|---|---|
| Full attention `[7,15,23,31]` | 有显式 q/k/v softmax | 主对齐候选 | 不跳过 window index 还原 |
| Window/SWA | 当前返回不是真实概率图 | 局部诊断、proxy 对照 | 不把 `attn_output+1e-17` 当 attention |
| GLA | 返回递归状态 | proxy/skip 对照 | 不对状态矩阵直接做二维 MSE |
| 语言末端 hidden state | 有 | 构造 `q_w` 和动作 query | 不把语言 self-attention 直接 reshape 成空间图 |

首个训练配置：

```yaml
alignment:
  mode: similarity_bridge
  layers: [23, 31]
  distance: normalized_mse
  teacher: lavender_word_map
  invalid_layer_policy: skip
  lambda: 0.05
```

VLM 普适性配置：

```yaml
alignment:
  mode: target_gradient
  students: [spikingbrain_vl, qwen25_vl, qwen3_vl, deepseek_vl2]
  teacher: lavender_word_map
  target: answer_logprob
  distance: normalized_mse
  lambda: 0.02
```

## 8. VLA 动作头

动作定义：

```text
delta_eef = [dx, dy, dz, droll, dpitch, dyaw, gripper]
```

建议结构：

1. 从最后语言/视觉融合 hidden state 取动作 query；
2. 经过两层 MLP 或轻量 Transformer action head；
3. 输出 7 维连续值；
4. 使用训练集统计量归一化；
5. 首版 `H=1`，后续可扩展 `[H,7]` action chunk。

损失：

```text
L_action = SmoothL1(pred_delta_eef, target_delta_eef)
L_smooth = mean(||a_t - a_{t-1}||_1)
L_total = L_action + λ_align * L_align + λ_smooth * L_smooth
```

夹爪分量可以使用回归或二分类；首版统一回归，若 LIBERO 中开合不稳定，再单独改为 BCE 并记录接口变化。

## 9. 模型适配层

### 9.1 Qwen2.5-VL / Qwen3-VL

适配目标是抽取视觉 token、语言 token 和答案目标，不直接假设其 attention 与 SpikingBrain 同构。首选学生归因是 `answer_logprob → visual_patch_hidden` 的 gradient×input。需要记录：

- 视觉 token 数量和动态分辨率；
- 视觉层索引；
- 是否能稳定得到每层 hidden state；
- 与 SpikingBrain 的层数、窗口机制和 full-attention 位置的差异；
- answer token 与 instruction 中对象词的 span 对齐规则；
- 梯度归因是否稳定，是否需要 Integrated Gradients。

Qwen2.5-VL 的论文版本登记为 arXiv:2502.13923；Qwen3-VL 的技术报告登记为 arXiv:2511.21631。它们首先用于 VLM grounding 普适性对照；动作实验只有在 VLM 阶段成立后再统一接 7D delta EEF head。

### 9.2 DeepSeek-VL2

重点比较高分辨率/多图 token 组织和专家路由。若模型接口只能提供最终 hidden state，则用最终答案 score 对视觉 hidden states 做归因，不强行提取不可解释的 attention。MoE/router 作为分析元数据，不作为 `L_align` 的主监督对象。

### 9.3 OpenVLA

作为 LIBERO VLA 基线，保留其原生动作表示做官方对照；同时增加一个统一 7D delta EEF adapter，单独报告“原生动作接口”和“统一接口”结果。学生归因优先来自 action token log-prob 或 delta EEF adapter loss 对视觉 hidden states 的 gradient×input。

### 9.4 π0 / LingBot

作为后置架构参考。π0 的连续动作专家可指导 action chunk；LingBot-VA 的因果视频-动作世界模型和 LingBot-VLA 2.0 的统一多 embodiment 动作表示可指导多帧输入和动作空间扩展。它们可以在后期作为 world-model teacher 候选，但第一阶段不把它们与扩散词图教师混在同一训练脚本中。

## 10. LIBERO 集成

必须实现以下接口：

```python
obs = env.reset()
instruction = task.language
action = policy.predict(obs.rgb, instruction, obs.proprio)
obs, reward, done, info = env.step(action)
```

工程要求：

- 记录任务 ID、随机种子、语言指令、初始状态；
- 记录每一步 7D action 和执行频率；
- 对 delta EEF 做 workspace、旋转和夹爪限幅；
- 保存成功/失败原因，而不是只保存最终 success；
- 测试集只用于最终一次报告。

## 11. 评价指标与对照组

VLM 阶段主指标：

- VQA / referring expression / grounding accuracy；
- pointing accuracy；
- 目标区域 IoU；
- 归因图熵；
- 正确教师图相对错图/错词教师图的增益；
- 不同 VLM 学生上的平均增益和方差。

VLA 阶段主指标：

- LIBERO success rate；
- 按任务类型分组的 success rate；
- delta EEF 的 MAE/RMSE；
- 轨迹平滑度；
- 目标物体区域归因 IoU/pointing accuracy。

必须有的对照：

1. VLM 普通 SFT / LoRA；
2. Lavender 原式 attention-to-attention 对齐；
3. 本项目 teacher-to-attribution 对齐；
4. SpikingBrain-VLA，无对齐；
5. 所有可用层统一对齐；
6. full-attention `[23,31]` 对齐；
7. 仅 `[31]`；
8. 随机层；
9. 教师词图打乱；
10. 错配图片的教师图；
11. window/SWA/GLA 伪 attention 直接 MSE；
12. 相似度桥 V0 与 gradient×input V1/V2。

## 12. 文献与模型版本记录

每个下载文件在 `references/README.md` 登记：

- 名称；
- 论文/仓库 URL；
- 版本或 commit；
- 下载日期；
- 本地文件名；
- SHA256；
- 许可证和使用限制；
- 在本项目中用于“教师、主干、基线还是架构参考”。

这样可以区分“参考模型看过”和“真正用于实验”的证据链。
