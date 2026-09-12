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

教师 registry 按“语义教师 + 动作相关性细化器”组织，而不是把多个大模型 loss 直接堆叠。首轮 VLM 只启用语义空间教师：

```yaml
teacher:
  semantic:
    type: lavender_stable_diffusion
    signal: word_cross_attention
    output: T_sem_word_image_region
    use_in_inference: false
  dynamics:
    enabled: false
    role: action_relevance_refiner
    candidates: [libero_demo_rules, lingbot_va, lingbot_video]
    signal: phase_or_future_progress_mask
  action_relevance:
    enabled: false
    role: action_relevance_refiner
    candidates: [libero_demo_rules, pi0, lingbot_vla, qwen_robotmanip]
    signal: region_relevance_mask_or_candidate_action_score
```

三类教师的定位：

| 教师 | 首轮状态 | 进入条件 |
|---|---|---|
| Lavender / Stable Diffusion | 启用 | 默认教师，证明 VLM grounding |
| Qwen/LLaVA/DeepSeek 离线归因 ensemble | 关闭 | 扩散教师定位失败或需要 sanity check |
| LIBERO demonstration 派生规则 | 关闭，但优先纳入 VLA 扩展 | 从专家轨迹构造阶段标签、接触点、目标进展和区域 relevance mask |
| LingBot-VA / LingBot-Video world model | 关闭，但纳入研究设计 | VLA smoke 成立后，用于未来状态、接触变化、目标状态接近度，细化 `T_sem` |
| π0 / LingBot-VLA / Qwen-RobotManip action expert | 关闭，但纳入研究设计 | 只评估候选动作或区域 relevance，不直接蒸馏其策略动作 |

教师输出必须保存：词/短语、token span、原图尺寸、教师图尺寸、归一化方式、生成模型版本、随机种子和文件 SHA256。Action-Relevance Refiner 还必须保存当前观测、候选动作、阶段标签、relevance mask、预测未来状态或候选动作分数，避免只保存一个不可审计的标量分数。

### 5.2 Action-Relevance Refiner 的接口

Action-Relevance Refiner 不直接监督最终动作，不替代行为克隆标签，也不把 π0、LingBot 或 Qwen-RobotManip 当作 policy teacher。它只生成一个 refinement mask：

`R_act(w,x,a,s) ∈ R^(H×W)`

用于把 `T_sem(w,x)` 从 object-level semantic map 细化为 phase-conditioned actionable map：

`T_AR(w,x,a,s) = Normalize(T_sem(w,x) ⊙ R_act(w,x,a,s))`

首选实现来自 LIBERO demonstration 的弱规则，而不是直接依赖大模型 teacher：

| 阶段 `s` | 规则信号 | 推荐 relevance 区域 |
|---|---|---|
| approach | EEF 靠近 source object | source object 整体位置 |
| grasp | gripper closing 且接近物体 | 可抓取区域、夹爪-物体接触区域 |
| lift | 物体随 EEF 上升 | 物体与夹爪接触区域 |
| place | 物体靠近 target receptacle | target opening、目标放置区域 |
| release | gripper opening 且物体在目标附近 | 物体与 receptacle 的相对位置 |

World/video-action 模型只作为第二实现，用于从未来状态预测中估计目标进展：

```python
def compute_action_relevance_from_world(world_model, obs_t, instruction, candidate_action, phase):
    """
    Returns:
        relevance_mask: FloatTensor[H, W]
        future_latent: optional predicted next or short-horizon visual latent
        goal_progress: optional scalar score measuring movement toward the language goal
        evidence: metadata for audit, e.g. predicted object/contact/state change
    """
```

Action expert 也只作为 refiner，用于评估候选动作或区域是否在可行操作流形上：

```python
def compute_action_relevance_from_action_expert(action_model, obs_t, instruction, candidate_action, phase):
    """
    Returns:
        relevance_mask: optional FloatTensor[H, W]
        candidate_score: feasibility or likelihood under the teacher action model
        evidence: action embedding or attribution metadata
    """
```

三个信号的实验目标不同：

- `T_sem` 检验词-区域 grounding；
- `R_act` 检验当前动作阶段下哪些语义区域真的与成功有关；
- `T_AR` 检验 action-relevant semantic map 是否优于静态 semantic map。

因此报告时必须分开列出 `T_sem only`、`T_sem + random R_act`、`T_sem + wrong-stage R_act`、`T_AR correct`，不能只报告一个混合 teacher 的最终分数。

### 5.2.1 Next Forcing / future-attribution extension

### 5.2.2 Attention Sink 处理原则

Attention sink 是少数特殊 token 或固定位置长期吸收注意力/归因质量的现象。它会让二维热力图看起来有高亮，却不代表模型真正使用了目标区域。因此 P3 先做 sink-free 诊断：统计特殊 token、边缘 patch 和高频 patch 的归因质量，并比较 mask 前后的空间熵、目标 IoU 与增强一致性。

主方法不把 sink removal 宣称为核心创新。只有当 sink 对结果有实质影响时，才启用明确的 mask、重归一化或 token-drop，并在 P8 报告 raw/sink-free 和多种 mask 的消融。这样可以区分“归因对齐带来的收益”和“去掉热点后指标变好”的预处理收益。

#### WAM/VAM 与 VLA 的接口

世界模型负责预测状态转移，视频生成模型负责生成未来观测，VLA 负责输出 delta EEF 动作。B 路线把前两者作为冻结的未来评估器：对 VLA 的候选动作 rollout 后提取未来视频归因 `R_future`，再与 `T_sem` 相乘得到 `T_AR`。该接口只提供 action relevance refiner，不改变 VLA 的基本动作头。

Next Forcing 只进入 B 路线的后期扩展。对当前观测、语言和候选动作，world model 预测多个时间尺度的未来视频 chunk：

```text
F^k_t = predicted future video chunk at horizon k
R_future^k(i) = attribution of F^k_t to visual patch i
R_future(i) = Σ_k β_k R_future^k(i)
T_AR = Normalize(T_sem ⊙ R_future)
```

首版不将 `R_future` 写入 D0/D1 的训练损失。先做离线分析：比较 `A_act(t)` 与 short/mid/long future attribution 的相关性，并使用 random-horizon、wrong-action、wrong-future 等负控。只有当未来归因和成功/失败的未来状态变化有稳定关系，才考虑把它作为 B 的 refinement mask。

### 5.3 主方法 D：Structure-Native Action Attribution

VLA 阶段的优先路线不是先构造外部动作区域，而是读取模型内部结构：

| 内部信号 | 记号 | 推荐提取方式 | 用途 |
|---|---|---|---|
| 语言归因 | `A_lang(w)` | word/span hidden state 到 visual tokens 的 similarity、attention 或 gradient×input | 被 Lavender `T_sem` 锚定 |
| 动作归因 | `A_act(a)` | action query attention、action token log-prob gradient、delta EEF loss gradient×activation | 表示动作决策依赖哪些视觉 token |
| 层级角色 | `layer_type` | SpikingBrain full/window/GLA，或其他 VLM 的视觉层/融合层 | 决定哪些层参与主监督 |

`A_act` 的默认优先级：

1. **delta EEF loss / action output gradient×activation**：默认实现，最普适。连续动作 VLA、action token VLA、Qwen/LLaVA/DeepSeek 的 VLM 归因都能复用“目标标量对视觉 token 求梯度”的接口。
2. **action token log-prob gradient**：适用于 OpenVLA 这类离散 action token 模型。
3. **action query attention**：如果模型天然有 action query，则作为更可解释的结构内生版本；但它不是默认假设，因为很多模型没有显式 action query。

首版最小实现：

```text
1. 用 Lavender 监督 A_lang：L_sem = D(A_lang, T_sem)
2. 用 action loss 对 visual tokens 求 A_act
3. 约束 A_act 不跑出 A_lang_union：L_contain = Σ_i A_act(i) · (1 - A_lang_union(i))
```

D 路线按两个版本推进：

**D0：静态 containment，首版必做。**

```text
VLM:    T_sem → A_lang
VLA-D0: A_act(t) ⊂ A_lang(source) ∪ A_lang(target)
```

D0 只要求动作归因主要落在语言相关区域并集内，不强行规定每个时间步必须看 source 还是 target。

**D1：phase-conditioned transition，后期扩展。**

```text
VLA-D1: A_act(t) follows source → source+target → target
```

D1 使用软阶段目标：

```text
A_phase(t) = α_t · A_lang(source) + (1 - α_t) · A_lang(target)
L_phase = D(A_act(t), A_phase(t))
```

`α_t` 不用硬 one-hot。推荐初值是 source phase `0.9`、mixed phase `0.5`、target phase `0.1`。距离函数 `D` 与 `L_sem` 保持一致，首选 normalized MSE，同时记录 KL/cosine。

phase 来源按优先级分三档：

| 版本 | 来源 | 规则 | 用途 |
|---|---|---|---|
| D1a | LIBERO 状态规则 | EEF-source 距离、EEF-target 距离、夹爪开合、source 高度、object-target 距离 | 快速 sanity check |
| D1b | 成功 demonstration 事件边界 | 自动检测 `t_grasp`、`t_lift`、`t_near_target`、`t_release`，再映射到 source/mixed/target | 推荐后期主扩展 |
| D1c | 模型内生 phase | 用 `mass_source=sum(A_act·A_lang(source))` 与 `mass_target=sum(A_act·A_lang(target))` 推断迁移状态 | 更普适，放在后期研究 |

D1b 的事件检测伪代码：

```python
t_grasp = first_t(gripper_closing and dist(eef[t], source[t]) < eps_src)
t_lift = first_t(source_z[t] - source_z[0] > eps_lift)
t_near_target = first_t(dist(source[t], target) < eps_tgt)
t_release = first_t(gripper_opening and dist(source[t], target) < eps_tgt)

phase[t < t_grasp] = "source"
phase[t_grasp <= t < t_near_target] = "mixed"
phase[t >= t_near_target] = "target"
```

如果事件缺失，退化成三阶段：before grasp、after grasp before near target、after near target。不要用固定轨迹百分比作为主实现，只能作为负控。

`L_phase` 必须配负控：反向 phase、随机 phase、固定百分比分 phase、正确 phase 但错 source/target 词图、只加 `L_phase` 不加 `L_contain`。

相关 phase / skill segmentation 方法已整理到 `references/phase_segmentation/phase_segmentation_methods.md`，包括 TACO、BUDS、CHAMP、RoboSegNet、Robo2VLM、ProcVLM 和 ROVER。D1 实现时优先参考 D1b demonstration event boundaries，不把 VAM/world model 作为默认 phase 来源。

这个版本的创新点是结构内生一致性：扩散教师只锚定语言证据，动作证据通过模型自己的 action query/action head 与语言证据保持一致。World model / VAM 不属于 D 的默认依赖，只能在 B 路线中作为 `R_act` refiner。

`A_lang` 和 `A_act` 必须先映射到同一个图像网格再比较。模型内部处理的不是原始像素，而是视觉 tokens；不同层或不同分支可能有不同 token 顺序、分辨率和窗口重排。例如 `A_lang` 可能来自 SpikingBrain 第 23 层 full-attention，还原后是 `16×16` patch 图；`A_act` 可能来自 action head 的最终视觉 token，可能是另一个顺序或分辨率。坐标桥的工作就是把它们都还原成同一张输入图像上的 `H×W` 空间图，否则 `L_contain` 会比较错位置。

### 5.4 B/C 扩展：动作相关区域的文献依据与首轮构造

相关工作里，“动作相关区域”通常不是凭 RGB 直觉画二维热力图，而是由几何、深度、像素动作值或接触状态构造：

| 路线 | 代表工作 | 区域/动作表示 | 对本项目的启示 |
|---|---|---|---|
| 像素级动作值 | FC-GQ-CNN / Dex-Net、Transporter Networks、CLIPort | 在图像或正交高度图上输出 grasp/place/pick 的像素位置或 Q map | `R_act` 可以是 action-value map，不必是语义 attention |
| 语言条件 pick-place | CLIPort、PerAct、RVT 等 | 语言条件下在 heightmap/voxel/view 上选择 pick/place/action token | 动作相关区域应随阶段从 source object 切到 target receptacle |
| 可供性 affordance | Where2Act、ActAIM 等 | 预测对象部件或关节处的可操作区域和动作方向 | 对 drawer/door 等任务，`R_act` 应偏向 handle、joint、contactable part |
| 接触/状态变化 | Contact-GraspNet、接触丰富操作和视觉触觉工作 | 预测接触点、抓取点或物体状态变化 | grasp/lift 阶段应用 EEF/接触附近区域，而不是整物体 mask |
| 动作流形/动作专家 | π0、Diffusion Policy、OpenVLA 类方法 | 输出连续动作、action chunk 或动作 token | 后续只用于候选动作 relevance，不作为策略蒸馏主教师 |

因此 B/C 路线中的 `R_act` 应声明为 **image-plane proxy**：由仿真 3D 状态、相机投影、对象 mask/depth/heightmap 生成的二维弱动作相关区域，不是精确 3D affordance。它只用于细化 `T_sem` 或诊断 `A_act`，不是主方法的必要条件。

推荐三档实现：

| 版本 | 构造方式 | 用途 |
|---|---|---|
| V0 bbox proxy | 对象 bbox + EEF 投影高斯 | 最快 smoke，噪声最大 |
| V1 mask proxy | 仿真 segmentation 或 SAM mask + EEF/target 投影高斯 | 首轮推荐 |
| V2 geometry/heightmap proxy | object pose + camera pose + depth/heightmap + gripper geometry | 更接近 Dex-Net/Transporter/CLIPort 传统，工程更重 |

grasp 阶段的 V1 mask proxy 可以写成：

`R_act(u,v)=Mask_source(u,v) · exp(-||[u,v]-Π(p_eef)||² / 2σ²)`

其中 `Π(p_eef)` 是 EEF 3D 位置投影到 RGB 图像平面的点。place 阶段则把高斯中心换成 target receptacle 的 opening/center/top region。这样二维高亮的依据是 3D 状态投影和对象 mask，而不是模型凭空猜接触点。

必须加入负控：

- `T_sem + random EEF point`；
- `T_sem + wrong-stage R_act`；
- `T_sem + wrong-object mask`；
- `T_AR correct`。

只有 `T_AR correct` 明显优于这些负控时，才能声称动作相关细化有效。

### 5.5 学生归因标准接口

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

以答案 token log-prob、类别分数或文本判断 score 为目标，计算视觉 patch hidden state 的 `gradient × activation`。这是 Qwen、LLaVA、DeepSeek 等没有可比 cross-attention 或 attention 语义不稳定时的首选接口。

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
  students: [spikingbrain_vl, qwen25_vl, qwen3_vl, llava16, llava_onevision]
  deferred_students: [deepseek_vl2]
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

### 9.2 LLaVA-1.6 / LLaVA-OneVision

适配目标是把 LLaVA 系列作为非 Qwen 系 VLM 骨干，验证方法是否依赖某一个 tokenizer、动态分辨率策略或视觉编码器设计。首选学生归因仍是 `answer_logprob → visual_patch_hidden` 的 gradient×input；若任务形式是 phrase grounding，则可把候选区域判断分数或 yes/no answer score 作为目标标量。需要记录：

- 视觉编码器类型、patch/grid 尺寸和 projector 输出 token 顺序；
- LLaVA-1.6 与 OneVision 的图像、多图或视频 token 组织差异；
- 是否能稳定取到 projector 后视觉 hidden states；
- 答案 token、对象短语和视觉 token 的坐标映射；
- 与 Qwen 系相比，归因图是否更依赖最后答案 token，还是更依赖中间 multimodal projector。

LLaVA-1.6/OneVision 只进入 VLM grounding 普适性验证，不在首轮承担 VLA 闭环任务。若它与 Qwen 系都能在同一 `T_sem` 下得到正确教师优于错图/随机教师的结果，才能较强地说明本方法不是针对 Qwen 或 SpikingBrain 特化。

### 9.3 DeepSeek-VL2

重点比较高分辨率/多图 token 组织和专家路由。若模型接口只能提供最终 hidden state，则用最终答案 score 对视觉 hidden states 做归因，不强行提取不可解释的 attention。MoE/router 作为分析元数据，不作为 `L_align` 的主监督对象。

### 9.4 OpenVLA

作为 LIBERO VLA 基线，保留其原生动作表示做官方对照；同时增加一个统一 7D delta EEF adapter，单独报告“原生动作接口”和“统一接口”结果。学生归因优先来自 action token log-prob 或 delta EEF adapter loss 对视觉 hidden states 的 gradient×input。

### 9.5 π0 / LingBot

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
