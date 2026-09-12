# 项目背景与研究定位：语言条件空间归因对齐

**版本**：Final design draft v1.0  
**日期**：2026-09-10  
**验证顺序**：VLM grounding → LIBERO VLA → 真实机器人（后续）
**动作接口**：delta EEF，固定为 7 维  
**主线**：保留原始 VLM 的“外部视觉/扩散注意力监督”，但把方法主张从“对齐某个模型的 attention”升级为“对齐语言条件空间归因”。先证明不同 VLM 能被统一到可比较的词级空间归因图，再把同一机制接到 VLA 动作预测。

## 1. 已冻结的项目边界

| 项目 | 最终约束 |
|---|---|
| 研究对象 | 语言条件空间归因对齐；SpikingBrain-VL 是主实验骨干 |
| 首个验证环境 | VLM grounding；随后进入 LIBERO 仿真；不先做真机 |
| 动作输出 | `a_t=[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]`，即 7 维 delta EEF |
| 视觉教师 | 首选 Lavender 离线 Stable Diffusion 词级空间注意力图 |
| 核心机制 | 将不同 VLM/VLA 的 attention、hidden-state gradient 或动作条件归因统一成空间图，再与教师词图对齐 |
| 参考模型 | Lavender、Qwen2.5-VL、Qwen3-VL、LLaVA-1.6 / LLaVA-OneVision、Qwen-RobotManip、DeepSeek-VL2、OpenVLA、π0、LingBot 系列 |
| GPU | 由服务器环境管理；文档不绑定型号、数量或显存 |
| 代码仓库 | `/home/bubble/类脑计算/VLM终局` |
| 文献归档 | `/home/bubble/类脑计算/VLM终局/references` |

这里的“最终版”指研究方案和实验接口已经固定，不表示结果已经完成。第一阶段先验证 VLM 层面的 grounding 和跨模型归因接口，第二阶段再验证 LIBERO 动作闭环。真机部署需要另行处理相机标定、控制频率、碰撞安全和执行器限幅。

## 2. 原始想法的来源与本次收敛

原始 VLM 想法来自：

- `/home/bubble/类脑计算/参考/vlm`
- `/home/bubble/类脑计算/idea初筛/vlm_lavender_spikingbrain`
- `/home/bubble/类脑计算/idea初筛/我自己目前想到的.md`
- `/home/bubble/类脑计算/idea初筛/08_lavender_spikingbrain_vlm_fast_iteration_plan.md`

原始思路不是简单地“给 VLM 增加一项注意力损失”，而是：

1. 用冻结的扩散模型提供词级、空间化的外部视觉先验；
2. 观察 VLM 内部不同层的注意力功能并进行选择性约束；
3. 避免把所有层压成同一种注意力分布；
4. 将有效的视觉-语言表征继续接到动作输出，形成 VLA。

本项目保留这条主线，但补上一个原方案中必须明确的接口问题：Lavender 产生的是 `word → image region` 教师图，而不同 VLM/VLA 的内部证据并不共享同一种 attention 形式。SpikingBrain 有 full/window/GLA 的混合层级，Qwen、LLaVA 和 DeepSeek 更适合从 hidden states 做梯度归因，OpenVLA 则可以围绕 action token 或 delta EEF adapter 做动作条件归因。因此最终方法增加一个**语言条件空间归因桥接层**，把不同模型的内部证据统一投影到图像坐标，再与教师图比较。

## 3. Lavender 机制的准确解释

本地 Lavender 实现位于：

`/home/bubble/类脑计算/参考/vlm/llama_finetune/src/llama_recipes/diffag/`

关键代码包括：

- `diffag_xattn_manager.py`：聚合 cross-attention、按词切分、归一化、调整到空间网格；
- `mllama/modeling_mllama.py`：将选定层的 attention map 与外部扩散图计算 MSE，并加入总损失。

其可复用的设计点是：

- `use_layer="last"|"select"|"show_all"`；
- `num_last_layers` 和 `attn_select_layer` 控制监督层；
- 对共享词语计算教师图与模型图之间的 MSE；
- 记录各层和各词的损失，支持层消融。

这说明“按网络层级选择注意力监督”不是事后包装，而是已有代码路径。最终项目保留 Lavender 的扩散教师思想，但不要求所有学生模型都有同构 cross-attention；学生端统一为“空间归因图”。

## 4. 教师选择原则

教师不做单一来源押注，但也不做粗糙的多教师蒸馏。核心教师始终是 Lavender / Stable Diffusion 词级 cross-attention，因为第一阶段要证明 VLM 的词-区域 grounding 改善，扩散模型天然提供 `word → image region` 的空间教师信号，且不依赖机器人动作数据。World model 或 video-action model 不作为 policy teacher，不直接教动作；它们只作为 **Action-Relevance Refiner**，把扩散语义图从“语言相关区域”细化成“当前动作阶段真正相关的区域”。

教师分三档管理：

| 教师类型 | 是否进入首轮 | 用途 | 风险 |
|---|---:|---|---|
| Stable Diffusion / Lavender 词级 cross-attention | 是 | VLM grounding 和 VLA 训练期空间教师 | 只表达静态词-区域关系，不懂动作动力学 |
| VLM 自监督归因教师，例如强 Qwen/LLaVA/DeepSeek 的离线归因 | 否，作为备选 | 若扩散图对真实图像物体定位不稳定，可做 ensemble 或 sanity check | 容易把学生模型偏差当教师 |
| Action-Relevance Refiner，例如 LIBERO 轨迹规则、LingBot-VA、π0、Qwen-RobotManip | 是，但只进入 VLA 扩展阶段 | 细化 `T_sem`，判断当前阶段哪些语义区域与动作成败有关 | 若直接蒸馏动作，会变成别人的 VLA 策略蒸馏 |

首轮 VLM 只使用扩散教师，避免把“空间 grounding 是否有效”和“动作相关性细化是否有效”混在一起。VLA 阶段同时保留 B/C/D 三条路线，但优先级不同：D 是主方法，B 是扩展，C 是诊断。这样最终都服务于同一点：让 VLA 内部的动作证据从语言相关进一步变成动作相关。

### 4.1 VLA 动作相关性的三条路线

| 路线 | 名称 | 核心机制 | 定位 |
|---|---|---|---|
| B | Action-Relevance Refiner | 用 world/action 模型或规则生成 `R_act`，将 `T_sem` 细化成 `T_AR` | 可选扩展 |
| C | LIBERO 状态投影弱 refiner | 用仿真 3D 状态、EEF/target 投影、mask/depth/heightmap 构造弱动作相关区域 | 诊断和 sanity check |
| D | Structure-Native Action Attribution Consistency | 利用模型内部 language attribution、action query/action head attribution 和层级结构做一致性监督 | 主方法 |

我们更倾向 D。B/C 不删除，因为它们能提供工程可行的对照和解释工具，但不能抢占主创新。主创新不是外部预测 grasp/affordance map，而是约束 VLA 自身结构中“语言证据 → 动作证据”的空间一致性。

### 4.2 语义教师到动作相关教师图

| 组件 | 记号 | 来源 | 监督对象 | 进入阶段 |
|---|---|---|---|---|
| 语义空间教师 | `T_sem(w,x)` | Lavender / Stable Diffusion cross-attention | 词或短语对应的图像区域 | VLM 与 VLA 主线 |
| 动作相关性细化器 | `R_act(w,x,a,s)` | LIBERO demonstration 规则、LingBot-VA、π0、Qwen-RobotManip | 当前阶段 `s` 和候选动作 `a` 下，哪些语义区域真正影响成功 | VLA 扩展 |
| 动作相关教师图 | `T_AR(w,x,a,s)` | `T_sem` 与 `R_act` 的组合 | 语言条件且动作相关的空间区域 | VLA 扩展 |

最终不是把多个教师 loss 简单相加，而是将语义教师图细化：

`T_AR(w,x,a,s) = Normalize(T_sem(w,x) ⊙ R_act(w,x,a,s))`

其中 `s` 表示动作阶段，例如 approach、grasp、lift、place、release。这样 world/action 模型不直接输出训练动作，只对“语义相关区域中哪些部分与当前动作阶段相关”给出弱监督或诊断。

对应损失写成：

`L = L_task + λ_sem L_sem + λ_ar L_action_relevance`

首轮 VLM 只开启 `L_sem`。LIBERO 主实验先开启 `L_sem`，随后在扩展组中使用 `T_AR` 替代或补充 `T_sem`。如果 `L_action_relevance` 带来提升，必须报告它主要修复的是接触、可达性、遮挡、阶段切换还是长时序失败，而不是简单提升物体定位。

### 4.3 主方法：结构内生动作归因一致性

D 路线的核心链路是：

`T_sem → A_lang → A_act`

其中 `A_lang` 是模型内部语言词或短语对视觉 token 的归因图，`A_act` 是动作 query、动作 token 或 action head 对视觉 token 的归因图。Lavender 只锚定 `A_lang`，动作监督主要来自模型内部结构一致性：

`L_sem = D(A_lang, T_sem)`

`L_contain = Σ_i A_act(i) · (1 - A_lang_union(i))`

`A_lang_union` 是 source object、target object 等语言相关区域的并集。`L_contain` 的含义是：动作决策使用的视觉证据不应大量跑到语言无关区域。这是 **D0：静态 containment**，也是首版 VLA 主方法。

后期再做 **D1：阶段条件动态迁移**。D1 不让 `A_act` 直接对齐单一静态 Lavender 图，而是让动作归因在语言相关区域之间随任务阶段迁移：

`source → source + target → target`

对 pick-place 任务，D1 的软目标写成：

`A_phase(t) = α_t · A_lang(source) + (1 - α_t) · A_lang(target)`

`L_phase = D(A_act(t), A_phase(t))`

其中 `α_t` 来自成功 demonstration 的事件边界，而不是首轮依赖 world model。推荐先检测 `grasp`、`lift/move`、`near_target`、`release` 等事件，再映射成软权重：source 阶段 `α≈0.9`，mixed 阶段 `α≈0.5`，target 阶段 `α≈0.1`。如果五阶段检测不稳定，就退化成三阶段：before grasp、after grasp before near target、after near target。

D1 有三种实现层级：

| 版本 | phase 来源 | 定位 |
|---|---|---|
| D1a | LIBERO 状态规则，例如 EEF/object 距离、夹爪开合、物体高度 | 快速诊断 |
| D1b | 成功 demonstration 的事件边界自动切分 | 后期主扩展，推荐优先实现 |
| D1c | 模型内部 `mass_source=sum(A_act·A_lang(source))` 与 `mass_target=sum(A_act·A_lang(target))` 发现 phase | 更普适的研究扩展 |

这一路线不需要外部抓取区域真值，也不直接蒸馏其他 VLA 的动作。B/C 只用来检查 `A_act` 是否落在合理阶段区域，或在 D 效果不足时作为扩展 refiner。World model / VAM 只属于 B 路线的后期 `R_act` 细化器，不是 D0/D1 的默认依赖。

`A_act` 的默认实现采用 **action loss gradient×activation**，因为它最普适：连续 delta EEF、action token VLA、Qwen/LLaVA/DeepSeek 这类无显式 cross-attention 或 attention 不可比的模型都能用同一个“目标标量对视觉 token 求梯度”的接口。若模型天然提供 action query attention，则作为更可解释的结构内生版本；若模型是 OpenVLA 类 action token 输出，则用 action token log-prob gradient。

## 5. SpikingBrain 的网络结构与可对齐对象

本地实现位于：

`/home/bubble/类脑计算/参考/spikingbrain仓库/SpikingBrain-7B/hf_7B_VLM/`

### 5.1 视觉层

- `window_size=112`，大多数视觉块使用局部窗口；
- `fullatt_block_indexes=[7,15,23,31]`，这些位置执行全局视觉注意力；
- 视觉 token 会按窗口重排，因此输出图必须根据 `grid_thw` 和窗口索引还原到原始 patch 网格；
- full-attention 层有显式 softmax 权重，具备直接提取 attention 的条件；
- window/SWA 路径使用 flash attention，当前返回值不是可解释的概率矩阵，不能误认为教师注意力。

### 5.2 语言层

- SWA/局部注意力承担高效的局部建模；
- GLA 是递归/线性注意力，返回状态而不是标准 softmax 矩阵；
- GLA 不能直接与 Lavender 的二维空间图做 MSE，应使用归因代理或跳过直接监督。

### 5.3 可检验假设

**H1：** full-attention 视觉块比 window/SWA 块更适合承载跨物体、跨区域的语言条件空间信息。  
**H2：** 只对齐 `[7,15,23,31]` 或其中后段层，比“所有层统一 MSE”更稳定。  
**H3：** 只有在加入语言条件空间归因桥后，词级教师图才会先提升 VLM grounding，再对 delta EEF 成功率产生可解释增益。
**H4：** 对 GLA/SWA 直接使用伪 attention 权重会造成错误监督；proxy/skip 应优于直接 MSE。  
**H5：** 对象定位和关系判断的改进应先体现在 LIBERO 的接近、抓取、放置子任务，再体现在整体成功率。

## 6. 最终方法：语言条件空间归因对齐

### 6.1 教师信号

对指令中的对象词或对象短语 `w`，Lavender 提供：

`T(w) ∈ R^(H_s×W_s)`

它表示扩散模型认为该词对应的空间区域。教师图只用于训练期，不进入部署时的动作推理路径。

### 6.2 统一归因对象

对任意模型 `m`，我们希望得到：

`G_m(w,x,y) ∈ R^(H×W)`

其中 `w` 是词或短语，`x` 是图像，`y` 在 VLM 阶段是答案 token、类别或文本判断目标，在 VLA 阶段是 action token、动作维度或 delta EEF loss。`G_m` 表示模型为了当前预测目标，在图像哪些区域上使用了与 `w` 相关的证据。

不同模型只负责实现自己的 `G_m` 提取器：

| 模型 | 学生归因图提取方式 | 首轮用途 |
|---|---|---|
| SpikingBrain-VL | full-attention 层 `[7,15,23,31]` 的语言/动作 query 到视觉 patch；同时实现 gradient×input 对照 | 主模型，做结构感知消融 |
| Qwen2.5-VL / Qwen3-VL | answer log-prob 或目标 token score 对视觉 patch hidden states 的 gradient×input | VLM 普适性验证 |
| LLaVA-1.6 / LLaVA-OneVision | answer log-prob、目标 token score 或判别式 grounding score 对视觉 patch hidden states 的 gradient×input；OneVision 可额外记录多图/视频 token 组织 | VLM 普适性验证，检验非 Qwen 系结构 |
| DeepSeek-VL2 | 最终答案 score 对视觉 token hidden states 的 gradient×input；MoE/router 只做记录 | VLM 普适性验证 |
| OpenVLA / OpenVLA-OFT | action token log-prob 或 delta EEF adapter loss 对视觉 hidden states 的动作条件归因 | VLA 普适性验证 |
| Qwen-RobotManip | 若 checkpoint 和接口可用，按动作/行为 score 做归因；否则只做接口和论文对照 | 后置机器人 VLA 参照 |
| LingBot / π0 | 不进入首轮 VLM 实验；后续作为 Action-Relevance Refiner 候选 | 后置扩展 |

所有 `G_m` 都必须映射回原图坐标，重采样到同一尺寸，并按有效区域归一化。这样比较对象不是某一种 attention kernel，而是统一坐标系下的语言条件空间证据。

### 6.3 SpikingBrain 的语言条件 patch 归因桥

对 SpikingBrain 第 `l` 个视觉层，记 patch 表征为：

`V_l ∈ R^(N_patch×d)`

从指令/当前动作查询得到 `q_w`，构造 patch 归因分数：

`A_l(w,i)=normalize(sim(q_w, V_l[i]))`

首选实现为可微的 cosine/dot-product 归因；备选实现为对动作 logit 的 gradient×input 或 integrated-gradient 归因。桥接层必须：

1. 保留语言条件，不能只使用无条件视觉 attention；
2. 按 SpikingBrain 的窗口索引将 token 映射回原始 patch 网格；
3. 将 `A_l(w,:)` reshape 并 resize 为 `T(w)` 的空间尺寸；
4. 记录该层来自 full-attention、window/SWA 还是 GLA；
5. 在训练和离线诊断中使用同一坐标变换。

### 6.4 层级监督策略

| 层类型 | 初始策略 | 原因 |
|---|---|---|
| full-attention `[7,15,23,31]` | 直接进行 bridge 后 MSE/KL | 有全局 receptive field，具备可解释的跨区域信息 |
| window/SWA | 先做局部归因；默认不做主损失 | 局部窗口可能看不到完整对象关系，且当前返回值不是概率 attention |
| GLA | proxy 或 skip | 没有标准二维 attention map |
| 语言末端层 | 只做动作条件归因诊断 | 不把语言 token 权重误当成视觉空间图 |

初始主实验使用 `[23,31]`，随后比较 `[7,15,23,31]`、仅 `[31]`、所有可用层和随机层选择。

### 6.5 损失函数

对每个对象词和被选中的层：

`L_align = Σ_l w_l · D(norm(A_l), norm(T))`

其中 `D` 首先采用 MSE，同时记录 KL/cosine 作为诊断。VLM 阶段总损失为：

`L = L_task + λ_align L_align`

`L_task` 是 caption、VQA、referring expression 或分类任务的原始损失。VLA 阶段总损失为：

`L = L_deltaEEF + λ_align L_align + λ_smooth L_smooth`

`L_deltaEEF` 为 7 维动作回归损失；`L_smooth` 约束相邻 delta EEF 的突变。`λ_align` 通过验证集选择，不在测试集调参。

## 7. 参考模型比较与使用边界

| 模型 | 结构特点 | 学生归因接口 | 本项目用途 |
|---|---|---|---|
| Lavender | Stable Diffusion 的词级 cross-attention 空间图 | 教师 `T(w)`，不做学生 | 首轮教师 |
| SpikingBrain-VL | full-attention + window/SWA + GLA 的异构层级结构 | full-attention map + gradient×input | 主模型，验证结构感知选择 |
| Qwen2.5-VL | 动态分辨率视觉编码、多尺度视觉 token、强多模态语言模型 | answer score 对视觉 hidden states 的 gradient×input | VLM 普适性对照 |
| Qwen3-VL | Interleaved-MRoPE、DeepStack、多层视觉特征注入和长视频理解 | answer score 对视觉 hidden states 的 gradient×input | VLM 普适性对照 |
| LLaVA-1.6 / LLaVA-OneVision | CLIP/SigLIP 视觉编码器经 projector 接入 LLM；OneVision 统一图像、多图和视频输入 | answer score 或 grounding 判别 score 对视觉 hidden states 的 gradient×input | VLM 普适性对照，避免结论只覆盖 Qwen 系 |
| DeepSeek-VL2 | 高分辨率/多图输入和混合专家式语言建模路线 | answer score 对视觉 hidden states 的 gradient×input | VLM 普适性对照 |
| Qwen-RobotManip | 基于 Qwen-VL 的机器人操作 VLA，统一表示、运动和行为对齐 | 动作/行为 score 归因，接口待审计 | 机器人操作 VLA 参照 |
| OpenVLA | VLM 主干接动作 token，直接面向机器人任务 | action token 或 delta EEF adapter 的动作条件归因 | VLA 普适性关键对照 |
| π0 | 预训练视觉语言模型 + 连续动作/flow action expert | flow/action expert 归因，后置 | 连续动作建模参照 |
| LingBot-VA | 因果视频-动作世界模型，视频动态和动作在交错序列中联合建模 | 未来状态/阶段相关 `R_act`，后置 | Action-Relevance Refiner 候选 |
| LingBot-VLA 1.0/2.0 | VLA 基础模型，2.0 支持多 embodiment 的统一动作表示并使用 Qwen3-VL 依赖 | action expert 归因，后置 | 多 embodiment/action chunk 参照 |

这里不把 Qwen、LLaVA、DeepSeek 或 LingBot 直接拼进 SpikingBrain。第一阶段先在 VLM 任务上验证同一教师、不同学生归因接口都能受益；VLA 阶段再使用 SpikingBrain-VLA 和 OpenVLA/OFT 做动作闭环验证。

## 8. Next Forcing 对本项目的启发与边界

Next Forcing 是一个因果 world model 训练框架，通过链式 Multi-Chunk Prediction 同时预测多个未来视频 chunk，主要解决自回归视频模型的 myopic supervision。它不是 phase-label 数据集，也不是语言条件空间归因方法。

对本项目的正确吸收方式是升级 B 路线，而不是替换 D：

- D0 仍然负责当前时刻的 `A_act` containment；
- D1 仍然负责 `source → mixed → target` 的阶段迁移；
- Next Forcing / VAM 后期产生多时间尺度未来预测归因 `R_future^short/mid/long`；
- 未来归因只用于检查或细化 `T_sem`，构造 `T_AR = Normalize(T_sem ⊙ R_future)`。

因此项目后期可以增加一个 **Future-Consistent Language-Action Attribution Refinement** 实验，但不能把它写成首轮主创新。否则研究问题会从“语言条件空间归因对齐”变成“world model 预测训练”，同时引入视频动作数据和额外 backbone 混淆。完整分析见 `references/next_forcing/next_forcing_analysis.md`。

## 8. VLM 优先验证

先在 VLM 上证明方法优越性，避免机器人控制噪声掩盖机制判断。首轮任务应选择能提供物体或区域证据的数据：

- **首选 Flickr30k / Flickr30k Entities**：Lavender 使用 Flickr30k 作为核心训练/验证数据，并已公开 Flickr1k Stable Diffusion attention map；若使用 Entities 标注，可做 phrase-level pointing/IoU；
- **备选 RLAIF-V-83K 的 1k attention 子集**：与 Lavender 公开数据一致，但更偏 VQA/偏好数据，区域标注不如 Entities 直接；
- RefCOCO/RefCOCOg 可作为后续更标准的 referring expression 扩展，不作为首轮阻塞项。

核心比较：

1. 普通 SFT 或 LoRA；
2. Lavender 原式 attention 对齐；
3. 本项目的语言条件空间归因对齐；
4. 错词、错图、随机教师图负控。

指标包括 VQA/grounding accuracy、pointing accuracy、目标区域 IoU、归因图熵、正确教师图相对错图的增益。只有 VLM 阶段显示正确教师图带来稳定收益，才进入大规模 VLA 训练。

## 9. VLA 任务定义

输入：

- 当前或短窗口 RGB 图像；
- 语言指令；
- 可选机器人本体状态（末端位姿、夹爪状态）。

输出：

`a_t=[Δx,Δy,Δz,Δroll,Δpitch,Δyaw,g]`

- 平移和旋转均为相邻控制周期的增量；
- `g` 为夹爪开合控制量；
- 统一归一化到训练范围，部署前再反归一化；
- 首个 smoke 实验采用单步 `H=1`，模型稳定后再评估动作 chunk。

评估必须区分：

1. 视觉语言理解是否改善；
2. delta EEF 数值误差是否下降；
3. LIBERO 任务成功率是否提升；
4. 提升是否来自层级对齐，而不是额外参数或数据量。

## 10. LIBERO 首阶段范围

先覆盖能暴露物体定位和空间关系的任务：

- 首轮只做一个 pick-place / put-object-into-container 任务；
- 通过后再加入颜色/位置组合指令；
- drawer/handle 任务后置，因为它更依赖部件级可供性和接触几何。

第一阶段不承诺真机泛化。真机阶段需要新增相机外参、动作频率、限位、延迟和安全策略，不能从 LIBERO 成功率直接推导。

## 11. 预期贡献与反证条件

预期贡献：

1. 将 Lavender 的词级扩散空间先验从“attention-to-attention”推广为“teacher map-to-spatial attribution”；
2. 给出跨模型归因接口：attention、hidden-state gradient、action-conditioned attribution；
3. 先在 VLM grounding 中证明跨架构有效性，再在统一 delta EEF 接口下验证 LIBERO VLA；
4. 通过随机层、错图、打乱词图、错误教师类型和无对齐基线排除伪增益。

以下结果会否定主假设：

- bridge 后的 `L_align` 降低，但 VLM grounding 没有提升；
- VLM 阶段成立但 LIBERO 成功率没有提升或显著下降；
- 只要增加参数/训练步数就能得到同样增益；
- 随机或错配教师图与正确教师图效果相同；
- 在多个 VLM 上只有 SpikingBrain 有效，Qwen/LLaVA 均无效且无接口原因解释；
- window/SWA/GLA 直接伪 attention 比结构化 full-attention 或梯度归因更好。

## 12. 相关文献与归档索引

完整 PDF、来源、下载日期和校验信息放在 `../references/`。核心文献如下：

1. Lavender: *Learning to Attend Better with Language-Conditioned Diffusion*，`https://arxiv.org/abs/2502.06814`
2. SpikingBrain: *SpikingBrain: ...*，`https://arxiv.org/abs/2509.05276`
3. Qwen2-VL，`https://arxiv.org/abs/2409.12191`（区分于 Qwen2.5-VL）
4. Qwen2.5-VL 官方仓库与技术报告入口，`https://github.com/QwenLM/Qwen2.5-VL`、`https://arxiv.org/abs/2502.13923`
5. Qwen3-VL 技术报告，`https://arxiv.org/abs/2511.21631`；官方仓库，`https://github.com/QwenLM/Qwen3-VL`
6. LLaVA-1.6 / LLaVA-NeXT 官方仓库，`https://github.com/LLaVA-VL/LLaVA-NeXT`；LLaVA-OneVision，`https://arxiv.org/abs/2408.03326`，官方入口同 LLaVA-NeXT
7. Qwen-RobotManip，`https://arxiv.org/abs/2606.17846`；官方仓库，`https://github.com/QwenLM/Qwen-RobotManip`
8. DeepSeek-VL2，`https://arxiv.org/abs/2412.10302`
9. DeepSeek-VL2 官方仓库，`https://github.com/deepseek-ai/DeepSeek-VL2`
10. OpenVLA，`https://arxiv.org/abs/2406.09246`
11. π0，`https://arxiv.org/abs/2410.24164`
12. LingBot-VA，`https://arxiv.org/abs/2601.21998`；LingBot-VLA，`https://arxiv.org/abs/2601.18692`
13. LingBot-VLA 2.0，`https://arxiv.org/abs/2607.06403`
14. Diffusion Policy，`https://arxiv.org/abs/2303.04137`
15. LIBERO，`https://arxiv.org/abs/2306.03310`
16. GLA，`https://arxiv.org/abs/2312.06635`
17. Attention Sinks，`https://arxiv.org/abs/2309.17453`
18. LoRA，`https://arxiv.org/abs/2106.09685`

## 13. 仍需在 P0 对齐的事项

下列选项已有默认值，可以直接开始文档后的工程准备；如果后续你要改变，只需在 P0 的 commit 中修改：

| 事项 | 默认值 |
|---|---|
| 动作预测步长 | `H=1` |
| 旋转表示 | `Δroll,Δpitch,Δyaw` |
| 主对齐层 | full-attention `[23,31]` |
| 备选层组 | `[7,15,23,31]`、`[31]`、随机层 |
| 主要距离 | 归一化 MSE，KL/cosine 仅诊断 |
| 视觉输入 | LIBERO RGB，先单帧或短窗口 |
| 本体状态 | 有则拼接，无则先做视觉语言动作基线 |
| 第一组 VLM 基线 | SpikingBrain-VL + 一个 Qwen 系模型 + LLaVA-1.6/OneVision；DeepSeek-VL2 后置 |
| 第一组 VLA 基线 | 无对齐 SpikingBrain-VLA + OpenVLA/OFT attribution adapter，用于普适性验证 |
| 首轮教师 | Lavender / Stable Diffusion 词级 cross-attention |
| Action-Relevance Refiner | 纳入 VLA 扩展阶段，生成 `R_act` 并构造 `T_AR` |
| 首轮 VLM 数据 | Flickr30k/Flickr30k Entities；优先复用 Lavender Flickr1k attention maps |
| 首轮 LIBERO 任务 | pick-place / put-object-into-container |
| `A_act` 默认接口 | action loss 或 action output 对 visual tokens 的 gradient×activation |
