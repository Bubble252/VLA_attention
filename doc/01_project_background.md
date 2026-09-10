# 项目背景与研究定位：基于层级注意力对齐的 SpikingBrain-VLA

**版本**：Final design draft v1.0  
**日期**：2026-09-10  
**验证顺序**：LIBERO 仿真 → 真实机器人（后续）  
**动作接口**：delta EEF，固定为 7 维  
**主线**：保留原始 VLM 的“外部视觉/扩散注意力监督”，把监督对象改造成适合 SpikingBrain 层级结构的语言条件 patch 归因图，最终服务于 VLA 动作预测。

## 1. 已冻结的项目边界

| 项目 | 最终约束 |
|---|---|
| 研究对象 | SpikingBrain-VL 主干 + VLA 动作头 |
| 首个验证环境 | LIBERO 仿真；不先做真机 |
| 动作输出 | `a_t=[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]`，即 7 维 delta EEF |
| 视觉教师 | Lavender 离线 Stable Diffusion 词级空间注意力图 |
| 核心机制 | 依据 SpikingBrain 的 full-attention、window/SWA、GLA 等层级结构进行选择性对齐 |
| 参考模型 | Lavender、Qwen2.5-VL、Qwen3-VL、DeepSeek-VL2、OpenVLA、π0、LingBot 系列 |
| GPU | 由服务器环境管理；文档不绑定型号、数量或显存 |
| 代码仓库 | `/home/bubble/类脑计算/VLM终局` |
| 文献归档 | `/home/bubble/类脑计算/VLM终局/references` |

这里的“最终版”指研究方案和实验接口已经固定，不表示结果已经完成。第一阶段只验证仿真闭环和机制有效性，真机部署需要另行处理相机标定、控制频率、碰撞安全和执行器限幅。

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

本项目保留这条主线，但补上一个原方案中必须明确的接口问题：Lavender 产生的是 `word → image region` 教师图，而 SpikingBrain 的视觉 full/window 注意力是 `patch → patch` 自注意力。二者不能直接逐元素比较。因此最终方法增加一个**语言条件 patch 归因桥接层**，先把动作条件下的 patch 重要性转换成教师图的空间分辨率，再做层选择性对齐。

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

这说明“按网络层级选择注意力监督”不是事后包装，而是已有代码路径。最终项目把这个选择器迁移到 SpikingBrain 的结构化层级上。

## 4. SpikingBrain 的网络结构与可对齐对象

本地实现位于：

`/home/bubble/类脑计算/参考/spikingbrain仓库/SpikingBrain-7B/hf_7B_VLM/`

### 4.1 视觉层

- `window_size=112`，大多数视觉块使用局部窗口；
- `fullatt_block_indexes=[7,15,23,31]`，这些位置执行全局视觉注意力；
- 视觉 token 会按窗口重排，因此输出图必须根据 `grid_thw` 和窗口索引还原到原始 patch 网格；
- full-attention 层有显式 softmax 权重，具备直接提取 attention 的条件；
- window/SWA 路径使用 flash attention，当前返回值不是可解释的概率矩阵，不能误认为教师注意力。

### 4.2 语言层

- SWA/局部注意力承担高效的局部建模；
- GLA 是递归/线性注意力，返回状态而不是标准 softmax 矩阵；
- GLA 不能直接与 Lavender 的二维空间图做 MSE，应使用归因代理或跳过直接监督。

### 4.3 可检验假设

**H1：** full-attention 视觉块比 window/SWA 块更适合承载跨物体、跨区域的语言条件空间信息。  
**H2：** 只对齐 `[7,15,23,31]` 或其中后段层，比“所有层统一 MSE”更稳定。  
**H3：** 只有在加入语言条件 patch 归因桥后，词级教师图才会对 delta EEF 成功率产生可解释增益。  
**H4：** 对 GLA/SWA 直接使用伪 attention 权重会造成错误监督；proxy/skip 应优于直接 MSE。  
**H5：** 对象定位和关系判断的改进应先体现在 LIBERO 的接近、抓取、放置子任务，再体现在整体成功率。

## 5. 最终方法：层级注意力对齐 VLA

### 5.1 教师信号

对指令中的对象词或对象短语 `w`，Lavender 提供：

`T(w) ∈ R^(H_s×W_s)`

它表示扩散模型认为该词对应的空间区域。教师图只用于训练期，不进入部署时的动作推理路径。

### 5.2 语言条件 patch 归因桥

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

### 5.3 层级监督策略

| 层类型 | 初始策略 | 原因 |
|---|---|---|
| full-attention `[7,15,23,31]` | 直接进行 bridge 后 MSE/KL | 有全局 receptive field，具备可解释的跨区域信息 |
| window/SWA | 先做局部归因；默认不做主损失 | 局部窗口可能看不到完整对象关系，且当前返回值不是概率 attention |
| GLA | proxy 或 skip | 没有标准二维 attention map |
| 语言末端层 | 只做动作条件归因诊断 | 不把语言 token 权重误当成视觉空间图 |

初始主实验使用 `[23,31]`，随后比较 `[7,15,23,31]`、仅 `[31]`、所有可用层和随机层选择。

### 5.4 损失函数

对每个对象词和被选中的层：

`L_align = Σ_l w_l · D(norm(A_l), norm(T))`

其中 `D` 首先采用 MSE，同时记录 KL/cosine 作为诊断。VLA 总损失为：

`L = L_deltaEEF + λ_align L_align + λ_smooth L_smooth`

`L_deltaEEF` 为 7 维动作回归损失；`L_smooth` 约束相邻 delta EEF 的突变。`λ_align` 通过验证集选择，不在测试集调参。

## 6. 参考模型比较与使用边界

| 模型 | 结构特点 | 与 SpikingBrain 的共同点 | 主要差别 | 本项目用途 |
|---|---|---|---|---|
| Lavender | Stable Diffusion 的词级 cross-attention 空间图 | 都能提供空间化语言视觉关系 | 是教师图生成器，不是 VLA 主干 | 产生 `T(w)` |
| Qwen2.5-VL | 动态分辨率视觉编码、多尺度视觉 token、强多模态语言模型 | 都需要处理视觉 token 与语言条件 | 无原生 delta EEF 头，视觉注意力接口与 SpikingBrain 不同 | VLM 表征和层选择参照 |
| Qwen3-VL | 新一代 Qwen 视觉语言模型，强调长上下文和复杂视觉理解 | 可作为语言条件和视觉 token 的强基线 | 具体 checkpoint 的视觉层接口需按官方实现确认 | 高质量 VLM 对照，不作为首个动作基线 |
| DeepSeek-VL2 | 高分辨率/多图输入和混合专家式语言建模路线 | 都强调视觉 token 到语言决策的高效路由 | 专家路由、token 组织和 SpikingBrain 的窗口/GLA 不同 | 比较 token 路由和多尺度输入 |
| OpenVLA | VLM 主干接动作 token，直接面向机器人任务 | 都是视觉、语言到动作 | 常见动作表示是离散 token，不是本项目的 delta EEF 回归 | LIBERO VLA 基线 |
| π0 | 预训练视觉语言模型 + 连续动作/flow action expert | 都把高层语义接到连续机器人动作 | 动作专家和训练目标不同，工程复杂度更高 | 连续动作建模参照，后置 |
| LingBot 系列 | 视频/世界模型/VLA 路线，重视时序和动作生成 | 与最终 VLA 目标一致，都需要时序动作决策 | 重点是视频预测或大规模动作先验，不等同于本项目的离线 attention 教师 | 研究定位和时序扩展参照 |
| SpikingBrain-VL | full-attention + window/SWA + GLA 的异构层级结构 | 本项目主干 | 缺少现成 delta EEF 头和词级空间教师桥 | 主模型 |

这里不把 Qwen、DeepSeek 或 LingBot 直接拼进 SpikingBrain。第一阶段只抽取它们的结构启发和可复现实验接口；主结果必须来自同一 LIBERO 设置下的 SpikingBrain-VLA 与明确的对照组。

## 7. VLA 任务定义

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

## 8. LIBERO 首阶段范围

先覆盖能暴露物体定位和空间关系的任务：

- 物体抓取与放置；
- 抽屉/柜门开合；
- 物体移动到指定容器；
- 多物体、颜色/位置组合指令。

第一阶段不承诺真机泛化。真机阶段需要新增相机外参、动作频率、限位、延迟和安全策略，不能从 LIBERO 成功率直接推导。

## 9. 预期贡献与反证条件

预期贡献：

1. 将 Lavender 的词级扩散空间先验接入 SpikingBrain 的异构层级视觉结构；
2. 给出“直接对齐、归因代理、跳过监督”的层级化设计；
3. 在统一 delta EEF 接口下验证对 LIBERO VLA 的影响；
4. 通过随机层、错图、打乱词图和无对齐基线排除伪增益。

以下结果会否定主假设：

- bridge 后的 `L_align` 降低，但 LIBERO 成功率没有提升或显著下降；
- 只要增加参数/训练步数就能得到同样增益；
- 随机或错配教师图与正确教师图效果相同；
- window/SWA/GLA 直接伪 attention 比结构化 full-attention 对齐更好。

## 10. 相关文献与归档索引

完整 PDF、来源、下载日期和校验信息放在 `../references/`。核心文献如下：

1. Lavender: *Learning to Attend Better with Language-Conditioned Diffusion*，`https://arxiv.org/abs/2502.06814`
2. SpikingBrain: *SpikingBrain: ...*，`https://arxiv.org/abs/2509.05276`
3. Qwen2.5-VL Technical Report，`https://arxiv.org/abs/2409.12191`
4. Qwen 官方仓库，`https://github.com/QwenLM/Qwen2.5-VL`
5. Qwen3-VL 官方仓库，`https://github.com/QwenLM/Qwen3-VL`
6. DeepSeek-VL2，`https://arxiv.org/abs/2412.10302`
7. DeepSeek-VL2 官方仓库，`https://github.com/deepseek-ai/DeepSeek-VL2`
8. OpenVLA，`https://arxiv.org/abs/2406.09246`
9. π0，`https://arxiv.org/abs/2410.24164`
10. LingBot/VLA 相关工作，见 `references/README.md`，下载前核对论文版本和标题
11. Diffusion Policy，`https://arxiv.org/abs/2303.04137`
12. LIBERO，`https://arxiv.org/abs/2306.03310`
13. GLA，`https://arxiv.org/abs/2312.06635`
14. Attention Sinks，`https://arxiv.org/abs/2309.17453`
15. LoRA，`https://arxiv.org/abs/2106.09685`

## 11. 仍需在 P0 对齐的事项

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
| 第一组 VLA 基线 | 无对齐 SpikingBrain-VLA、OpenVLA 接口对照 |

