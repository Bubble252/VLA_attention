# 4. System 与 Method 设计

## 4.1 输入与符号

当前观测为 $x_t$，语言指令为 $w$，第 $i$ 个视觉 token 为 $h_i$。动作输出统一为

$$
a_t = [\Delta x,\Delta y,\Delta z,\Delta r_x,\Delta r_y,\Delta r_z,g]
\in \mathbb{R}^{7}.
$$

Lavender 提供语义教师图

$$
T_{\mathrm{sem}}(w,x)\in\mathbb{R}^{H\times W},
$$

所有学生归因图最终恢复到输入图像坐标。

## 4.2 Spatial Attribution Bridge

### Token 到 patch 网格

- 读取 `grid_thw`、patch 数和视觉 token 顺序；
- 对 SpikingBrain 逆转 window/SWA reorder；
- 记录 full-attention、window/SWA、GLA layer type；
- 将每层 token 归因 reshape 为 patch 图并 resize 到统一 `H×W`；
- 对 padding、无效词和不存在对象做 mask。

### 语言归因 `A_lang`

根据模型能力选择：

- full-attention：语言 query 到视觉 patch 的显式权重；
- 通用接口：答案 log-prob/grounding score 对视觉 hidden state 的 gradient×activation；
- 相似度 proxy：`|cos(q_w,h_i)|`，只作诊断。

归一化后得到 $A_{\mathrm{lang}}(w,x)$。训练损失为

$$
L_{\mathrm{sem}}=\sum_{w,l}m_w\,D\!\left(\operatorname{Norm}(A_l(w,x)),\operatorname{Norm}(T_{\mathrm{sem}}(w,x))\right).
$$

首选 `D=MSE`，同时记录 KL 和 cosine。

## 4.3 后期可选：SpikingBrain 的类脑结构选择

本节不参与首轮主方法验收。主线模型按实际视觉/语言/动作接口选择层，不搬用下面的层号；SpikingBrain 有空再做。

SpikingBrain 同时包含局部窗口、全局 attention 和 GLA。本文不把所有层视为同一种 attention：

- full-attention `[7,15,23,31]`：候选主监督层，优先 `[23,31]`；
- window/SWA：局部诊断和 gradient proxy，不直接把返回值当概率图；
- GLA：只用 proxy/skip，不 reshape 状态矩阵；
- 脉冲活动：记录 firing/activity sparsity、归因熵和层间稳定性，作为类脑分析指标。

## 4.4 D0：Structure-Native Action Attribution Consistency

动作归因默认使用动作损失或动作输出对视觉 token 的 gradient×activation：

$$
A_{\mathrm{act}}(i)=\left|\frac{\partial L_{\mathrm{action}}}{\partial h_i}\odot h_i\right|.
$$

若为离散 action token，使用 token log-prob gradient；若存在 action query attention，作为结构内生对照。

定义语言相关区域并集：

$$
A_{\mathrm{lang}}^{\cup}(i)=\max_{w\in\{\mathrm{source},\mathrm{target},\mathrm{tool}\}}A_{\mathrm{lang}}(w,i).
$$

D0 containment：

$$
L_{\mathrm{contain}}=\frac{1}{N}\sum_i A_{\mathrm{act}}(i)\left(1-A_{\mathrm{lang}}^{\cup}(i)\right).
$$

### D0-CF：动作条件反事实证据细化（DVD 启发，候选扩展）

DVD 的可迁移启发是：视觉证据应由下游任务决定，并通过教师信号教给接收原始观测的学生。本文不照搬 Grounding DINO 框增强图像；候选做法是用冻结的 VLA-BC 基线和专家动作构造动作条件的反事实证据图，再监督学生的内部动作归因。这样可以补足纯 `A_lang` 并集的限制：夹爪、接触边缘或目标开口可能是动作必需证据，却不是指令名词的显著区域。

对冻结参考策略 $\pi_0$、当前观测 $x_t$、语言指令 $w$、本体状态 $q_t$、专家动作 $a_t^*$ 和 patch 干预算子 $M_i$，定义移除第 $i$ 个视觉 patch 后专家动作损失的增量。遮挡只作用于图像 patch；计算前后固定同一指令、本体状态和目标动作：

$$
\Delta_i^{\mathrm{cf}}(x_t,w,q_t,a_t^*) =
\max\!\left(0,\,
 L_{\mathrm{act}}\!\left(\pi_0(M_i(x_t),w,q_t),a_t^*\right)
-L_{\mathrm{act}}\!\left(\pi_0(x_t,w,q_t),a_t^*\right)
\right),
$$

$$
R_{\mathrm{act}}^{\mathrm{cf}}(i)=
\operatorname{Norm}_i\!\left(\Delta_i^{\mathrm{cf}}\right).
$$

`R_act^cf` 由冻结的 $\pi_0$ 离线生成并缓存，训练时作为 stop-gradient 教师图；全零或有效质量过低的图跳过该样本，避免归一化噪声。学生图仍由当前 VLA 的 action loss gradient×activation 计算。候选支持区域为语言实体证据与动作反事实证据的软并集：

$$
U_t(i)=1-\left(1-A_{\mathrm{lang}}^{\cup}(i)\right)
\left(1-R_{\mathrm{act}}^{\mathrm{cf}}(i)\right),
\qquad
L_{\mathrm{support}}=
\frac{1}{N}\sum_i A_{\mathrm{act}}(i)\left(1-U_t(i)\right).
$$

这把 DVD 的“任务相关视觉线索”接入本文 `T_sem → A_lang → A_act` 链路：动作相关的非名词线索可以进入支持区域，但须有反事实动作证据。它不使用 GT box 坐标，也不声称反事实图是真实因果区域。框/分割标注如用于离线评测，只作为评测真值，不参与教师图构造。

**进入训练的 gate：** 先在冻结 B0 策略上离线生成小样本反事实图，验证正确专家动作、错动作、随机 patch、背景 patch 和不同 mask 算子之间有可区分的响应；并检查 action-relevant occlusion 是否比背景 occlusion 更能恶化专家动作预测。若图对 mask 方式高度敏感、无法区分正确/错动作，或主要突出背景捷径，则停止该路线，只保留 D0/D1 与干预评测。通过后再比较 B4（D0）与 B5（D0-CF）；B5 是候选扩展，不替换当前 D0 主线。

总损失：

$$
\begin{aligned}
L={}&L_{\mathrm{task}}+\lambda_{\mathrm{sem}}L_{\mathrm{sem}}\\
&+\lambda_{\mathrm{contain}}L_{\mathrm{contain}}+\lambda_{\mathrm{smooth}}L_{\mathrm{smooth}}.
\end{aligned}
$$

启用 D0-CF 时，以 `L_support` 替代纯语言区域 containment，或将两者作为消融比较；不能无条件叠加强惩罚。首版 `H=1`，不引入 action chunk；`λ` 由验证集选择。

## 4.5 D1：阶段条件动态归因

从成功 demonstration 推断事件：`t_grasp`、`t_lift`、`t_near_target`、`t_release`。首版使用三阶段：

```text
source-dominant → mixed → target-dominant
```

软目标：

$$
A_{\mathrm{phase}}(t)=\alpha_t A_{\mathrm{lang}}(source)
 +(1-\alpha_t)A_{\mathrm{lang}}(target),
$$

$$
L_{\mathrm{phase}}=D\big(A_{\mathrm{act}}(t),A_{\mathrm{phase}}(t)\big).
$$

五阶段仅在事件检测稳定后启用。反向 phase、随机 phase、固定时间百分比和错 source/target 是必需负控。

## 4.6 B 路线：WAM/VAM 的未来归因扩展

WAM/VAM 给定 `x_t,w,a_{t:t+H}` 预测未来视频/latent `F^k_t`。先定义未来目标标量 `S^k`（例如目标接近度或接触进展），再得到：

$$
r^k_{\mathrm{grad}}(i)=
\left|h_i\odot\frac{\partial S^k}{\partial h_i}\right|,
$$

$$
r^k_{\mathrm{occ}}(i)=
\left|S^k(x_t)-S^k(\operatorname{mask}_i(x_t))\right|,
$$

$$
R_{\mathrm{future}}=\operatorname{Norm}\left(\sum_k\beta_k r^k\right),
\qquad
T_{\mathrm{AR}}=\operatorname{Norm}
\left(T_{\mathrm{sem}}\odot R_{\mathrm{future}}\right).
$$

B 只在 D0/D1 成立后进行离线分析，加入 wrong-action、wrong-future、random-horizon 和低置信度回退。若 WAM/VAM 不接受动作条件，不称为动作后果评估器。

## 4.7 Sink-free 与证据等级

- raw attention：结构线索；
- gradient×activation：输出条件代理；
- patch occlusion：外部干预证据。

检查 BOS/CLS/register/边缘 patch sink，比较 raw 与 sink-free，但不默认将 sink-free 图用于训练。任何“模型使用了碗”的结论必须有 occlusion 或语言/state 反事实支撑。

## 4.8 借鉴机制与方法边界

本文借鉴已有工作时，区分“增强核心归因链路”和“改变问题定义的独立方法”。

### 4.8.1 BlindVLA-style representation trust region

BlindVLA 的中层 patch-feature retention 作为 `B1` baseline。通过记录 action fine-tuning 前后的视觉表征漂移、feature collapse、attention sink mass 和 VL-Think 保留结果，判断 `L_sem` 或 `L_contain` 的收益是否只是因为视觉表征没有遗忘。

在 G1 通过后，可做一个独立的漂移感知消融：

$$
L_{\mathrm{ret}}=\sum_l w_l\,\operatorname{mean}_i
\left[1-\cos(P_l(h^{\mathrm{student}}_{l,i}),z^{\mathrm{teacher}}_{l,i})\right],
$$

其中 $w_l$ 只由 validation calibration 上的 representation drift 冻结得到。该变体不是主方法；它用于检验固定 retention 与按层 trust region 的差别。

### 4.8.2 Next Forcing-inspired multi-layer attribution

Next Forcing 的 multi-layer feature fusion 可迁移成归因层聚合，而不引入完整 MCP：

$$
A_{\mathrm{lang}}=\operatorname{Norm}\left(\sum_l\alpha_l A_{\mathrm{lang}}^l\right),
\qquad
A_{\mathrm{act}}=\operatorname{Norm}\left(\sum_l\beta_l A_{\mathrm{act}}^l\right).
$$

首版使用固定早/中/后层平均；只有在 validation 上完成预注册校准后才允许固定非均匀权重。该机制是 layer-sensitivity 稳定性消融，不改变 `T_sem\rightarrow A_lang\rightarrow A_act` 的研究问题。

### 4.8.3 LIT-inspired action-prior and counterfactual controls

LIT 的 paired budget、task-preserving visual intervention 和 goal-changing language intervention 进入评测协议。它们分别验证：

- 无关视觉变化是否改变动作；
- 语言目标变化时动作是否正确重定向；
- 方法增益是否只来自 success 或视觉图可视化。

LIT 的 image-free action prior 作为独立 `E-LIT-prior` 对照：

```text
Stage A: language + proprioception + terminal goal -> action prior
Stage B: visual conditioning + our attribution losses
```

它不并入 B0–B4，因为 Stage A 改变 action expert 初始化和总训练预算。完整 latent bottleneck 也不作为本文主方法，否则无法区分 LIT 的通道约束和本文的空间归因约束。

### 4.8.4 Future-consistent extension

WAM/VAM 阶段先冻结策略，计算：

$$
R_{\mathrm{future}}^{\mathrm{short}},
R_{\mathrm{future}}^{\mathrm{mid}},
R_{\mathrm{future}}^{\mathrm{long}},
$$

并验证它们与 $A_{\mathrm{act}}$ 的相关性和未来 patch occlusion 响应。只有 wrong-future、shuffled-future 和 wrong-action 负控通过后，才考虑：

$$
L_{\mathrm{future}}=\sum_k\gamma_k
D(A_{\mathrm{act}},R_{\mathrm{future}}^k).
$$

因此，Next Forcing 式 multi-horizon `R_future` 归因和长时 WAM/VAM 仍是后期扩展，不进入历史 OpenVLA B0--B4。后文 LC-PEA 使用的 short-horizon future latent 是一个独立、最小化的 predictive auxiliary；它不继承 Next Forcing 的完整多 chunk 生成机制，也不等同于长时 `R_future` teacher。

### 4.8.5 LC-PEA：语言条件预测证据对齐（新主方法候选）

静态 `A_lang→A_act` containment 容易把语言实体区域误当成全部动作证据，也不能表达物体、夹爪和目标容器随动作变化的动态关系。为使 VLM 阶段的空间证据直接服务于 manipulation 能力，本文将主方法候选更新为 **LC-PEA（Language-Conditioned Predictive Evidence Alignment）**。它不要求语言热图直接规定动作，而是让语言指定的视觉证据负责预测短时未来的任务相关 latent，再由共享表征支持原生 action objective。

令在线视觉编码器输出当前多视角 token $H_t=\{h_{t,i}\}_{i=1}^{N}$，语言条件证据为 $A_{\mathrm{lang},t}(i)$。为保留非语言实体但对动作必需的上下文，定义带均匀泄漏的软权重：

$$
q_{t,i}=\operatorname{Norm}_i\left[(1-\eta)+\eta A_{\mathrm{lang},t}(i)\right],
\qquad 0\leq\eta\leq1,
$$

$$
z_t^{\mathrm{ev}}=\sum_i q_{t,i}h_{t,i}.
$$

预测器读取目标相关证据、完整视觉摘要、本体状态和当前 policy 产生的 action-plan latent $u_t$，预测未来 $k$ 步的任务相关表征：

$$
\hat z_{t+k}^{\mathrm{ev}}
=P_\theta\!\left(z_t^{\mathrm{ev}},\operatorname{Pool}(H_t),s_t,u_t\right).
$$

$u_t$ 优先取 action head 输出前的原生计划表征；若骨干没有可比 latent，则使用当前 policy 的预测 action chunk，经轻量 adapter 编码。主方法不把专家未来动作直接输入 predictor，以免仅凭 action label 重建未来而绕过视觉证据。teacher-forced expert action 只作为诊断上界，并与 action-drop、action-shuffle 和无 action 条件版本共同报告。

冻结或 EMA target encoder 从真实未来观测获得 stop-gradient 目标。未来证据权重使用同一指令重新计算，防止学生只预测全局背景变化：

$$
z_{t+k}^{*}
=\operatorname{sg}\!\left[
\sum_j q_{t+k,j}^{*}E_{\mathrm{target}}(x_{t+k})_j
\right].
$$

首版预测损失采用归一化 cosine/MSE 二选一，并在 validation 上冻结，不同时搜索多种复杂目标：

$$
L_{\mathrm{pred}}
=\frac{1}{|K|}\sum_{k\in K}
D\!\left(\operatorname{Norm}(\hat z_{t+k}^{\mathrm{ev}}),
          \operatorname{Norm}(z_{t+k}^{*})\right),
$$

$$
L_{\mathrm{LC\text{-}PEA}}
=L_{\mathrm{action}}+\lambda_{\mathrm{pred}}L_{\mathrm{pred}}.
$$

首版 $K$ 只取与原生 action chunk 对齐的 short-horizon offsets；完整 WAM、像素视频生成和长时规划不进入主损失。policy action head 始终读取完整视觉和 proprioception，$q_t$ 只作用于辅助预测分支。这样可以复用 VLM 的 `T_sem→A_lang` 结果，同时避免错误语言图硬遮掉夹爪、接触边缘、障碍和容器开口。

最关键的 matched controls 是：普通 JEPA（去掉语言权重）、静态语义对齐（去掉未来预测）、BlindVLA retention、wrong-word、wrong-image、future shuffle、uniform/full-image 权重和面积匹配随机图。只有 LC-PEA 相对普通 JEPA 与 retention 均有动作/rollout 增量，而且正确证据优于负控，才支持“语言条件动态表征”这一机制。`I_bind`、patch intervention 和 source/target phase 曲线用于解释涨点是否来自正确目标绑定，不能替代 success。

### 4.8.6 方法版本与历史边界

```text
Static-Ours = L_sem + D0 A_act containment（历史候选/机制 baseline）
LC-PEA = L_action + language-conditioned short-horizon latent prediction
LC-PEA-R = LC-PEA + BlindVLA-style L_ret
```

LC-PEA 是新的性能主方法候选，须先完成独立新颖性审计和最小实验 gate；写入计划不等于已经证明创新或涨点。`Static-Ours` 保留为直接归因约束 baseline，`LC-PEA-R` 用于检验动态语义预测是否在通用视觉 retention 上仍有增量。长时 WAM/VAM、闭环 latent planning 和 action-conditioned video generation 继续作为后期扩展，不与首版 LC-PEA 混合。
