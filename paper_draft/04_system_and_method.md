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

\[
L_{sem}=\sum_{w,l}m_w\,D(\operatorname{Norm}(A_l(w,x)),\operatorname{Norm}(T_{sem}(w,x))).
\]

首选 `D=MSE`，同时记录 KL 和 cosine。

## 4.3 SpikingBrain 的类脑结构选择

SpikingBrain 同时包含局部窗口、全局 attention 和 GLA。本文不把所有层视为同一种 attention：

- full-attention `[7,15,23,31]`：候选主监督层，优先 `[23,31]`；
- window/SWA：局部诊断和 gradient proxy，不直接把返回值当概率图；
- GLA：只用 proxy/skip，不 reshape 状态矩阵；
- 脉冲活动：记录 firing/activity sparsity、归因熵和层间稳定性，作为类脑分析指标。

## 4.4 D0：Structure-Native Action Attribution Consistency

动作归因默认使用动作损失或动作输出对视觉 token 的 gradient×activation：

\[
A_{act}(i)=\left|\frac{\partial L_{action}}{\partial h_i}\odot h_i\right|.
\]

若为离散 action token，使用 token log-prob gradient；若存在 action query attention，作为结构内生对照。

定义语言相关区域并集：

\[
A_{lang}^{\cup}(i)=\max_{w\in\{source,target,tool\}}A_{lang}(w,i).
\]

D0 containment：

\[
L_{contain}=\frac{1}{N}\sum_i A_{act}(i)(1-A_{lang}^{\cup}(i)).
\]

总损失：

\[
L=L_{task}+\lambda_{sem}L_{sem}+\lambda_{contain}L_{contain}+\lambda_{smooth}L_{smooth}.
\]

首版 `H=1`，不引入 action chunk；`λ` 由验证集选择。

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
