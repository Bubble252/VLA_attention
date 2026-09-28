# P6：VLA B0–B4 protocol

> 2026-09-28 执行状态：并行非 GPU 准备已获授权；cache 验收与资源释放后可执行 B0 工程验证。见 [非 GPU 队列](non_gpu_preparation.md)、[官方源码审计](oft_libero_static_audit.md)、[三类 cache 与 Plus 边界](cache_preparation.md)。静态证据确认 OFT 原生 L1、8×7 chunk；后文旧 MSE/H=1 示例不覆盖此真实接口。协议/adapter 骨架不等于真实 B0 已可运行。

状态：**工程准备中，尚未运行首个 B0**。本协议服务于当前 VLM/VLA goal 的第 15 步：在 VLM 主线仍收集正式证据的同时，准备 OpenVLA/OFT + LIBERO 的可恢复 action smoke。最终验收明确包含 GPU 阶段：真实 OFT P1、20–50 step B0、checkpoint restore、held-out action prediction 和少量官方 rollout；CPU 静态审计、缓存生成和脚本测试不能替代这些 GPU 证据。任何 smoke、接口审计或离线误差都不等于 VLA 方法有效。

## 1. 研究问题与首轮边界

### 数据集角色边界

本项目不是所有实验都使用 COCO，也不是所有实验都复刻 Lavender 的数据。不同数据集服务不同问题，不能合并成一个训练集或平均分：

| 数据轨道 | 数据集 | 用途 | 是否进入首轮 VLM 训练 |
|---|---|---|---|
| Lavender-style caption 训练 | Flickr30k 原始 image-caption pairs | V1–V4 的 caption SFT；V3/V4 在同一 caption token 上施加语义归因约束 | 是 |
| 词—区域空间验证 | Flickr30k Entities | teacher calibration、pointing/mass-in-box/IoU、wrong-word/wrong-image/random controls | 否；box 只用于 calibration/test |
| 指代表达迁移 | RefCOCOg | 检验更复杂 referring expression 的外部泛化 | 否，仅评测 |
| 表征诊断 | COCO 固定类别子集 | 复刻 BlindVLA 的 t-SNE、linear probe、kNN 和 feature drift 诊断 | 否，仅可视化/诊断 |
| 下游能力评测 | COCO Captions、VQAv2、TextVQA、POPE、MME、WorldMedQA-V | 检验空间约束是否影响 caption/VQA/OCR/幻觉/综合能力 | 否，仅官方评测 |
| VLA 机制与闭环 | LIBERO、后续 LIBERO-Plus/LIBERO-PRO | action prediction、归因干预和官方 rollout success | 不属于 VLM 训练 |
| 真实视觉离线泛化 | DROID | 7D delta EEF offline action prediction、camera/object/scene/operator 分组 | 不属于首轮 VLM 训练 |

VLM 主训练采用的是 **Lavender 风格的 image-caption SFT 形式**，但主空间验证使用 Flickr30k Entities；COCO 只承担 BlindVLA-style 表征诊断和部分能力评测。VLA 阶段转向 LIBERO/DROID，不能用 COCO t-SNE 替代动作或闭环证据。

P6 只回答一个可证伪问题：在同一个 VLA 骨干、同一动作头、同一数据和同一训练预算下，语言条件空间归因是否能改善动作预测和 LIBERO 闭环，并且增益是否超出 BlindVLA-style visual feature retention。

首轮冻结的工程口径如下；带 `TO_FREEZE` 的字段必须由真实接口审计或数据扫描产生，不能手填成看似合理的值。

| 项目 | 首轮口径 | 证据要求 |
|---|---|---|
| VLA 骨干 | 首轮冻结为 OpenVLA/OFT，记录精确 checkpoint 和代码 commit；π0.5、MolmoAct2、LingBot-VLA 保留到后续跨结构轮次 | `p1_interface.json`、来源和许可 |
| BlindVLA retention teacher | VLA B1/B3 首轮使用 BlindVLA 官方默认 `C-RADIOv3-L`；DINOv2 作为 teacher swap，不能把 DINO 结果称为官方 BlindVLA 复现 | teacher revision、预处理、patch grid、projector、cache SHA |
| 主 benchmark | 标准 LIBERO，一个 pick-place/put-object-into-container 任务 | 环境列表、任务 ID、版本、episode manifest SHA |
| OOD/泛化 | LIBERO-Plus、LIBERO-PRO、SimplerEnv、DROID offline 后置 | 不阻塞首个 B0；分别记录数据版本和 split |
| 观察 | 官方 OpenVLA/OFT LIBERO 配置的 RGB、语言指令、原生 proprioception 和 wrist-camera 输入；不得静默删掉模型必需输入 | processor/action adapter audit |
| 动作 | 统一报告为 7D delta EEF：`[dx, dy, dz, droll, dpitch, dyaw, dgrip]` | 必须记录原生动作、单位、旋转表示、夹爪符号、限幅和归一化 |
| action chunk | 正式 OpenVLA/OFT LIBERO B0–B4 使用 checkpoint/evaluator 的原生 horizon（当前草案 `H=8`）；`H=1` 仅作 P1/B0 smoke、首步归因和 restore 诊断 | action tensor shape、官方配置和 rollout 执行证据 |
| 首轮训练 | 先 offline action prediction，再做短闭环 rollout | checkpoint restore 和离线评测先通过 |
| seeds | B0–B4 共享预注册 seeds `17/29/41`；B0 smoke 可先用 17 | 训练配置和日志 |
| 输出位置 | `experiment_workspace/results/P6_vla_b0_b4/`，与 VLM/F0 分开 | 每组不可覆盖、带 config/manifest/Git SHA |

`7D delta EEF` 是统一报告接口，不代表 OpenVLA/OFT 原生一定就是该表示。若原生 action 不是 7D delta EEF，adapter 必须给出可逆转换和数值单元；无法证明转换等价时，保留原生结果并停止跨动作头比较。

动作单位和归一化冻结原则：训练与 rollout 保留 OpenVLA/OFT 原生 action 表示；统一报告时平移优先转换为米、旋转优先报告为 axis-angle/弧度、夹爪保留原生符号。限幅沿用官方 action head，归一化沿用 checkpoint 或数据集提供的 `unnorm_key`/action statistics，禁止手工 min-max。P1 必须用真实 LIBERO demonstration 完成归一化→反归一化→再归一化的 round-trip，并验证单步平移、旋转和夹爪方向；通过前不填写具体单位、符号或范围。

### 1.1 StarVLA 的依赖边界

当前 P6 主线**不要求 StarVLA 作为 VLA 模型骨干**。首轮模型是官方 OpenVLA/OFT checkpoint，训练、归因和评测通过本项目自己的 `openvla_oft` adapter 接入；StarVLA 不进入 B0–B4 的模型定义，也不作为论文主线的隐含依赖。

StarVLA 源码已作为可选参考固定归档于 `references/repos/starvla`（commit `4507931a625c844404c4a76128fb116536d8ca7c`）。它支持 Qwen-VL 与 OFT、FAST、PI/flow-matching、GR00T 等动作头组合，适合后续建立独立的“VLM + action head”扩展组；该扩展不与 P6 主表混合。

### 1.2 后续 VLA 轮次

首轮 B0–B4 只使用 OpenVLA/OFT，确保动作头、训练预算和 LIBERO evaluator 固定后，先回答主方法是否有效。其他模型保留在后续轮次：

- **第二轮：π0.5**，优先验证连续 flow/diffusion action interface 下的可迁移性；
- **第三轮：MolmoAct2**，验证不同视觉语言和动作头组合；
- **扩展轮：LingBot-VLA**，验证视频/多 embodiment 条件下的迁移；
- **独立对照：StarVLA**，仅在研究 VLM-to-VLA 或跨 action-head 适配时使用，不并入 OpenVLA/OFT 主表。

后续模型必须各自完成 checkpoint、processor、native action、visual token、归一化、chunk 和 evaluator 的 P1 审计；跨模型只比较相对各自 matched baseline 的增量，不直接比较原生 success 绝对值。

**模型选择规则：已有原生 VLA 形态时，不再套用 StarVLA。** 如果某个模型系列已经公开了可复现的 VLA 版本、原生 action head 和对应训练/评测接口，应优先使用该系列的原生 VLA 形态。例如 OpenVLA/OFT、π0/π0.5、MolmoAct2、LingBot-VLA 等，应直接审计其官方 VLA 接口；不能先取其 VLM 再额外接一个 StarVLA action head，并把结果当成该系列的标准 VLA。StarVLA 只用于两种情况：一是该 VLM 系列没有可用的原生 VLA 版本，需要构建独立的 VLM-to-VLA 扩展；二是专门研究相同 VLM 在不同 action head 下的可迁移性。此类结果必须单独编号、单独报告，不能与原生 VLA 结果混在主表中。

需要把三个层次分开：

| 层次 | P6 首轮选择 | StarVLA 的位置 |
|---|---|---|
| 模型/动作头 | OpenVLA/OFT 原生 policy/action head | 不是首轮骨干 |
| 实验 adapter/损失 | 本项目 `adapters/openvla_oft.py`、B0–B4 trainer 和归因损失 | 不从 StarVLA 脚本直接 import |
| 运行环境/数据工具 | 官方 LIBERO、项目独立环境和可复核 evaluator | 只有在兼容性审计通过时，可借用 StarVLA 环境中的 CUDA/数据工具 |

服务器上的 `/root/code/Starvla`、`/root/starvla_cu124` 只能作为已有环境或参考代码进行**只读审计**。其中的 SpikingBrain、Robotwin 或旧项目训练脚本不得直接运行、修改或复制到 P6；`server/01_vla_attention_training.md` 中已经记录这一边界。若以后要研究 StarVLA 自己的模型，必须另建一个 adapter 和独立实验组，重新完成 checkpoint、视觉 token、原生动作和 evaluator 的 P1 审计，不能把它与 OpenVLA/OFT 的 B0–B4 结果混在一张主表中。

只有在以下条件同时满足时，StarVLA 才可被临时用作运行时：它能加载已冻结的 OpenVLA/OFT checkpoint，processor、LIBERO 观测、动作单位和 checkpoint restore 与 P1 结果逐项一致，并且版本和 commit 被写入配置。否则应在 VEPFS 建立本项目独立环境。该边界保证即使 StarVLA 不可用，P6 的模型定义和论文结论仍然成立。

## 2. B0–B4 主矩阵

所有组必须使用同一个模型初始化、LIBERO episode manifest、语言模板、图像预处理、动作归一化、H、optimizer、训练 steps、seed 和 evaluator。只有下表中标出的损失或 teacher 开关可以变化。

| 组 | 训练内容 | 目的 | 关键比较 |
|---|---|---|---|
| B0 | 原生 BC/SFT，只有 action loss | 建立动作能力和 action head 基线 | 所有后续组对 B0 |
| B1 | B0 + BlindVLA-style frozen visual feature retention | 排除“收益只是视觉表征保持” | B1−B0 |
| B2 | B0 + `T_sem → A_lang` | 测语言条件空间教师对动作学习的独立作用 | B2−B0 |
| B3 | B1 + `T_sem → A_lang` | 测 retention 和语义归因是否互补 | B3−B1、B3−B2 |
| B4 | B3 + D0 soft action-evidence containment | 测动作归因是否进一步留在语言允许区域 | B4−B3 |

其中：

- `A_lang` 由与 VLM 主线相同的冻结 `T_sem` 和学生语言短语归因生成；学生图使用 phrase score 对 visual tokens 的 gradient×activation，不能把某个模型的 raw attention 当成通用教师。
- `A_act` 首版定义为固定动作目标的 action loss 对 visual tokens 的 gradient×activation。若模型天然有 action query attention，只额外记录作解释分支，不把它作为跨模型必需接口。
- 层级分工固定为：B1 retention 参考 BlindVLA 使用中间视觉层作为通用视觉表征保持目标；B2/B3 的 `A_lang` 与 B4 的 `A_act` 使用 action-head 输入前的 action-relevant visual tokens。其他层只做固定 checkpoint、单 seed 的 attribution sensitivity，不重复完整 B0–B4。
- D0 使用 stop-gradient 的语言支持图 `M_lang`，对动作图在语言支持区域外的质量施加软约束；首版不加入 `L_phase`、`R_future`、world-model refiner 或 action expert 蒸馏。动作图约束的候选形式记录为：

  `L_contain = mean_t sum_i A_act(t,i) * (1 - M_lang(i))`。

  正式实现须同时报告 `A_act` 的归一化、阈值/软 mask、温度和是否对 EEF/目标区域使用额外允许项；没有这些 metadata 的结果不能进入主表。

## 3. 训练和评测顺序

### 3.0 VLM→VLA transition：SAEB-T0（后置于 B0，但优先于 dynamic query）

ALN-P3 更适合被借鉴为 VLM→VLA 的过渡范式：冻结的 VLM 语义分支在训练期提供 `A_lang`/semantic latent，原生 VLA 保留自己的 action head 和 action loss，通过轻量阶段桥接学习语言证据到动作证据的对应关系。它不能替代 B0–B4，也不能把 ALN-P3 的 CLIP/BEV 对齐目标直接搬到 LIBERO。

#### ALN-P3 是否要求同一架构

不要求。ALN-P3 的原始设定本身就是异构的：快速 P3 感知/预测/规划系统与慢速 MLLM 并非同一个网络；作者通过 MLP、Holistic Token Mixer、learnable prompt/attention pooling 和 CLIP 语义空间建立桥接。因此可迁移的是“异构系统通过语义桥对齐”的范式，而不是某个 hidden-state 层的逐元素匹配。

对我们的跨架构 VLM→VLA，必须区分三种情况：

| 情况 | 是否可行 | 需要什么 |
|---|---|---|
| 同一 VLM backbone 加原生 action head | 最容易 | 可共享 token provenance，直接做 evidence/latent bridge |
| 不同 VLM 与 VLA backbone，例如冻结 Qwen/LLaVA 接 OpenVLA | 可行，推荐先做 | 不对齐 raw hidden state；分别生成 `A_lang`/`A_act`，再映射到公共图像网格或低维 stage latent |
| 没有可比视觉 token、grid 或输出语义的模型 | 不宜作为首轮 | 需要额外蒸馏数据、投影器和行为验证，不能仅凭 cosine loss 宣称迁移 |

因此 SAEB-T0 不应写成“VLM hidden state 对齐到 VLA hidden state”，而应写成：

```text
frozen VLM → phrase evidence A_lang / pooled semantic latent z_lang
native VLA → action evidence A_act(t) / action-stage latent z_act(t)
shared bridge → normalized image evidence space + optional stage latent space
```

空间证据桥比 latent 桥更具跨架构普适性，因为它只要求 token provenance 能回到图像坐标。latent 桥则需要独立的 stage/event 对应和 projector 校准，必须通过 latent-only、map-only、phase-shuffle 及 action/occlusion controls 后才能解释为证据迁移，而不能把它当作通用架构兼容性证明。

SAEB-T0 的最小形式为：

```text
frozen VLM(image, instruction) → A_lang, z_lang
native VLA(image, instruction, state) → A_act(t), action, z_act(t)
weak LIBERO phase(source/mixed/target) → stage bridge
L = L_action + λ_sem L_sem + λ_contain L_contain + λ_stage L(z_act(t), z_lang(phase(t)))
```

`L_stage` 是低权重辅助项；空间 `A_lang/A_act` 和 patch occlusion 仍是主要机制证据。teacher、semantic cache 和 bridge head 只在训练时存在，推理时移除。首先只做一个小规模 T0（单任务、单 seed、固定 checkpoint）验证过渡接口，再决定是否进入三 seed。

SAEB-T0 的必要 controls：冻结 VLM、随机 semantic latent、phase shuffle、wrong-word/wrong-image、只用 latent 不用空间 map、只用空间 map 不用 latent。若 latent-only 与空间证据结果相同，不能声称空间归因是必要机制；若 SAEB-T0 只改善 latent alignment 而不改善 action/occlusion/rollout，保留为 transition engineering result，不升级为核心创新。

XYZ-Drive-inspired dynamic query 仍放在 T0/D0 通过后的 D1-query：它要回答的是“当前状态或 action prefix 是否能动态选择当前证据”，而不是“VLM 如何接入 VLA”。

### 3.1 P1 interface audit

- [ ] 审计 OpenVLA/OFT checkpoint revision、外部代码 commit、许可和 processor 版本；不从 `references/` 目录直接 import 模型。
- [ ] 找到 visual hidden states 与 patch/token provenance，生成动态 grid（禁止根据 token 数盲猜正方形）。
- [ ] 分别记录 retention 中间层与 action-relevant 层的 token provenance、grid 和坐标桥；两者不要求是同一层。
- [ ] 找到原生 action target、action loss、动作 chunk 和所有归一化常数。
- [ ] 分开记录 network action horizon 与 rollout execution chunk，不能把二者混写成一个 `H`。
- [ ] 用一个固定 LIBERO observation 计算一次有限 action loss、一次有限 gradient×activation 图和一次重复运行；记录 `action_shape`、grid、梯度有限性和 repeatability。
- [ ] 通过 checkpoint save/restore 后重新得到相同 action shape 和有限 loss。

P1 输出必须是机器可读 JSON，至少包含：

```json
{
  "model_id": "openvla-oft",
  "model_revision": "TO_FREEZE",
  "external_code_commit": "TO_FREEZE",
  "sample_id": "TO_FREEZE",
  "native_action_schema": "TO_FREEZE",
  "reported_action_schema": "delta_eef_7d",
  "action_shape": [7],
  "chunk_horizon": 1,
  "visual_grid": {"height": "TO_FREEZE", "width": "TO_FREEZE", "token_indices": []},
  "scalar_definition": "mean squared error on fixed native action target",
  "gradient_finite": true,
  "repeatability": "TO_MEASURE",
  "normalization": {"translation_unit": "TO_FREEZE", "rotation_unit": "TO_FREEZE", "gripper": "TO_FREEZE"}
}
```

### 3.2 LIBERO manifest

- [ ] 列出当前安装版本真正可用的 pick-place 任务，确定一个 task ID；不能把“pick-place”文字描述当作 task ID。
- [ ] 固定训练 episode、评测 episode、环境 seed、相机视角、语言模板、控制频率、最大 horizon 和 success evaluator。
- [ ] 对每个 episode 记录图像/状态来源、episode ID、任务 ID、语言、动作 shape、数据版本和 SHA256。
- [ ] train/eval episode 按 episode ID 隔离；评测 episode 不用于 teacher、超参或 checkpoint 选择。
- [ ] 首轮 offline split 和 rollout split 分开；offline 预测的 action error 不能替代 rollout success。

首轮 manifest 规则：smoke 使用 `train/offline_eval/rollout = 8/4/4`；G1 最小正式实验使用 `100–200/25–50/25–50`。三份 manifest 按 episode ID 严格不相交，B0–B4 完全共享同一版本和顺序；rollout split 不参与 teacher、超参或 checkpoint 选择。每次 manifest 变更都生成新的实验 ID、JSONL、SHA256、数据/环境 revision 和生成脚本 Git SHA，禁止覆盖旧结果。

正式训练预算至少为 10,000 个 optimizer updates。先按有效 observation-action transitions 和 effective batch 计算自然 epoch steps；若不足 10,000，则按固定 manifest 的确定性重复/采样补足，并记录 `natural_steps_per_epoch`、`dataset_passes`、transition 访问次数和重复规则。所有 B0–B4 与三 seed 使用相同的 steps、顺序和访问规则；smoke 仍固定 30 个 optimizer steps。

VLA-specific loss 草案固定为：`lambda_retention=0.05`、`lambda_containment=0.02`，并分别在前 500/1,000 个 optimizer steps 线性 warm-up；`lambda_sem` 继承 VLM 阶段冻结的 teacher 配置，但必须在 VLA smoke 中重新检查 loss scale。B1/B3 的 retention、B4 的 containment 只作用于允许训练的 projector/策略参数；各项 loss 和 optimizer-parameter gradient norm 分开记录，统一使用 optimizer 参数范围的 gradient clipping。30–100 step smoke 后若 action loss、finite、梯度比例或 restore gate 不通过，必须调整并重新记录，不能靠增大权重制造收益。

### 3.3 B0 smoke 与恢复

执行顺序固定为：1 个 episode 数据读取 → 20–50 optimizer steps → 保存 → 新进程 restore → held-out offline action prediction → 最多少量 rollout。只有以下条件全满足才允许 B1：

- loss、action output 和 gradient×activation 全部 finite；
- action shape、旋转和夹爪符号通过 adapter 校验；
- restore 后 config/manifest/model SHA 一致；
- offline action prediction 的样本数和 episode identity 可复核；
- rollout 使用官方 LIBERO evaluator，保存每一步 action、状态和失败原因。

首轮 OpenVLA/OFT 的 chunk 规则：P1 和 B0 smoke 保留原生完整 action chunk；正式 LIBERO B0–B4 暂定使用官方 `H=8`；rollout 每次预测后执行官方规定的 chunk，再重新观测。`A_act(t)` 保存 chunk 内每个时间步的图，并额外报告首步和 chunk 平均，不能为了简化归因把正式训练或 rollout 改成 `H=1`。

首轮相机配置固定为官方 LIBERO 的主第三人称相机加 wrist camera；相机顺序、名称、分辨率、center-crop、颜色格式和是否启用 proprioception 必须写入 P1 report 与 episode manifest。除非 P1 证明该 checkpoint 不接受其中某一路输入，否则不得静默移除 wrist camera。

官方 evaluator 规则：rollout success 只能由冻结版本的官方 LIBERO evaluator 判定；offline action MSE/MAE 与 rollout success 分开报告。每个 rollout episode 保存 evaluator version/commit、task_id、episode_id、environment seed、success、steps、每步 action、关键状态摘要和 failure reason。官方 evaluator 的 trial 数、最大 horizon、action repeat、控制频率和输出字段在 101 恢复后从实际代码读取并写入 manifest；在此之前不得用自定义 success 规则替代。

### 3.5 Visual token provenance 与 patch grid

所有归因图必须保存可追溯的 token provenance，而不是只保存一维 token 分数。每个视觉 token 至少记录来源模块、原始 patch/token index、二维坐标或覆盖区域、是否经过 spatial merge/projector、camera view 和 frame index。原生 patch grid 必须来自 processor、vision backbone 或 forward metadata；禁止仅根据 token 数量盲猜正方形。

层级分工固定为：B1 retention 使用 BlindVLA 风格中间视觉层；B2/B3 的 `A_lang` 与 B4 的 `A_act` 使用 action-head 输入前的 action-relevant visual tokens。其他层只做固定 checkpoint、单 seed 的 attribution sensitivity，不重复完整 B0–B4。

`T_sem`、`A_lang`、`A_act` 可以拥有不同原生 grid，统一只在规范化图像坐标中比较。坐标桥固定记录输入 resize/crop、padding、merge、view/frame 合并规则和目标评测网格；首轮目标网格暂定 `16×16`，仅作为公共评测网格，不代表任何模型的原生 grid。正式 P1 必须用一个已知位置的遮挡 patch检查归因能否回投到正确图像区域。

### 3.4 B1–B4 与 controls

B1 先跑通并确认 optimizer 参数组只有允许训练的 projector/策略层；冻结 teacher、retention target 和视觉 backbone。随后在完全匹配的 B0/B1 数据和预算上跑 B2/B3/B4。

每个语义或动作归因组至少保留：

1. 正确词/正确图 teacher；
2. same-image wrong-word；
3. wrong-image；
4. 面积匹配 random map；
5. object-only、object+EEF（仅作为动作相关性诊断，不作为训练真值）；
6. raw-attention proxy（若模型提供 attention），并记录其与 gradient×activation 的差异。

动作图必须再做 patch occlusion：遮挡每个 patch，测动作输出或 action loss 的变化，报告 `A_act` 与 `Δa` map 的 pointing、mass-in-region 和 rank correlation。该干预是机制证据，不能被写成额外训练标签。

## 4. 主要指标、gate 和解释边界

### 4.1 BlindVLA-style 表征可视化与量化诊断

需要加入 t-SNE，但它是**表征变化的诊断图**，不能作为 VLA success 或 grounding 的主指标。BlindVLA 的做法是从 COCO 中选择固定类别，抽取 OpenVLA 中间视觉特征，比较原生/微调模型的聚类结构，并辅以 linear probing。我们采用其设计，但将比较矩阵扩展为：

| 版本 | 含义 | 是否必须 |
|---|---|---|
| `VLM-pretrained` | 与 VLA 初始化对应的原生视觉语言模型/Prismatic 视觉语言底座 | 必须；作为上游表征参考 |
| `VLA-pretrained` | 未进行当前任务 action fine-tuning 的原生 VLA checkpoint | 必须；区分 VLM 与 VLA 初始化差异 |
| `B0` | 只用 action loss 微调后的 VLA | 必须；测普通 action fine-tuning 的表征漂移 |
| `B1-C` | B0 + C-RADIOv3-L BlindVLA-style retention | 必须；测官方 retention 机制是否保留聚类/可迁移视觉特征 |
| `B3-C` | B1-C + `T_sem → A_lang` | 必须；测语义证据是否在 retention 之外带来额外结构 |
| `B4-C` | B3-C + action-evidence containment | 推荐；观察动作约束是否破坏通用视觉结构 |

`VLM-pretrained` 和 `VLA-pretrained` 只有在输入处理、视觉 token 定义和抽取层确实可比时才放入同一张图；否则分面展示，不强行把不同 feature space 混合。t-SNE 的主比较在同一学生 backbone 的 `VLA-pretrained → B0 → B1-C → B3-C → B4-C` 之间完成。

#### 特征与数据冻结

- 复刻 BlindVLA 的中间视觉层作为主层，首选 layer 16；真实 checkpoint 的层号和 token provenance 由 P1 写入配置。
- 只使用视觉 patch feature，不使用带类别词的 text-object token 作为主结果；另保留 `vis_mean` 或 `vis_pool_attn` 作为敏感性图。
- 所有版本使用完全相同的 image ID、类别标签、图像预处理、prompt、patch pooling 和 feature normalization。默认采用 COCO `cup/bottle/knife` 三类，先固定每类相同数量；不得为某个版本单独筛样本。
- t-SNE 前先对所有版本的 pooled patch feature 使用同一 PCA 到 50 维，再使用相同 seed、perplexity、iteration 和 joint fitting。不得分别对每个模型单独拟合后比较坐标位置。
- 原生 VLM 与 VLA 的 hidden dimension 不同时，不直接拼接。主方案改为将各模型 feature 通过冻结、预注册的 C-RADIOv3-L teacher space 做 linear probe/nearest-neighbor 对照；t-SNE 只在各自空间内分面展示。

#### 必须同时报告的量化结果

t-SNE 图只用于直观展示，主表至少报告：

1. frozen feature 到 `VLA-pretrained` 的 cosine drift；
2. frozen feature 到 C-RADIOv3-L 的 patch cosine retention；
3. fixed train/test split 上的 linear probe accuracy；
4. kNN accuracy、class-wise silhouette 和同类/异类距离比；
5. 与 Flickr30k/RefCOCO grounding 或 LIBERO action attribution 的相关性（仅报告相关性，不声称因果）。

预期的解释边界是：B0 若出现类别混合或线性 probe 下降，说明 action fine-tuning 造成视觉表征漂移；B1-C 若恢复 feature retention，说明 BlindVLA 机制确实工作；B3-C 若在 B1-C 表征基本保持时仍改善语言/动作 evidence 和任务指标，才支持我们的语义证据机制具有独立增量。t-SNE 聚类变紧本身不能证明 grounding 或闭环成功率提升。

#### 图组布局

建议每个 checkpoint 使用相同布局：

```text
row 1: VLM-pretrained | VLA-pretrained | B0
row 2: B1-C          | B3-C           | B4-C
row 3: C-RADIO teacher reference + linear-probe / drift summary
```

每个点代表一张图像的 pooled visual feature，颜色表示类别，点形状表示 checkpoint；不要把不同模型的 t-SNE 坐标误读为绝对空间距离。另存一组以 image ID 对齐的 before/after drift 图，避免只展示类别聚类。

#### 与 BlindVLA 的差异

BlindVLA 的 t-SNE 主要支持“普通 SFT 会损伤可迁移视觉表征，而 alignment 可以缓解”这一诊断。我们的扩展增加了 `B3-C/B4-C`，检验“语言条件空间 evidence 或动作 evidence 在保持通用表征后是否还有独立收益”。如果 B3-C 的 t-SNE 与 B1-C 几乎相同但 grounding/action 指标更好，这不是失败，反而说明主方法改变的是输出条件 evidence，而不是简单重塑全局 feature geometry。

所有 t-SNE、PCA、linear probe 配置、样本 manifest、feature cache SHA 和图像生成脚本都必须写入 run artifact；t-SNE 图不能覆盖正式 checkpoint 结果，也不能作为单独 Go/No-Go gate。

主结果按同一 action head 报告：LIBERO success rate、平均完成步数、失败类型、offline action MSE/MAE、动作图 pointing/mass-in-box/soft-IoU、训练吞吐和达到固定 success 的 steps。动作图指标属于机制证据，不能单独替代闭环 success。

| Gate | 通过条件 | 失败时的表述 |
|---|---|---|
| 工程 | B0 restore、有限 loss、动作 shape/单位/官方 evaluator 全通过 | 接口未打通，暂停扩展 |
| 基线 | B1 与 B0 在同预算下可复现且 retention target 有效 | 不能声称复现 BlindVLA-style baseline |
| 语义机制 | B2 或 B3 的 `A_lang` 指标和至少一项动作指标优于对应 matched baseline；正确 teacher 优于负控 | 仅说明 teacher 可测，不能说语义约束有效 |
| 动作机制 | B4 相对 B3 在动作归因干预和至少一项 offline/rollout 指标有增量，且不是随机 mask 增益 | D0 作为诊断，不升级为核心创新 |
| 泛化 | LIBERO-Plus/LIBERO-PRO 或 DROID 的预注册分组趋势复现 | 只报告 ID 机制结果，不声称 OOD/真实泛化 |

多 seed 用 image/episode cluster bootstrap；不同模型和动作头不得直接比较绝对 success。B4 不通过时不能进入 D1 phase、WAM/VAM、RoboTwin 或真机扩展。

## 5. 代码和 artifact 结构

首轮实现按以下边界组织；模型外部仓库只由 adapter 调用：

```text
src/vla_attention/
  adapters/openvla_oft.py       # processor、hidden states、native action/loss、restore
  attribution/action_gradient.py # action loss -> visual token gradient×activation
  losses/retention.py            # BlindVLA-style feature retention
  losses/semantic.py             # frozen T_sem -> A_lang bridge
  losses/action_containment.py   # D0, stop-gradient M_lang
  trainers/vla_bc.py             # B0-B4 shared loop and step logging
  evaluation/libero.py           # official evaluator wrapper and interventions
configs/vla/p6_b0_b4.json        # one matrix, immutable revisions/manifest references
experiment_workspace/results/P6_vla_b0_b4/
  p1_interface/
  manifests/
  B0/ B1/ B2/ B3/ B4/
```

每个组必须保存 `config.json`、Git SHA、model/external-code revision、teacher revision、manifest SHA、seed、effective batch、max steps、每步 JSONL 曲线、checkpoint restore report、失败 episode 和 evaluator 版本。checkpoint、大 cache、LIBERO 数据和视频不得进入 Git。

## 6. 当前未决项与解决证据

| 未决项 | 不能靠猜的原因 | 解决动作 | 状态 |
|---|---|---|---|
| OpenVLA/OFT 精确 checkpoint/commit | 不同 revision 的 action head 和 processor 可能不同 | P1 审计后写入 config | 未解决 |
| LIBERO task ID/版本 | 同名 pick-place 任务可能对应不同初始状态和语言 | 枚举环境并生成 manifest | 未解决 |
| 原生 action 是否可逆映射到 7D delta EEF | 直接 reshape 会改变物理含义 | 读取官方 action spec，做 round-trip 检查 | 未解决 |
| 旋转和 gripper 符号/限幅 | 符号错会产生假性 action error 和 rollout 失败 | 单步专家 action 与反归一化对照 | 未解决 |
| visual hidden-state 层和 grid | 模型可能有双视觉塔或非方形 token | adapter 保存 token provenance 与坐标桥 | 未解决 |
| A_act 的 scalar | action token、action output、action loss 不等价 | 首版固定 action loss；若不支持则记录阻塞 | 已选候选，待审计 |
| B0–B4 effective batch/max steps | 显存和 action chunk 决定可行 batch | 20–50 step GPU smoke 后冻结 | 未解决 |
| rollout 时长和 success seed | 长 horizon 会混入控制频率和随机性 | 先 offline，再固定少量官方 rollout | 未解决 |

这些未决项不会阻塞本地代码骨架和配置准备，但在 P1/manifest 证据产生前，不允许提交 B0–B4 长训练。

## 7. 参考论文后的颗粒度冻结

本协议采用四层推进，避免把工程 smoke、机制验证、OOD 泛化和 WAM 动态扩展混成一个结论。

### G0：接口工程 gate

- OpenVLA/OFT、一个 LIBERO pick-place/put-object-into-container task；
- 1 个 episode 读取、20–50 steps、checkpoint restore；
- 验证 native action schema、7D delta EEF round-trip、visual grid、finite action loss 和 `A_act`；
- G0 通过只代表接口可运行，不代表方法有效。

### G1：最小 idea gate

- B0–B4 使用固定 train/eval/rollout manifest，seed `17/29/41`；
- 同一 action head、语言模板、图像预处理、预算和 official evaluator；
- 报告 offline action MSE/MAE、短 rollout success、action attribution pointing/mass/soft-IoU 和 patch-occlusion rank correlation；
- correct teacher、wrong-word、wrong-image、area-matched random 和 raw-attention proxy 必须成组保留；
- 只有正确 teacher 的归因质量、`B2/B3` 对 matched baseline 的增量、`B4` 对 `B3` 的非随机增量同时满足预注册 gate，才进入 G2。

### G2：论文级机制与 OOD

- LIBERO 四个官方 suite 做 ID 闭环；
- LIBERO-Plus 按每类 task-preserving perturbation 分开报告；
- 增加 VL-Think 风格的 VL 保留诊断、task-preserving visual intervention 和 goal-changing language intervention；
- DROID 只做 episode-level/scene-level offline action prediction 和归因迁移，不能把 offline error 写成闭环 success；
- 所有结果拆成归因、动作、OOD、VL retention、效率五类表。

### G3：跨结构与动态教师

- 先迁移到 π0.5，再考虑 MolmoAct2；
- WAM/LingBot-VLA/Next Forcing 只在 G2 通过后加入；
- RoboTwin 和真机不阻塞首轮，也不作为 B0–B4 的替代 benchmark。

### 外部 baseline 的编号边界

BlindVLA-style retention 是 `B1` 主矩阵；LIT 的 spatial-goal action prior + latent interface 是独立 `E-LIT` 外部 baseline，不并入 B0–B4；Next Forcing 只作为 `F0` future-attribution diagnosis 和 `F1` refiner 扩展。这样可以分别回答“视觉表征保持”“语言条件归因”和“未来动力学监督”三个问题。

## 7.1 借鉴机制的实施优先级

为了保持增量链路可解释，借鉴机制按以下顺序推进：

| 优先级 | 机制 | 首次出现位置 | 目的 | 是否改变主方法 |
|---|---|---|---|---|
| P0 | BlindVLA retention drift、VL 保留和 sink 诊断 | B1/G1 | 监控视觉遗忘，排除 retention 解释 | 否 |
| P0 | LIT 的 paired budget、task-preserving visual intervention、goal-changing language intervention | G1/G2 | 证明结果不是只靠 success 或漂亮热图 | 否 |
| P0 | Next Forcing 的 multi-layer attribution aggregation | layer-sensitivity 小消融 | 减少单层归因不稳定 | 否 |
| P1 | image-free action-prior 对照 | `E-LIT-prior` | 分离 action prior 与空间归因收益 | 否，独立 baseline |
| P1 | stop-gradient semantic interface | `D-interface` | 检验语言支持视觉摘要是否改善 action head | 是扩展变体，不进入 B0–B4 |
| P2 | short/mid/long `R_future` consistency | F0/F1 | 检验动作归因是否与未来动力学一致 | 后期 WAM/VAM 扩展 |
| P2 | action noise-level attribution | π0.5/flow/diffusion 后期 | 检查不同 action denoising level 的归因稳定性 | 只适用于原生连续动作头 |

首轮禁止同时启用 retention、semantic、phase、future、pose 五类损失。任何新增机制必须先作为单独对照或诊断跑通，再进入下一层；否则无法判断增量来自哪条链路。

## 8. 记录与 Git 操作

每完成 P1、manifest、B0 smoke、restore、B1、B2–B4、controls 或评测的 10/25/50/75/100% 节点，在 `experiment_workspace/PROGRESS.md` 新增记录：时间、实验 ID、Git SHA、远端 PID/job、输入 manifest/SHA、model/teacher revision、seed/steps、日志和结果路径、成功/失败样本、是否 non-final、下一步。

代码/配置变更只提交可审计文件，例如：

```bash
git add experiments/P6-vla-b0-b4/protocol.md configs/vla src/vla_attention experiment_workspace/PROGRESS.md
git commit -m "docs: freeze vla b0-b4 interface and gates"
git push origin main
```

提交前运行 `git diff --check`、JSON 解析和相关 pytest；不得用 `git add -A` 把 checkpoint、cache、数据或其他未审计修改带入提交。

## 9. 101 不可用时的工作边界

101 的 TCP 端口可达不等于 SSH 服务可用。若 SSH 在 banner/kex 阶段超时或被远端关闭，本地只继续完成协议、纯 Python 合同测试、StarVLA 静态审计、manifest/checkpoint artifact schema 和可恢复上传包；不猜测 OpenVLA/OFT 的 native action、LIBERO task ID、动作单位或 rollout 结果。具体队列和恢复后的固定顺序见 `experiments/P6-vla-b0-b4/101_outage_readiness.md`。

在 101 恢复后，必须先完成只读 SSH、上传 SHA 校验、OpenVLA/OFT P1、LIBERO episode manifest、20–50 step B0 smoke、checkpoint restore 和 held-out offline prediction，才允许提交 B1–B4 长训练。
