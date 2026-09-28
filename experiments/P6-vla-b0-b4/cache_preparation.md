# 三类 VLA cache：输入冻结与 LIBERO-Plus 边界

2026-09-28。用户确认三类都纳入计划；准备接口不代表现在同时加入训练。

| 类别 | 生产者 / 输入 | 消费者 | 前置条件 |
|---|---|---|---|
| semantic_map | 冻结扩散模型 / LIBERO 实际相机帧、指令短语 | B2/B3/B4 | 词位、抽取阶段、增强/坐标桥审计和机器人图像校准 |
| visual_features | C-RADIOv3-L / 同一实际相机帧 | B1/B3/B4 | 权重 revision、特征分支、预处理、patch grid/维数与 projector 定义 |
| SAEB evidence/latent | 冻结 VLM / 同帧、同指令、明确输出目标 | ALN-inspired 后续扩展 | VLM checkpoint、输出标量、层、pooling、phase 来源冻结；当前不执行 SAEB |

Flickr30k cache 不可充当 LIBERO 的 teacher 输出。可复用经验证的提取代码和配置，不能跳过机器人观测上的有效性校验。`A_act` 等训练中随学生参数变化的量必须在线计算；仅冻结 teacher 信号可缓存。

每条输入清单包含 benchmark/version、task、episode、timestep、camera/frame、图像内容 SHA、指令与 phrase/token-span、预处理/augmentation SHA、teacher revision、extractor/code SHA、signal type、normalization、输出 SHA。随机增强改变输入时必须同步变换具有有效几何对应的空间图，或重算 teacher；颜色增强后的 feature 不默认等于原 feature。主相机和 wrist 分别保留来源。

LIBERO base 训练、Plus OOD 评测严格隔离。Plus 普通策略 rollout 不需要任何外部 teacher cache；若为诊断提取 Plus 图/特征，单独设 evaluation-only cache，不调参、不选择模型、不回流训练。同 task/episode 名称不等于同输入；仅当某信号依赖的所有输入内容、预处理、teacher、抽取参数一致时才可复用。例如语言变化不一定影响纯视觉 C-RADIO feature，但一定需要重新检查语言条件图。依赖 state/phase 的 latent 即使图像相同，也不能忽略状态变化。

闭环轨迹由当前 policy 产生，不可把 demonstration cache 按 timestep 生搬到 rollout。需分析闭环证据时，对实际观测在线提取或事后按日志重算。

准备顺序：真实 episode manifest → 各 teacher 单样本验收 → 缓存合同与小样本一致性 → 资源可用时生成。三类分别版本化，不能因一种生成完毕就称三种就绪。
