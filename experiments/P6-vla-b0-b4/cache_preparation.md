# 三类 VLA cache：输入冻结与 LIBERO-Plus 边界

2026-09-28。用户确认三类都纳入计划；准备接口不代表现在同时加入训练。

当前执行更新：用户已授权暂停旧错误Flickr job，保留产物；SD重建阶段提取修复后，两条样本与确定性单图参考误差均为0（阈值未放宽）。锁见 `configs/teachers/cache_reconstruction_v3.json`。C-RADIO真实8帧smoke后已完成全部27条train轨迹的5040帧特征缓存，并逐条校验通过。LIBERO语义图采用相同5040帧×source/target=10080张图，先验收16条smoke后后台队列生成；不能将导出帧数当生成完成数。

SAEB只记录输入合同，尚无已选定的VLM输出标量、层与checkpoint；不生成任意latent代替待定目标。SD图的finite/hash/参考一致仅是工程门禁，旧1000图混阶段校准不能沿用，新语义质量仍待重新评估。

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

## 安全增量原则：semantic 去重只能先做隔离 pilot

2026-09-29 的远端 CPU 审计确认，LIBERO semantic manifest 的 `10080` 行由 `5040` 个严格二元组构成。每个二元组共享同一张图像、完整 instruction、camera、episode、timestep、teacher revision、随机种子和数值配置，只在 `role/phrase/sample_id` 上不同。因此可以把一次完整的 teacher 计算结果视为共享证据张量：

```text
H(I, c) = full-caption inversion/null-text/final-reconstruction evidence
A_phrase(I, c, p) = Normalize(Mean_{j in token_span(p)} H(I, c, :, j))
```

这只是计算因子分解，不改变 teacher、loss 或监督定义。正式实现仍必须为 source 和 target 分别保存独立的 map、metadata、phrase occurrence、token span、role 和 sample_id。不能因为共享 `H` 就合并两个样本的 metadata，也不能把 phrase 从 key 中删除后忽略 occurrence 或坐标来源。

在当前正式 cache 仍运行时，遵循以下不可变规则：

1. 旧 cache、已有 map、metadata、progress、failures、日志和 manifest 只读保护，不删除、不移动、不重命名、不覆盖。
2. 不在 parity 通过前停止或重启正式 worker。
3. pilot 必须写入新的版本目录，例如 `semantic_shared_pilot_v4_pair_parity_seed17/`，不写入 `semantic_train_task0_v3`。
4. pilot 只取两个已有 source/target pair；必须复用正式 FP32 SD1.5、20 diffusion steps、inner_steps=10、CFG=7.5、16×16、100 tensors、deterministic math SDPA 和相同 extractor。
5. parity 至少检查 attention tensor 数量、有限重建 MSE、map finite、空间尺寸、token span、metadata provenance、map SHA、max/mean absolute error 和归一化后误差。误差非零时先解释来源，不能直接放宽阈值。
6. 两张 GPU 都忙时不启动 pilot，不为了 pilot 打断正式 worker；只做 CPU manifest 审计、快照和文档更新。
7. parity 通过后只生成切换和 rollback 方案。正式切换必须使用新版本目录，保留旧目录为 immutable legacy；不得自动删除旧 cache，也不得修改 Flickr、C-RADIO 或 B0 waiter。

本次安全快照和 manifest audit 位于：

```text
experiment_workspace/results/cache_safety_snapshots/20260929T155338Z/
```

正式 semantic cache 仍以原 worker 的输出为准；pilot 通过前，任何“去重后全量完成”的说法均无效。
