# 模型、baseline、benchmark：供筛选的详细建议

依据本地 PDF 的摘要筛查及 LIT 实验段落。这里给出选择，不替用户冻结；代码、权重、许可证和最新线上结果未在本轮重新核验。SpikingBrain 后置已确认，其余推荐不等于决定。

## 1. 模型选择：按接口而非名气计数

| 候选组合 | 能回答的问题 | 优势 | 代价与进入条件 |
|---|---|---|---|
| OpenVLA/OFT 单骨干 + Qwen/LLaVA VLM | 最小可行的语义→动作归因链 | 数据/代码已部分归档，便于先做 smoke | 不足以证明跨 VLA；OpenVLA token 与 OFT 动作头要分别审计 |
| OpenVLA/OFT + π0 或 π0.5 | 离散/回归与 flow 动作机制差异 | 连续生成动作是重要普适性测试 | 必须确认 OFT 版本与 head；噪声和 flow time 的归因需定义 |
| π0 系 + MolmoAct2 | shared-token/MoT 与逐层 KV cross-attention | 最接近 LIT 的结构覆盖方式 | 两套系统成本；需核对相同数据训练能力及 checkpoint |
| OpenVLA/OFT + MolmoAct2 | 一个可运行底座加逐层 KV 条件结构 | MolmoAct2-DROID 清洗方法值得借鉴 | 防止预训练包含测试轨迹，不能将其数据直接随机切帧 |
| 现有首个 VLA + LingBot-VLA | 不同 VLM 底座与多 embodiment 表示 | 后期可接视频/几何辅助任务 | 仅在 7D delta EEF 子设置可比时运行 |
| 一种 VLA + 一种 WAM | 接口能否跨预测式和响应式策略 | 拓展方法覆盖面 | 额外 future model 能力混杂；WAM 作为被测学生与外部教师分开 |

**推荐顺序**：先从已有本地生态挑一个可运行的 VLA；用 20–50 个固定样本做 hidden-state/动作归因、高阶梯度与噪声复现检查；再选结构差异大的第二个。这里样本数是诊断建议，不是论文统计量。π0.5 与 π0、OpenVLA 与 OFT 不能在题目中随意互换。

VLM 侧建议 Qwen2.5 或 Qwen3 先二选一，再加 LLaVA-1.6/OneVision。若接口只有视觉 self-attention，不能直接命名为语言条件 attention。同一 Qwen 家族多个规模不替代跨结构证据。

## 2. 先区分两种训练问题

**已有 policy checkpoint 微调（建议首轮）**：实际研究如何改进已有 VLA。paired baseline 和我们方法必须从同一 checkpoint 开始，保留同样动作头。

**预训练 VLM + 随机动作专家（LIT 的选择）**：更适合隔离动作 prior 和视觉接口如何形成，但动作学习难度与资源更大。LIT 的 π0.5 等架构结果不等于官方 policy checkpoint 微调结果。

这两条可以分别报告，不能混在一张表宣称全面优于他人。相同步数也不等于相同 FLOPs；归因训练的高阶梯度、额外教师前向需要记录 wall-clock、显存和教师缓存成本。

## 3. Baseline 分四类，按要排除的解释选择

### 3.1 最小方法矩阵

| 编号 | 条件 | 回答什么 |
|---|---|---|
| B0 | 同初始化的原生 BC/SFT | 是否真的改进动作或 grounding |
| B1 | BC + L_sem | 是否所有收益已由 VLM 语义保持解释 |
| B2 | BC + L_sem + L_contain（D0） | 动作归因约束的增量 |
| B3 | 与 B2 等额外计算/参数的简单 feature anchoring | 是否只是通用正则/额外算力 |
| B4 | 等数据量/预算的视觉增强 | 是否只是对背景过拟合的常规修复 |
| B5 | 错词/错图/随机教师，保持图面积与平滑度可比 | 是否利用了正确语义，还是任意稀疏 prior 都有效 |

### 3.2 强相邻方法

- Anchor-Align：表征锚定 + 同观测动作方向语言标签。
- Don't Blind Your VLA / ReVLA：对齐或恢复视觉表征。
- PosA-VLA：pose-conditioned spatial attention。
- BridgeVLA：显式 heatmap/空间输出桥。
- LIT：受限视觉通道 + 空间目标监督 + action prior。

选择与主假设最接近的 1–2 个完整方法，另做简化机制对照。简化复现必须叫 inspired baseline，不使用原论文方法名暗示完整复现。C 的 EEF 高斯也不是对原论文的等价实现。

### 3.3 动态扩展

只在 D0 有结果后比较 D0、memory-only、D0+memory、phase-only、D0+phase。对照 AVA-VLA；motion teacher 参照 MotionEnhancer。时间窗口、帧数、action chunk 和物理预测时长匹配。

### 3.4 未来扩展

比较静态教师、直接 future-feature alignment、R_future spatial refinement；再做 wrong-action、wrong-future、同 horizon 重复与多 horizon。Fast-WAM 提醒分清训练期 future supervision 和推理期 imagination；ImageWAM/LaWAM 提供不解码完整视频的选项；Robust-WAM 是强 feature-alignment 对照。

## 4. Benchmark 菜单和证据强度

| 候选 | 在我们论文中的作用 | 是否建议首轮 | 不可据此声称 |
|---|---|---|---|
| Flickr30k Entities | 短语区域定位，教师/学生图评价 | 推荐最小 VLM 组合 | 准确热力图就是动作因果证据 |
| RefCOCO/RefCOCOg | 多实例、关系、较长 referring expression | 可替换或扩充 | 与前者不匹配 split 的绝对分数可直接比较 |
| VQA 保留集 | 机器人微调是否损伤回答能力 | 小规模建议 | 所有 VLM 能力均被保留 |
| LIBERO 单 pick-place | 坐标/动作/闭环 smoke | 必需启动实验 | 单任务即普适机器人能力 |
| LIBERO 正式 suites | ID 闭环动作能力 | 主结果建议 | ID 高分排除了 shortcut |
| LIBERO-Plus | 视觉及其他任务保持扰动下零样本闭环 | 推荐 OOD 候选，待用户确认 | 每类扰动对每个模型必然有效 |
| LIBERO-PRO | 更强调目标/任务组合变化的候选 | 语言跟随主张强时考虑 | 与 Plus 相同性质或相同实现 |
| CALVIN | 长程时序与指令执行 | D1/记忆主线时考虑 | 单步 delta EEF 已解决长时域 |
| SIMPLER | ReVLA/Don't Blind 路线 OOD 比较 | 替代评估方向 | 与 LIBERO 同机器人或同控制器 |
| RLBench+COLOSSEUM / GemBench | 与 BridgeVLA 的多视角/3D 路线配套 | 若转 3D 再考虑 | RGB-only 方法与点云方法资源一致 |
| DROID | 真实数据离线预测、保留能力与域间泛化 | 用户已确定必须使用 | 离线动作误差等于真实机器人成功率 |
| 实体机器人 | 最终闭环外部效度 | 后置 | 只有漂亮视频就已统计验证 |

LIBERO-PRO 等条目依据本地 Anchor-Align 摘要定位，具体任务构造、版本和下载入口仍待复現前核查。不要求一次跑所有 benchmark。

## 5. DROID：可执行的候选协议

1. 下载后先审计字段：episode、语言、相机、robot state、动作单位/坐标、时间戳、采集分组和可用标签。没有可靠 object/operator ID 就不虚构该 split。
2. 所有 split 在 episode 层及更高分组进行；同一轨迹不能帧级拆分。对基础模型预训练数据重叠做声明，无法排除时不声称严格未见。
3. 首轮 pick-place 子集规模由筛选结果和训练曲线决定。此前 1,500/250/400 只是建议，不是冻结配置。
4. 建议先保留原始控制频率与时间戳。若降到 5/10 Hz，重新定义动作时域与组合方式，不能每隔几帧抽一个仍代表原控制周期的 delta。
5. 优先单帧/H=1 接口诊断，再独立比较多帧/chunk；不能把四帧收益算给 D0。7D 只约束表示类别，具体旋转、坐标系和 gripper 编码需审计。
6. 离线报告各动作分量误差、夹爪、语言反事实、视觉保留与归因稳定性。物理 rollout 成功率另在 LIBERO/真机测。

## 6. 从这批论文引出的必要技术审计

- action loss 的梯度突出拟合误差，action output 的梯度突出输出敏感性；二者不能都叫同一个“决策贡献”。动作分量尺度也需统一。
- 如果把 gradient×activation 本身作为训练损失，对参数求导可能需要二阶梯度；先验证可微、开销和 kernel 支持。只训练一个看起来正确的 heatmap head 不能证明 action head 使用它。
- 语言图可以很宽，containment 也会很小；增加面积/熵匹配对照，stop-gradient 锚定或其他避免双方共同退化的设计待选择。
- source/target 并集不总包含夹爪、障碍物、接触边缘；严格包含是待检验假设，宜提供 soft leakage/可信区域限制的备选，而非把合理越界一律惩罚。
- occlusion 会造成分布外输入，不能无条件称“金标准”。同时使用模糊、替换、匹配面积随机遮挡，固定生成模型噪声；干预后保留任务可解性。
- state 遮挡和机械臂图像遮挡不同；对 proprioception 的合理依赖并不天然是捷径。采用有效范围内的配对状态变化，区分失去控制必需输入与错误依赖。
- StreamingLLM 保留 sink；我们的可视化重归一化不代表应修改 attention kernel。不要自动删除所有边缘 patch。

## 7. 逐步结论与失败分支

| 步骤 | 如果结果支持 | 如果不支持 |
|---|---|---|
| 坐标桥/目标标量审计 | 进入小数据训练 | 修接口，不调大对齐权重掩盖错误 |
| 语义教师优于错图与 feature baseline | 支持空间语义有增量 | 先检查教师/目标，不急上动作 |
| D0 优于 L_sem only 与简单正则 | 支持动作证据约束有贡献 | 收窄主张，分析是否发生图退化 |
| ID 保持、task-preserving OOD 改善 | 支持鲁棒性 | 不写泛化结论，查看分维度负结果 |
| 换目标会正确改变动作 | 支持语义敏感性 | 稳定可能只是忽略所有输入 |
| DROID 分组测试改善 | 支持真实数据离线泛化 | 保留仿真结论，不宣称真实控制 |
| 第二接口复现趋势 | 有限范围跨结构证据 | 限定模型家族，不转回 SpikingBrain 主线 |
| B/D1 超过相关强对照 | 可考虑升级扩展 | 写负结果/未来工作，不堆损失 |

报告至少使用独立训练重复、配对评估条件、逐任务/逐扰动结果和不确定性区间。没有计算预算确定前不强制一个虚构的样本量；正式评估前预先冻结统计协议。

## 8. 供用户确认的少量决策

- [x] SpikingBrain 后置，不是主线；DROID 必须使用。
- [ ] 第一 VLA：OpenVLA/OFT、π0 系、MolmoAct2、LingBot-VLA 中哪个接口先审计通过就优先试验，还是先指定一个？
- [ ] 第二 VLA：优先结构差异，还是优先已有可运行工程？
- [ ] OOD 主 benchmark：推荐 LIBERO-Plus；若重心是语言目标变化，是否补 LIBERO-PRO？
- [ ] 论文重心：先 D0 输出条件空间约束，还是以记忆/阶段为主？推荐先 D0，D1/B 保留选择。
- [ ] 主线 strong baseline：Anchor-Align 与 PosA-VLA 优先，还是 LIT 通道约束优先？应结合最终主张选择。

这些选项均可在接口诊断后确认；本轮不把任何推荐偷偷改成固定模型组合。
