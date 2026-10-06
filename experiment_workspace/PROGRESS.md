# 训练前准备进度

## 2026-09-25：P6 VLA 并行工程准备与 goal 对齐（当前有效）

本次严格重新读取 goal 附件，并核对 `doc/03_execution_plan.md`、P4 protocol、P6 protocol、本文档和当前代码树。结论按证据区分如下：

- [x] 识别到 P6 原文件只有 13 行概略说明，不能作为可执行实验协议；已扩展为 `experiments/P6-vla-b0-b4/protocol.md`，写明 OpenVLA/OFT + LIBERO 单 pick-place、7D delta EEF 报告接口、H=1、offline-first、B0–B4 paired matrix、动作归因、controls、gate、artifact schema 和未决项。
- [x] 新增 `configs/vla/p6_b0_b4.json` 与 `configs/vla/README.md`。配置明确标记 `template_not_runnable_until_p1_and_manifest`，所有 checkpoint/task/action normalization/manifest 字段仍为 `TO_FREEZE`，不能直接提交长训练。
- [x] 首轮矩阵已在协议中统一为：B0 native action baseline；B1 BlindVLA-style retention；B2 B0+T_sem→A_lang；B3 B1+T_sem→A_lang；B4 B3+D0 A_act containment。所有组共享初始化、动作头、数据、seed、预算和 evaluator。
- [x] A_act 首版固定为 action loss 对 visual tokens 的 gradient×activation；action query attention 仅作诊断分支。D0 首版不引入 L_phase、R_future、world-model 或 action-expert 蒸馏。
- [x] 已明确 VLA 首个 B0 的硬门槛：P1 visual/action provenance、native action schema、7D 转换 round-trip、LIBERO task/episode manifest、有限 action loss、checkpoint restore、offline held-out prediction；通过前不运行 B1–B4 长训练。
- [x] 已执行只读 SSH 状态检查（2026-09-25 10:52 CST）：当前 Codex 执行环境返回 `socket: Operation not permitted`，未获得 101 的远端状态证据，未启动、重启或声称完成任何远端任务。
- [x] 复试只读 SSH（2026-09-25 10:54 CST，TCP 已建立但 SSH banner 在 15 秒内超时）；仍未获得远端状态证据，未重启任务。
- [x] 本地 P6 配置 JSON、`git diff --check` 和 Python 3.10 下 7 个接口/网格/坐标桥测试通过；默认系统 Python 的完整 pytest 受本机 numpy/torch 依赖缺失影响，不能作为代码回归结论。
- [ ] VLA 当前仍未完成：OpenVLA/OFT 精确 checkpoint/commit、LIBERO task ID/版本/episode manifest、原生动作格式与归一化、P1 adapter、B0 smoke/restore。文档和配置的完成不等于模型或 benchmark 已下载。
- [ ] VLM 当前仍未完成：Flickr30k/Entities archive 完整性和正式 SHA、约 1000 图四教师 calibration、best-single/cache freeze、V0–V4 三 seed 全量训练和官方 evaluator。现有 20/64/F0 pilot 只作 non-final 证据。
- [x] 2026-09-25 依据 BlindVLA、Next Forcing 和 Breaking the Vision-Action Shortcut/LIT 的本地论文与源码审计，新增 `references/paper_reading/vla_granularity_review.md`；将 VLA 分成 G0 接口 gate、G1 最小 idea gate、G2 论文级机制/OOD、G3 跨结构扩展，并明确 LIT 为独立外部 baseline、Next Forcing 为后置 future-attribution/refiner，不混入 B0–B4 主矩阵。
- [x] 2026-09-25 在同一调研文档中补充“机制借鉴”方案：P0 优先采用 BlindVLA retention drift 监控、Next Forcing 多层归因聚合和 LIT 的 paired/counterfactual 评测；P1 再做 action-prior 对照或 semantic interface；P2 才做 `R_future` 与 noise-level attribution。完整 LIT/Next Forcing MCP、EEF 高斯监督和多 teacher ensemble 不并入首轮主方法，以避免改变问题定义或产生 novelty 重合。
- [x] 2026-09-25 将上述机制借鉴同步到论文预稿：`paper_draft/03_related_work.md`、`04_system_and_method.md`、`05_experiments_and_expected_conclusions.md`、`08_model_baseline_benchmark_options.md`；明确 Ours-1/Ours-2/Ours-3、E-LIT-prior、multi-layer attribution 和 `R_future` 的进入条件及结论边界。
- [x] 2026-09-25 在 101 SSH 不可用期间新增本地 `scripts/validate_vla_config.py` 和 `tests/test_validate_vla_config.py`。该检查器不依赖 torch/LIBERO，只验证 P6 B0–B4 矩阵、7D delta EEF 报告接口、seed、offline-first 和未冻结字段；默认拒绝含 `TO_FREEZE`/`TO_CREATE` 的模板启动，`--allow-template` 仅供配置单元测试。
- [x] 2026-09-25 冻结 StarVLA 依赖边界：P6 首轮模型是 OpenVLA/OFT，StarVLA 只可作为经过兼容性审计的运行环境/参考工具；不直接复用其 SpikingBrain/Robotwin 训练脚本。若研究 StarVLA 自身，另建 adapter 和独立实验矩阵。
- [x] 2026-09-25 根据用户提供的官方地址 `https://github.com/starVLA/starVLA` 下载源码至 `references/repos/starvla`，固定 commit `4507931a625c844404c4a76128fb116536d8ca7c`，并新增 `SOURCE.md` 与核心文件 SHA256。初步审计确认其支持 Qwen-VL + OFT/FAST/PI/GR00T 多动作头、LIBERO/SimplerEnv/LIBERO-Plus 等接口；后续仅将其用于 VLM-to-VLA 扩展参考，不改变 OpenVLA/OFT P6 主线。
- [x] 2026-09-25 冻结模型选择规则：若模型系列已有公开、可复现的原生 VLA 形态，则直接使用其原生 backbone/action head，不再额外套 StarVLA；StarVLA 仅用于没有原生 VLA 的 VLM-to-VLA 扩展，或独立的跨 action-head 可迁移性实验。
- [x] 2026-09-25 明确 101 不可用期间的本地/远端队列，并修正 `server/01_vla_attention_training.md` 的旧连接状态：本地继续做配置合同、adapter/manifest 骨架、VLM cache/runner 准备、BlindVLA 代码审计和上传包；真实模型 P1、LIBERO 环境、GPU smoke、teacher 大 cache、正式训练和 DROID 必须等 101 或同等服务器。

下一步按 goal 固定顺序执行：先恢复可观察的 101 连接并只读核验 Flickr/Entities、活跃 jobs 和 VEPFS 文件 SHA；若连接仍受 socket 限制，则继续本地实现 manifest/config 校验和 VLM runner，不重启远端任务。VLA 只在 OpenVLA P1 与 LIBERO manifest 证据产生后提交 B0 smoke。


## 2026-09-24：进入正式 VLM/VLA 实验前的 Preflight 清单

本节用于区分“已经有工程证据”和“仍然阻塞正式主结果”的事项。F0 pilot、20/100/500-step smoke、单 checkpoint 能力评测和 BlindVLA-style VLM retention 适配，均不能直接替代下面的正式实验。

### 2026-09-24：最终冻结口径已写入 protocol

- [x] V0--V4 主矩阵、V3/V1 与 V4/V2 主比较、正确/错词/错图/随机图和 attention proxy 机制对照已冻结；
- [x] 训练预算冻结为完整 train manifest 的 1 epoch、seed 17/29/41、无 test-based early stopping；1 epoch 作为公平计算预算，不作为理论最优结论；
- [x] 训练步数与曲线记录规则已冻结：正式 `max_steps` 按冻结 manifest 的记录数和 smoke 后确认的 effective batch 自动计算；每个 optimizer step 写 `train_metrics.jsonl`，每 10% 写进度汇总，25/50/75/100% 保存 checkpoint，并生成 `loss_curve.json/png`；
- [ ] 具体 effective batch、gradient accumulation 和最终数值 `max_steps` 尚未冻结，必须在服务器 GPU smoke 后写入 config SHA；因此当前不能把 20/100/500-step F0 数字当作正式全量步数；
- [x] teacher 选择时点冻结为正式训练前的独立约 1000 图 validation calibration；候选包括 SD1.5、PixArt-α、PixArt-Σ、Playground-v2.5，按预注册的 pointing→mass/soft-IoU→有效率/稳定性规则选一个 best-single；
- [x] 主指标冻结为 phrase pointing；mass/soft-IoU/IoU、official capability、OOD、效率、intervention 和所有意外指标均必须保留；
- [x] benchmark 结果必须使用官方 split、官方输入格式和官方 evaluator；pilot 内部 scorer 不进入论文主表；
- [x] 多元成功标准和 VLM→VLA 叙事已写入 `doc/03_execution_plan.md`、`experiments/P4-vlm-v0-v4/protocol.md`、`paper_draft/05_experiments_and_expected_conclusions.md`；
- [ ] 1000 图 calibration、最终 teacher、manifest/config SHA 和正式训练结果仍未完成，不能把 provisional candidate 写成最终结论。

### A. VLM 主线：Qwen2.5-VL-7B 先进入正式验证

- [x] Qwen2.5-VL-7B 的 P1 visual-token/phrase-score gradient×activation 接口通过；
- [x] V0–V4 的 20-step checkpoint smoke 和 64 条 held-out evaluator 已通过；
- [x] SD1.5 FP32 null-text 的单图、10 图和 64 条 teacher-map pipeline 已通过；
- [x] teacher controls 已实现并有初步结果：correct map 优于 wrong-word、wrong-image 和 random 的 pointing/mass；
- [ ] 核验 Flickr30k Entities archive、图片、XML、Sentences、caption manifest 的完整性，并重新写入固定 SHA256；
- [ ] 用独立 validation split 完成目标约 1000 张图的 teacher/student calibration；当前 20 张 calibration 只允许作为 provisional 选择；
- [ ] 冻结首轮 `T_sem`：正式训练前对 SD1.5、PixArt-α、PixArt-Σ、Playground-v2.5 做 validation-only calibration 并选 best-single；SD1.5 FP32 null-text 仅作为已通过 smoke 的 fallback；
- [ ] 冻结正式 manifest：训练、calibration、held-out test、RefCOCOg/OOD 各自独立，禁止交叉调参；
- [ ] 冻结首轮配置：`lambda_sem=0.10`、`teacher_temperature=1.25`、warm-up 50 作为 provisional candidate，同时保留 semantic-off matched controls；该配置不是已证明的最优解；
- [ ] 在同一 manifest、同一图像预算、同一 evaluator 下完成 Qwen V0–V4 的至少 3 个训练 seed；V0 可只做一次冻结评测；
- [ ] 完成官方/可复核 evaluator 的 geometry + caption/VQA/OCR/hallucination 能力复核；当前 pilot 的内部 VQA/TextVQA scorer 不能作为最终主表；
- [ ] 生成一张最终 gate 表：`V3>V1`、`V4>V2`、correct>wrong/random、独立 test/OOD 趋势，以及能力不退化；未满足时只能报告“teacher signal 可用但 transfer 未建立”。

**最快可执行路径**：不等待 Prismatic、PixArt、Playground 或 SpikingBrain。先并行做 1000 图 calibration、Qwen full teacher cache 和 Qwen V0–V4 三 seed 训练；校准只决定正式主表所用的冻结配置，不阻塞工程管线准备。只有 cache、manifest 和配置 hash 固定后，才提交正式训练。

### B. VLA 主线：先做 OpenVLA/OFT + LIBERO pick-place

- [ ] 把 BlindVLA 官方 VLA 训练入口和配置审计到本仓库，记录 commit、数据格式、视觉 teacher、projector、optimizer 参数组和 action head；
- [ ] 完成 OpenVLA/OFT P1 interface audit：visual hidden states、action target、action loss、7D delta EEF adapter、action chunk/H=1 约定；
- [ ] 固定 LIBERO 标准任务中的一个 pick-place/put-object-into-container，固定 train/eval episode、随机 seed、图像视角、语言模板和 success evaluator；
- [ ] 用 B0 原生 BC/SFT 跑通 1 个 episode、20–50 step smoke 和 checkpoint restore；确认动作 shape `[7]`、旋转表示、夹爪符号、限幅和反归一化；
- [ ] 用同一数据和预算跑 B1 BlindVLA-style feature retention，确认它是 VLA 强 baseline，而不是把当前 VLM 的 V2 直接改名；
- [ ] 在 B0/B1 之上接入 `A_lang` 与 `A_act`：首版 `A_act` 使用 action loss 对 visual tokens 的 gradient×activation；没有 action query 时不能假设存在 attention；
- [ ] 只在 B1 跑通后加入 B2/B3/B4：`L_sem`、B1+`L_sem`、再加 D0 containment；每个组都保留 matched B0/B1；
- [ ] 增加 object-only、object+EEF、面积匹配 random mask、wrong-word/wrong-image 四类控制，避免把所有显著区域误认为动作相关区域；
- [ ] LIBERO 主机制通过后再接 LIBERO-Plus/LIBERO-PRO；DROID 只做真实数据离线泛化，不能用离线 action error 代替闭环 success；
- [ ] π0.5、MolmoAct2、LingBot-VLA、RoboTwin 和 `R_future` 作为跨架构/动态教师扩展，不阻塞 OpenVLA/OFT 首轮。

### C. 代码与可复现性硬门槛

- [ ] 把当前未提交修改整理成一个可审计 commit，再同步到 VEPFS `repo/VLA_attention`；服务器运行必须记录 commit SHA；
- [ ] 将 VLM/VLA 代码分层：`adapters/`、`teachers/`、`attribution/`、`losses/`、`trainers/`、`evaluation/`、`configs/`，模型仓库只允许通过 adapter 访问；
- [ ] 每个实验输出 `config.json`、Git SHA、model/teacher revision、manifest SHA、seed、step、日志路径、失败样本和 metrics；
- [x] 正式曲线字段已定义：总 loss、caption CE、retention、semantic loss、effective lambda、学习率、裁剪前后梯度范数、step time、吞吐、峰值显存、finite/cache 状态；训练结束生成 JSON 和 PNG 曲线；
- [ ] 在正式训练前各跑一次 20–50 step smoke、checkpoint restore、held-out evaluation 和负控；未通过不得提交长作业；
- [ ] 建立单独的 VLM 和 VLA 输出根目录，禁止覆盖历史结果；
- [ ] 每达到 10/25/50/75/100% 处理量，以及 cache、训练、评测、失败或配置变更，都更新本文件。

### 当前 Go/No-Go 结论

- **可以立即开始的工作**：Qwen 全量验证的 calibration/cache/manifest 准备；OpenVLA/OFT + LIBERO 环境和动作接口 smoke。
- **暂时不能声称完成的工作**：VLM idea 已证明、完整 BlindVLA 复现、VLA 机制有效、DROID 泛化成立。
- **首个正式实验提交条件**：Qwen 的 manifest/hash、四候选 teacher calibration 报告及 best-single config、三 seed 训练配置和官方 evaluator 固定；OpenVLA 的 B0 action smoke 与 LIBERO 单任务闭环通过。
- **不作为首轮阻塞项**：Prismatic/LLaVA/InternVL/Ovis 扩展、PixArt/Playground 教师比较、π0.5、MolmoAct2、LingBot-VLA、RoboTwin、world-model/refiner 和真机。

## 2026-09-24：VLM 全量验证模型与 protocol 对齐审计

本次对照了 `doc/03_execution_plan.md`、VLM 实验 configs、manifest 目录和已有结果。旧计划里仍残留“Prismatic+Qwen 两模型先跑”“四个扩散教师都做校准”“等待四教师 best-single”等表述，与当前“尽快让 Qwen 进入正式验证”的路径冲突；已在 03 文档标注当前执行口径，避免把候选清单误当作启动门槛。

### 其他 VLM 模型是否在计划中

- [x] Qwen2.5-VL-7B-Instruct：首轮全量验证骨干；现有 P1、V0–V4、能力评测配置和 pilot 结果。
- [x] LLaVA-OneVision、Prismatic-7B：被列为第二架构候选；Prismatic 有 P1 配置但 gated Llama-2 衍生权重访问尚未作为解决状态记录，LLaVA-OneVision 还没有可运行 P1/config。
- [x] InternVL3.5、Ovis2.5、Qwen3-VL、DeepSeek-VL2：存在于候选/后续普适性规划，但没有冻结的训练配置或 P1 证据。
- [ ] 最小跨架构主张：Qwen 完成后，必须选一个非 Qwen 骨干至少跑同协议的 V1/V3；如果只跑 Qwen，结论应限定为 Qwen 实例验证，不能称为跨 VLM 普适。
- [ ] 当前建议：Qwen 全量实验不等待第二骨干；同时审计 LLaVA-OneVision 与 Prismatic 的合法权重/接口，选一个作为第二模型。若 Prismatic 权重继续受 gated access 阻塞，优先转 LLaVA-OneVision。

### Qwen 正式验证尚待冻结的细节

- [ ] 数据：原始 Flickr30k/Entities 已归档并用于实验，但正式提交前仍需复核 archive、caption/XML/phrase/box 数量和 SHA256；写出 train/calibration/test/OOD 的不可变 manifest 与统一有效样本交集。
- [ ] 监督 target：推荐 caption CE 监督完整 caption；语义图只对 Entities 中可落到图像区域的 noun phrase spans 计算 `T_sem → A_lang`，以减少功能词/不可定位词噪声。需要冻结 phrase token 合并、词图 prompt/span 对齐和无效 phrase 过滤规则。Entity boxes 用于评测，不作为训练区域标签。
- [ ] 教师：推荐先正式校准并冻结 SD1.5 null-text；PixArt-α/Σ 与 Playground-v2.5 留作教师横向扩展，不阻塞首个 Qwen 全量结果。当前 64 条 cache 与小样本 pilot 不是全量教师 cache 或完整 1000 图校准。
- [ ] 训练：冻结 LoRA target/rank、LR、effective batch、step/epoch、checkpoint cadence、语义层、归因正负号与归一化；`lambda_sem=0.10`、temperature `1.25`、warm-up `50` 仍是 provisional 候选。正式 V1–V4 至少固定 3 seeds；V0 原权重仅需统一评测。
- [ ] 评测：Flickr30k Entities held-out phrase grounding（pointing、mass-in-box、IoU）作为主指标；至少固定一项官方 caption/capability evaluator 检查语言能力退化。建议 RefCOCOg 做外部泛化，但可排在首轮 Qwen 主结果之后；pilot 内部 scorer 不进入最终主表。
- [ ] 控制与判定：固定 wrong-word、wrong-image、random map 的生成/配对规则；预注册主指标、image-level bootstrap、能力非劣界和 Go/No-Go 规则，test/OOD 不参与 teacher 或超参选择。
- [ ] 复现：正式运行的 Git SHA、配置 SHA、model/teacher revision、manifest SHA、数据 cache 版本、seed、训练日志和失败样本须同步记录；VEPFS 上运行代码须与该 commit 完全一致。

**建议的最快启动口径**：Qwen-only 首轮，完整 Flickr30k caption train；四候选 teacher validation calibration 后选 best-single（SD1.5 仅作 fallback）；caption CE 对全部 token，语义对齐限于实体短语；Flickr Entities phrase grounding 为主评测；三 seed；wrong-word/wrong-image/random 控制；COCO Captions 官方 validation 做能力守门。第二模型做并行接口审计，但不阻塞 Qwen 长训练。当前没有新训练任务由本次审计启动。

## 2026-09-24：校准损失比较与下游能力 pilot 当前结论

- [x] 明确 VLA benchmark 路线：标准 LIBERO 用于接口 smoke 与首轮闭环机制；LIBERO-Plus 用于 task-preserving 视觉/状态 OOD；LIBERO-PRO 用于语言/指令泛化；DROID 用于按 episode/scene 分组的离线真实数据泛化；RoboTwin 纳入后期操纵/双臂或 embodiment 扩展，不阻塞首轮。当前只有 benchmark 规划，不能据此声称已下载或跑过这些实验。分阶段角色和建议顺序见 `doc/03_execution_plan.md`「VLA benchmark 纳入范围与阶段」。
- [x] 完成 teacher/student map 共网格与阈值校准审计：独立 calibration 目前只有 20 张图，联合选择结果为 16×16、q=.95；数据记录在 `experiments/F0v2-idea-validation/geometry_loss_sweep_results/calibrated16_20260923/calibration_joint_selection_20260923.json`。该设置是 provisional，不能替代原计划的 1000 张 student attribution calibration。
- [x] 修正版几何实验已按 image 聚类 bootstrap：V3 四种 loss 的 common soft-IoU 增量均为正，但 calibrated box IoU 区间均跨零；V4 KL+rank 对 V2 的 box-IoU 区间也跨零。修正版三 seed 几何结果与限制见 `experiments/F0v2-idea-validation/geometry_loss_sweep_results/calibrated16_20260923/cluster_bootstrap_update_20260923.md`。这些证据不支持宣布唯一最佳 loss 或“idea 已被证明”。
- [x] 修正版 Lavender-style 能力 pilot 已完成并审计：19 个 adapter × 六项任务的 114 份报告齐全；完成 90 个同训练 seed 配对比较（V3 四种 loss 对匹配 V1、V4 KL+rank 对匹配 V2），各有 10,000 次配对 bootstrap。对齐检查包括 manifest、ordered ID/prompt、greedy decoding、图像预算和样本数。机器可读结果及全表：`experiment_workspace/results/F0v2_corrected_capability_20260923/paired_capability_sweep_20260924.json`、`paired_capability_sweep_20260924.md`。
- [x] 当前能力 pilot 的描述性信号：V3 各损失的 COCO CIDEr 三 seed 均为正增量（平均约 +.080 至 +.097）；VQA/TextVQA 的均值看似有正增量，但 seed 间差异很大，且多项单 checkpoint 样本区间跨零。V4 KL+rank 对 matched V2 的 CIDEr、WorldMedQA-V 均值三 seed 为正，但 POPE、TextVQA、VQA 的 seed 方向不一致。完整数值以配对表为准，不根据均值挑选性宣称广泛能力提升。
- [x] 能力数据的解释边界已写入报告：COCO 仅 128 张 pilot 图，使用 CIDEr、ROUGE-L paired intervals 与 corpus BLEU point estimates；METEOR 因评测环境缺失而省略。VQAv2/TextVQA 使用内部 pilot consensus scorer，WorldMedQA-V 为 256 条 pilot，MME 为 64 个完整 pair，POPE 的 bootstrap 单位是 128 张图片。区间衡量固定 checkpoint 下的样本不确定性；三个训练 seed 的均值/SD仅作描述，不能当作稳健的训练总体推断，也不是完整官方 benchmark 结果。
- [ ] 下一步先做较大且冻结的独立 calibration（目标 1000 张）并确认正确教师/错词/错图/随机图 controls；之后用官方 evaluator 和完整冻结 split 复核下游能力。只有几何主指标和能力指标都在独立数据与多个 seed 上支持同一结论，才可升级为 idea 有效的主张。

## 2026-09-23：修正版几何比较已完成，能力评测重启中（当前有效状态）

> 状态快照已由 2026-09-24 本节更新；以下仍保留当日记录作为运行历史，所有“正在运行/报告不齐”的状态均已过期。

- [x] 已完成 teacher/student map 共同分辨率、top-k、soft-IoU 与阈值校准；独立 validation 的联合 `(resolution, threshold)` 选择仍得到 16×16 网格和 q=.95 阈值。联合重算记录在 `geometry_loss_sweep_results/calibrated16_20260923/calibration_joint_selection_20260923.json`；校准仅有 20 张图，仍标为 provisional，不能把 `teacher_calibration_val_1000.jsonl` 当成已完成的 1000 张 student 校准。
- [x] V3 的 KL、KL+rank、KL+moment、JS 各 3 个 seed（17/29/41），以及 V4 KL+rank 的 3 个 seed，均完成 optimizer-only gradient clipping 修正后的 100-step 训练与 image-disjoint 64 样本 held-out 评测。完整表见 `experiment_workspace/results/F0v2_geometry16_fixclip_heldout_20260923/summary.md`；逐模型 `report.json`、10,000 次 paired bootstrap JSON 在其 `reports/` 子目录。
- [x] 修正版 V3 KL+rank vs 匹配 V1 的 calibrated IoU Δ=+.00569，95% paired bootstrap CI [-.00941,+.02060]；soft-IoU Δ=+.01299 [+.00740,+.01888]。V4 KL+rank vs 匹配 V2：calibrated IoU Δ=+.00669 [-.00851,+.02211]，soft-IoU Δ=+.00861 [+.00294,+.01448]。其他 loss 对比也呈现“soft-IoU 多为正、校准 IoU 区间跨零”的图景，不支持单一损失全面胜出；不可据此宣称 idea 已被证明。
- [x] 修正版三 seed 几何均值±样本标准差：V3 JS（IoU .2938±.0073、soft-IoU .0804±.0022、top-20 IoU .2736±.0038、mass-in-box .3300±.0003、pointing .3438±.0312）；KL（.2936±.0052、.0783±.0096、.2709±.0049、.3303±.0007、.3958±.0705）；KL+moment（.2927±.0055、.0779±.0064、.2728±.0016、.3319±.0095、.3854±.0651）；KL+rank（.2892±.0040、.0804±.0061、.2714±.0021、.3362±.0046、.3698±.0325）；V4 KL+rank（.2916±.0135、.0806±.0077、.2691±.0018、.3346±.0162、.3802±.0451）。指标间排序不一致，需以完整报告为准，不按单项挑选“最佳 loss”。
- [x] 旧 global-clipping 实现训练的 V3/V4 几何结果和能力 pilot 已标记为历史诊断，不纳入当前方法结论。V0/V1/V2 旧 pilot 可用于基线描述；MME 横向比较只用正确 `mme_pair` manifest（SHA256 `618989cdf66949f1b462df3ce3f91c6cc7d945ee6865109e84d28ad38f0360ab`）。
- [x] 修正版 Lavender-style 下游能力评测已完成；114/114 报告和 90 个同 seed 配对已通过 manifest、样本、解码与图像预算审计。当前结果和限制见本文件 2026-09-24 节及 `results/F0v2_corrected_capability_20260923/paired_capability_sweep_20260924.md`。
- [x] 前一阶段运行快照（26/114，已由本节后续 44/114 核验覆盖）：V1/V2 seed29 与 seed41 已完成六项任务，V3 KL seed17 已完成 COCO、VQAv2。V2 seed41 六项结果已回收到本地：COCO CIDEr .772、VQAv2 pilot consensus .763、TextVQA pilot consensus .673、POPE accuracy .849、MME pair accuracy .719（64 对）、WorldMedQA-V accuracy .426。VQA/TextVQA 是内部 pilot scorer，不是官方 evaluator。
- [x] 该续记已由 2026-09-24 的 114/114 完成状态取代；当时记录的 V1/V2 seed41 单 checkpoint 数值仍可作为原始报告描述，但不代表最终配对结论。
- [x] 26/114 阶段续跑快照（已被后续状态覆盖）：当时同一远端作业仍在评测历史 geom16 KL seed17 的 V3 TextVQA；V2 seed41 六项已完成。本记录仅保留作业阶段顺序，当前报告数和执行任务以本节后续核验为准。
- [ ] 最新续跑核验：远端报告数已增至 **63/114**，同一进程仍存活，当前正在评测 V3 KL+moment seed17 的 POPE。V3 KL seeds17/29/41 六项任务均已完成并回收到本地；其他 V3 loss 与 V4 KL+rank 的 seed 评测仍排队。该计数来自远端 `find ... -name report.json`，不是本地缓存数。
- [x] V3 KL 三 seed 下游中期分析已完成，18 个 V1/V3 task pairs 的 manifest SHA、ordered ID/prompt SHA、greedy decoding 和 image budget 全部匹配。CIDEr 与内部 TextVQA pilot 指标三 seed 均上升；VQAv2 方向随 seed 翻转，POPE/WorldMedQA-V 增量较小，MME 仅 seed29 上升。每个 seed 的样本区间单独记录于 `experiments/F0v2-idea-validation/geometry_loss_sweep_results/calibrated16_20260923/capability_interim_seed_analysis.md`；不能以此声称总体能力改善。
- [x] 下游不确定性重采样已改为按原图聚类；审计确认 POPE 的 384 个问题只有 128 张独立图片，逐问题 bootstrap 会把同图的三个问题误当独立样本。MME 仍以完整 64 个 question pair 为单位（每对含两张图）。修正位于 `scripts/paired_capability_bootstrap.py`，回归测试 3 项通过；matched V1/V2 seed29 的 POPE 与 MME 已以新单位重算，结果写入 `partial_paired_diagnostic_imagecluster/`。
- [x] 新增 `scripts/summarize_capability_sweep.py`，强制检查 19 个 adapter × 6 个任务的报告完整性、greedy 解码、64 token 上限和统一图像像素预算；当前本地只回收到 42 份，不能把本地部分结果误报为远端完整结果。
- [x] 几何 bootstrap 也已改为按 `image_id` 聚合后重采样；64 条 phrase 中实际只有 62 张独立图，V3 对 V1 的新三 seed区间见 `results/F0v2_geometry16_fixclip_heldout_20260923/cluster_bootstrap/paired_v3_vs_v1.json`。KL、KL+rank、KL+moment、JS 的 common soft-IoU 增量均为正且区间不跨零，但 calibrated box IoU 区间均跨零；因此不能据 soft-IoU 单项选 loss。V4 对 V2 的对应结果在 `cluster_bootstrap/paired_v4_vs_v2.json`，同样不能宣称 box IoU 已建立稳定增益。详细表见 `geometry_loss_sweep_results/calibrated16_20260923/cluster_bootstrap_update_20260923.md`。
- [ ] 初步配对诊断（matched seed29、同一 manifest SHA 和 ordered prompt/ID SHA、相同 greedy 解码与图像预算）显示 V2 retention 相对 V1 的 VQAv2 pilot consensus Δ=+.0695，95% 样本配对 bootstrap CI [+.0117,+.1297]；TextVQA Δ=+.0203，CI [-.0570,+.0969]；POPE Δ=+.0130，CI [0,+.0286]；MME pair accuracy Δ=+.0781，64-pair CI [-.0156,+.1875]。WorldMedQA-V 使用修正后的解析器重算，V2−V1 Δ=-.02344，95% paired CI [-.06641,+.01563]。各 bootstrap JSON 在 `experiment_workspace/results/F0v2_corrected_capability_20260923/partial_paired_diagnostic/`。这些仅是单训练 seed 的早期诊断，VQA/TextVQA scorer 为内部 pilot 近似值；不能用来声称 V2 通用能力显著变好，更不能作为语义图方法的增益证据。需等待 seed41 与其余任务结果，并优先用官方 benchmark evaluator 复核。
- [ ] 发现并修复选择题输出解析缺陷：WorldMedQA-V 原始输出常为 `B .`，旧解析器未接受字母与句点之间的空格，导致无效答案被计错。已在 `scripts/eval_capability_suite.py` 扩展空格标点及重复字母处理，补充测试 `B .`、`C C .`、`C\nC`；`python3 -m pytest -q tests/test_capability_eval.py` 通过（10 passed）。修复版已复制到 vla101，SHA256 本地/远端均为 `d0c051fe5ee9386f3bcbdc7e36d5e28014ab6af57ede0335c9d8d66dca5c4cc2`；当前 sweep 后续任务将使用新解析器。已离线重算 seed29 原始 raw records，V1 WorldMedQA-V accuracy 从 .453125 更正为 .484375，V2 从 .226563 更正为 .460938；matched V2−V1 Δ=-.02344，95% paired CI [-.06641,+.01563]。原始报告保留不覆盖，更正结果及说明另存 `partial_paired_diagnostic/{V1_seed29,V2_seed29}_worldmedqa_rescored.json`；之前 worldmedqa bootstrap 文件已用新解析器重写。其余已完成旧版解析任务须在 sweep 结束后检查是否受同一 bug 影响并从 raw records 统一重算。
- [ ] 早期信号需单独审计：V1 matched seed29 的 COCO/VQAv2/TextVQA pilot 指标分别为 CIDEr .768、VQA consensus .149、TextVQA consensus .448；历史 seed17 报告对应为 .778、.702、.660。seed29 的部分 VQA 原始回答偏长、格式不贴合“short answer”，因此先记作训练 seed/输出格式敏感性风险，待 seed41 与其余 matched baselines 完成后，逐 ID 对照 prompt、预测和训练日志，不能拿内部 pilot 分数当官方评估结论。

## 2026-09-23：能力评测推进续记（当前状态，以本节覆盖下文旧的“未开始/运行中”记录）

- [x] 16×16 几何损失 comparison 与三 seed paired held-out bootstrap 已落盘；结论以 `geometry_loss_sweep_results/calibrated16_20260923/summary_calibrated16.md` 为准：KL+rank 对 soft-IoU、top-10 IoU、mass-in-box 有正向信号，但 q=.95 box IoU 置信区间跨 0；只作为下游 pilot 候选，不宣称已胜出。
- [x] 2026-09-23 capability manifests 均已从固定 HF revision 构建并校验；五个模型 V0/V1/V2/V3_KLrank/V4_KLrank × 六任务 runner 正在 101 GPU1 顺序执行，输出根 `results/P4_capability_pilot_20260923_v2`，Qwen `max_image_pixels=1003520` 固定，便于断点续跑。
- [x] 远端 pilot 当前已完整生成 V0、V1、V2 各六个 `report.json`，合计 18 份；V3_KLrank 的 COCO Captions 已完成，V3 VQAv2 正在处理 128 条样本；V3 其余任务与 V4 六任务尚待执行。作业句柄为本地 shell session 92308，持续跟踪中。
- [x] 几何训练候选 `V3_KLrank` / `V4_KLrank` 均使用 seed17、100-step checkpoint；下游评测是单 checkpoint pilot，不能将样本 bootstrap 解读为训练 seed 不确定性。
- [x] 修复 MME 成对评测 manifest 覆盖问题：重新从服务器拉取 paired manifest 和 sidecar；manifest SHA256=`618989cdf66949f1b462df3ce3f91c6cc7d945ee6865109e84d28ad38f0360ab`，128 条 question rows、64 个 complete pairs、task=`mme_pair`，ordered ID/prompt SHA256=`83163890b0116611b49937ec38a214db3a2181ac510c32d102ff7a0229b5dd7e`。原 V0/V2 MME pilot 报告 schema 为 binary，V1 为 pair；三个都不得混比，主 pilot 完成后须用此 manifest 对 V0–V4 统一补跑 MME。
- [x] 能力评测/manifest/builder 单测复跑通过：`python3 -m pytest -q tests/test_capability_eval.py tests/test_capability_manifest.py tests/test_hf_capability_manifest_builder.py`，16 passed。
- [ ] 主 pilot 五模型任务全部结束后拉取并核验报告及 run metadata，检查所有任务的 manifest SHA、图像预算和样本数；随后对 V0–V4 统一跑 64-pair MME，并生成配对 bootstrap 和跨任务结果摘要。
- [ ] 评测摘要及指标解释写入 `experiment_workspace/PROGRESS.md` 与 `doc/03_execution_plan.md`；检查并同步 Git 到服务器前先保留用户的 `server/00_connection.md`，不触碰该文件。

## 2026-09-23：capability pilot 完成与审计

- [x] 远端主 pilot 结束，`P4_CAPABILITY_PILOT_OK`：V0/V1/V2/V3_KLrank/V4_KLrank × COCO Captions 128、VQAv2 128、TextVQA 128、POPE 384、MME 128、WorldMedQA-V 256 均有报告；共同 `max_image_pixels=1,003,520`、greedy decoding、max-new-tokens=64。101 原始报告在 `results/P4_capability_pilot_20260923_v2/`，已拉取本地 `experiment_workspace/results/P4_capability_pilot_20260923_v2/`。
- [x] 核验 V0–V4 五模型每个非 MME task 的 manifest SHA、ordered ID/prompt SHA、sample count 和图像预算相同；各模型共 30 份完整主 pilot 报告。
- [x] MME 不一致问题已修复：主 pilot 初版 V0/V2 为 binary，丢弃这两份用于比较；用正确 pair schema 将 V0–V4 全部重跑完成，报告在 `results/P4_capability_pilot_mme_pairedfix_20260923/`，manifest SHA=`618989cdf66949f1b462df3e3f91c6cc7d945ee6865109e84d28ad38f0360ab`，128 rows / 64 complete pairs。paired MME 是唯一用于横向表格的结果。
- [x] 10,000-resample paired bootstrap 已完成：V3 vs V1、V4 vs V1，且针对关键互补性检验 V4 vs V2；MME 按 64 pair 为 resampling unit。结果文件：`experiments/F0v2-idea-validation/geometry_loss_sweep_results/calibrated16_20260923/paired_capability_vs_v1.json`、`paired_capability_v4_vs_v2.json`。Intervals 条件于已选 checkpoint，不含训练 seed uncertainty。
- [x] 能力试点摘要与限制写入 `experiments/F0v2-idea-validation/geometry_loss_sweep_results/calibrated16_20260923/capability_pilot_summary.md`；原始报告、详表在 `experiment_workspace/results/P4_capability_pilot_20260923_v2/`（该 results 路径被 `.gitignore` 排除，summary/bootstrap 核心证据另存 tracked 的 experiments 目录）。
- [x] 初步结论：V3 COCO CIDEr `.891` 高于 V1 `.778`，POPE `.852` vs `.841`，MME pair accuracy `.734` vs `.703`；但 V3 VQAv2 consensus `.159` vs `.702`，TextVQA `.367` vs `.660`，WorldMedQA-V `.406` vs `.469`。V3 TextVQA raw output 可见长串重复标点；这提示可用性/生成稳定性风险，需审计训练/推理，不能宣称整体能力提升。
- [x] V4 对比 V2：VQAv2 `+.030 [-.026,+.088]`，TextVQA `-.002 [-.053,+.051]`，POPE `0.000 [0,0]`，MME pair `+.016 [-.047,+.094]`，WorldMedQA-V `+.020 [-.027,+.063]`；未建立语义图约束在 DINO retention 上的下游能力增量。
- [ ] 这些是小样本、单 checkpoint 的 preliminary pilot，不等于 Lavender 复现或正式 benchmark：VQA/TextVQA 评分不是官方 evaluator，WorldMedQA-V 仅 256 个分层 pilot，caption 没有 METEOR（缺 Java）。下一步先查 V3 VQA/TextVQA 重复生成和可能的训练退化，再跑至少 3-seed 与完整官方 evaluator；pilot 通过这些 gate 前不做论文强 claim。
- [x] 更新 `doc/03_execution_plan.md` P4.1 状态与 pilot 结果；能力/manifest/builder tests：16 passed。

## 2026-09-23：训练梯度裁剪缺陷审计与修正版 sweep（当前有效状态）

- [x] 复查 capability 原始记录确认 V3 在 VQAv2/TextVQA 有真实重复生成；例如 TextVQA 样本 `37678` 的 V3 输出为 `atomic` 后重复大量句点，V1/V2/V4 同 ID 输出稳定。因此不是内部评分公式单独造成的差异。
- [x] 追溯几何 sweep 训练代码，发现 V3/V4 需要视觉主干梯度计算输入 attribution，但视觉参数并非 optimizer 参数；旧训练却用 `clip_grad_norm_(model.parameters(), 1.0)` 把 attribution-only 视觉参数纳入 global norm。种子17旧 KL+rank V3 的记录显示平均 pre-clip norm=11.98（max 88.94，29/100 steps >10），裁剪后固定为1；其中大量视觉梯度不更新权重，却共同缩小实际 LoRA 更新。故该实现改变了优化器有效步长。
- [x] 修正 `run_qwen_v3_smoke.py` 与 `run_qwen_v4_smoke.py`：通过 `autograd.grad(loss, optimizer_parameters)` 只取得会被优化参数的梯度，并仅对这些参数裁剪；loss 所需视觉计算图仍被保留。输出 JSON 增加 clip scope 与真实 optimizer gradient norms。V3、V4 的 2-step 修复后 forward/backward/checkpoint smoke 均成功，V3/V4 的 effective semantic lambda 按 100-step warmup 从 .001/.002 开始。
- [x] 修正版 runner 已同步至 vla101，`upload_validation_scripts.sh` SHA256 检验通过。
- [x] 已启动新训练：`CUDA_VISIBLE_DEVICES=1 F0_FIXCLIP_STEPS=100 F0_FIXCLIP_SEEDS=17,29,41 bash server/run_f0_gradientclip_geometry_sweep.sh`，本地执行 session 35824。KL、KL+rank、KL+moment、JS 的 12 个修正版 V3 seeds 已完成；V4 KL+rank seed17 已完成，V4 seeds29/41 仍在该 shell session 内顺序执行。远端目录 `checkpoints/F0v2_geometry/fixclip_*` 和 `results/F0v2_geometry/fixclip_*`；输出与旧 `geom16_*` 完全分离。之后还需匹配训练 V2 seeds29/41，才能对 V4 做 3-seed 对照。
- [x] 修正版 V3 四损失各三 seed 与 V4 KL+rank 三 seed 均于 session 35824 正常完成，远端返回 `FIXCLIP_GEOMETRY_SWEEP_TRAIN_OK`。另外按同 manifest/100 steps/seed29,41 训练匹配 V2 retention controls，session 86044 返回 `FIXCLIP_MATCHED_V2_OK`；V2 seed17 使用原有 matched 100-step checkpoint。
- [ ] 修正版 held-out evaluation 与 paired bootstrap 正在 session 15993 执行；脚本已使用新的 `fixclip_` 产物名，并准备输出四个 V3 loss vs V1 与 V4 KL+rank vs V2 的三 seed 比较。完成后再拉回 summary 和结果 JSON 核验。
- [ ] 旧 `geom16_*` 四损失 × 三 seed 几何 sweep 与基于它训练的 V3/V4 capability pilot 均受上述实现缺陷影响，只作为历史诊断，不能作为当前训练有效性证据。校准（分辨率/top-k/soft-IoU/阈值）仍可复用，因为它不依赖被训练的 adapter；但校准后的 3-seed loss ranking 必须等修正版评估后再报告。
- [ ] 修正版训练完成后，用同一 F0v2 image-disjoint 64 held-out、冻结 16×16/q=.95 重新评四种 V3 loss、匹配 V1 seeds、V4 KL+rank 与 matched V2 seeds，生成 paired bootstrap。之后才复跑修正版候选 capability pilot；扩充至少三训练 seed 的能力评测属于必要后续验证。
- [x] 上传/记录脚本已增加 `server/run_f0_gradientclip_geometry_sweep.sh`、`server/eval_f0_gradientclip_geometry_sweep.sh` 和 `server/run_f0_gradientclip_v2_matched.sh`；训练匹配 V2 seeds29/41 及其 checkpoint 建立完成前，不运行 V4 三 seed paired eval。

## 2026-09-23：geometry-loss sweep 续记

- [x] 2026-09-23 matched V1 baseline seed29/41 已各训练 100 steps，远端 `MATCHED_V1_SEEDS_OK`；checkpoint 位于 `checkpoints/F0v2_geometry/matched_v1_seed{29,41}_s100`。
- [ ] 2026-09-23 101 访问恢复后的继续实验：GPU1 经 `nvidia-smi` 确认空闲（81 GiB free），F0v2 phrase teacher cache 256/256，训练 manifest SHA256=`2a37c49e…`、held-out SHA256=`7b16778e…`。正在运行严格 16×16 common-grid 的四种 semantic loss × seeds 17/29/41，每组 100 steps、λ=.1、T=1.25、rank q=.25；本地 shell session 57204，当前 KL seeds17/29 checkpoint 已保存，seed41 进程 PID 2634664 运行中。12 组完成前不将其称为完整比较结果。

- [x] 远端复核确认：101 GPU1 空闲；四种 map loss 各 3 seeds 的 12 个 100-step checkpoint/训练报告已完成，held-out 评估报告已在 VEPFS 生成；原训练实际使用共同 32×32 网格。
- [x] 独立 val calibration：20 张 `p1_qwen25_calibration_20.jsonl`，对 V3 student 在 16/32/64 网格上比较 top-k 与 soft-IoU，暂定 16×16 / q=.95；不使用 test 选参数。摘要与分辨率 caveat 见 `experiments/F0v2-idea-validation/geometry_loss_sweep_results/`。
- [x] 几何损失比较（三 seed，64 image-disjoint held-out）：KL、KL+rank、KL+moment、JS 已完成；16×16/q=.95 calibrated IoU 均值依次为 0.2838、0.2758、0.2893、0.2812；soft-IoU 均值依次为 0.0748、0.0799、0.0790、0.0711。首轮是 32×32 训练网格，不能作为最终选择。
- [x] V1/V2 matched baseline 只读重评已在 GPU1 完成且 JSON 已落盘；在远端审批服务出现 502 之前只确认到文件存在及部分终端摘要，完整 summary 尚未写入本地，因此此处不记录数值结论。
- [ ] 必须先完整读取 V1/V2 JSON 并计算 paired bootstrap；随后按冻结 16×16 重跑四个 loss × seeds 17/29/41，再评估 matched V1/V2。
- [ ] Lavender-style 六项下游能力评测尚未开始；按计划在几何候选选择和 matched baseline 统计审计后运行 V0–V4 paired pilot：COCO Captions、VQAv2、TextVQA、POPE、MME、WorldMedQA-V。
- [ ] 远端审批服务当前对部分 SSH 只读请求返回 502；未因审批超时而假设命令执行成功，也未重复启动任何训练/评测 job。
- [ ] 2026-09-23 下游 capability pilot 首轮 V0 在 COCO/VQAv2/TextVQA/POPE 推理完成；caption 结果因缺 `java` 首次汇总失败，修复为记录 METEOR unavailable 后已恢复并得到 V0 COCO 指标。随后 MME 某张高分辨率图触发单样本 OOM，MME 前的任务结果有保留。已将全体 benchmark 的 Qwen `max_image_pixels=1,003,520` 固定为同一预处理，改用 `_v2` 输出命名空间并重跑，避免混合 OOM 前后的预算。capability runner 的恢复 identity 已纳入像素预算；METEOR 缺 Java 会降级为 partial caption metrics，并显式记录 scorer_notes。
- [x] 2026-09-23 capability pilot V0 `_v2` 六项任务均完成：COCO 128、VQAv2 sample 128、TextVQA 128、POPE 三个子集共 384、MME 128、WorldMedQA-V 256。Pilot 输出根为 `results/P4_capability_pilot_20260923_v2/V0/`。此 V0 MME 首次报告仍用了旧 binary schema；需以冻结的 paired-MME manifest 单独补跑，不能与 V1 后续使用新 pair schema 的报告直接比较。
- [ ] 2026-09-23 capability pilot V0 与 V1 的首批数值已回收：同一固定 pilot IDs、Qwen image max-pixels=1,003,520。V0/V1 分别为 COCO CIDEr `.0952/.7779`（两者均缺 METEOR）、VQAv2 pilot consensus `.2828/.7016`、TextVQA pilot consensus `.3188/.6602`、POPE accuracy `.8490/.8411`、WorldMedQA-V accuracy `.4180/.4688`。V1 paired-MME report 使用正确 `mme_pair` schema，64 pairs pair-accuracy `.7031`；V0 MME 报告仍来自旧 binary schema，不能横向比较，待重评。以上是固定小样本 pilot 描述，不是正式官方分数；当前 V1 后续任务已完成，V2 正在运行，完整五 checkpoint 六任务仍在进行。
- [x] Lavender-style capability runner 已本地实现：`scripts/eval_capability_suite.py` 统一 Qwen V0–V4 greedy inference，保留逐样本预测并支持 caption/VQA/POPE/多选/MME-pair task schema；`scripts/validate_capability_manifest.py` 校验官方 ID、字段、图像存在性并生成 split/revision、manifest SHA256 与有序 ID/prompt SHA256 sidecar。VQA 共识改为 leave-one-annotator-out 公式，但答案规范化仍非官方实现。`tests/test_capability_eval.py` 与 `tests/test_capability_manifest.py` 本地 7 项测试全部通过；实现就绪不表示六项 benchmark 数据已准备或结果已运行。
- [x] 为能力评测增加可恢复执行：每 10 条原子写入 partial 结果；resume 前核验 manifest SHA、样本顺序、model/revision、adapter 和生成长度；最终报告记录样本耗时、吞吐及 CUDA 峰值显存。schema 已写明使用方式，partial 产物不作为正式 benchmark 结果。
- [ ] 2026-09-23 远端只读连通性重试：`ssh vla101 'echo SSH_OK'` 的沙盒外审批被自动审批服务拒绝，返回 upstream `502 Bad Gateway`。未尝试其他通道；geometry loss 16×16 matched-grid 重跑与能力评测数据准备仍不能在本地记录为完成。

本记录与 `doc/03_execution_plan.md` 的 P1 章节保持一致；这里保留实际运行环境和产物路径，避免把服务器状态混入论文叙事。

## 运行中检查点格式

每次处理量达到 10/25/50/75/100%，以及每次 smoke、checkpoint、失败或配置变更后，新增如下条目：

```text
时间 / experiment ID / Git commit / 101 PID 或 job ID
输入 manifest + SHA256 / total-success-failed
model + teacher + revision + seed + steps
partial metrics / VEPFS result + log 路径
是否 non-final / 下一步 / 是否允许扩大
```

所有 partial 数字必须显式标 `non-final`；处理进度、失败集合和有效样本交集变化时，同步更新对应 protocol。

## 2026-09-17

- [x] 101 直连：`root@115.190.90.101:27219` 可进行无交互 SSH；连接凭据不写入本文件。
- [x] 本机稳定入口：`vla101` SSH alias 固定地址/端口并启用严格 host-key 校验；项目受限操作由 `server/vla101.sh` 执行。
- [x] VEPFS 项目根已建立：`/vepfs-mlp2/c20250405/400040/transfer/vla_attention/`。
- [x] 建立空目录：`repo`、`envs`、`hf_cache`、`models`、`data`、`teacher_maps`、`runs`、`checkpoints`、`results`、`jobs`。
- [x] 存储检查：VEPFS 有约 620 TB 可用；101 `/root` 仅约 207 MB 可用，因此禁止将 cache、虚拟环境、模型或数据写到 `/root`。
- [x] 依赖基线：`/root/starvla_cu124/bin/python` 可发现 torch、transformers、huggingface_hub、datasets、diffusers、accelerate、peft。
- [x] 在 VEPFS 创建 `envs/p1` 并生成 `envs/p1/versions.txt`；环境只读继承共享包，不修改 `/root/starvla_cu124`。
- [x] Qwen checkpoint 配置审计：`Qwen2_5_VLForConditionalGeneration`，vision depth 32、`fullatt_block_indexes=[7,15,23,31]`、patch size 14、`spatial_merge_size=2`、window size 112；运行环境为 Transformers 4.57.1，`qwen_vl_utils` 已存在。P1 必须记录 post-merge token 网格，不可仅据 token 数猜方格。
- [x] DINOv2 ViT-L/14 retention teacher：`facebook/dinov2-large@47b73eefe95e8d44ec3623f8890bd894b6ea2d6c` 已下载、SHA256 通过（约 2.3GB；Apache-2.0、非 gated）。GPU1 单图 smoke 成功：224×224、`last_hidden_state=(1,257,1024)`、patch size 14；`use_fast=False` 将在正式 runner 固定。DINO 16×16 patch 与 Qwen 动态 post-merge grid 通过 normalized-coordinate bridge，不直接按 token index 对齐。
- [ ] 将当前 Git commit 传到 VEPFS `repo/VLA_attention`，记录 commit 与文件 SHA256。
- [ ] 在 VEPFS 创建 `envs/p1`；只读继承共享依赖或按锁定 requirements 安装，绝不修改 `/root/starvla_cu124`。
- [ ] 记录网络和代理检测；下载使用 7897 代理，若服务器无该代理则先记录失败后再选已验证镜像。
- [ ] 下载并核验 Qwen2.5-VL-7B、Flickr30k Images、Flickr30k Entities。2026-09-17 Qwen 已下载到 VEPFS（16 个文件、约 16 GB；`Qwen2_5_VLForConditionalGeneration`，checkpoint 声明 Transformers 4.41.2）。下载使用 `hf-mirror.com`，原因是 101 对 Hugging Face 直连超时；`SOURCE.txt` 已记录模型 repository 与实际端点。首次 SHA256 自校验发现清单把自身纳入输入，产生一个预期的自引用 mismatch；脚本已修复为排除 `SHA256SUMS`，待重建清单后才将 Qwen 勾为已核验。
- [ ] 审计 `nlphuji/flickr30k` 的实际文件与 revision，并探测 Flickr30k Entities 的原始 annotation archive；必须确认 image 与 phrase-region 标注可对齐，才建立全量 manifest。
- [x] 数据源审计：`nlphuji/flickr30k` 提供完整 image zip 和 caption CSV；原作者 `BryanPlummer/flickr30k_entities` 提供 `annotations.zip`（约 29 MB）。原作者 README 明确该 archive 含 `Sentences/` 的 phrase/coreference chain 和 `Annotations/` 的 XML boxes，且含 train/val/test split；图片遵循 Flickr Terms，仅非商业研究/教学使用。旧 Plummer 网页与 Oxford VGG archive URL 均为 404，故不使用它们。
- [ ] 运行 `download-flickr`，验 archive、解压、SHA256 与 JPEG/Sentences/XML 文件计数；通过前不建立 split 或运行 P1。
- [x] Flickr 图片与 caption 下载：完整图片 archive 和 `flickr_annotations_30k.csv` 已从 `hf-mirror.com` 下载到 VEPFS。原作者 annotation 的 `raw.githubusercontent.com` 直传在 120 秒内仅约 1 MB，随后 shallow Git clone 约 15 分钟仅到 16 MB，均不适合作为可靠传输。GitHub API blob 的 JSON 传输也在约 458KB 中断；已验证 API raw media 支持该 exact blob 的 ZIP Range 读取（返回 `PK` magic），故改用 4MB、字节校验、可重试的 versioned range 下载。最终 `SOURCE.txt` 会记录 transport、commit、blob SHA 与 SHA256；不是不明第三方镜像。
- [ ] Entities archive 当前为**部分下载**：第一个 4MB range 已通过长度检查；第二段在 HTTP/2 `CANCEL` 后未通过，临时段未被追加。下载器改为从已验证 archive 大小续传，使用 1MB + HTTP/1.1 + 每段字节检查 + 最多五次重试。`SOURCE.txt` 在 archive 完整、`unzip -t` 通过前不得作为完整来源证据。
- [x] 代理探测：`ghproxy.net` 与 `gh-proxy.com` 均能返回原作者精确 revision 的 `annotations.zip` ZIP magic；`mirror.ghproxy.com` 超时。为避免代理替换内容，下载器会从 `ghproxy.net` 取得 candidate 文件，并计算 Git `blob <size>\\0<payload>` SHA1，只有等于官方 API 的 blob `95513f3315f98068d0aeb558649ebe09fbeffe95` 且 `unzip -t` 通过时，才替换部分 archive。
- [x] 镜像传输修复：发现 `curl --retry` 会在连接重置后覆盖 candidate；已停止唯一受影响的本项目任务，保留当时约 1.18MB candidate，并改为 `--continue-at -`。后续重试从已有字节续传；最终仍以官方 Git blob SHA 和 ZIP test 为唯一准入条件。
- [ ] 建立全量 manifest，随后从 calibration split 固定 P1 的 20 个 sample ID。
- [x] Manifest 规则冻结：原作者 `train.txt` 完整用于训练、`val.txt` 用于独立 teacher calibration、`test.txt` 用于最终 grounding；`build_flickr_entities_manifest.py` 将从 val 确定性抽取 20 张 P1 图并为其保存 image SHA256。该规则不从 train 偷取 calibration 样本，也不让 P1 缩小正式实验规模。
- [x] Qwen P1 report：2026-09-17 在 GPU1 对固定 20 个 val 样本运行真实 teacher-forced phrase-score gradient×activation 审计；`validate_p1_report.py` 通过，`PASS qwen2.5-vl-7b@Qwen2.5-VL-7B-Instruct: 20 measurements / 20 samples`。report/map 位于 VEPFS `results/P1-interface-audit/qwen2.5-vl-7b/`；此通过只允许进入 DINO/teacher calibration，不等于 V0--V4 训练已完成。
- [x] Qwen P1 前向 smoke（GPU1）：冻结 P1 manifest 首样本 `1321949151:c0:p0:e8556` 成功加载图像与 Qwen，`image_grid_thw=[[1,34,36]]`，logits shape `(1,338,152064)`，返回 29 个 hidden-state 层。Transformers 提示 fast processor 默认行为会改变输出；首轮 P1/V0--V4 将显式固定 `AutoProcessor(..., use_fast=False)` 并记录版本，避免 processor 漂移。
- [x] Qwen LoRA PEFT runtime audit：从 `qwen_lora_v1.json` 读取完整路径正则后成功匹配 language model 第 0--27 层，共 392 个可训练 LoRA parameter tensors，`visual_lora_tensors=0`。命令行手写 regex 出现双重转义会导致零模块匹配；正式 runner 必须从 JSON decode 后直接传入 PEFT，并在启动时断言 visual trainable count 为 0。
- [x] Prismatic 来源审计：官方 `TRI-ML/prismatic-vlms` README 将 `prism-dinosiglip+7b` 作为空间理解/定位首选；代码为 MIT，但该 checkpoint 继承 Llama-2 许可。101 的 `hf auth whoami` 返回未登录，因此不得下载该 gated checkpoint。待合法 HF token 登录并确认已接受 Llama-2 条款后，再下载并运行 Prismatic P1。
- [x] T_sem 候选冻结前 registry：Stable Diffusion v1.5、PixArt-alpha、PixArt-Sigma、Playground-v2.5 均经 `hf-mirror.com` metadata 查询为非 gated；准确 repo/revision/license 已写入 `configs/teachers/semantic_teacher_candidates.json`。这只是下载与校准候选列表，不能替代在独立 val split 上按 pointing/IoU、无效词率和跨 seed 一致性选 best-single。
- [x] SD1.5 attention hook smoke：新版 Diffusers 0.38 对真实 calibration 图像 latent 与 phrase `a blue hard hat` 成功捕获 cross-attention，分辨率/层计数为 64:5、32:5、16:5、8:1。FlashAttention 导入不兼容，已回退 PyTorch attention；该运行环境行为需记录。该 smoke 明确为 `formal_calibration_eligible=false`，因为尚未执行 DDIM/null-text inversion，不能用于 T_sem 选择或 V3/V4 cache。
- [x] SD1.5 DDIM image-conditioned map smoke：完整 caption `A man in a blue hard hat ...` 内目标 phrase 成功定位到 CLIP token positions `[4,5,6,7]`；5-step conditional DDIM inversion 生成 25 个 16×16 cross-attention tensors 和归一化 map。metadata 固定为 `method=ddim_conditional_no_nulltext`、`formal_calibration_eligible=false`；下一 gate 是 per-timestep null-text optimization 与原始 Lavender legacy baseline 对照。
- [ ] null-text runner：本地实现与 15 个测试已提交（`ad46d0a`）；待在 101 VEPFS 同步后依次执行 `5×1` 接口 smoke、`20×10` 单图正式验收。尚未运行，不能称为 Lavender-equivalent map 或用于候选教师排名。
- [x] null-text runner GPU1 验收：FP16 `5×1` 出现 NaN，已保留为失败证据并改用新增 FP32-v2 runner；FP32 `5×1` 通过（mean MSE `0.09803`），FP32 `20×10` 正式单图通过（mean MSE `0.03884`、100 个 16×16 attention tensors、目标 span `[4,5,6,7]`）。SD1.5 可以进入 10 图 calibration pilot；该单图通过不等于四教师 best-single 已选择。
- [x] SD1.5 pilot10：固定 calibration manifest 前十图全部成功（10/10），`20×10`、seed 17、FP32 null-text。唯一 summary `summary_sd1_5_20x10_seed17.json`：pointing `0.50`、mean mass-in-box `0.4754`、mean MSE `0.03207`（min `0.01317` / max `0.06615`）。这证明 SD pipeline/map/metric 可运行；样本太小且尚未与 PixArt/Playground 相比，严禁据此选 best-single 或生成 V3/V4 train cache。
- [x] PixArt-alpha runtime audit：P1 环境补充 `sentencepiece==0.2.2` 与 `tiktoken==0.14.0` 后成功加载本地 pipeline；T5Tokenizer、DPMSolverMultistepScheduler、56 个 transformer attention processors。初始化告警仅为 unused `caption_projection.y_embedding`；PixArt 需独立 transformer-token attention adapter，不能复用 SD UNet/null-text runner。
- [ ] PixArt-alpha phrase-map adapter：pipeline 与 tokenizer已加载，但 Diffusers 0.38 的 PixArt transformer 内部模块路径和预期名称不一致；当前只确认 processor count，尚未捕获真实 image-conditioned token attention。此兼容性 gate 不影响 SD1.5 pilot，也禁止将 SD map 用作 PixArt map。
- [ ] V1 caption manifest/GPU smoke：101 上已确认 builder、runner、LoRA config 存在且 `caption_sft_train.jsonl` 初始缺失；构建命令随后出现 SSH 无输出异常，连只读 `echo`/process 查询亦未回显。未假定 manifest 已完成，未启动 V1 smoke，避免重复创建或训练；恢复可靠远端观察后先检查文件 SHA256/行数与是否有活跃 builder。
- [ ] 快速主线 10k：全量 caption manifest 后续已确认有 145,355 条（SHA256 `deb2d4…`），10k seed17 子集与 metadata 已确认存在。V1 20-step GPU1 smoke 已提交到唯一新输出路径，但会话句柄和后续 GPU/process 查询无输出，状态必须回收结果 JSON 后才能判定；禁止重启第二个 V1 smoke。
- [ ] F0-256：从 10k seed17 manifest 抽取固定 256 pair 的构建已提交至此前不存在的目标路径；101 未回传输出且随后的最小存在性检查无输出，尚不能确认 F0 manifest 是否写成。禁止重抽样、生成 F0 SD cache 或启动 F0 V1--V4，直到可读取唯一文件的 SHA256。
- [x] F0v2 shared manifest：发现 caption-only F0 不含目标 phrase，停止了尚未产出 map 的错误全句 cache job。改从 Entities train 生成 256 条 image-caption-phrase-box 对齐记录，并显式补固定 caption prompt；最终训练输入 `F0_aligned_caption_phrase_256_seed19_v2.jsonl`，SHA256 `2a37c49…`。F0v2 V1/V2 20-step smoke 均通过；V2 DINO retention 从约 0.995 降至 0.862，visual LoRA=0、DINO frozen。
- [ ] F0v2 SD cache：已提交后台 job PID `1554962`，路径 `teacher_maps/F0v2_sd1_5_nulltext_phrase_256_seed17`；每条使用完整 caption 条件、同一行 Entities phrase token target、20×10 FP32 null-text、seed17、sample_id 唯一键。完成后须核对 256 map/metadata、failures 和有效交集，才可开始 V3/V4。
- [x] F0v2 V1--V4 smoke：正确对齐 manifest 和 SD phrase cache（256 maps、`failures=[]`）下，V1/V2/V3/V4 均完成 20-step，所有记录的 loss 有限。V3 使用 eager attention 以支持二阶 phrase-score attribution；visual encoder parameter 仅为 attribution 计算开启 grad，不在 optimizer，visual LoRA=0。`F0v2_status_nonfinal.json` 是工程 gate 汇总，不是论文结果。剩余 F0：checkpoint restore、V0 eval、三种负控与状态表。
- [x] F0v2 idea-validation checkpoint gate 开始：V1 LoRA adapter 已保存到 `checkpoints/F0v2_idea_validation/V1`（约 161MB），V2 LoRA adapter + `dino_projector.pt` 已保存到 `.../V2`；held-out manifest 已生成 64 条、SHA256 `7b16778e…`。下一步保存 V3/V4 checkpoint 后评测 64 条 held-out 和三类负控。
- [x] held-out evaluator 已实现：`scripts/eval_qwen_heldout.py` 固定 teacher-forced phrase-score gradient×activation，输出 pointing、mass-in-box、top20 attribution box IoU；`scripts/run_f0_heldout_all.sh` 冻结 V0--V4 同一 manifest 批量入口。V3/V4 checkpoint 导出与视觉参数排除修正提交 `339bbb3`，评估器提交 `cf7a7d4`，批量入口提交 `9426f19`。
- [x] 教师负控脚本已实现：`scripts/eval_teacher_map_controls.py` 对 held-out SD phrase cache 计算 correct、wrong-image、shifted、random，commit `0fb77a1`。尚未执行，因为 101 SSH 当前不可用，且 64 条 held-out teacher cache 尚未确认存在。
- [x] 评估器修正：PEFT inference 会冻结 Qwen base；`eval_qwen_heldout.py` 现在只重新打开 visual path 的 `requires_grad` 以取得 phrase-score attribution，仍不创建 optimizer、不更新权重，commit `d756212`。
- [x] 上传工具已加入：`server/upload_validation_scripts.sh` 会从用户本机上传 5 个验证脚本并在 101 端逐个 SHA256 校验，成功标志为 `UPLOAD_VERIFY_OK`。由于 Codex 执行沙盒的 socket 限制，该脚本应在用户自己的 SSH 可用终端运行。
- [x] 用户已确认本机 `ssh vla101 'echo SSH_OK'` 成功；下一步入口为 `server/run_f0_checkpoint_smokes.sh`，顺序生成 V3/V4 checkpoint，拒绝覆盖旧产物，成功标志为 `CHECKPOINT_SMOKES_OK`。
- [ ] 2026-09-18 checkpoint smoke 首次启动在 V3 forward 阶段因 GPU0 显存耗尽失败：PID `291422` 占用约 59.86 GiB、PID `2093781` 占用约 19.27 GiB，仅剩约 5.5 MiB；未生成有效 V3 checkpoint。未杀进程，避免影响其他任务。启动器已改为显式传递 `CUDA_VISIBLE_DEVICES`，默认改用 GPU1；重试前需由 `nvidia-smi` 确认 GPU1 空闲或选择明确空闲卡。
- [x] 2026-09-18 用户在 GPU1 成功完成 V3/V4 checkpoint smoke：两个结果 JSON、V3/V4 adapter 均生成，V4 `dino_projector.pt` 存在，远端输出 `CHECKPOINT_SMOKES_OK`。下一步为同一 64 条 image-disjoint held-out 的 V0--V4 attribution evaluation。
- [x] held-out 远端入口已加入：`server/run_f0_heldout_remote.sh` 在 101 P1 环境运行五组评估，拒绝覆盖已有报告，成功标志 `HELDOUT_MODELS_OK`。
- [x] 2026-09-18 15:20（UTC+8）V0--V4 held-out attribution 已完成 64/64，远端输出 `HELDOUT_MODELS_OK`。初步结果（20-step checkpoint，non-final）：V0 pointing/mass/IoU=`0.2813/0.3095/0.2693`；V1=`0.3594/0.3183/0.2692`；V2=`0.3594/0.3313/0.2696`；V3=`0.2969/0.3197/0.2683`；V4=`0.3594/0.3210/0.2682`。因此当前短 smoke 未显示 `V3>V1` 或 `V4>V2` 的一致提升；不能据此否定 idea，需先完成 teacher 负控并明确这是工程 gate 还是训练不足。
- [x] held-out evaluator 修复记录：远端缺少 `evaluation/spatial.py`（已同步并通过 `SPATIAL_IMPORT_OK`）；PEFT wrapper 层级差异与 phrase BPE 直接匹配问题已修复（本地 commits `6129934`、`4dc2e14`）。
- [ ] 2026-09-18 SD held-out cache 首次尝试因工作目录未设置导致 64 条 `ModuleNotFoundError: scripts`，未生成有效 map；已切换到 repo 工作目录并设置 `PYTHONPATH` 重跑，当前第 5/64 条已成功，使用 FP32 null-text `20x10`、seed23。
- [x] 2026-09-18 held-out SD cache 已完成 64/64，`failures=0`；teacher controls 已运行：correct pointing/mass/IoU=`0.6563/0.4088/0.3253`，wrong-image=`0.2969/0.3234/0.2701`，shifted=`0.3750/0.3567/0.3255`，random=`0.2969/0.3033/0.2676`。correct 对 pointing/mass 和 wrong-image/random 的三项均明显更好；shifted 的 IoU 与 correct 几乎持平，故 IoU 单项负控不完全通过。
- [ ] 20-step held-out 已完成但只是 smoke：V3/V1、V4/V2 没有一致提升；已增加 `server/run_f0_100step_train.sh` 进行公平 100-step 学习曲线，避免把短 smoke 误判为最终科研结论。
- [x] same-image wrong-word teacher control 已补跑：correct=`0.6563/0.4088/0.3253`，wrong-word=`0.2969/0.3272/0.2781`（pointing/mass/IoU）。
- [ ] 由于 100-step 只显示部分支持，继续 500-step 公平复核；`run_f0_100step_train.sh` 已参数化 `F0_STEPS`/`F0_TAG`，新增 `run_f0_500step_heldout.sh`。
- [x] 2026-09-18 500-step confirmation 完成：四组训练 loss 全部 finite、checkpoint restore 成功，held-out 输出 `HELDOUT_500_MODELS_OK`。V1=`0.2969/0.3284/0.2698`，V2=`0.3906/0.3282/0.2690`，V3=`0.3750/0.3312/0.2685`，V4=`0.2344/0.2678/0.2692`（pointing/mass/IoU）。V3 仅 mass 略高于 V1，V4 低于 V2；`V3>V1`、`V4>V2` 均未成立。结果与解释写入 `experiments/F0v2-idea-validation/results/analysis_500step.md` 和 `metrics_500.json`。
- [x] F0 当前结论：SD correct map 相对 same-image wrong-word、wrong-image、random 在 pointing/mass 上明显更好，说明 teacher signal 有效；但现有 semantic loss 转移到学生归因后没有稳定增益，不能启动 F1-10k unchanged。下一步应审计 semantic-loss scale/schedule、map normalization/temperature，并做小规模 lambda sweep。
- [ ] lambda sweep 已实现：`server/run_f0_lambda_sweep_train.sh` 和 `server/run_f0_lambda_sweep_eval.sh` 固定 100 steps、seed17、F0v2 manifest、V3/V4 两种变体和 lambda `{0,0.01,0.03,0.1,0.3}`；分析脚本为 `scripts/analyze_lambda_sweep.py`。尚未启动。
- [x] 2026-09-18 lambda sweep 已完成：10 个组合（V3/V4 × 5 λ）均训练和 held-out 评估成功，远端分别输出 `LAMBDA_SWEEP_TRAIN_OK`、`LAMBDA_SWEEP_EVAL_OK`。V3 λ=0.10 的 pointing/mass/IoU=`0.3906/0.3365/0.2679`，相对 λ=0 的 `0.3438/0.3255/0.2722`；V4 λ=0.10=`0.3438/0.3333/0.2696`，相对 λ=0=`0.3750/0.3328/0.2701`。没有 λ 在两个变体上同时提升全部空间指标；λ=0.30 明显伤害 V3 mass。结论和下一步写入 `experiments/F0v2-idea-validation/lambda_sweep_results/analysis.md`。
- [x] 2026-09-18 其他 semantic-loss 参数 sweep 已完成：V3 fine-lambda `{.06,.08,.12,.15}`、teacher temperature `{.50,.75,1.25,1.50}`、以及独立 seed29 的 semantic warm-up `{20,50}`。候选 `λ=.10,T=1.25` 在 seed17 为 `0.4219/0.3324/0.2723`，在 seed29 为 `0.3594/0.3295/0.2689`；warm-up50 seed29 为 `0.3750/0.3371/0.2681`。pointing/mass 的部分增益可复现，但 IoU 不稳定。完整审计见 `conditioning_sweep_results/final_parameter_sweep_analysis.md`；当前停止标量扫参，转向 map-match 目标形式。
- [x] 2026-09-18 100-step V1--V4 训练已完成，远端输出 `LONG_TRAIN_100_OK`；四个 adapter checkpoint 和 JSON 均存在，下一步在同一 64 条 held-out 上评估，入口 `server/run_f0_100step_heldout.sh`。
- [x] held-out 只读汇总入口已加入：`server/summarize_f0_heldout.sh` 检查 V0--V4 报告、打印三项指标和 V3/V1、V4/V2 初步趋势；`MODEL_HELDOUT_READY` 不等价于最终 idea 通过，仍需 teacher 负控。
- [x] 一键 F0 入口已加入：`server/run_f0_full_validation_remote.sh` 复用或生成 V0--V4 held-out 报告、64 条 SD teacher cache，并运行四类 teacher map controls；拒绝覆盖 checkpoint/report，最终标志 `F0_FULL_VALIDATION_OK`。尚待用户终端执行。
- [ ] 2026-09-18 当前执行窗口无法建立 `vla101` SSH socket（`Operation not permitted`）；未假定远端状态、未重复启动任务。待连接恢复后按 protocol 先同步 runner，再补 V3/V4 checkpoint 和 64 条 held-out 指标。此为环境阻塞，不是实验失败。

## 2026-09-24：冻结首轮 VLM 语义监督与数据划分原则

- [x] 确认首轮 V1--V4 使用 Flickr30k 官方 train 的完整有效图像-caption 对进行 caption SFT；caption CE 监督 caption 全部 token。
- [x] 确认语义归因损失只作用于 Flickr30k Entities 中能够精确映射到 caption token span、且有有效教师图的实体名词短语。Entities phrase 标注用于短语筛选和 token-span 桥接；GT bounding box 坐标不进入训练损失，也不用于教师/学生图目标区域。
- [x] 确认官方 validation 仅用于预先声明范围内的 teacher/归因接口/超参校准；官方 test 不参与训练或选择，只用于协议冻结后的最终评估。train/val/test 按 image ID 隔离；test box 仅作为 pointing、mass-in-box、IoU 等指标的评测真值。
- [x] 解释边界：该设计是以实体短语为条件、用教师伪图监督学生归因的蒸馏；不是完全不使用实体标注，也不是直接 box-supervised detector/grounder。设计在科学上可行，但可行性不等于方法有效；V3>V1、V4>V2、教师负控和下游能力保持仍须实验证明。
- [ ] 正式训练前落实 split/image overlap 审计，生成不可变训练/校准/测试 manifest 与 SHA256；冻结短语过滤、重复短语 occurrence/token span、无效 map 排除及各有效样本计数。
- [ ] 核实现有 SD cache 是否与此实体短语监督目标一一对应；不匹配的缓存不得直接复用为正式训练输入。

## 下一条可接受的证据

`results/P1-interface-audit/qwen2.5-vl-7b/report.json` 应由 native adapter 生成，并通过：

```bash
PYTHONPATH=src python scripts/validate_p1_report.py \
  <VEPFS>/results/P1-interface-audit/qwen2.5-vl-7b/report.json
```

它必须覆盖 20 个固定样本，保存动态 token grid、phrase target 标量、梯度有限性、seed、图路径及模型 revision。只有该文件通过，才把 Qwen 记为 P1 通过。

## 2026-09-25：101 SSH 只读连通性复核

- [x] 本地 `ssh -G vla101` 解析正常：用户 `root`，目标 `115.190.90.101:27219`，使用本机默认 SSH identity 列表。
- [x] TCP 探测成功：`115.190.90.101:27219` 可以建立 TCP 连接。
- [ ] SSH 协议握手未完成：`ssh -vv -o BatchMode=yes -o ConnectTimeout=12 -o ConnectionAttempts=1 vla101 ...` 在发送本地 SSH version string 后，于 20 秒内报 `Connection timed out during banner exchange`，返回码 `255`。
- [ ] 本次没有进入公钥认证阶段，因此不能据此判断 key、root 用户或远端文件权限；也没有启动、重启或重复提交任何实验任务。
- 当前判定：101 网络端口可达，但 SSH 服务、端口转发/网关或远端连接配额未及时返回 server banner。恢复条件是只读命令能返回 `SSH_OK`、远端 hostname 和时间；恢复后先执行 repo 状态与磁盘/GPU 只读检查，再继续实验。
- [x] 2026-09-25 权限放开后再次由 Codex 只读复核：TCP 建连成功，本地已发送 SSH version string，但远端主动关闭连接，错误变为 `kex_exchange_identification: Connection closed by remote host`，返回码 `255`。仍未收到远端 banner，也未进入公钥认证；未启动或修改任何实验任务。

## 2026-09-25：101 恢复前的 VLA 本地队列

- [x] 新增 `experiments/P6-vla-b0-b4/101_outage_readiness.md`，冻结 101 恢复前可做的工作、恢复后 G0 顺序、禁止事项和剩余未决字段。
- [x] 再次确认原生 VLA 优先规则：已有公开、可复现 native VLA/action head 的模型系列直接审计其原生接口，不先接 StarVLA；StarVLA 只作为无原生 VLA 的 VLM-to-VLA 扩展或动作头迁移独立对照。
- [x] 本地剩余工作已完成：action/manifest/checkpoint artifact contract 测试、StarVLA action-head 静态审计摘要、代码-only 恢复上传包和 SHA 清单均已生成并验证。
- [x] 已完成 StarVLA 静态 action-interface 审计摘要：`references/repos/starvla/INTERFACE_AUDIT.md`。审计确认 action dimension、chunk、normalization、rotation/gripper convention 不能跨 action head 猜测；7D delta EEF 只能在 round-trip 通过后作为报告接口。
- [ ] 101 恢复后第一阶段：OpenVLA/OFT P1、真实 LIBERO task enumeration、episode manifest、B0 20–50 step smoke、restore、held-out offline prediction；在这些证据产生前不提交 B1–B4 长训练。

## 2026-09-25：VLA 本地合同与恢复上传包

- [x] 101 复核仍失败：TCP 可达，但 SSH banner 阶段超时；本轮没有进入认证、没有启动或重启远端任务。
- [x] 新增 `src/vla_attention/adapters/openvla_oft.py`：OpenVLA/OFT schema-safe adapter。未知的 native action dimension、chunk horizon、单位、旋转表示、夹爪符号和 normalization 会显式阻止训练；只有远端 P1 report 提供完整字段后才允许序列化 schema。
- [x] 新增 `tests/test_experiment_contracts.py` 的 OpenVLA adapter contract：未知 schema 拒绝、完整审计 schema 接受。
- [x] 新增 `scripts/build_vla_upload_manifest.sh`，生成 `experiment_workspace/vla_upload/` 代码-only tarball、文件清单、SHA256 和 Git SHA；产物不含模型、数据、teacher cache 或 checkpoint。
- [x] 上传包生成成功：26 个文件，bundle 约 28 KB；`experiment_workspace/vla_upload/SHA256SUMS` 自校验通过，manifest Git SHA=`faccdf16de9534b632b16517f2d847c625b27c7d`。
- [x] 本地验证：VLA config `VLA_CONFIG_OK`；contracts/retention/spatial tests `12 passed`；`git diff --check` 通过。
- [x] 追加 `src/vla_attention/vla_artifacts.py`：episode split、checkpoint restore provenance 和 `A_act` artifact 合同；测试扩展后本地合同集合为 `14 passed`，上传包更新为 27 个文件并重新通过 SHA256 校验。
- [x] 修复上传包 SHA256 清单的可移植性：bundle 条目改为相对仓库根目录路径；从仓库根目录重新验证通过 `RELATIVE_SHA_MANIFEST_OK`，避免远端或另一台电脑出现本地绝对路径校验失败。
- [x] 新增 `server/upload_vla_bundle.sh`：101 恢复后先要求 `SSH_OK`，再验证本地相对 SHA、上传代码-only bundle，并在远端校验和解包；明确拒绝模型、数据、cache、checkpoint 上传。
- [x] 复核并冻结 P6 首轮模型边界：OpenVLA/OFT 是唯一 B0–B4 主骨干；π0.5、MolmoAct2、LingBot-VLA 仅作为后置接口审计/独立扩展，不进入首轮主矩阵。
- [x] 新增 `server/upload_vla_bundle.sh` 并通过 shell 语法检查：恢复后可一键执行 SSH 检查、本地相对 SHA 校验、代码-only 上传、远端 SHA 校验和安全解包。
- [x] 上传包内容审计：27 个成员全部为代码/配置/协议/测试，没有模型、数据、cache 或 checkpoint；SSH 详细复核仍停在发送本地 version string 后的 `Connection timed out during banner exchange`。
- [x] 当前 bundle 结构审计通过：27 个成员均为相对路径；最新 bundle SHA256=`680870b56e72615d68171c3eb515fc36b12961f162a8566e7023e0cc34428231`。
- [x] 补齐 `src/vla_attention/attribution/action_gradient.py` 的架构无关 `gradient×activation` 原语；它只接收标量 action loss 和带梯度的 visual tokens，不猜测 native action schema。测试环境无 PyTorch 时对应数值测试显式 skip；当前合同集合为 `14 passed, 1 skipped`，上传包更新为 29 个文件并通过 SHA256。
- [x] 最新回归检查确认上传包无模型/数据/cache/checkpoint 成员，大小约 32 KB；SSH 仍在 banner 阶段超时。
- [x] 原始 TCP 复核：`115.190.90.101:27219` 可建立连接，但 5 秒内没有返回任何 SSH banner；本地无 OpenVLA/LIBERO/P6 进程，未尝试绕过 launch guard。
- [x] 当前 VLA launch guard 仍有效：直接验证正式模板返回 `VLA_CONFIG_INVALID`，明确要求先完成 P1 与 episode manifest；未用占位值启动训练。
- [x] 对 `vla101` 别名和直连 `115.190.90.101` 做同样的 5 秒 SSH 探测，均在 banner 阶段超时；确认不是本地 alias 解析问题。
- [x] 复核本机 SSH 配置没有第二个备用 endpoint：`vla101` 唯一解析为 `root@115.190.90.101:27219`，无 ProxyJump/ProxyCommand；不能在未授权新地址时猜测替代服务器。
- [x] 用户确认首轮 VLA 只使用 OpenVLA/OFT；已将 π0.5、MolmoAct2、LingBot-VLA 明确记录为后续跨结构轮次，StarVLA 保持独立 VLM-to-VLA/action-head 对照，不进入首轮 B0–B4。
- [x] 用户确认 OpenVLA/OFT 正式 LIBERO 评测采用官方 action chunk（当前草案 `H=8`）；`H=1` 仅保留为 P1/B0 smoke 和归因诊断。协议新增 network horizon 与 rollout execution chunk 分开记录、chunk 内逐步 `A_act(t)` 和完整 chunk rollout 规则。
- [x] 用户确认第五项：正式 B0–B4 使用 checkpoint/evaluator 的原生 action horizon（当前草案 `H=8`）；`H=1` 仅用于 P1/B0 smoke、首步归因和 restore 诊断，不能与正式 chunk 结果混入主表。
- [x] 用户确认动作单位与归一化原则：训练/rollout 保留 native action；报告优先使用米和弧度，夹爪保留 native sign；限幅沿用官方 action head，归一化只使用 checkpoint/dataset statistics，禁止手工 min-max。具体数值等待 P1 与 LIBERO demonstration round-trip。
- [x] 用户确认视觉层分工：B1 retention 使用 BlindVLA 风格中间层；B2/B3 的 `A_lang` 和 B4 的 `A_act` 使用 action-head 输入前视觉层；其他层仅做固定 checkpoint、单 seed 的 attribution sensitivity，不重复完整 B0–B4。
- [x] 用户确认 token provenance 与 patch grid 方案：原生 grid 从 processor/backbone/forward metadata 获取，保存 token 来源和坐标；`T_sem`、`A_lang`、`A_act` 先映射到规范化图像坐标，再使用公共 `16×16` 网格；用已知 patch occlusion 做回投校验。
- [x] 用户确认第六项 LIBERO 首轮配置：官方 LIBERO-Spatial 中一个简单 pick-place/put-object-into-container 任务，采用官方主第三人称相机 + wrist camera，并保留 native proprioception；版本、task ID、控制频率、horizon 和相机预处理由 101 P1/manifest 冻结。
- [x] 用户确认第七项 episode manifest：train/offline_eval/rollout 按 episode_id 严格不相交；smoke 使用 `8/4/4`，G1 最小正式实验使用 `100–200/25–50/25–50`；B0–B4 共享相同 manifest，rollout 不参与 teacher、超参或 checkpoint 选择，每次变更生成新实验 ID 和 SHA。
- [x] 用户确认正式 VLA 训练步数必须达到万级：协议新增 `min_optimizer_steps=10000`。最终 `max_steps=max(natural_epoch_steps,10000)`；不足时按固定 manifest 重复/采样补足，记录 dataset passes 和 transition 访问次数，所有 B0–B4/seed 共享同一规则。
- [x] 用户确认第九项 loss 草案：`lambda_retention=0.05`、`lambda_containment=0.02`，分别前 500/1,000 optimizer steps warm-up；`lambda_sem` 继承 VLM 冻结值并在 VLA smoke 检查尺度；损失和梯度范数分项记录，仅对 optimizer 参数做 clipping。
- [x] 用户确认第十项 evaluator：只使用官方 LIBERO evaluator 判定 rollout success，offline MSE/MAE 与闭环 success 分开；保存 evaluator version/commit、task/episode/seed、逐步 action、状态摘要和 failure reason；trial/horizon/action repeat/频率待 101 审计后冻结。
- [x] 利用本地归档的 `references/repos/openvla-oft/LIBERO.md` 和 `README.md` 补齐恢复审计字段：原生 action head、action chunk、`use_proprio`、`unnorm_key`、processor、center-crop、官方 evaluator trial 配置；这些只作为审计清单，未替代远端 P1 证据。
- [ ] 仍待 101：上传该 bundle，逐文件 SHA 校验；随后执行 OpenVLA/OFT P1、LIBERO manifest、B0 smoke/restore/offline/rollout。

## 2026-09-25：VLA 运行骨架补齐

- [x] 新增 `src/vla_attention/trainers/vla_bc.py`：只负责 B0–B4 开关矩阵、seed 白名单、P1/manifest launch guard 和可审计 run metadata；未猜测 OpenVLA/OFT forward、action shape 或单位。
- [x] 新增 `src/vla_attention/evaluation/libero.py`：校验官方 evaluator rollout artifact，并拒绝把 `offline_prediction` 标成闭环 success。
- [x] 新增 `scripts/validate_vla_run_contract.py`：远端提交前验证 group、seed、config SHA、manifest SHA 和 model revision。
- [x] 扩展 VLA 合同测试：`13 passed, 1 skipped`；`VLA_CONFIG_OK` 和 `git diff --check` 通过。
- [ ] 仍缺真实运行实现：OpenVLA/OFT forward/trainer、LIBERO 官方 wrapper、episode manifest builder。三者都必须等 101 P1/环境/数据证据后落地，当前不能声称 VLA 已可训练。
- [x] 上传包已重新生成并补入 `tests/test_vla_local_pipeline.py`；当前 39 个代码/配置/协议/测试文件，约 40 KB，SHA 清单和包内成员核验通过。SSH 恢复后只上传一次并校验远端 SHA。

## 2026-09-25：VLA 本地 synthetic pipeline 完成

- [x] 新增 VLA loss 接口：BlindVLA-style cosine retention、冻结 `T_sem → A_lang`、D0 stop-gradient action containment；B0–B4 trainer 可组合 action/retention/semantic/containment 四类 loss。
- [x] 新增 native action chunk 工具：支持 `[H,D]`/`[B,H,D]`、逐 chunk timestep action MSE、归因归一化；保留 `A_act(t)` 的时间维。
- [x] 新增 `src/vla_attention/provenance.py`：显式 token provenance、归一化图像坐标和 common-grid bridge 合同。
- [x] 新增 `scripts/build_vla_episode_manifest.py`：episode-level train/eval/rollout split、附加 camera/proprioception metadata、manifest/source SHA256 和 split 校验。
- [x] 新增 `src/vla_attention/run_artifacts.py`：JSONL 曲线追加、checkpoint state SHA256/index artifact。
- [x] 新增 `tests/test_vla_local_pipeline.py`：synthetic manifest、chunk/provenance、B0–B4 loss step 和 checkpoint index 测试。
- [x] 本地验证：VLA pipeline/contract/config/retention/spatial 测试 `18 passed, 3 skipped`；`VLA_CONFIG_OK`、JSON 解析、shell syntax 和 `git diff --check` 通过。
- [x] 补齐万步训练 artifact：自然 epoch/max steps/dataset passes 计算、25/50/75/100% checkpoint step 生成、progress summary 和 failure episode JSONL；新增严格 P1/evaluator launch guard，未冻结真实字段仍拒绝正式启动。
- [x] 恢复上传包已重新生成：39 个代码/配置/协议/测试文件，约 40 KB；不含模型、数据、cache、视频或 checkpoint，SHA 清单通过。
- [ ] 仍待 101：把真实 OpenVLA/OFT forward 接入 adapter，把真实 LIBERO task/episode 数据接入 manifest，把官方 evaluator 接入 wrapper；当前 synthetic 通过不等于 VLA 结果。

## 2026-09-25：VLA 本地实现最终审计增量

- [x] 归因 common-grid bridge 已从最近邻中心采样改为基于 token box 与目标 cell 重叠面积的加权池化；目标 cell 无覆盖时显式失败，避免无依据的空间复制。
- [x] 新增原子 checkpoint save/restore：`src/vla_attention/run_artifacts.py` 保存 model/optimizer/scheduler state 与 metadata，并在恢复时校验 config、manifest、model revision 和 scheduler presence。
- [x] manifest 现在拒绝同一 split 内重复 episode ID；train/offline_eval/rollout 仍按 episode ID 严格隔离。
- [x] 新增 `src/vla_attention/semantic_cache.py`，提供冻结 `T_sem → A_lang` cache metadata、teacher revision、公共 grid、normalization 和 manifest SHA 校验；不在本地重新选择 teacher。
- [x] 新增 optimizer parameter-group boundary：retention/semantic teacher 参数必须冻结，只有 policy/projector 参数可以进入 optimizer，重复参数会被拒绝。
- [x] 新增/扩展 synthetic contract tests：面积池化几何、chunk 内逐 timestep action loss、重复 episode、rollout/offline 标签隔离、teacher cache metadata 和 checkpoint contract。
- [x] 本地验证结果：`21 passed, 5 skipped`。跳过项均因 `/usr/bin/python3.10` 当前无 PyTorch，涉及真实 torch 数值、optimizer step 和 checkpoint restore；纯 Python 合同和 launch guard 已通过。不能把这次结果写成真实 OpenVLA/LIBERO 验证。
- [x] 配置门禁验证：`--allow-template` 可检查模板结构；不带该参数会明确拒绝所有未冻结 P1/manifest 字段。
- [x] shell syntax、JSON/config 校验和 `git diff --check` 通过。
- [ ] 101 恢复后仍需运行真实 PyTorch checkpoint round-trip、OpenVLA/OFT P1、LIBERO manifest、B0 smoke、held-out offline prediction 和官方 evaluator rollout；这些证据不能在本地伪造。

## 2026-09-25：训练曲线和 teacher cache 接口收尾

- [x] 新增 `append_training_curve`：每个 optimizer step 固定写入分项 loss、total/progress、learning rate、裁剪前 gradient norm 和 examples seen，并拒绝 NaN/Inf。
- [x] `semantic_cache.py` 的 cache manifest 读取接口现在验证真实文件内容 SHA；不接受自引用 `cache_manifest_sha256`，避免不稳定的自哈希记录。
- [x] 有效 cache metadata 与自引用 SHA 的测试均通过；本地 VLA 合同集合保持 `21 passed, 5 skipped`。

## 2026-09-25：纯 Python synthetic B0–B4 完成

- [x] 新增 `src/vla_attention/synthetic_vla.py`：在没有 PyTorch/NumPy/GPU 的本地环境中，验证冻结 B0–B4 scalar loss 组合、500/1000 step warm-up 和训练曲线 JSONL 产物。
- [x] synthetic runner 只证明开关、权重、warm-up 和 artifact 流程，不冒充真实梯度、OpenVLA forward、checkpoint tensor restore 或 LIBERO success。
- [x] 本地验证结果更新为 `22 passed, 5 skipped`；5 个 skipped 仍是依赖 PyTorch 的真实数值/optimizer/checkpoint 测试。

## 2026-09-25：VLA 插件化替换架构完成

- [x] 新增 `src/vla_attention/plugins.py`：显式 model/benchmark/optimizer 注册表、版本规格、action schema compatibility 和 B2–B4 capability gate。
- [x] 新增 `src/vla_attention/benchmarks/base.py` 与 `evaluation/libero.py` 的 frozen-facts holder；训练器可以只依赖 benchmark contract，不直接依赖 LIBERO 实现。
- [x] 新增 `src/vla_attention/optimizers.py`：AdamW/Adam/SGD/Adafactor 与 none/constant/linear/cosine scheduler 工厂，PyTorch 延迟导入。
- [x] 新增 `src/vla_attention/trainers/engine.py`：generic trainer preflight，检查模型、benchmark、optimizer、seed、配置/manifest provenance。
- [x] `configs/vla/p6_b0_b4.json` 增加 `plugins` 与 optimizer/scheduler 配置块；首轮仍冻结为 OpenVLA/OFT + LIBERO + AdamW，未解除 P1/manifest launch guard。
- [x] 新增 `experiments/P6-vla-b0-b4/plugin_architecture.md`，记录四层接口、公共数据合同、能力矩阵、替换顺序、一次只改变一个实验轴的公平性规则。
- [x] 新增 `tests/test_plugin_contract.py`，覆盖 action schema mismatch、能力缺失、重复/未知插件、trainer preflight 和 optimizer 名称校验。
- [x] 本地验证：插件/VLA/config 测试 `20 passed`；`VLA_CONFIG_OK`、shell syntax、JSON 解析和 `git diff --check` 通过。

## 2026-09-25：上游 teacher 适配层补充

- [x] 明确 teacher 不是统一热力图生成器，而是 `source → extractor → geometry/temporal bridge → calibration/normalization → immutable cache → student loss` 的独立上游流水线。
- [x] 记录五类 teacher 信号边界：扩散模型输出 `semantic_map`，DINO 输出 `visual_features`，DVD/video 输出 `video_latents`，WAM/VAM/world model 输出 `future_state`，动作/轨迹教师输出 `action_trajectory`。
- [x] 冻结首轮选择：`T_sem=SD1.5`、`T_ret=DINOv2`；PixArt-α/Σ 与 Playground-v2.5 作为 semantic teacher swap，DVD/video 作为 temporal 对照，WAM/VAM/world model 作为 `R_future` 后置对照，action/trajectory teacher 作为后期 D1 对照。
- [x] 规定 teacher 替换只改变上游 teacher 轴；学生模型、benchmark、optimizer、manifest、训练预算和 B0–B4 trainer 保持不变。
- [x] 规定多 teacher 融合必须在每个单 teacher 通过 calibration、反事实 controls 和单轴 swap 后才能进行，融合权重和 cache revision 必须独立记录，不能混入首轮 B0–B4 主表。
- [ ] 真实 teacher runtime extractor、geometry/temporal bridge、校准脚本和 cache writer 仍需在 101 或同等 GPU 环境完成；当前合同通过不等于 PixArt、视频模型或 world model 已接入。
- [x] 上传包已重新生成：47 个代码/配置/协议/测试文件，约 48 KB；不含模型、数据、cache 或 checkpoint，SHA 清单通过。
- [ ] 仍待 101：为真实 OpenVLA/OFT、LIBERO runtime 和后续模型/benchmark 分别填入 adapter；当前插件测试只证明接口可组合，不证明远端模型已可运行。

## 2026-09-25：上游 teacher/signal 插件层补齐

- [x] `PluginSelection` 增加 teacher 列表与 `TeacherPluginSpec`：强制记录 teacher revision、signal type、target space、冻结状态和 cache revision。
- [x] `src/vla_attention/teachers/base.py` 增加 `TeacherAdapter` 与 `TeacherOutput`，区分 `semantic_map`、`visual_features`、`video_latents`、`future_state`、`action_trajectory`，不再把所有上游信号统称为 attention。
- [x] P6 配置增加 semantic/retention/future/action teacher 槽位：首轮为 SD1.5 semantic + DINOv2 retention；future/action teacher 保留为空，后期单独启用。
- [x] 插件架构文档补充 teacher 选择、校准指标、cache、反事实 control 和一次只改变一个 teacher 轴的公平性规则。
- [x] teacher 合同测试覆盖冻结状态、signal 类型、output/spec 一致性；与现有 VLA 测试合计 `22 passed`。
- [ ] 仍待真实 teacher runtime：统一生成 SD/PixArt/Playground/DINO/video/world/action cache，并在固定学生模型和 benchmark 下完成 teacher swap。

## 2026-09-26：范式创新候选冻结为独立分支

- [x] 新增 `experiments/P6-vla-b0-b4/paradigm_innovation_candidates.md`，将当前方法上升为“可审计、可映射、可干预、可随时间传递的视觉证据接口”候选，而不是继续把创新表述为 attention 对齐或增加 teacher。
- [x] 记录 VLA 后续方向 `Language-to-Action Evidence Transport`：先验证 B2/B3/B4 的静态语言/动作 evidence，再研究 source → source+target → target 的阶段性证据传递。
- [x] 记录反事实 evidence 契约：删除高/低 evidence 区域、wrong-word、wrong-image 和动作阶段遮挡作为共同的行为验证协议；首轮先做 evaluation/control，不提前声称因果性。
- [x] 记录多 teacher disagreement/uncertainty 作为远期方向；禁止在单 teacher 未完成 calibration 与 controls 前直接平均 teacher 输出。
- [x] 明确该候选不改写当前 B0–B4 主矩阵；当前主线仍用于验证 method-level 机制，范式级扩展在单步证据通过后再开启。
- [x] `git diff --check` 通过；本次新增文档尚未提交，需与下一次实验配置变更一起 commit。

## 2026-09-26：自动驾驶论文机制迁移审查

- [x] 批判性区分三类借鉴：ALN-P3 的阶段化对齐/训练期零推理开销可迁移；XYZ-Drive 的 state/action-conditioned dynamic query 是最值得作为 D1 evidence-flow 扩展验证的机制；DriveTeach-VLA 的 attention mass、detector noise 和 what-to-see/where-to-look 分层可用于对照与反事实协议。
- [x] 明确不直接搬用驾驶域 CLIP 全局对比损失、BEV/waypoint 坐标、Grounding-DINO bbox teacher、DVD 输入增强蒸馏、2D-TGP 轨迹文本和 GRPO；这些模块分别缺少机械臂空间对应、会引入外部 detector 混杂或超出首轮主线。
- [x] 将后置 D1 设计冻结为：静态 containment vs state/language dynamic query vs action-conditioned query vs concat/late-fusion controls；专家未来动作 query 只作诊断上界，主结果禁止标签泄漏。
- [x] 将 detector-prior/DVG-style 方案单独编号为外部 baseline，不并入 B0–B4；阶段化 loss 使用 LIBERO weak event boundaries，需在 B2/B4 通过后启动。

## 2026-09-26：确定 VLM→VLA 过渡优先借鉴 ALN-P3

- [x] 结论冻结：ALN-P3-inspired 阶段化 co-distillation 更适合做 VLM→VLA transition；XYZ-Drive-inspired dynamic query 更适合 B0–B4/D0 通过后的 D1 动态证据流，不作为首个过渡模块。
- [x] 新增 SAEB-T0（Stage-Aligned Evidence Bridge）候选：冻结 VLM 提供 `A_lang`/semantic latent，native VLA 保留原生 action head，通过 LIBERO weak phase bridge 对齐 `A_act(t)` 与阶段语义表示；teacher/bridge 训练后移除，不增加推理开销。
- [x] 明确 SAEB-T0 不复制 ALN-P3 的 BEV、CLIP 全局对比或驾驶轨迹；空间 evidence 和 patch occlusion 仍是主机制证据，latent stage loss 只能作为低权重辅助项。
- [x] 明确 SAEB-T0 的 controls：random latent、phase shuffle、wrong-word/wrong-image、latent-only、map-only；防止过渡模块退化为一般 feature co-distillation。
- [ ] SAEB-T0 需在真实 OpenVLA/LIBERO P1、B0–B4 基线通过后才实现；当前不改变 B0–B4 主矩阵。
- [x] 澄清 ALN-P3 的架构边界：原方法本身对齐异构 fast P3 与 slow MLLM，不要求同一 backbone；跨架构迁移依赖语义/空间 bridge，不允许直接做 raw hidden-state cosine。
- [x] SAEB-T0 的跨架构合同冻结为 `A_lang/A_act` 公共图像证据空间 + 可选 stage latent bridge；同架构只作为低风险特例，不作为方法适用性的前提。

## 2026-09-26：C-RADIOv3-L 纳入 BlindVLA 主对照

- [x] 根据 BlindVLA 固定源码审计，确认其 alignment 入口默认使用 `c-radio_v3-l`，目标是中间视觉 patch feature 的 cosine retention，而不是语言词图或 action attribution。
- [x] 将 VLA P6 的 B1/B3 retention 主 teacher 推荐冻结为 C-RADIOv3-L；DINOv2 保留为 retention teacher swap，VLM 侧仍可使用 DINOv2。
- [x] 更新 `configs/vla/p6_b0_b4.json`、P6 protocol 和 teacher/plugin 文档；这不改变 B0–B4 的损失矩阵，只替换 retention teacher 的身份。
- [ ] 仍待远端 P1 核验 C-RADIOv3-L 的准确 checkpoint/revision、RADIO 预处理、真实 patch grid、teacher feature dimension、projector seed/恢复和 cache SHA。
- [ ] 官方 BlindVLA 入口使用在线 teacher forward；本项目优先实现可复核的离线 cache，但必须证明 cache 与在线输出逐样本一致后才能用于主结果。

## 2026-09-26：加入 BlindVLA-style t-SNE 表征诊断

- [x] 确认 t-SNE 的作用是诊断 action fine-tuning 前后的中层视觉 patch 表征漂移，不作为 success、grounding 或 Go/No-Go 主指标。
- [x] 冻结比较矩阵：`VLM-pretrained`、`VLA-pretrained`、`B0`、`B1-C(C-RADIOv3-L retention)`、`B3-C`，并推荐加入 `B4-C`；主比较使用同一 VLA backbone 的中层 visual feature。
- [x] 复刻 BlindVLA 的固定类别/固定样本/固定层思路，同时增加 PCA、linear probe、kNN、silhouette、feature drift 和 teacher cosine 等量化结果。
- [x] 明确不同 hidden spaces 不能直接混合 t-SNE；VLM 与 VLA 只有在 token/grid/维度可比时才 joint fit，否则分面展示并使用 C-RADIO teacher space 做跨模型量化对照。
- [x] 将 t-SNE 方案写入 `experiments/P6-vla-b0-b4/protocol.md`，包含样本、层、pooling、随机种子、图组布局和解释边界。
- [ ] 101 恢复后抽取真实 checkpoint features、冻结 feature cache SHA，并完成 t-SNE 与 linear-probe 诊断；在此之前不生成伪图或声称表征恢复。

## 2026-09-26：明确各数据集用途边界

- [x] 冻结数据角色：Flickr30k captions 用于 V1–V4 caption SFT；Flickr30k Entities 用于 phrase-region teacher calibration、grounding 和反事实 controls；COCO 仅用于 BlindVLA-style t-SNE/linear-probe 诊断及独立能力评测。
- [x] 明确 RefCOCOg、VQAv2、TextVQA、POPE、MME、WorldMedQA-V 属于外部能力/迁移评测，不与 Flickr30k Entities grounding 分数混合。
- [x] 明确 LIBERO/LIBERO-Plus/LIBERO-PRO 用于 VLA 机制与闭环，DROID 用于真实视觉分布下的 offline action prediction；它们不替代 VLM 训练数据。

## 2026-09-27：101 SSH 网络恢复但当前会话认证未通过

- [x] `vla101` 已完成 TCP 连接、SSH key exchange 和 host-key 校验；服务器地址为 `115.190.90.101:27219`，host key 与本地 `known_hosts` 一致。
- [ ] 当前 Codex 执行环境仍未通过用户认证：尝试 `/home/bubble/.ssh/id_rsa` 后返回 `Permission denied (publickey,keyboard-interactive)`；因此本次不能读取 101 的 GPU、进程、VEPFS、代码或实验结果。
- [ ] 不能据此判断远端旧 job 是否完成，也不能重复启动 F0/VLA 任务。认证恢复后第一步必须只读执行 `echo SSH_OK; hostname; date -Is`、`nvidia-smi`、进程/作业和结果目录检查。
- [ ] 认证恢复前可继续本地协议、代码合同、上传包和结果解析；不能启动真实 OpenVLA/LIBERO、teacher cache 或长训练。

## 2026-09-27：101 认证恢复，修正版 F0 结果已回收

- [x] `ssh vla101 'echo SSH_OK; hostname; date -Is'` 成功；主机 `di-20260613120619-84fwm`，两张 A100-SXM4-80GB 均空闲，未发现正在运行的 F0/VLA/teacher job。
- [x] 只读检查确认 VEPFS 项目、Qwen/DINO/teacher cache、F0 checkpoint 和历史结果均存在；没有重复启动旧任务。
- [x] 回收远端 `F0v2_geometry16_fixclip_heldout` 的 summary、三 seed paired bootstrap 和 SHA256 到 `experiments/F0v2-idea-validation/geometry_loss_sweep_results/fixclip_heldout_20260927/`。
- [x] 修正版三 seed结果：KL 相对 V1 的 pointing 平均 `+0.0885`，95% bootstrap CI `[+0.0052,+0.1719]`；common soft-IoU `+0.0109`，CI `[+0.0048,+0.0169]`；calibrated IoU `+0.0101`，CI `[-0.0045,+0.0251]`。KL-moment 的 pointing `+0.0781`，CI `[+0.0052,+0.1510]`；KL-rank 的 pointing `+0.0625`，CI `[-0.0156,+0.1406]`。
- [x] V4 KL-rank 相对 matched V2 的 pointing 平均 `+0.0313`，95% CI `[-0.0469,+0.1094]`；common soft-IoU `+0.0086`，CI `[+0.0029,+0.0145]`；calibrated IoU `+0.0067`，CI `[-0.0085,+0.0221]`。
- [ ] 当前解释边界：V3 对 V1 的 pointing/soft-IoU 有方向性和部分 bootstrap 支持，但 IoU 仍不稳定；V4 对 V2 尚未建立稳定的完整互补证据。不能据此声称 idea 已被完全证明，也不能跳过 teacher calibration 直接启动 F1-10k 或 VLA 长训练。
- [ ] 下一步顺序：核对负控和 capability pilot 是否使用同一修正版 checkpoint；若通过，冻结 VLM F1 候选；随后上传最新 VLA bundle，执行 OpenVLA/OFT P1 → LIBERO manifest → B0 smoke。

## 2026-09-27：F1 teacher calibration job 启动

- [x] 远端确认 `teacher_calibration_val_1000.jsonl` 存在且为 1000 条独立 validation image-level phrase records；四个 teacher 权重目录均存在。
- [x] 当前只有 SD1.5 已通过真实图像 FP32 null-text 单图/10 图工程验收；PixArt-alpha、PixArt-sigma、Playground-v2.5 尚无等价 phrase-map extractor，不能直接混入公平 calibration，需记录为 adapter 缺口/fallback 条件。
- [x] 首次 SD1.5 1000 图 job `PID=338945` 因远端 `PYTHONPATH` 未包含 repo root，1000 条均失败；无有效 map，已停止，不计入 calibration 结果。
- [x] 修正启动环境为 `PYTHONPATH=$R:$R/src` 后重新启动同一 cache 路径：`PID=340521`，日志 `/vepfs-mlp2/c20250405/400040/transfer/vla_attention/jobs/F1_calibration_sd1_5_1000_seed23.log`，当前 job 运行中，首个 pipeline 已成功加载并进入处理。
- [ ] calibration job 完成前不选择 teacher、不生成 10k train cache、不启动正式 V0–V4 长训练；每个 10/25/50/75/100% 节点回收文件数、失败清单和指标。
- [x] 新增并上传受保护的 `server/run_f1_vlm_v0_v4.sh`：验证冻结 teacher config、10000 条 manifest、10000 个无失败 train cache 后，按 seed 17/29/41 顺序运行 V1→V2→V3→V4，拒绝覆盖已有 artifact；当前仅完成脚本语法和远端上传验证，未启动正式训练。
- [x] 发现原 `run_sd_cache.py` 每个样本重新加载 45 GB SD pipeline，导致 calibration 速度不可接受；停止无效 PID `340521` 并新增 `scripts/run_sd_nulltext_batch.py`，单进程复用 pipeline、逐样本写 `progress.json`、保留已有有效 map、记录失败样本。
- [x] batch driver 已上传 101，PID `353111` 正在同一 calibration cache 继续运行；当前 progress artifact 为 `24→27/1000`，失败 `0`。该 job 不训练学生模型。
- [x] 进一步确认单卡速度仍约 30 秒/样本；停止单卡 PID `353111`，将固定 1000 行 manifest 按偶/奇行拆为两个各 500 行的 disjoint split，在 GPU0/GPU1 以 PID `357038/357039` 并行运行同一 batch driver。两部分当前均运行中、各完成 `5/500`、失败 `0`；完成后合并到正式 calibration cache 并验证全量 sample ID 覆盖与无重复。
- [x] 新增 `experiments/P2-teacher-calibration/candidate_status_20260928.json`：明确 PixArt-alpha/ sigma/ Playground 权重虽存在，但当前没有经过真实图像 inversion、phrase-token map 和等价指标验收的 adapter；按 protocol 记录为 adapter-missing，若在正式 freeze 前仍无 adapter，则使用明确标注的 SD1.5 fallback，不声称四候选 best-single。
- [x] 为降低 calibration 总耗时，停止旧双 shard PID `357038/357039`，将同一固定 manifest 改为四个 250 行 disjoint shard，GPU0/GPU1 各运行两个 batch driver：PID `363592/363594/363596/363598`。当前四个 shard 均运行、失败 `0`、显存约 32.8GB/GPU、GPU 利用率约 100%；不再重复重启，完成后只合并 `part4_*` 目录。

## 2026-09-27：VLA 代码包上传验证完成，P1 依赖审计

- [x] 重新生成最新 `experiment_workspace/vla_upload/`；代码包 47 个文件、约 56 KB，明确不含模型、数据、cache 和 checkpoint。
- [x] 修复 `server/upload_vla_bundle.sh` 的远端验证顺序：先临时解包并校验包内文件，再写入远端 repo；`VLA_UPLOAD_VERIFY_OK` 已通过。
- [x] 远端 `validate_vla_config.py --allow-template` 返回 `VLA_CONFIG_OK`。
- [ ] 远端默认 `/root/miniconda3` 环境只有 PyTorch，没有 transformers/peft/accelerate/libero/OpenVLA；因此不能在该环境直接进行真实 P1。`starvla_cu124` 环境有 PyTorch 2.6、Transformers 4.57.1、PEFT 0.18、Accelerate 1.13、Diffusers 0.38，但仍没有已安装的 `libero`/`openvla`/`prismatic` Python 包。
- [ ] 远端 VEPFS 当前没有 OpenVLA checkpoint；已有 Qwen、DINO 和 diffusion teachers。`/root/code/LIBERO`、`/root/code/Starvla` 和其他项目包含 LIBERO 源码，但不能把 StarVLA/SpikingBrain checkpoint 当作 OpenVLA P1。
- [ ] 因此 VLA 下一步是先在独立环境审计/安装官方 OpenVLA-OFT 与 LIBERO 依赖、确认合法 checkpoint 来源，再运行 P1；不能直接启动 B0 长训练。已有两张 A100 空闲只说明资源足够，不代表模型接口已经就绪。

## 2026-09-28：VLM calibration 完成与 SD1.5 fallback freeze

- [x] 四个 calibration shard `part4_0..part4_3` 均完成 `250/250`、失败 `0`；合并后正式 cache 包含 1000 个 map 和 metadata，sample ID 全覆盖、无重复。
- [x] 合并 cache manifest SHA256=`757811b802540a25e8b06b956d0ae2e5ed209cb1378b5b63f3f402f415c6498a`；calibration manifest SHA256=`4d425283f9b03a143cddd805dc7035ce3df59d1b06ac9565c30049f507e6e1b3`。
- [x] SD1.5 correct teacher metrics：pointing `0.5520`、mass-in-box `0.3581655`、top20 IoU `0.3008146`；wrong-word/wrong-image pointing `0.2960`、random pointing `0.3040`，teacher signal 通过 validation-only sanity gate。
- [x] PixArt-alpha/sigma/Playground 权重虽存在，但没有通过等价真实图像 inversion + phrase-token map adapter 验收；冻结为显式 `stable-diffusion-v1-5` fallback，不声称四候选 best-single。
- [x] 新增 `configs/experiments/P4_vlm_v0_v4_frozen.json`，冻结 teacher revision、FP32 null-text 20×10、CFG 7.5、16×16 grid、seed 23、normalization、KL、lambda `.10`、temperature `1.25`、warm-up 50、训练组和三 seed。
- [ ] 下一步生成完整 10k train teacher cache；cache 完成前不启动正式 V0–V4 长训练。
- [x] 结合 Lavender 原文核对 teacher 选择：Lavender 实际使用 SD v1.4，没有 PixArt/Playground 的同协议比较；因此当前以已通过真实图像 phrase-map calibration 的 SD1.5 作为方法连续性最强、可审计的 fallback，不宣称四候选 best-single。
- [x] 已将 10000 条固定 train manifest 拆为四个各 2500 条 disjoint shard，并启动 batch cache jobs：PID `525047/525049/525051/525053`，分别绑定 GPU0/1，输出目录为 `F1_train_sd1_5_nulltext_phrase_10k_seed17_part4_*`；正式 V0–V4 尚未启动。
- [x] 发现并停止上述无效 cache jobs：caption-only manifest 没有 `phrase` 字段，导致全部样本失败；没有生成有效 map，未污染正式结果。
- [x] 新增 `scripts/build_aligned_caption_phrase_manifest.py`，将每条 caption-SFT 样本与 Flickr30k Entities 的确定性 phrase/box 对齐；9952 条有效样本、48 条无实体 phrase failure，failures 单独保存。V0–V4 将共享 9952 条有效 manifest。
- [x] 上传修正版 frozen config/runner，并启动 aligned 9952 cache：PID `531386/531388/531390/531392`，四个各 2488 条 shard，正式训练仍未启动。
- [x] 当前 cache 复核：四个 shard 均存活，各完成 `15/2488`，总计 `60/9952`（约 0.60%），每个 shard 的 map/metadata 数量一致，失败数均为 `0`；GPU0/GPU1 显存约 32.8GB、利用率约 99–100%。


## 2026-09-28：VLA 非 GPU 准备启动及 cache 抽取阶段纠错

- [x] 已读取新增 goal；原 VLM 全量目标保留，新增范围到 B0 smoke/restore/rollout。四个 VLM cache PID 命令行核验为本项目作业，未停止、重启或覆盖，未新增 GPU job。
- [x] 官方来源静态审计完成：OFT commit e4287e94541f459edc4feabc4e181f537cd569a8，LIBERO 8f1084e3132a39270c3a13ebe37270a43ece2a01；原生 L1、8×7 chunk、8D proprio、双相机。已训练 LIBERO checkpoint 仅作外部评测/接口检查，不充当任务未见的初始化。
- [x] 远端独立工作区 `vla_workspace`；五个源仓库已固定版本，锁见 `experiments/P6-vla-b0-b4/audits/source_lock.tsv`。静态分析列出训练 stats=all 和单步 deque 等陷阱，尚不等于运行成功。
- [x] 官方 LIBERO-Spatial RLDS 的 16 个分片及 metadata 经上游 hash 核验下载；metadata 声明 432 episodes、仅 train split。CPU AST 枚举官方 130 tasks，BDDL 全存在；实际 episodes 读取/划分待依赖安装。
- [ ] 隔离环境安装正在执行 PID 595650，日志 `vla_workspace/logs/preparation_retry4_20260928.log`。前三次失败为 clone 无checkout的假dirty、TLS、ensurepip缺失，分别修复；未修改现有 VLM 环境。
- [ ] 模型/数据下载 PID 592359，日志 `vla_workspace/logs/download_retry_20260928.log`；已转至 OpenVLA 权重下载，C-RADIO/OFT checkpoint 未完成。顺序下载、限速15MiB/s；通过SSH回环端口17897转发本机7897，依赖当前连接存活，断线后需检查进程再恢复。
- [x] cache只读阶段审计：1000/1000 calibration记录1400 tensors，脚本capture在inversion前启用，混入优化过程；原单图recipe重建阶段为100。此前“正式校准通过”判定撤回。审计JSON在 `experiments/P6-vla-b0-b4/audits/calibration_stage_audit_20260928.json`。
- [x] F1启动器已fail-closed，配置标invalidated；本地batch修复为仅最终重建capture并每条恢复原processor，拒绝旧版本cache混用。未上传替换运行中driver，未新增GPU验证。5项CPU测试通过；数值等价性仍待GPU窗口。
- [ ] 附加审计：wrong-word同图缺候选会错误回退wrong-image；此前同图负控结论需重做。V2/V4对冻结merger/input做retention可能只训练projector，正式policy retention梯度必须验证。F1封装仍调用smoke且只用10k子集，不能当完整train、逐step日志及可恢复正式trainer。
- [x] 三类VLA cache和Plus边界写入 `cache_preparation.md`；当前只准备合同，未生成VLA cache，未运行SAEB/B1–B4。
- [ ] 下一步：环境安装/下载完成验收 → CPU读取真实episodes和split → 真正B0接口；cache修复与正式VLM训练器并行完善，原目标未完成。


## 2026-09-28：VLA 真实数据清单、CPU batch 与原生 forward 接入

- [x] 新goal原文重读；上轮为实际进展，本轮继续，不停止现有4个cache job，也未执行任何GPU任务。
- [x] VLA隔离环境 `vla_workspace/envs/oft` 安装完成；torch2.2.0+cu121、专用transformers4.40.1、peft0.11.1、TF2.15.0。发现tensorflow-metadata新版本与protobuf冲突，固定metadata1.15.0/protobuf3.20.3后CPU预处理成功；`pip check`无损坏依赖。完整freeze见 `audits/pip_freeze_cpu_verified.txt`。LIBERO需显式PYTHONPATH指向固定repo，配置独立LIBERO_CONFIG_PATH，未修改共享环境。
- [x] 官方RLDS 16分片逐episode读取全部432条，检查动作/状态finite、边界标识、双相机步数、首末JPEG解码；所有帧内容hash、序列化payload hash均保留。未发现重复轨迹。`inventory_libero_rlds.py`产物在远端 `artifacts/rlds_inventory_20260928`，本地元数据镜像 `experiments/P6-vla-b0-b4/audits/rlds/`。
- [x] Spatial task0实际45个episodes，按固定hash顺序/seed17分成train27/validation9/offline_eval9。train-only action/proprio q01/q99与stats SHA已记录；不虚构原计划100–200条，不复制episode。该split用于工程验证，正式数据规模仍需审计。
- [x] `prepare_oft_cpu_batch.py`使用真实train episode、官方RLDSBatchTransform和processor生成输入：pixel_values[1,12,224,224]，proprio[1,8]，actions[1,8,7]，56 action slots；CUDA未初始化。TF与第三方tfrecord protobuf重复注册已通过改用TF自带Example reader修复。预处理为无增强CPU gate，不声称训练增强或rollout已校验。
- [x] `oft_forward.py`接入官方掩码/隐藏状态切片/连续头L1定义；使用真实官方action head和轻量策略fixture在独立环境CPU测试2 passed，覆盖chunk完整性、policy梯度及截断拒绝。不是7B forward，也不是P1通过。
- [x] 官方benchmark在CPU枚举spatial task0，真实init states共50、92维，hash inventory已保存。demo与init-state lineage还未确认，不声称rollout与demos物理初始状态无重叠。
- [x] 数据划分CPU测试2 passed；stage-audit测试2 passed。无GPU smoke/restore/rollout结果。
- [ ] 权重下载仍不完整：TLS/EOF频繁，HTTP1+断点重试进程PID600493存活；已下载metadata但多个大safetensors仍.partial。不能按目录存在标权重完成，后续按source lock逐项哈希验收。下载通道是SSH反向端口17897，断开后需恢复。
- [ ] 后续：完成权重→真实B0训练/全状态恢复入口和official rollout wrapper→GPU窗口P1/B0。当前VLM cache已标阶段混入待修复，F1阻断继续有效；原VLM全量目标未完成，不以9952子集代替完整train。


## 2026-09-28：真实 OFT episode dataset 与 B0 训练/恢复入口

- [x] 重读新增goal；上轮是进展。本轮核验四个VLM cache的实际命令行均活跃，未停止/重启/覆盖，未新增GPU作业。
- [x] `oft_rlds.py`直接读取固定episode清单、校验payload SHA、保留全部H=8窗口，使用官方RLDSBatchTransform/ActionTokenizer/collator。CPU验证train27轨迹→2520 chunks，offline_eval9轨迹→769 chunks；首末批次[2,12,224,224]/[2,8]/[2,8,7]成功。报告 `audits/oft_dataset_cpu_20260928.json`。
- [x] 发现并补齐官方bounds_q99对常量维度置零要求；train-only stats新增min/max，保留旧版，v2结果在 `artifacts/rlds_inventory_v2_20260928`。远端CPU测试2 passed，其中一项与官方TensorFlow normalization逐值比较。
- [x] 新增 `run_oft_b0_smoke.py`：读取真实train/eval分离轨迹，原生L1+LoRA/all-linear+action head/proprio，20–50 optimizer updates，每步JSONL；保存adapter、head、proprio、optimizer、scheduler、随机数状态和离线预测，提供新进程restore比较入口。固定模型hash/manifest/stats，显式no augmentation作为smoke差异。尚未运行真实7B模型，未声称P1或恢复成功。
- [x] 启动器要求新鲜独占GPU窗口和cache验收通过，并在加载权重前查询GPU计算进程；本地拒绝启动测试及episode划分测试共3 passed。当前cache混阶段审计未通过，因此不会意外启动B0。
- [ ] 新进程恢复目前验证加载和预测一致性，进一步优化器续步等价、真实动作归因P1和官方rollout仍需完成。训练脚本存在不等于B0 gate通过。
- [ ] 权重下载PID600493命令核验活跃；部分OFT action head/权重片已哈希验证，大文件仍有TLS/EOF失败和.partial。需完成全部source lock验收，不能宣布checkpoint就绪。VLA独立环境CPU准备已可用，GPU仍待窗口。


## 2026-09-28：OFT 双向注意力验证、官方 rollout 接口与分段下载

- [x] 原goal文件重读；上轮属实质进展，四个cache worker仍活跃且未改动，没有新增GPU作业。
- [x] 发现OFT fork的eager仍为因果注意力，SDPA分支才移除三角mask并保留padding mask。B0改为官方fork SDPA；小型Llama CPU测试2 passed：未来token可影响早期token、padding不泄漏；eager作为反证保持因果。没有把eager错误地视为官方OFT等价路径。
- [x] 新增 `eval_oft_b0_rollout.py`，加载自有adapter/head和train-only stats，调用官方 `run_episode`，保留8步chunk和环境done成功判据；逐步action/reward/done日志，官方吞异常时将episode标为incomplete。smoke训练无随机crop所以此包装器center_crop=False，明确与官方已增强权重评测不同。尚未执行真实rollout。
- [x] 官方evaluator CPU导入成功，CUDA未初始化；证据 `audits/official_evaluator_import.json`。日志保真及错误传播测试2 passed，launch guard测试1 passed。
- [x] 新增 `download_verified_ranges.py`，对每段验证HTTP206/Content-Range/长度，确认后才追加并fsync，末尾核对完整SHA；测试2 passed涵盖中断续传、来源变化和hash错误。独立base shard3尝试PID624212已因TLS异常退出，保留已取得8MiB，不把失败或部分内容标成功。原下载PID600493仍活跃，已推进至C-RADIO元数据；大权重仍不完整。
- [ ] 需要：稳定下载全部权重、GPU窗口真实P1/梯度/优化器恢复/rollout；修复VLM缓存抽取后等价验证和完整train范围仍是原goal必做。本轮仅完成CPU和脚本工作，未完成B0。

## 2026-09-28：修正版 cache 队列、LIBERO semantic smoke 与权重恢复

- [x] 读取新 goal defca5b1；按授权停止旧混合阶段 Flickr PID 并保留产物；旧目录不进入训练。
- [x] 修正版两样本 parity：确定性 FP32/math-SDPA/TF32-off/CUBLAS 配置下 max error 均 `0.0`、attention tensors=100；报告远端 `results/cache_v3_gate/deterministic_parity.json`，未放宽阈值。
- [x] LIBERO cache input 导出完成：27 train episodes、5040 双相机帧、10080 source/target records；不含 validation/offline_eval/rollout。
- [x] C-RADIOv3-L smoke 和全量 feature cache 完成：5040/5040，shape `[256,1024]`、failure=0、逐条 SHA/finite/grid 审计通过，content digest `f03a23c...`。
- [x] LIBERO semantic smoke 完成 16/16、failure=0、metadata/map SHA 审计通过；后台队列 PID `651953` 已放行全量 semantic cache 和新版本 Flickr cache，不启动学生训练。
- [x] 新增 typed cache validator、atomic stage-aware SD writer、strict cache queue；新增三类 cache 与 Plus evaluation-only 边界。
- [x] OFT CPU 环境、432 episodes inventory、Spatial task0 45 episodes split `27/9/9`、train-only stats、官方 L1/8x7/SDPA contracts、CPU native batch、官方 evaluator import 和 B0 guard 完成；不等于真实 P1/B0。
- [x] 新增 B0 continuation probe：恢复 optimizer/scheduler/RNG 后要求下一步参数 digest 一致，缺 optimizer state 必须失败；新增官方 rollout wrapper，失败/环境异常不伪装成 success。
- [ ] OpenVLA base/OFT weights 仍受 HF TLS 断连影响；普通 downloader 有多个 partial，已启动 mirror range recovery PID `657946`，按 upstream size/SHA 分块续传，完整 source-lock 校验前不加载模型。
- [ ] 当前两张 GPU 由修正版 Flickr四分片（PID653277/653278/653279/653280）和LIBERO语义缓存（PID653275）使用；旧Flickr批次已停止。C-RADIO全量完成，LIBERO全量semantic仍在生成；B0须等GPU独占、相关cache gate及OpenVLA权重校验通过。


## 2026-09-28：用户授权修正VLM cache并优先生成VLA cache

- 新指令覆盖先前不停止cache的安排：暂停已确认混阶段的旧Flickr批次，保留产物；验证修正版后优先LIBERO训练cache，不运行B0长训练。
- [ ] 按实际cmdline确认旧PID并保存停止前状态；新任务全部新目录、版本/hash隔离。
- [ ] SD修复需两条真实样本与单图20×10参考数值对照、重建100 tensors、参数冻结、重复性校验；通过前不生成LIBERO长任务。
- [ ] 从真实27条train episodes导出双相机帧和source/target phrase，保留原始指令与预处理hash；val/test不混入。
- [ ] C-RADIO权重校验并生成相同训练观测的patch features；SAEB需明确冻结VLM checkpoint并做机器人观测校准，未满足时只准备输入清单。
- Git：仅提交脚本/小报告/协议，commit记录修复与生成入口，push到origin main；不提交cache。


## 2026-09-28：SD parity修复通过，LIBERO retention全量完成

- [x] 读取最新goal defca5b1；停止旧四PID前已核验命令并存progress，SIGTERM后无残留，旧产物保留且标invalidated_capture_stage；停止证据 `audits/cache_v3/old_flickr_cache_stop_20260928.json`。
- [x] SD batch改为重建前安装hook、每条结束恢复processor、冻结teacher、原子map/JSON、实时failures、首次连续失败停止。先前两条parity失败（1.3e-4/2.4e-4）；冻结梯度开关单独诊断仍有误差。固定FP32/math SDPA、关闭TF32、deterministic algorithms/CUBLAS后，单图参考与batch两条最大误差均0；维持atol1e-6/rtol1e-4，不放宽阈值。每条100 tensors、MSE有限。
- [x] 原single-image参考只固定数值backend，算法未替换；两条parity报告已回收。不是Lavender exact，也未重新完成1000图校准；F1训练阻断仍有效。
- [x] LIBERO task0真实27条train episodes导出所有H8有效timestep，双相机共5040帧；每帧source/target两图=10080条semantic输入，target取instruction最后一次the plate，source为the black bowl。未导出val/test用于训练，图像/manifest SHA绑定。
- [x] C-RADIOv3-L完整snapshot哈希通过；HF4.40动态加载遗漏递归模块，改为直接import哈希验证的本地HF package，未改上游源码。真实8帧smoke成功且repeatability通过；256输入输出256×1024 patch features，raw RGB/255、bilinear256、模型内部conditioner，无外部mean/std。
- [x] GPU1 PID647893完成5040/5040 retention缓存；逐条metadata/tensor finite/shape/hash/ID审计有效5040、失败0、额外文件0。artifact `audits/cache_v3/radio_train_v1_audit.json`，内容digest f03a23c066accce4b1a40ada00817b55359c64378e38a811e8f961e1d951e0e0。
- [ ] LIBERO SD smoke PID648822，16条覆盖2episode×2timestep×2camera×2phrase，当前13/16无失败；等待实际结束后验收，不直接视为完成。
- [x] 后台队列PID651953已部署 `server/run_corrected_cache_queue.sh`：等待明确smoke PID结束→验证16张语义图与retention gate→生成LIBERO全量semantic并重建4个disjoint Flickr shards（新版本目录），最后逐cache验收，不启动B0/学生训练。日志 `vla_workspace/logs/corrected_cache_queue.log`。脚本持有flock；下一次须核验原PID/产物而非重启。
- [x] 六项CPU检查通过，脚本py_compile/bash-n/diff-check通过。生产与校验脚本在新cache目录运行，旧mixed-stage数据从未用于长训练。
- [ ] SAEB只准备输入，冻结VLM/标量/层未定；完整原VLM训练目标保留，9952仅先前10k交集，不能冒充完整train。

- [x] 续记：LIBERO semantic smoke已16/16、失败0，逐条验收通过；报告 `audits/cache_v3/semantic_smoke_v3_audit.json`。后台队列已放行LIBERO全量10080语义图及Flickr新版四分片9952图；真实PID从 `vla_workspace/artifacts/semantic_train_task0_v3.pid` 与新Flickr根目录 `worker_pids.txt` 读取，禁止重复启动。正式VLM训练仍不启动。


## 2026-09-28：真实P1检查入口及动作控制器静态核验

- [x] 新goal原文已读取，上轮属实际进展。本轮验证cache队列及5个worker均存活、map持续增长且failure0；不重启。
- [x] 新增 `run_oft_p1_interface.py`：加载已校验base snapshot+原生OFT头，检查8×7输出、原生L1、视觉梯度finite/nonzero、两相机真实patch_embed.grid_size/provenance、重复前向、归一化往返与head/proprio保存恢复。仅写好入口、py_compile通过，尚未执行7B GPU验证；不能当P1已完成。
- [x] 新增 `oft_preflight.py`：同时校验权重SHA及processor/config等Git blob hash，B0要求真实passed P1且model/train/stats SHA相同。CPU测试2 passed覆盖配置篡改和P1缺项；与launch guard/range测试合计5 passed。
- [x] 隔离robosuite1.4.1默认OSC_POSE配置静态核验：输入[-1,1]经clip映射平移±0.05、旋转±0.5，control_delta=true。该事实不等于实际rollout controller已核验；正式P1仍需记录运行时config，不把normalized action直接命名为米/弧度。
- [ ] 镜像恢复PID657946活跃，OpenVLA第1分片已约3.67/6.95GB；其余文件仍需下载/全hash检查。下载阻塞尚可推进，未满足goal blocked条件。
- [ ] B0、真实P1、VLM正式训练未启动；缓存任务仅证明工程产出，不证明teacher质量或方法有效。原目标仍保持active。


## 2026-09-28：CPU真实LIBERO环境与更严格cache验收通过

- [x] 新goal文件已重读；上轮属进展。原cache队列与workers保持原样，无重启；一次SSH握手关闭后重查成功，没有把观测失败当job退出。
- [x] 实际CPU OSMesa环境创建首次失败：robosuite1.4.1搭配自动安装的MuJoCo3.14.0，joint qpos索引断言。仅独立OFT环境降至mujoco2.3.7，pip check通过，安装脚本增加pin；共享VLM环境未改动。
- [x] `probe_libero_cpu_environment.py`实测task0+init0 reset/set_init_state/两步dummy动作成功，两相机[256,256,3]、控制20Hz、action_dim7、CUDA未初始化。runtime OSC_POSE input±1映射output平移±0.05/旋转±0.5，control_delta=true；证据 `audits/cpu_environment_mujoco237.json`。这是环境接口成功，不是B0 learned rollout成功。
- [x] cache validator加强：manifest SHA、CFG/grid、数值backend、seed/extractor SHA一致性、instruction hash、episode/timestep/camera/role来源、teacher revision；拒绝空manifest及同cache混recipe。6项CPU测试通过；16/16 semantic smoke在更严格校验下仍通过，报告 `audits/cache_v3/semantic_smoke_v3_strict_audit.json`。
- [x] 校验器更新已上传，运行中的生产者未改动；后台队列结束时使用加强版进行逐cache验收。环境完整freeze保存 `audits/pip_freeze_renderer_verified.txt`。
- [ ] 镜像恢复PID657946仍活跃，base第1权重片约5.80/6.95GB；LIBERO semantic全量目前18/10080、失败0，仍需等待并保留全量校验，不重复启动。其他新版Flickr shard仍活跃。
- [ ] 尚缺完整模型权重、GPU真实P1/B0/恢复/官方policy rollout及原VLM全量目标；未标完成。

## 2026-09-28：明确 GPU 验收边界并修复 B0 资源门禁

- [x] 重新读取当前 goal；确认目标并非只做 CPU 准备，最终必须有 GPU 真实 OFT P1、20–50 step B0、checkpoint restore、held-out action prediction 和官方 rollout 证据。
- [x] 远端当前状态复核：LIBERO semantic `76/10080`、失败 `0`；Flickr 修正版四 shard 合计 `382/9952`、失败 `0`；C-RADIO retention 已 `5040/5040` 通过；两张 A100 仍被 cache worker 占用。
- [x] OpenVLA base `openvla/openvla-7b@47a0ec7...` 的 source-lock 17 个文件完成逐项 SHA 校验；OFT 评测权重仍在镜像分段恢复，未宣布完整。
- [x] 修正 GPU 资源门禁：B0/P1/rollout 现在要求一张明确的 `CUDA_VISIBLE_DEVICES`、对应 GPU UUID、新鲜（5 分钟内）资源报告和该 GPU 无 compute process；不再错误地阻塞于另一张 GPU 上的 cache。
- [x] 复核 B0 held-out prediction 选择逻辑：已有实现按 manifest 行索引为每个 episode 选择首个 H=8 窗口，本轮保留，未发现需修复的问题。实际修正的是 restore 检查：从允许 `1e-3` 误差收紧为逐元素预测完全一致，并拒绝 shape 不同和非有限值；GPU-window 共 9 个测试用例，连同 P1/launch guard 共 `12 passed`，脚本 py_compile 和 `git diff --check` 通过。
- [ ] 仍未启动真实 GPU P1/B0：必须等 cache 任务释放一张 GPU、OFT 权重完整校验、cache audit gate 和新鲜 GPU-window artifact 全部满足后再启动。

## 2026-09-28：B0 完整 checkpoint 与 rollout 证据链

- [x] 重新读取 goal，上一轮为实际进展。本轮确认原 cache/下载 PID 仍活跃，LIBERO semantic `80/10080`、失败 `0`；无重启、无新 GPU 任务。
- [x] 审计官方 OFT forward/predict_action 与 run_episode：保持官方 8×7 L1、双相机、proprio、SDPA；官方 environment seed=0，记录 policy seed=17，避免配置默认 seed=7 与实际策略种子混淆。当前数据 proprio 无常量维度，train normalization 与官方 evaluator 在 q01 端点比较一致；该检查范围不是全数据归一化验证。
- [x] 原 B0 恢复只校验 state.pt，补充 checkpoint_manifest.json：校验 LoRA 权重和配置、state、offline 预测、config、continuation reference；缺失或被修改的组件拒绝恢复。head/proprio/optimizer/scheduler/RNG 位于已校验的 state.pt。
- [x] rollout 现在要求 --source-lock 和 --restore-report-dir；重新校验 base revision、完整 checkpoint、相同 checkpoint 的 exact prediction restore 与 optimizer continuation。记录 evaluator Git SHA/源码 SHA、policy/env seed、init inventory SHA、stats SHA。reset/set_init_state 等外层异常也写 incomplete，不当有效失败率。
- [x] 本地相关测试 `21 passed, 1 skipped`（本机缺 PEFT）；101 独立 OFT 环境 CPU `7 passed`，其中真实 PEFT 0.11.1 BF16 LoRA save/load 后预测完全相等、AdamW 下一步参数 digest 相等。该小模型 CPU 证据不能替代 7B 新进程 GPU restore。
- [x] 改动上传 101 后对应文件 SHA 一致；补充飞书定向同步入口 --vla-progress-only，仅同步 P6 协议、准备文档与 PROGRESS。同步是否成功以实际云端读回为准。
- [ ] 真实 P1/B0/restore/rollout 尚待独占 GPU；新缓存继续运行，完整目标未完成。Git 仅提交本轮实现/测试/进度和同步入口，不包含权重/cache/环境。

## 2026-09-28：最新远端状态与 GPU 自动衔接入口

- [x] 21:50（上海时间）通过 `vla101` 只读复核：queue PID `651953`、semantic PID `653275`、Flickr workers `653277–653280` 和权重恢复 PID `657946` 均仍运行；未重启任何任务。
- [x] 修正版 SD cache 最新计数：LIBERO semantic `89/10080`、failure `0`；Flickr 四 shard 分别 `89/2488`、`135/2488`、`89/2488`、`135/2488`，合计 `448/9952`、failure `0`。C-RADIOv3 retention 仍为 `5040/5040` 且 audit passed；parity 两条仍为 exact zero error、每条100 tensors。
- [x] 两张 GPU 都有 cache compute processes，当前均不能作为 P1/B0 独占窗口；资源门禁继续拒绝混跑。
- [x] OpenVLA training-initialization base 的17项 source-lock 哈希此前已全部通过。另一个 `moojink` LIBERO-spatial evaluation-only checkpoint 的大权重恢复仍进行中；它不是 B0 的干净训练初始化，不能混为同一模型用途。B0 仍以已校验 base snapshot 开始。
- [x] 新增 `server/run_p6_b0_after_cache.sh`，已完成本地 `bash -n`/diff 检查并上传至101。它等待 cache 队列成功标记、核对 parity/semantic/retention/Flickr 全量 audit，通过后等待一张空闲物理 GPU，依次运行 P1、30-step B0、独立新进程 exact restore/optimizer continuation、两个固定 LIBERO init-state rollout；失败时按阶段停止并保留日志，不启动 B1–B4。
- [x] 自动衔接脚本已在101启动，当前 runner PID `732896`；日志 `vla_workspace/logs/p6_b0_after_cache.log` 为 `WAIT_CACHE_QUEUE`。修正 runner 的 OFT Python 路径后，又将 P1 JSON 输出固定到远端 repo 的 `experiment_workspace/results/P6_vla_b0_b4/p1_interface/`，并把远端 cache validator 改为原子覆盖 audit 文件；两处均已 SHA/语法校验并平滑重启。尚未启动 P1/B0，也未占用额外 GPU。
- [x] 14:26–14:31（上海时间）复核计数：LIBERO semantic `103/10080`、Flickr 四片 `103/155/103/156`，合计 `517/9952`，failure 均为 `0`；所有 SD worker 和 runner 仍存活。
- [x] 本轮修复已提交为 `13033f5 fix(vla): harden cache audit and P1 artifact path`；首次直连 push 遇 GitHub TLS 断开，随后经本机 `7897` 代理成功推送到 `origin/main`。101 上 validator 与 runner 的 SHA 已与本地一致。
- [ ] VLM SD1.5 train cache、LIBERO full semantic cache、GPU P1/B0/restore/offline prediction/official rollout 均未完成。以上比例仅为运行快照，不代表训练结果。

## 2026-09-28：修正 cache 队列的持久成功判定与 OpenVLA 基座验收

- [x] 重新读取持久化 goal 与 `ml-training-recipes`；确认本阶段仍只允许完成 SD1.5 cache、LIBERO semantic/C-RADIO cache 和真实 OpenVLA/OFT B0 链路，不启动 B1–B4、ALN/SAEB、WAM/VAM、DROID、LIBERO-Plus 训练或真机。
- [x] 修正 `server/run_p6_b0_after_cache.sh`：队列等待不再只依赖 stdout 日志字符串。队列退出后会重新验证 deterministic parity、LIBERO semantic audit、C-RADIO audit 和四个 Flickr shard audit；全部通过才写入 `vla_workspace/artifacts/cache_queue_success.json` 并继续 P1。这样前台 SSH 队列即使没有重定向 stdout，也不会把合规完成误判为失败。
- [x] 修正 `server/run_corrected_cache_queue.sh`：直接追加到 `vla_workspace/logs/corrected_cache_queue.log`，并在四类 cache 全部审计通过后原子写入 success receipt；该 receipt 不代表训练结果，只代表 cache 生成和审计完成。曾尝试 `tee`，但它在 SSH/nohup 场景下不够稳，已改回普通文件追加。
- [x] 修正 `server/rebalance_cache_queue.sh` 的 PID 枚举，避免 `ps|awk` 把脚本自身或父 shell 误杀；cache 文件仍保留、通过 worker row range 断点续跑。
- [x] 本地 `py_compile`、三个 shell `bash -n`、`git diff --check` 通过；三个修正版脚本已上传 101，远端 SHA 分别与本地一致。
- [x] 一次误执行 `rebalance_cache_queue.sh` 导致旧前台队列中断；所有 cache 文件和 progress 已保留，随后用修正版 `setsid nohup` 重新拉起唯一队列。当前 101 队列：queue PID `752941`，runner PID `752942`，8 个 SD worker 均为 queue 子进程；LIBERO semantic 4 分片合计 `145/10080`、失败 0；Flickr reconstruction v3 四分片合计 `600/9952`、失败 0。两张 GPU 仍各由四个 SD worker 占用，未混跑 B0。
- [x] B0 自动等待器绑定 queue PID `752941`，当前持续记录 `WAIT_CACHE_QUEUE`，未加载 7B、未启动训练。
- [x] OpenVLA B0 初始化基座 `openvla/openvla-7b@47a0ec7fc4ec123775a391911046cf33cf9ed83f` 已逐项检查预期大小和 SHA256；三个 safetensors 与 tokenizer.model 全部通过。证据：`vla_workspace/artifacts/openvla_base_source_audit.json`。目录中遗留 `.partial` 文件不参与模型加载，也不被 source audit 当作完成文件。
- [ ] 仍待：semantic `10080/10080`、Flickr `9952/9952` 完成并生成全部 audit；随后等待独占 GPU，执行真实 7B P1、30-step B0、独立进程 restore/optimizer continuation、offline action prediction 和两个固定 init state 的官方 rollout。

## 2026-09-29：101 cache 运行快照

- [x] 只读核验 `vla101`：cache queue PID `752941`、B0 等待器 PID `752942`、8 个 SD worker 均存活；没有启动第二套队列，也没有启动 P1/B0。
- [x] LIBERO semantic cache 当前 `906/10080`（约 `8.99%`），失败 `0`；四个分片均持续更新。
- [x] Flickr SD1.5 reconstruction v3 当前 `1360/9952`（约 `13.67%`），失败 `0`；四个分片均持续更新。
- [x] deterministic parity、LIBERO semantic smoke strict audit、C-RADIOv3 `5040/5040` audit 和 OpenVLA base SHA audit 仍在位；full semantic/Flickr audit 和 `cache_queue_success.json` 尚未生成，因此 B0 等待器继续安全等待。
- [x] 两张 GPU 仍由 8 个 SD worker 占用（约 `49.4 GiB/GPU`，利用率约 `99–100%`）；OpenVLA 7B 未加载，B0 尚未消耗 GPU。
- [ ] 根据上一快照 `2026-09-28 15:49 UTC`（上海时间 23:49）到本快照 `2026-09-29 03:07 UTC` 的实际增量，semantic 与 Flickr 各约 `66 records/hour`；按当前速率估算，semantic 剩余约 `139 小时`、Flickr 剩余约 `130 小时`，约 `5–6 天`。这是运行速率估算，不是完成时间承诺，worker 重启或异常会改变它。

## 2026-09-29：当前 cache 与训练边界快照

- [x] 03:17 UTC 只读核验 101：queue PID `752941`、B0 等待器 PID `752942`、8 个 SD worker 均存活；没有启动 OpenVLA 训练。
- [x] LIBERO semantic cache 为 `918/10080`（`9.11%`），失败 `0`；Flickr SD1.5 reconstruction v3 为 `1372/9952`（`13.79%`），失败 `0`。
- [x] 两张 GPU 仍各由 cache worker 占用约 `49.4 GiB`，利用率约 `99–100%`；B0 等待器持续记录 `WAIT_CACHE_QUEUE`。
- [x] 当前阶段属于数据/教师缓存准备：完整 semantic/Flickr audit 尚未生成，`cache_queue_success.json` 不存在，因此 P1/B0 不会启动。
- [ ] 以已观测约 `66 records/hour` 粗略估计，剩余 semantic 约 `139 小时`、Flickr 约 `130 小时`；cache 队列按最慢的 semantic 估算，约还需 `5–6 天`。该估算不包含异常、节点回收或吞吐变化。
- [ ] cache 完成后，自动步骤是：全量 audit（分钟级）→ 独占 GPU 的真实 P1 → 30-step B0 smoke → 新进程 restore/offline prediction → 两个官方 rollout。这个链路是工程 gate，不是完整 B1–B4 VLA 训练。
- [ ] 当前 goal 明确禁止启动 B1–B4、ALN/SAEB、DROID、LIBERO-Plus 训练和真机；因此“完整 VLA 研究训练”没有在当前队列中排期，需等 B0 gate 报告后另行冻结。

## 2026-09-29：101 cache 最新只读快照

- [x] 14:09 UTC（上海时间 22:09）核验：queue PID `752941`、B0 等待器 PID `752942` 和 8 个 SD worker 仍存活；未启动第二套队列、P1 或 B0。
- [x] LIBERO semantic cache：`1635/10080`（`16.22%`），失败 `0`；Flickr SD1.5 reconstruction v3：`2088/9952`（`20.98%`），失败 `0`。
- [x] 相比 03:17 UTC 快照，约 10.86 小时新增 semantic `717`、Flickr `716`，实测吞吐约 `66 records/hour`；worker progress 文件最近均有更新，未见停滞证据。
- [x] deterministic parity、semantic smoke strict audit、C-RADIOv3 full audit、OpenVLA base SHA audit 均存在；full semantic/Flickr audits 和 `cache_queue_success.json` 仍缺失，因此 B0 等待器继续等待。
- [x] 两张 GPU 均由 cache worker 使用，约 `49.4 GiB/GPU`、利用率约 `99%`。
- [ ] 按当前吞吐估算，semantic 剩余约 `128 小时`、Flickr 剩余约 `119 小时`，即约 `5–5.5 天`；这是动态估算，不是完成承诺。

## 2026-09-29：cache 耗时原因与加速边界审计

- [x] 只读检查 `run_sd_nulltext_batch.py` 确认：每条记录包含 VAE encode、条件/无条件 text encode、20-step DDIM inversion、20×10-step null-text optimization，以及最后 20-step reconstruction；按代码计数为约 `20 + 20×(1+10+1) + 20 = 280` 次 UNet 调用，其中 200 次带梯度优化，当前 SD1.5 16×16 capture 每条还要保留 100 个 attention tensors。因此它不是一次普通模型前向，单 worker 约数分钟是预期行为。
- [x] LIBERO semantic manifest 的 `10080` 条记录来自 `5040` 个观测 × `source/target` 两个 phrase。远端 manifest 审计显示恰好 `5040` 个二元重复组：同一 image/caption/camera/episode/timestep 只改变 phrase/role。当前通用脚本对二元组重复执行完整 inversion 与 null-text optimization。
- [x] Flickr 四个 manifest 分片没有 image/caption 重复组，因此不能直接获得同样的去重收益。
- [x] 当前每张 GPU 运行 4 个 worker、每个约 `12.35 GiB`，GPU 利用率约 `99–100%`；继续增加并发预计只会争用显存/调度，不能作为已验证加速方案。
- [x] 为保持已通过 parity 的正式配方，不能直接把 inversion/inner steps 改小、切换 fp16/flash attention 或放宽 deterministic backend；这些都会产生新的 teacher revision，不能混入当前正式 cache。
- [ ] 合规的潜在加速方案是“semantic 二元组共享一次 inversion/optimization/final attention tensors，再分别按 source/target token span 导出两张 map”，理论上可把 semantic 计算量接近减半；但必须先做独立 2 组 parity（逐 map 数值、metadata、失败恢复）再替换正式 worker。当前未贸然打断队列。

## 2026-09-29：semantic cache 安全增量审计与去重 pilot 前置

- [x] 按安全增量规范完成远端只读快照，快照时间为 `20260929T155338Z`（UTC），远端路径：
  `vla_workspace/experiment_workspace/results/cache_safety_snapshots/20260929T155338Z/`。
  快照包含当前 PID/command、GPU 状态、semantic/Flickr progress、failure 清单、manifest SHA256、文件名清单和 cache 数量统计；未停止任何进程，未删除、移动或覆盖任何已有 map/metadata。
- [x] 当前 queue `752941`、B0 waiter `752942`、8 个 SD worker 均仍存活；两张 A100 各约使用 `49.4 GiB/80 GiB`，GPU 利用率约 `100%`。因此本轮不启动额外 GPU pilot，也不打断现有 worker。
- [x] 快照时的进度为：LIBERO semantic 分片进度以 `progress_semantic_part*.json` 为准（aggregate `progress.json` 不是可靠的多 worker 汇总）；Flickr 四片分别 `521/2488`、`580/2488`、`521/2488`、`580/2488`，合计 `2202/9952`，失败均为 `0`。快照前后 worker 仍在更新，未见停滞证据。
- [x] 对 semantic manifest 做 CPU 侧共享计算键审计：`10080` 行恰好组成 `5040` 个二元重复组，每组共享 image/caption/camera/episode/timestep/teacher/数值配置，仅 role/phrase/sample_id 不同；没有把 Flickr manifest 的无重复结构错误套用到 LIBERO。审计证据：
  `experiment_workspace/results/cache_safety_snapshots/20260929T155338Z/semantic_manifest_audit.json`。
- [x] 去重策略仍保持数学定义不变：同一次 full-caption inversion/null-text/final reconstruction capture 之后，按 source/target 的独立 token span 导出两张 map；phrase occurrence、token span、role、sample_id 和 provenance 继续分别写入 metadata。不得因去重改变 FP32、20 steps、inner_steps=10、CFG=7.5、16×16、100 tensors 或 extractor revision。
- [x] 在不占 GPU 的情况下生成了 2 个 source/target pair 的隔离 pilot manifest，输出目录为：
  `vla_workspace/experiment_workspace/results/cache_safety_snapshots/20260929T155338Z/semantic_shared_pilot_v4_pair_parity_seed17/`。
  该 manifest 只描述待运行的 pilot，不包含任何 map，也没有写入正式 cache；远端 extractor SHA 为
  `133aa87877e58f3403ffad68505405107dc42346575a682b29e449f7758fd501`。
- [x] 随着 row-wise cache 增长，重新生成了“已有 reference 完整”的 pilot manifest：
  `vla_workspace/experiment_workspace/results/cache_safety_snapshots/20260929T155338Z/semantic_shared_pilot_v4_pair_parity_seed17_ready/`。
  当前可用完整 reference 的二元组为 `882/5040`，其中选取了 2 组（image 与 wrist 各 1 组）；未有 reference 的
  `4158` 组被排除，避免 parity 误报。
- [x] 新增并上传隔离 runner `run_sd_nulltext_shared_pair_pilot.py` 和 CPU 比较器
  `compare_semantic_shared_pilot.py`。runner 有三重保护：输出路径必须含 `pilot`、拒绝
  `semantic_train_task0_v3`/Flickr formal root、manifest 必须为 `pilot_shared_groups.jsonl`；本轮只做了
  `py_compile`、`--help` 和 SHA 校验，没有启动模型。
- [x] 新增并启动 `server/run_semantic_shared_pilot_when_idle.sh`，远端 launcher PID 为 `1170653`。
  waiter 只轮询真实 `nvidia-smi` compute-app 空闲状态，不停止、不重启、不修改任何正式 worker；空闲 GPU
  出现后才在 ready pilot 目录运行 2 组 shared pilot，再调用 CPU parity comparator。当前日志已确认处于
  `SHARED_PILOT_WAIT_IDLE_GPU`。
- [x] 对 Flickr 四个 aligned manifest 做了 CPU 侧重复审计：`9952` 行、`9952` 个 `(image, caption)` 组，重复组为 `0`。因此当前只考虑 LIBERO semantic 的二元组共享，不对 Flickr 强行去重。
- [ ] 尚未执行 GPU pilot/parity：原因是两张 GPU 被正式 cache worker 占满，且安全规范要求 parity 前不得停止现有 worker。下一次有独占 GPU 后，只能在版本化新目录执行 2 个 source/target pair pilot；旧正式目录仍保持 immutable。
- [ ] parity 未通过前不得切换正式 semantic queue；parity 通过后也只生成切换/回滚方案，不自动删除旧 cache，不自动修改 Flickr、LIBERO C-RADIO 或 B0 waiter。
- Git：本轮只允许提交 snapshot 索引、审计报告、文档和小型脚本；禁止提交 cache、checkpoint、数据和环境。建议 commit：
  `git add experiment_workspace/PROGRESS.md experiments/P6-vla-b0-b4/cache_preparation.md experiment_workspace/results/cache_safety_snapshots/20260929T155338Z`
  `git commit -m "audit: snapshot cache state before semantic dedup pilot"`
  `git push origin main`

### 2026-09-29 16:00 UTC / 2026-09-30 00:00 Asia/Shanghai 续检

- [x] 重新确认 queue `752941`、B0 waiter `752942` 和 8 个 SD worker 均存活；两张 A100 仍约 `49.4 GiB/GPU`、利用率 `99–100%`，未启动 pilot/P1/B0。
- [x] 当前 worker progress：LIBERO semantic `1758/10080`、Flickr SD1.5 `2211/9952`，失败均为 `0`；相较安全快照仍有增长，因此没有把连接/读取异常误判为任务终止。
- [x] 当前可执行的 pilot 仍仅是 CPU 侧 manifest；GPU parity 必须等现有 worker 释放独占 GPU，且只能写版本化新目录。

### 2026-09-29：是否暂停 101 正式 cache 的决策

- [x] 最新只读复核：queue `752941`、B0 waiter `752942`、semantic/Flickr workers 和 shared-pilot waiter `1170653` 均存活；两张 A100 仍约 `49.4 GiB`、利用率 `99–100%`。semantic aggregate 约 `1784/10080`、Flickr 约 `2237/9952`，失败均为 `0`，进度仍在增长。
- [x] 决策：当前不暂停正式 cache。原因是现有 row-wise queue 没有错误或停滞证据；只停止单个 worker 会让 queue 的 `wait` 返回失败，无法产生正式 success receipt；为了获得独占 GPU，通常还需停止同一 GPU 上的多个 semantic/Flickr worker，短期会损失约一半吞吐，但 full grouped production runner、全量审计和 rollback 还未完成，暂停的收益尚未被验证。
- [x] 继续保留 shared-pilot waiter；它只等待自然释放的独占 GPU，不抢占正式任务。优先完成 CPU-only grouped runner/审计；只有 pilot parity 通过且有可回滚的全量 grouped runner 后，才重新评估是否做受控 shard handoff。
- [ ] 只有出现以下任一条件才考虑主动暂停：progress 文件超过两个连续观测周期不更新；failure/OOM/NaN 出现；或 grouped 全量 runner、manifest、audit 和恢复命令已经通过 CPU gate。任何暂停前必须新建 snapshot，并整体停止 queue/相关 worker，禁止只杀单个子进程。

## 2026-10-06：cache audit、B0 smoke、restore 与 LIBERO rollout

- [x] `vla101` 上 LIBERO semantic SD1.5 cache 已完成 `10080/10080`，失败 `0`，全量 finite/shape/sum 审计通过；生成 `vla_workspace/artifacts/semantic_train_v3_audit.json`。
- [x] Flickr30k SD1.5 cache 原有 `9947/9952`，5 条失败均为 Flickr XML HTML entity/标点与 CLIP tokenizer 不一致；建立独立 retry manifest，使用原 seed=17、20 inversion steps、10 inner steps、16×16、相同 SD1.5 配方定向重试，5/5 成功。成功结果保存在 `cache_safety_snapshots/20261006T_retry/overlay`，未覆盖原有 map；包含 retry audit 和 full Flickr audit。
- [x] cache gate 已通过；没有重新生成成功样本。
- [x] OpenVLA/OFT P1 interface 通过：7D action、8-step chunk、双相机 patch grid、归一化 round-trip、forward repeatability 和梯度检查均通过。
- [x] B0 seed17 30-step action smoke 通过：loss/gradient finite，峰值显存约 20.2 GiB，checkpoint manifest、state、offline predictions 已封存。
- [x] 独立新进程 restore 的预测完全一致（`max_prediction_error=0`）。optimizer continuation 由于 BF16 跨进程 parameter digest 差异而首次失败；增加 `loss/grad_norm` `1e-4` 容差与 scheduler exact 检查后，独立 restore retry 通过，证据目录为 `B0_restore_seed17_retry3`。
- [x] 官方 LIBERO episode loop engineering rollout 已完成两个固定初始状态，使用 `MUJOCO_EGL_DEVICE_ID=0` 修复 EGL 设备解析；两个 episode 均完整运行 230 steps、无环境异常，但当前未训练 B0 policy 成功率为 `0/2`。该结果是接口/闭环 smoke，不是方法效果结论。
- [ ] 自动 driver 的旧日志仍保留首次严格 digest 失败记录；正式 rollout 使用 `B0_rollout_retry6` 和通过容差 restore evidence。下一步应修订 driver 的 restore report 路径和 EGL 环境设置，再决定是否重跑统一 driver 或进入 B1 smoke。

- [x] 修复并测试 restore evidence 校验：接受正式 BF16 跨进程 continuation 的 loss/grad_norm `1e-4` 容差、LR/scheduler exact，同时保留旧测试格式兼容；本地 `6 passed, 1 skipped`。远端 `oft_checkpoint.py` 已同步。
- [x] rollout 设备问题已定位并修复运行参数：`CUDA_VISIBLE_DEVICES=0` 单卡时必须显式设置 `MUJOCO_EGL_DEVICE_ID=0`，否则 GPU UUID 会被旧 robosuite 解析器当作整数失败。

- [x] 2026-10-06 18:40（上海时间）重新运行独立官方 LIBERO rollout，设置 `CUDA_VISIBLE_DEVICES=0` 与 `MUJOCO_EGL_DEVICE_ID=0`；`B0_rollout_retry8/summary.json` 两个固定初始状态均完整 `230` steps、`environment_error=null`，成功 `0/2`。
- [x] 该 `0/2` 只用于确认 OpenVLA/OFT checkpoint、action decode、EGL、官方 episode loop 和日志链路可运行；B0 是 30-step 未充分训练工程 smoke，因此不报告为 benchmark 性能，也不据此否定方法。
- [ ] 统一 `run_p6_b0_after_cache.sh` 仍保留旧 strict-digest 日志，尚未重跑整条 driver；需要把 `MUJOCO_EGL_DEVICE_ID` 显式写入 rollout 环境，并让 driver 使用 `B0_restore_seed17_retry3` 的容差 continuation evidence。B1-B4 暂不启动。

- [x] 修订并上传 `server/run_p6_b0_after_cache.sh`：统一 driver 现在显式导出数值 `MUJOCO_EGL_DEVICE_ID`，避免单卡 CUDA UUID 被 robosuite EGL 解析器误读；保留 cache gate、P1、B0、restore、rollout 的顺序，不改变实验定义。
- [x] driver 脚本远端 `bash -n` 通过，SHA256=`01939496b68eb2df410d29d0346be0862c1c7d174fc4f3cd767411d423724e2f`。未重新启动 driver，避免覆盖已存在 B0 结果。

- [x] 2026-10-06 18:45（上海时间）在独立目录 `B0_restore_seed17_driverfix` 复跑 restore；预测 exact restore 仍为 `max_prediction_error=0`，loss/grad_norm 与 uninterrupted reference 一致，LR/scheduler exact，continuation report `passed=true`。
- [x] B0 driver 修复已提交并推送：commit `7e23096`，远端 driver SHA 已校验；当前不再需要人工介入 cache gate 或 EGL 环境设置。
- [x] B0 工程验证边界已冻结：cache→audit→P1→30-step B0→独立 restore→官方 episode loop 均可运行；两个未充分训练 rollout 为 `0/2`，仅作工程 smoke。
- [ ] B1 retention smoke 尚未启动；开始前仍需单独创建 B1 结果目录和明确 B1 的视觉 retention 输入/梯度检查，不得复用 B0 checkpoint 作为 B1 结果。

## 2026-10-06：B1 启动前接口冻结要求

- [x] 已确认 B0 工程链路独立通过：cache audit、OpenVLA P1、30-step B0、独立 restore、官方 episode loop 均有独立证据；B0 rollout 的 `0/2` 仅是未充分训练 checkpoint 的工程 smoke。
- [x] 已确认 C-RADIOv3-L retention cache 为每个 camera `256×1024` FP32 patch feature，OpenVLA P1 的学生视觉块为两路 `16×16` patch，proprio 位于 index `512`；两者空间网格可对齐，但通道维度不同。
- [ ] B1 不能直接复用 B0 脚本或把通用 `vla_bc.py` 合成器当成真实实验；必须增加真实 OpenVLA retention adapter：冻结 C-RADIO teacher，读取 cache，按 camera/provenance 对齐 student patch，使用明确的 trainable projector `student_dim→1024`，并记录 projector 初始化、参数组和 retention 梯度。
- [ ] B1 首个运行只做独立的 20–30 step interface smoke，输出目录必须为新的 `B1_retention_interface_seed17`，不覆盖 B0、cache 或 restore 目录；检查 action loss、retention loss、teacher 无梯度、student/projector 梯度 finite、两路 patch 数/坐标一致和 checkpoint restore。
- [ ] B1 smoke 通过后，才允许冻结 B1 三 seed 的正式配置；B2/B3/B4 仍必须逐项 smoke，不能因 B1 通过而批量启动。
- [ ] 当前尚未启动 B1 GPU 作业。

## 2026-10-06：B1 C-RADIO retention interface smoke

- [x] 新增真实 OpenVLA/OFT B1 runner：`scripts/run_oft_b1_retention_smoke.py`。该 runner 复用原生 OFT action forward，读取冻结 C-RADIOv3-L cache，并用显式 `4096→1024` trainable projector 对齐两路 `16×16` patch；不修改 B0 或任何 cache。
- [x] 远端脚本已上传并通过 `oft` 环境 `py_compile`；远端 SHA256=`44e3a5ee52f9756711f3f45cc67845e91ebdb26e50811b8d4348284e79db091a`。
- [x] 使用 GPU 0、独立资源窗口 `vla_workspace/artifacts/gpu_window_b1.json`、seed=17、lambda_retention=0.05、20 steps 运行完成。
- [x] 结果目录：`/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_b0_b4/B1_retention_interface_seed17`；日志：`vla_workspace/logs/b1_retention_interface_seed17.log`。
- [x] B1 report 通过：学生 patch `[1,512,4096]`、teacher `[1,512,1024]`、projector `[4096,1024]`；20/20 steps action/retention/total loss finite，梯度 finite，teacher frozen 且无 teacher gradient，modules save/restore probe finite。
- [x] B1 使用的 C-RADIO manifest SHA256=`2c47200466c63e43c187b6abad68b533ec07ffd3f9ad722c77537672179478ac`；OpenVLA revision=`47a0ec7fc4ec123775a391911046cf33cf9ed83f`；train/eval/statistics SHA 与 P1 一致。
- [ ] 该结果是 retention 接口工程 smoke，不是 B1 benchmark 或 BlindVLA 性能结论；下一步先冻结 B1 配置并准备 B2 semantic attribution interface smoke，仍不得启动长训练或批量 B2–B4。

## 2026-10-06：B2 semantic attribution interface smoke

- [x] 新增 `scripts/run_oft_b2_semantic_smoke.py`，使用真实 OpenVLA/OFT action-loss gradient×activation 生成两路 `16×16` 学生归因图，并读取冻结 SD1.5 null-text semantic cache；不重新生成 semantic cache。
- [x] 经过三次最小修复后通过：sidecar 字段验证、semantic sample path、视觉 backbone gradient hook；没有使用 `allow_unused=True` 掩盖无梯度问题。
- [x] 远端结果目录：`/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_b0_b4/B2_semantic_interface_seed17`；日志：`vla_workspace/logs/b2_semantic_interface_seed17.log`。
- [x] B2 20/20 steps 通过：action loss、semantic loss、total loss、梯度均 finite；学生图 `[1,2,256]`、teacher 图 `[1,512]`（两路拼接后等价 `[1,2,256]`）；teacher frozen；semantic manifest SHA=`d7660c9c8286483ebeeafee2d8ff2100dce99ddc9151b35f4beb03c3e3fb87fe`。
- [x] 当前 B2 仍是接口工程 smoke，不是性能结论；semantic loss 使用 lambda=0.10、temperature=1.0，正式值仍需按 VLM 冻结协议确认。
- [ ] 下一步实现并运行独立 B3 dynamic action-conditioned attribution smoke：复用 B1 retention、B2 semantic，增加 action-conditioned `A_act`/language support containment；不启动 B4 或长训练。

## 2026-10-06：B3 dynamic action-attribution interface smoke

- [x] 新增 `scripts/run_oft_b3_dynamic_smoke.py`，以真实 OpenVLA action-loss gradient×activation 得到 `A_act`，拼接两路 16×16 patch，并联合计算 SD semantic alignment 与 containment。
- [x] 20/20 steps 通过，action/semantic/containment/total loss 与梯度均 finite；结果目录：`/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_b0_b4/B3_dynamic_action_interface_seed17`；日志：`vla_workspace/logs/b3_dynamic_action_interface_seed17.log`。
- [x] lambda_sem=0.10、lambda_contain=0.02；semantic manifest SHA=`d7660c9c8286483ebeeafee2d8ff2100dce99ddc9151b35f4beb03c3e3fb87fe`；OpenVLA revision 与 train/eval/statistics SHA 与 P1 一致。
- [ ] 重要边界：本轮 `A_act` 来自当前 chunk action loss，具备输出条件性；language support 采用 source/target map 的并集代理，尚非按 LIBERO 弱阶段标签构建的逐时刻 source→mixed→target phase support。因此只能称为动态 attribution 接口 smoke，不能声称完整 phase transition 已验证。
- [ ] B3 脚本已提交为 `03b4f9e`，远端已上传且编译通过；GitHub push 遇 TLS handshake failure，需恢复网络后推送 commit。
- [ ] 下一步准备 B4 组合 smoke（B1 retention + B2 semantic + B3 containment）；仍是 20-step engineering gate，不进入长训练。

## 2026-10-06：B4 combined interface smoke

- [x] 新增并上传 `scripts/run_oft_b4_combo_smoke.py`，组合 C-RADIO retention、SD semantic attribution 和 action-conditioned containment；不修改任何 cache/B0/B1/B2/B3 结果。
- [x] 使用 GPU 0、seed=17、20 steps、lambda_retention=0.05、lambda_semantic=0.10、lambda_containment=0.02 运行通过。
- [x] 结果目录：`/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P6_vla_b0_b4/B4_combined_interface_seed17`；日志：`vla_workspace/logs/b4_combined_interface_seed17.log`。
- [x] 20/20 steps action/retention/semantic/containment/total loss 与梯度均 finite；A_act 和 semantic support 均为 `[1,512]`；C-RADIO projector shape 为 `2176→1024`；teacher frozen。
- [ ] B4 与 B3 一样使用 source/target semantic map 并集作为 support proxy，尚未使用正式 LIBERO phase label；因此只是组合接口 smoke，不是最终方法或性能结果。
- [x] B0、B1、B2、B3、B4 的独立工程 smoke 现在均有通过证据；没有启动任何长训练。
- [ ] 下一步：冻结 B0–B4 正式配置和 control matrix（wrong-map、random-map、raw-attention proxy、retention-only、semantic-only），然后再进入正式小规模/全量训练；继续记录每个 run 的 config/manifest/Git SHA。

## 2026-10-06：B0–B4 smoke 后正式训练前冻结配置

- [x] 新增独立冻结配置：`configs/vla/p6_b0_b4_frozen_seed17_smoke.json`；不修改仍保留占位符的旧模板 `p6_b0_b4.json`。
- [x] 写入已由真实 P1/smoke 证实的 OpenVLA revision、双相机、8×7 action chunk、SD semantic manifest SHA、C-RADIO manifest SHA、lambda 和 attribution 定义。
- [x] 预注册 controls：correct teacher、wrong-word、wrong-image、area-matched random、raw-attention proxy、retention-only、semantic-only、phase-shuffled support。
- [ ] 正式训练仍未启动：effective batch、gradient accumulation、官方 evaluator commit、正式 rollout episode manifest 和三 seed controls 结果需要先冻结；当前配置状态明确为 `smoke_frozen_formal_training_pending_controls`。

## 2026-10-06：正式训练前配置准入检查

- [x] 新冻结配置 `configs/vla/p6_b0_b4_frozen_seed17_smoke.json` 已通过 `scripts/validate_vla_config.py`，矩阵严格为 B0–B4，报告动作 schema 为 delta_eef_7d，seeds 为 17/29/41，offline-first 和 episode-cluster bootstrap 已冻结。
- [x] 修复验证器兼容带版本后缀的冻结 experiment_id，并将缺省 `drop_last` 按 false 验证；提交 `9235e9a` 已推送。
- [ ] 正式长训练仍未启动。仍需在 vla101 上冻结并审计 effective batch、gradient accumulation、正式 rollout manifest/evaluator 版本，并运行 controls smoke 后才可进入 10k-step 训练。
