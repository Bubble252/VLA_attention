# P6 VLA B0–B4：可冻结字段与未决项（2026-10-07）

本文件把“已经有真实证据支持的配置”和“还不能靠猜冻结的配置”分开。它是审计记录，不是正式训练启动令牌；正式启动仍必须经过 `validate_vla_run_contract.py` 和各项 launch gate。

## 1. 现在可以冻结的内容

### 1.1 实验范围

- 首轮骨干固定为 **OpenVLA/OFT 原生 VLA**；不把 StarVLA 作为隐含动作头，也不把 π0.5、MolmoAct2、LingBot-VLA 混入首轮主表。
- 首轮 benchmark 固定为 **LIBERO-Spatial 的 task0 pick-place**，使用 RGB、语言指令和原生 proprioception，双视角为 `image`（第三人称）与 `wrist_image`（腕部相机）。
- 首轮先做 offline action prediction 和接口/恢复验证；闭环 rollout 单独过 provenance gate，不能用工程 smoke 代替正式结果。
- B0–B4 共享同一 backbone、episode manifest、动作头、seed、训练预算和 evaluator；只改变 retention、semantic 和 containment 开关。

### 1.2 模型、代码和数据版本

| 字段 | 冻结值 | 证据 |
|---|---|---|
| OpenVLA checkpoint | `47a0ec7fc4ec123775a391911046cf33cf9ed83f` | `p6_b0_b4_formal_preflight_20261006.json`、P1/B0/B1–B4 reports |
| OpenVLA/OFT 外部代码 | `e4287e94541f459edc4feabc4e181f537cd569a8` | `PROGRESS.md` 2026-10-06 版本审计 |
| LIBERO inventory | `rlds_inventory_v2_20260928` | 101 上的固定 manifest |
| train manifest | SHA256 `695681afb03012bd18c34d09c9a4fbf15ca489605d4463cbf891cd7213bca921` | `vla_b0_offline_preflight_audit_20261006.json` |
| offline eval manifest | SHA256 `c93ae5afeb919de8fb148bfa1f935173d04f3ac0502b85e3b6916afb15e9dd0b` | 同上 |
| action statistics | SHA256 `21692284b4cbf9cdbdb2b3e2060ac4e62924eabf2b6db1881292e47f3b830397` | 同上 |
| rollout init inventory | SHA256 `c4dae51e9b48616af4f2a38463821a851a91c31b8dce63f1bcc75bde08c03a7f` | `formal_rollout_entry_audit_20261006.json` |

### 1.3 观察和图像预处理

以下链路已有 101 上的真实源码和文件 hash，可以固定到配置中：

1. LIBERO 环境图像为 `256×256×3`；manifest 中两个相机的 shape 均已核验。
2. 官方 `run_libero_eval.py` 的 `prepare_observation` 读取 `image` 和 `wrist_image`，调用 `resize_image_for_policy`。
3. `resize_image_for_policy` 使用 TensorFlow `encode_jpeg → decode_image → tf.image.resize(method=lanczos3, antialias=True)`，目标尺寸为 `224×224`。
4. OpenVLA processor 的 `preprocessor_config.json` 和 `processor_config.json` 已在 101 上固定；对应 hash 为：
   - `preprocessor_config.json`: `bd024783fb785cbdfe35b710b621c5e8678d4c157a9ba625675e67d40d1757da`
   - `processor_config.json`: `167e026d8121f6cb0bf5c18310b97ba8139152f970822bb3824213bd37dc7a76`
5. evaluator 源文件 hash 为 `701a20b6ca2942aa36e0e7d567b25647268b936a23af79c5893534f10bec7802`；相关 preprocessing helper 的 hash 已记录在远端审计命令输出中。

因此，`camera_views`、相机顺序、原始分辨率、resize 算子、目标分辨率和 processor revision 已经可以冻结。尚未冻结的是“正式训练脚本是否逐字复用同一 preprocessing wrapper”，这是实现一致性检查，不是研究方向选择。

### 1.4 动作接口

- 原生 LIBERO action 为 `7` 维、`H=8` chunk，原生训练目标为 OFT continuous action 的 L1 regression。
- 统一报告接口固定为 `delta_eef_7d = [dx, dy, dz, droll, dpitch, dyaw, dgrip]`。
- 训练和策略内部保留 checkpoint 的归一化；LIBERO 常量已确认为 `BOUNDS_Q99`，不能手工 min–max。
- 报告时将平移解码为米、旋转报告为 axis-angle/弧度，夹爪沿用官方符号和 `process_action` 的转换。
- 正式训练/rollout 使用 `H=8`；`H=1` 只保留给首步归因、restore 和接口诊断。

### 1.5 teacher、归因和 cache

- retention teacher 固定为 `nvidia/C-RADIOv3-L`，revision `9d0413465e8a91e67bbf2c1ad342815478d1b906`，输出为双相机拼接的 `512×1024`（每视角 `256×1024`）patch features；radio cache manifest SHA 为 `2c47200466c63e43c187b6abad68b533ec07ffd3f9ad722c77537672179478ac`。
- semantic teacher 在首轮操作上固定为 **SD1.5 null-text reconstruction fallback**，公共 map grid 为 `16×16`，LIBERO semantic cache manifest SHA 为 `d7660c9c8286483ebeeafee2d8ff2100dce99ddc9151b35f4beb03c3e3fb87fe`，cache 审计为 `10080/10080` 有效、finite/load failure 为 0。
- 这表示“首轮用哪个 teacher”已经可以落地；它不等价于声称 SD1.5 在四个候选中获得了正式 best-single。PixArt-α/Σ 与 Playground-v2.5 目前缺少同一 inversion/map adapter 的等价验收，作为后续 teacher-swap。
- `A_act` 首版固定为 OpenVLA visual backbone 输出上的 **action-loss gradient × activation**；当前真实 hook 路径为 `policy.base_model.model.vision_backbone`，双视角共 `512` patch。raw attention 只作 diagnostic proxy。corrected B3 smoke 已按该定义实现 retention+semantic、无 containment。
- B1 retention 使用显式 `4096→1024` trainable projector；teacher 参数、semantic cache 和 radio cache 都不进入 optimizer。

### 1.6 优化和训练预算

首轮候选工程值为：AdamW、cosine scheduler、`lr=5e-4`、`weight_decay=0.01`、effective batch `1`、`gradient_accumulation=1`、seed `17/29/41`、4 个确定性 pass、`10080` optimizer steps，checkpoint 在 `2520/5040/7560/10080` 保存；不使用 early stopping。batch=1 不是经过 batch-size sweep 证明的最优值，而是当前与“小数据集 4 pass”及“至少 10k optimizer updates”同时一致的候选值。

已有 B0 smoke 只验证了 batch=1、accumulation=1 可运行，峰值显存为 `21,618,780,160` bytes（约 `20.2 GiB`）；B1–B4 interface smoke 也只用单样本 collate。该结果支持工程可运行性，不证明 batch=1 必需或最优，也不证明更大 batch 在带归因保留图的 B2–B4 上可运行。`lambda_retention=0.05`、`lambda_containment=0.02` 和 warm-up `500/1000` 已有 smoke 依据，可以作为首轮 VLA 候选固定。`lambda_sem=0.10` 已继承 VLM 冻结协议；其 temperature 仍有一个需要单独解决的冲突，见第 3 节。

## 2. 已经有证据、但不能当成正式效果的内容

- B0 独立 50-step offline preflight：finite train/eval、9 个 held-out episode、checkpoint restore 前置条件通过；action L1 仅是工程基线，不是 LIBERO success。
- B0 restore：新进程 prediction equality 为 exact，最大误差 `0.0`。
- B1 当前 radio manifest 修正后的 20-step retention interface smoke：通过，teacher frozen，`[256,1024]` target 和 `4096→1024` projector shape 一致。
- B2、旧 B3、B4 以及负控的 20-step 运行均有产物，但这些产物是接口 smoke，不能转写成性能或机制结论。
- corrected B3 `retention+semantic, containment=0` 的 `temperature=1.0/1.25` 两个 20-step smoke 均通过；报告保存在 `experiment_workspace/vla_remote_results/20261007_b3_temperature/`。这只证明矩阵和数值接口可运行，不是性能结论。
- 现有 cache 均通过 tensor shape、finite、load 检查；Flickr retry overlay 没有覆盖原 cache，不能把 retry 元数据残留误写成数组损坏。

## 3. 现在还不能冻结的内容

### 3.1 B3 矩阵和旧脚本不一致（已修正，旧结果保留）

协议矩阵定义：

| 组 | retention | semantic | containment |
|---|---:|---:|---:|
| B0 | 0 | 0 | 0 |
| B1 | 1 | 0 | 0 |
| B2 | 0 | 1 | 0 |
| B3 | 1 | 1 | 0 |
| B4 | 1 | 1 | 1 |

旧的 `scripts/run_oft_b3_dynamic_smoke.py` 实际加的是 `semantic + containment`，没有 radio retention，因此旧产物不能证明协议意义上的 B3。已新增 `scripts/run_oft_b3_retention_semantic_smoke.py`，严格实现 retention+semantic、无 containment，并保留旧 B3 产物不覆盖。

### 3.2 `teacher_temperature` 的协议冲突（已冻结）

VLM 冻结配置是 `temperature=1.25`，历史 VLA B2/B3/B4 smoke 曾使用 `temperature=1.0`。已在同一 seed、相同 manifest/order、相同 20-step corrected B3 上比较两个值；两者均 finite，未出现明显梯度或 loss 数值异常。按预注册规则冻结 `temperature=1.25`，因为它与 VLM 协议一致。该 smoke 不用于选择性能最优值。

### 3.3 正式 rollout 的 demo provenance

train/eval episode disjointness、rollout init 唯一性和官方 evaluator 源码均已验证，但 rollout inventory 目前明确写着 `Demo-to-init-state linkage not verified`。在获得初始状态来源与 demo 文件的可追溯映射前，只能做 offline B0–B4，不能报告 LIBERO 闭环 success。

### 3.4 evaluator 的“smoke 次数”和“正式次数”

源码默认 `num_trials_per_task=50`；当前工程配置曾使用 10 作为快速 gate，task0 的最大控制 horizon 为 `220 + 10` 等待步，action chunk 为 8。建议冻结为：`10` 只用于 engineering smoke，`50` 用于最终官方结果；两者必须分开命名，不能混进同一主表。

### 3.5 长训练入口还没有和 B0–B4 矩阵完全对齐

当前有 B0 smoke、B1–B4 interface scripts、loss/adapter 合同和配置，但还没有一个经过 B3 修正、控制项、checkpoint cadence、逐 step metrics 和恢复逻辑统一验收的正式 `B0–B4 10080-step` launcher。因此“参数已知”不等于“可以提交长作业”。

### 3.6 effective batch 与训练暴露量的绑定

当前 train manifest 有 `2520` 个完整 H=8 transition。effective batch=1 时，每个 pass 是 `2520` optimizer updates，4 个 pass 正好是 `10080` updates。若 batch=2 且仍只训练 4 个 pass，则是 `5040` updates；若坚持 `10080` optimizer updates，则数据会被重复约 8 个 pass。gradient accumulation 也会改变 effective batch 和 optimizer update 数，不能只改一个字段而保持“训练预算相同”的说法。

因此 batch=1 的当前理由是与已约定的更新数和数据遍历次数匹配，并且是唯一经过真实 B0 smoke 验证的 batch；显存读数只证明该配置的运行峰值。若要把 batch 2/4 作为正式候选，需先明确比较口径：固定数据遍历次数，还是固定 optimizer updates，并对 B1–B4 的 attribution backward 单独做 OOM/吞吐 smoke。不能仅凭 B0 的显存余量推断 B2–B4 的 batch 安全。

## 4. 推荐的收敛顺序

1. 建立并验证统一正式 B0–B4 launcher。
2. 先提交 offline B0–B4 长训练；正式 rollout 等 demo provenance 解决后再开。
3. evaluator 配置明确区分 `10` 次 engineering smoke 与 `50` 次最终 official trials。

## 5. 已确认边界与剩余 gate

已确认采用“**SD1.5 fallback + offline B0–B4 先行 + B3 修正和 temperature 小 smoke 后提交 10k**”的路线；PixArt-α/Σ、Playground-v2.5 作为后续 teacher-swap，正式 rollout success 等 provenance 修复后再做。剩余事项是可执行 gate，不是待确认的研究方向：

1. corrected B3 和 semantic temperature gate 已完成，正式配置冻结为 `temperature=1.25`；
2. 完成统一正式 launcher 和 offline B0–B4 训练；
3. 保持 batch=1 作为首轮候选，除非新增 batch-size 实验改变了训练预算口径并有真实 B1–B4 显存/稳定性证据。
