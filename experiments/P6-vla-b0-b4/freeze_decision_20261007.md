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
- `A_act` 首版固定为 OpenVLA visual backbone 输出上的 **action-loss gradient × activation**；当前真实 hook 路径为 `policy.base_model.model.vision_backbone`，双视角共 `512` patch。raw attention 只作 diagnostic proxy。
- B1 retention 使用显式 `4096→1024` trainable projector；teacher 参数、semantic cache 和 radio cache 都不进入 optimizer。

### 1.6 优化和训练预算

首轮可先固定以下工程值：AdamW、cosine scheduler、`lr=5e-4`、`weight_decay=0.01`、`batch=1`、`gradient_accumulation=1`、seed `17/29/41`、4 个确定性 pass、`10080` optimizer steps，checkpoint 在 `2520/5040/7560/10080` 保存；不使用 early stopping。

smoke 的峰值显存约 `21.62 GB`，支持 `batch=1, accumulation=1`。`lambda_retention=0.05`、`lambda_containment=0.02` 和 warm-up `500/1000` 已有 smoke 依据，可以作为首轮 VLA 候选固定。`lambda_sem=0.10` 已继承 VLM 冻结协议；其 temperature 仍有一个需要单独解决的冲突，见第 3 节。

## 2. 已经有证据、但不能当成正式效果的内容

- B0 独立 50-step offline preflight：finite train/eval、9 个 held-out episode、checkpoint restore 前置条件通过；action L1 仅是工程基线，不是 LIBERO success。
- B0 restore：新进程 prediction equality 为 exact，最大误差 `0.0`。
- B1 当前 radio manifest 修正后的 20-step retention interface smoke：通过，teacher frozen，`[256,1024]` target 和 `4096→1024` projector shape 一致。
- B2、旧 B3、B4 以及负控的 20-step 运行均有产物，但这些产物是接口 smoke，不能转写成性能或机制结论。
- 现有 cache 均通过 tensor shape、finite、load 检查；Flickr retry overlay 没有覆盖原 cache，不能把 retry 元数据残留误写成数组损坏。

## 3. 现在还不能冻结的内容

### 3.1 B3 矩阵和脚本不一致（最高优先级）

协议矩阵定义：

| 组 | retention | semantic | containment |
|---|---:|---:|---:|
| B0 | 0 | 0 | 0 |
| B1 | 1 | 0 | 0 |
| B2 | 0 | 1 | 0 |
| B3 | 1 | 1 | 0 |
| B4 | 1 | 1 | 1 |

但当前 `scripts/run_oft_b3_dynamic_smoke.py` 实际加的是 `semantic + containment`，没有 radio retention；它不能证明协议意义上的 B3。`run_oft_b4_combo_smoke.py` 才包含三项 loss。因此需要新增/修正一个“retention + semantic、无 containment”的 B3 smoke，并重新写 audit；历史 B3 产物保留，不覆盖。

### 3.2 `teacher_temperature` 的协议冲突

VLM 冻结配置是 `temperature=1.25`，当前 VLA B2/B3/B4 smoke 命令使用过 `temperature=1.0`。这不是可以凭经验挑一个的细节：它会影响 semantic loss 的尺度和与 VLM 的可比性。建议在同一 seed、同一 batch、20 step 做 `1.0 vs 1.25` paired interface smoke，只看 finite、loss ratio、gradient ratio 和 restore，不按短期 action loss 选优；随后把一个值写入唯一正式 config。

### 3.3 正式 rollout 的 demo provenance

train/eval episode disjointness、rollout init 唯一性和官方 evaluator 源码均已验证，但 rollout inventory 目前明确写着 `Demo-to-init-state linkage not verified`。在获得初始状态来源与 demo 文件的可追溯映射前，只能做 offline B0–B4，不能报告 LIBERO 闭环 success。

### 3.4 evaluator 的“smoke 次数”和“正式次数”

源码默认 `num_trials_per_task=50`；当前工程配置曾使用 10 作为快速 gate，task0 的最大控制 horizon 为 `220 + 10` 等待步，action chunk 为 8。建议冻结为：`10` 只用于 engineering smoke，`50` 用于最终官方结果；两者必须分开命名，不能混进同一主表。

### 3.5 长训练入口还没有和 B0–B4 矩阵完全对齐

当前有 B0 smoke、B1–B4 interface scripts、loss/adapter 合同和配置，但还没有一个经过 B3 修正、控制项、checkpoint cadence、逐 step metrics 和恢复逻辑统一验收的正式 `B0–B4 10080-step` launcher。因此“参数已知”不等于“可以提交长作业”。

## 4. 推荐的收敛顺序

1. 修正并运行真正的 B3 retention+semantic smoke；不改任何 cache，也不覆盖旧 B3 结果。
2. 做 `temperature=1.0/1.25` 的 20-step paired smoke，冻结一个值。
3. 把上面第 1 节写入单独的 offline candidate config，保留第 3 节四个 gate 为显式状态。
4. 先提交 offline B0–B4 长训练；正式 rollout 等 demo provenance 解决后再开。
5. 最终 evaluator 使用 50 trials；10 trials 仅用于入口和故障排查。

## 5. 需要共同决定的边界

我建议采用“**SD1.5 fallback + offline B0–B4 先行 + B3 修正和 temperature 小 smoke 后提交 10k**”的路线。这样不等待 PixArt/Playground，也不把未验证 rollout 混入主结论。真正需要你确认的只有两点：

1. 是否接受先把 SD1.5 作为首轮 VLA 的 operational teacher，四 teacher 横向比较放到扩展实验；
2. 是否接受把“正式 rollout success”后移到 provenance 修复后，而先完成 offline B0–B4。

这两点若接受，剩余工作就是可执行的工程 gate，不再需要重新讨论方法方向。

