# OpenVLA-OFT × LIBERO：B0 前置静态审计

审计日期：2026-09-28。范围：本地官方源码、现有项目接口与配置；没有模型加载、GPU 使用、训练、模拟器 rollout、远端修改或 Git commit。此报告属于文档优先的准备阶段，不代表 P1/GPU smoke 已通过，也不是论文结果复现。PaperReader 子任务在本次按父任务要求限缩为代码证据审计，未重新做全文论文提取。

## 1. 结论与版本

可冻结的源码事实是：LIBERO 的 OFT 原生预测为 **8 步 × 7 维连续动作、8 维 proprio、双相机、L1 行为克隆损失**。真正的 B0 仍缺起始模型 snapshot、真实演示文件及 episode 划分、可执行训练接口、环境 lock 和随后单独授权的 GPU/rollout gate。现有 `OpenVLAOFTAdapter` 只是 schema 边界，不是模型训练器。

| 源码 | 本地 HEAD（完整 SHA） | 来源 / 状态 |
|---|---|---|
| OpenVLA-OFT | `e4287e94541f459edc4feabc4e181f537cd569a8` | `https://github.com/moojink/openvla-oft.git`；审计开始时 `git status --short` 为空 |
| LIBERO | `8f1084e3132a39270c3a13ebe37270a43ece2a01` | `https://github.com/Lifelong-Robot-Learning/LIBERO.git`；审计开始时 `git status --short` 为空 |

以下 `OFT:` 指 `references/repos/openvla-oft/`；`LIBERO:` 指 `references/repos/libero/`；行号均对应上表 commit。Git SHA 是本地 `git rev-parse HEAD` 的直接输出；HF checkpoint revision 与依赖 fork SHA **尚未由本报告核验**。

## 2. 原生动作、损失与训练配置

| 内容 | 已核验实现 | 证据 |
|---|---|---|
| chunk / action / proprio | `NUM_ACTIONS_CHUNK=8`、`ACTION_DIM=7`、`PROPRIO_DIM=8` | `OFT:prismatic/vla/constants.py:26` |
| 平台选择 | 从 `sys.argv` 字符串推断平台，找不到时默认 LIBERO；不是 checkpoint 自描述配置 | `OFT:prismatic/vla/constants.py:49` |
| 输入时序 | 1 帧当前观察，对应当前动作及未来 7 步 | `OFT:prismatic/vla/datasets/datasets.py:130` |
| 轨迹末端 | 当前实现将轨迹有效长度减去 7；长度 T 产生 T−7 个完整 chunk，不应自行复制末动作补满；T<8 必须显式处理 | `OFT:prismatic/vla/datasets/rlds/traj_transforms.py:25` |
| 连续动作头 | 最后一层 56 个 action token hidden states，每 7 个拼接，2-block MLPResNet 输出每步 7 维 | `OFT:prismatic/models/action_heads.py:84`；`OFT:vla-scripts/finetune.py:374` |
| B0 原生损失 | `torch.nn.L1Loss()(ground_truth_actions, predicted_actions)`，默认全 batch/chunk/dim mean；不是 MSE，也不是 token CE | `OFT:vla-scripts/finetune.py:386` |
| 并行预测 | 插入 56 个占位 action token 与 stop token；需要 OFT 的双向注意力 fork | `OFT:prismatic/extern/hf/modeling_prismatic.py:734`；`OFT:pyproject.toml:50` |
| 归一化 | q01/q99 bounds；前 6 维 action 归一化，夹爪不做同样的 bounds 归一化 | `OFT:prismatic/vla/constants.py:30`；`OFT:prismatic/vla/datasets/rlds/oxe/materialize.py:35` |
| gripper | RLDS 转换后 0=close、1=open；rollout 映射到 ±1 并翻转，环境 −1=open、+1=close | `OFT:prismatic/vla/datasets/rlds/oxe/transforms.py:827`；`OFT:experiments/robot/libero/run_libero_eval.py:265` |
| proprio | eef xyz + quaternion 转 axis-angle 3 维 + 两个 gripper qpos；共 8 维 | `OFT:experiments/robot/libero/run_libero_eval.py:253`；`OFT:experiments/robot/libero/libero_utils.py:63` |
| 控制器 | 默认 `OSC_POSE`、20 Hz；载入 robosuite 默认 controller config | `LIBERO:libero/libero/envs/env_wrapper.py:17`、`:27`、`:47` |

**单位仍不能写死为米/弧度。** proprio 的 axis-angle 定义已核验；这不等于动作旋转分量已经证实为 Euler 或未经缩放的弧度。OSC_POSE 输入缩放、output limits 和实际控制器版本需要对安装后的 robosuite 配置核验。项目配置当前的 `[droll, dpitch, dyaw, dgrip]` 只是草稿报告字段；最后一维实际为绝对夹爪命令，不是夹爪增量。训练保留官方原生表示，单位转换须经 roundtrip 后才作报告。

官方 LIBERO recipe：基座 `openvla/openvla-7b`，L1=true、diffusion=false、FiLM=false、两相机、proprio=true、LoRA rank 32、每 GPU batch 8、lr=5e−4、100k steps 后降 10 倍、150k 训练步、image_aug=true。证据：`OFT:LIBERO.md:98`。LoRA target 是 `all-linear`，alpha=min(rank,16)：`OFT:vla-scripts/finetune.py:845`。优化器是 AdamW，调度器 **MultiStepLR，不是 cosine**：同文件 `:935`。项目可另定短预算 B0，但必须把这些差异标为项目实验设置，不能称为完整复现官方 150k 结果。

## 3. 相机及增强链

官方 rollout 顺序为：

1. 从模拟器读取 `agentview_image`、`robot0_eye_in_hand_image`，默认渲染 256×256；两者均 `img[::-1, ::-1]`，即旋转 180°。证据：`OFT:experiments/robot/libero/libero_utils.py:18`、`:33`。
2. 各图经过 **JPEG encode/decode → TensorFlow Lanczos3 antialias resize → round/clip → uint8**，至 224×224；不能假设 PIL bilinear 等价。证据：`OFT:experiments/robot/openvla_utils.py:520`；`OFT:experiments/robot/robot_utils.py:33`。
3. `center_crop=True`：保留中心 **90% 面积**，边长比例 sqrt(0.9)，再通过 TF crop_and_resize 至 224×224。证据：`OFT:experiments/robot/openvla_utils.py:546`、`:596`。
4. 交给 checkpoint 对应 processor。输入顺序 first `full_image`、then `wrist_image`，处理后的 tensors 按 dim=1 拼接；prompt 为 `In: What action should the robot take to {task_label.lower()}?\nOut:`。证据：同文件 `:745`–`:770`。

训练的 RLDS reader 使用 primary→wrist，`image_aug=True` 包含面积固定 0.9、ratio=1 的随机裁剪及 brightness 0.2、contrast/saturation [0.8,1.2]、hue 0.05，顺序见 `OFT:prismatic/vla/datasets/datasets.py:117`、`:150`。训练 augmentation 的随机 crop 与评估 center crop 必须分别记录。对已转换的 RLDS 图片不能无证据再旋转一次；必须从实际记录核验存储方向。归因图/teacher map 必须继承实际图像几何变换，相机不可交换。

## 4. 起始 checkpoint 与污染边界

**B0 训练初始化：** 官方训练入口用 `openvla/openvla-7b`，随后初始化 OFT 的 continuous action head / proprio projector，并注入 LoRA；见 `OFT:LIBERO.md:100`、`OFT:vla-scripts/finetune.py:834`、`:877`。应冻结基座 HF commit、各权重 SHA256、processor/tokenizer/custom code，并对 B0–B4 使用相同初始化种子和同一初始 head/projector。

**公开 LIBERO OFT checkpoint：** `moojink/openvla-7b-oft-finetuned-libero-{spatial,object,goal,10}` 已分别对对应完整套件训练，联合 checkpoint 已对四套件训练。证据：`OFT:LIBERO.md:40`–`:50`。它们适合官方 inference/rollout sanity check、外部已训练 baseline；若作为 B0–B4 起点，只能声称“从 LIBERO 已微调策略继续训练”，不能声称当前 held-out 演示对初始化未见。

本次可以确认**训练套件重叠风险**；没有 checkpoint 的逐 episode lineage，不能捏造具体 episode 已重叠或已消除重叠的结论。事后拆分同一批 demos 无法消除既有 checkpoint 的训练暴露。基座 OpenVLA 的预训练数据与实际选定测试集是否重叠，也应单独做 provenance 审计；仅因使用 base model 就宣称完全无污染，同样没有证据。

加载公开 OFT 模型还需要单独的 `action_head--*_checkpoint.pt` 与 `proprio_projector--*_checkpoint.pt`，并非只下载 safetensors 即可。Spatial/Object/10 对应 150000、Goal 对应 50000、四套件联合对应 300000：`OFT:experiments/robot/openvla_utils.py:413`、`:495`。归一化 stats key 如 `libero_spatial_no_noops` 必须匹配真实数据和模型。

## 5. 任务 ID、真实数据及 split

任务身份必须至少记录 `(suite, task_order_index, task_id, task_name, bddl_sha256)`。默认 `task_order_index=0` 是 `[0..9]`；其他顺序会改变数值 ID 的含义：`LIBERO:libero/libero/benchmark/__init__.py:83`、`:111`。在默认顺序下：

| suite | task_id | canonical task_name | 源码 |
|---|---:|---|---|
| libero_spatial | 0 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | `LIBERO:libero/libero/benchmark/libero_suite_task_map.py:3` |
| libero_spatial | 1 | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 同文件 `:4` |
| libero_spatial | 2 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 同文件 `:5` |
| libero_object | 0 | pick_up_the_alphabet_soup_and_place_it_in_the_basket | 同文件 `:15` |

Spatial 的 task 0 可作为最小准备候选，但这里不替用户冻结最终任务。`libero_10` 是 LIBERO-Long 的套件名，不能理解为“task_id=10”；`libero_90` 的训练映射没有在 OFT 上述默认 modified datasets 中注册，不能仅换字符串就训练。

原始/再生成 LIBERO 是每任务 `<suite>/<task_name>_demo.hdf5`，内部 `data/demo_i` 下含 `actions`、`states`、`robot_states`、`rewards`、`dones`，以及 `obs/{agentview_rgb,eye_in_hand_rgb,ee_states,gripper_states,...}`。证据：`LIBERO:libero/libero/benchmark/__init__.py:144`；`OFT:experiments/robot/libero/regenerate_libero_dataset.py:178`。再生成过程去除 no-op 并只保留成功演示；no-op 条件考虑前 6 维范数 <1e−4 及夹爪是否变化：同文件 `:46`。丢弃哪些帧与 demos 需要有原始索引映射。

OFT 训练消费的不是任意 HDF5，而是 TFDS/RLDS：`libero_spatial_no_noops`、`libero_object_no_noops`、`libero_goal_no_noops`、`libero_10_no_noops`，官方来源是 `openvla/modified_libero_rlds`（`OFT:LIBERO.md:29`）。RLDS 标准化前的观察 keys 是 `image`、`wrist_image`、`state`；另外需要 `action` 和 `language_instruction`。`state[:6]` 和 `state[-2:]` 成为 proprio，见 `OFT:prismatic/vla/datasets/rlds/oxe/configs.py:645` 与 `.../transforms.py:827`。

**划分不是已经解决的事：** 当前 reader 直接要求 `train` / `val` split，不会自动创建项目要求的 episode manifest；是否存在 `val` 必须检查下载数据的 `dataset_info.json`。证据：`OFT:prismatic/vla/datasets/rlds/dataset.py:233`。更关键的是，默认统计量从 `split="all"` 计算（同文件 `:202`），若项目定义严格 held-out offline eval，应先完成 episode 隔离，再从 train-only 计算 action/proprio stats 或明确披露既有 stats 的范围。默认 restructure 还会丢掉未显式保留的 episode metadata（同文件 `:182`），因此必须在此前建立可追踪 episode→record 清单，不能在打乱后的 transition 层临时拆分。

本次未打开真实 HDF5/TFRecord；实际 episodes 数、成功比例、原始 ID→RLDS ID 映射、帧方向、train/val metadata、磁盘完整性均未核验。原方案单任务 100–200 train episodes 是否可获得，不能从任务列表推断，须按真实数据量修订，禁止复制同一 demo 充当新 episode。

## 6. 评估契约与一个单步陷阱

官方默认每任务 50 trials，10 步 no-op 稳定物体；Spatial/Object/Goal/10/90 的动作上限分别 220/280/300/520/400，另外加稳定步数。`OFT:experiments/robot/libero/run_libero_eval.py:62`、`:112`、`:316`。每个 chunk 的 8 个动作依次各执行一次 `env.step`，不是重复一个动作 8 次；成功来自环境 `done`：同文件 `:327`–`:353`。环境 seed 固定 0（`libero_utils.py:24`），而各 trial 默认取 `initial_states[episode_idx]`（`run_libero_eval.py:392`）。这些 init-state ID/hash 应加入 rollout manifest。

**不要直接把 `num_open_loop_steps` 改为 1 做单步 smoke。** 当前 evaluator 用 `deque(maxlen=num_open_loop_steps)`，随后 `extend(actions)`；模型输出 8 步时 maxlen=1 会保留**最后一个动作**，不是当前第一个动作。证据：同文件 `:302`、`:341`。最小 smoke 仍保留原生 8×7 预测；如确需只执行首步，必须显式切片并测试，另记实验差异。

官方主循环默认跑全 suite 的所有 tasks，没有独立单 task 参数（同文件 `:479`）。单任务最小 rollout 需一个有记录的 wrapper/过滤入口；降低 trials 数不等于只跑一个任务。当前阶段没有执行 rollout。

## 7. 依赖与隔离要求

| 层 | 代码要求 / 风险 | 证据 |
|---|---|---|
| Python | 官方结果环境 3.10.14；setup 创建 Python 3.10 | `OFT:LIBERO.md:86`；`OFT:SETUP.md:7` |
| PyTorch | torch 2.2.0、torchvision 0.17.0、torchaudio 2.2.0 | `OFT:pyproject.toml:47` |
| 模型框架 | `moojink/transformers-openvla-oft` fork（说明文档称 v4.40.1），peft 0.11.1、timm 0.9.10、tokenizers 0.19.1、sentencepiece 0.1.99 | 同文件 `:41`；`OFT:LIBERO.md:86` |
| RLDS | tensorflow 2.15.0、tensorflow_datasets 4.9.3、tensorflow_graphics 2021.12.3、`moojink/dlimp_openvla` fork | `OFT:pyproject.toml:52` |
| 训练扩展 | flash-attn 2.5.5，先安装 packaging/ninja | `OFT:SETUP.md:19` |
| LIBERO for OFT | robosuite 1.4.1、bddl、easydict、cloudpickle、gym、imageio[ffmpeg] | `OFT:experiments/robot/libero/libero_requirements.txt:1` |
| 原 LIBERO 依赖冲突 | 原 requirements pin robosuite 1.4.0、transformers 4.21.1、numpy 1.22.4；不能整体覆盖 OFT 环境 | `LIBERO:requirements.txt:1` |

两个 Git 依赖都没有在 pyproject URL 中 pin commit；必须解析并冻结其具体 SHA。此任务不安装/升级共享 VLM 环境；OFT 独立环境还要记录 CUDA、驱动、flash-attn wheel/build、MuJoCo、EGL/OSMesa、FFmpeg、所有未 pin 包和实际 import probe。CPU 准备阶段禁止导入会抢占 GPU 的训练/eval入口；没有用静态声明替代运行验证。

## 8. 真正启用 B0 的逐项要求

- [x] 冻结官方源码 SHA，识别原生 action/loss/camera contract；产物为本报告。Git：由主任务最终只 add 本报告和相关准备文件，使用 `docs(vla): audit OFT and LIBERO prerequisites` 类 commit；审查后按会话授权推送指定工作分支。本子任务不 commit/push。
- [ ] 冻结**训练初始化**与**已训练推理验收**两个模型用途；下载并校验 base snapshot，记录 HF revision、模型/processor/custom-code SHA，核验预训练暴露范围。Git：仅提交模型 manifest，不提交权重；commit `chore(vla): pin model provenance`，推送同一工作分支。
- [ ] 获取实际单任务数据，核验 schema、episode 数、相机方向、HDF5→RLDS lineage；冻结 train/offline-eval/rollout 清单，train-only stats，剔除失败/短轨迹的清单。Git：只提交 manifests/schema/摘要，commit `data(vla): freeze episode splits and statistics provenance`，推送同分支。
- [ ] 建独立 OFT 环境，pin 两个 fork SHA，记录完整 lock 与 CPU imports/data batch probe。Git：提交安装说明/lock/CPU 检查结果，commit `build(vla): isolate and lock OFT runtime`，推送同分支。
- [ ] 实现真实训练 adapter/入口：官方 L1、8×7、双相机+proprio、可追踪 batch、LoRA+head+projector 保存恢复、episode 隔离；把项目 cosine/预算等差异显式冻结。当前 `src/vla_attention/adapters/openvla_oft.py:50` 与 `src/vla_attention/evaluation/libero.py:35` 只是校验边界。Git：提交实现及有意义的契约验证，commit `feat(vla): connect native OFT B0 training`，推送同分支。
- [ ] **待后续授权 GPU 时**，做模型 forward/backward、8×7 输出、loss/梯度有限性、action normalize roundtrip、checkpoint 恢复 smoke；后续才执行少量固定 init states 的官方 rollout，确认成功记录来自环境。Git：只提交运行 config/摘要/产物路径，commit `test(vla): record B0 smoke and restore gate`，推送同分支。当前授权不包含运行这些 GPU 步骤或长训练。
- [ ] 在同起点、同 episodes、同增强、同 seed/预算的条件下冻结 B0–B4；B0 不依赖 diffusion teacher cache 完成，但 B1–B4 对应 teacher/归因门槛仍按主协议执行。Git：commit `experiment(vla): freeze paired B0-B4 protocol`，推送同分支。

每一步的 commit/push 是执行计划，不表示已执行；分支、remote 与是否最终推送由父任务已有授权及项目规则确定。禁止把大模型、数据、teacher cache 或检查点纳入 Git。

## 9. 未决事实与本报告限制

HF 模型/数据 snapshot、实际可用演示数量及 split、controller action 物理缩放、依赖 fork commit、真实 processor 内部视觉 grid、模型显存、checkpoint 恢复、模拟器可用性与任何 success rate 均未由本次静态阅读确认。官方文档中的性能/VRAM数值不能替代本机测量。本报告建议保留这些事项为未完成，不能将“准备文件生成”升级成“B0 已运行”。
