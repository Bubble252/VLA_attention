# VLA Attention：101 服务器连接与训练启动

**状态**：连接已验证；训练尚未提交。  
**最后只读验证**：2026-09-17。  
**范围**：本项目的 VLM/VLA attribution 实验；不复用其他项目的训练配方。

## 1. 已验证的 101 环境

| 项目 | 结果 | 备注 |
|---|---|---|
| SSH | 可连接 | `root@115.190.90.101:27219`；密钥/密码不写入本仓库 |
| Python | `/root/starvla_cu124/bin/python` | `torch 2.6.0+cu124`，CUDA 可用 |
| GPU | 2 × NVIDIA A100 80GB | 仅供调试/部署；大训练走火山队列 |
| StarVLA 代码 | `/root/code/Starvla` | 参考已有 Robotwin/LIBERO 模板，不能直接复用 |
| 共享数据根 | `/vepfs-mlp2/c20250405/400040/transfer/` | 训练数据、权重、cache、logs 应放这里 |
| `volc` CLI | 当前未找到 | 在 101 上提交 8 卡前必须由有权限者安装并配置 |
| `/root` 空间 | 约剩 206 MB | 禁止在 `/root` clone 大仓、建环境、放 cache 或 checkpoint |

GPU0 在验证时已有约 61GB 显存被其他进程占用，GPU1 基本空闲。未获得任务归属或资源许可前，不在本机启动占用 GPU 的实验。

## 2. 连接命令

本地电脑执行：

```bash
ssh -p 27219 root@115.190.90.101
```

自动化、非交互验证可使用：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 \
  -p 27219 root@115.190.90.101 \
  'hostname; whoami; nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader'
```

不要将 SSH 私钥、密码、AK/SK、W&B key、Hugging Face token 或飞书 token 写入项目、shell history、YAML 或 Git。

### 稳定的本机入口（2026-09-17 已配置）

本机 `~/.ssh/config` 已增加不含凭据的 `vla101` 别名，固定 `115.190.90.101:27219`、`root`、`BatchMode=yes` 和 `StrictHostKeyChecking=yes`；对应主机指纹已存在于本机 `known_hosts`。私钥仍由用户的本机 SSH agent/既有配置管理，不进入本仓库。

本仓库的受限入口是：

```bash
cd /home/bubble/类脑计算/VLM终局
bash server/vla101.sh status
```

它不提供任意远程 shell，只允许 `status`、`bootstrap`、`freeze-env` 和单一官方 Qwen 下载。每个命令只触及 `/vepfs-mlp2/c20250405/400040/transfer/vla_attention`；不写 `/root`，不处理凭据，不下载未冻结的数据集，不提交训练。

```bash
# 依次执行并记录输出
bash server/vla101.sh bootstrap
bash server/vla101.sh freeze-env
bash server/vla101.sh download-qwen
```

`download-qwen` 的唯一模型仓库是官方 `Qwen/Qwen2.5-VL-7B-Instruct`。101 不使用 7897：脚本先测试 Hugging Face 直连；若官方端点不可达，才显式使用 `https://hf-mirror.com`，并在 checkpoint 同目录的 `SOURCE.txt` 和 `SHA256SUMS` 记录实际端点与文件校验。镜像只是传输源，模型仓库名、revision 和校验文件必须照实记录。

## 3. 101 在本项目中的角色

```text
本地电脑：文档、代码编辑、GitHub push、轻量单元测试
101：共享数据/权重/缓存、接口审计、提交火山任务、查看日志
火山 8 卡：正式训练
101 GPU：小样本接口检查或部署调试，需先确认空闲资源
```

现有 `/root/code/Starvla/examples/Robotwin/train_files/submit_volc_*` 脚本属于其他项目。它们包含 SpikingBrain/Robotwin 专属路径、恢复 checkpoint、数据混合和共享软链接操作；**禁止直接运行、修改或套用到本项目**。

## 4. 项目在共享盘的推荐布局

在管理员确认目录和写权限后，使用 VEPFS，不使用 `/root`：

```text
/vepfs-mlp2/c20250405/400040/transfer/vla_attention/
├── repo/                 # GitHub 工作树或只读代码快照
├── envs/                 # 项目独立 conda/venv，不修改 starvla_cu124
├── hf_cache/             # HF / diffusion / teacher 权重 cache
├── models/               # 手工下载的 checkpoint，按模型/版本/sha256 分目录
├── data/
│   ├── flickr30k_entities/
│   ├── refcocog/
│   ├── libero/
│   └── droid/
├── teacher_maps/         # 固定 teacher 配置生成的离线图
├── runs/                 # TensorBoard/W&B 本地日志
├── checkpoints/          # 大文件，不进 Git
├── results/              # 小型 CSV/JSON/图表摘要可回传 Git
└── jobs/                 # 火山 YAML、启动脚本、任务 ID、日志索引
```

本地仓库另有 `experiment_workspace/`，用于保存 manifest、冻结配置和小型结果摘要；其目录结构与 VEPFS 运行目录对应。代码实现留在仓库 `src/`，大文件只留在 VEPFS，避免和文档、参考资料或其他项目混放。

首次建立前先检查：

```bash
ssh -p 27219 root@115.190.90.101 \
  'df -h /root /vepfs-mlp2/c20250405/400040/transfer; test -w /vepfs-mlp2/c20250405/400040/transfer && echo VEPFS_WRITABLE'
```

不要在未确认配额前创建大目录、复制 checkpoint 或下载数据。

## 5. 开始训练前的五个 gate

### Gate A：项目与环境隔离

- [ ] GitHub 仓库 clone 到经确认的 VEPFS 项目目录；
- [ ] 新建项目专属环境，记录 Python、Torch、CUDA、Transformers、Diffusers、PEFT 版本；
- [ ] `HF_HOME`、`TRANSFORMERS_CACHE`、`TORCH_HOME`、`TMPDIR` 指向 VEPFS；
- [ ] 不修改 `/root/starvla_cu124`，除非管理员明确指定它为共享可修改环境。

### Gate B：模型和 teacher 冻结

- [ ] VLM：Prismatic-7B + Qwen2.5-VL-7B；
- [ ] VLA：OpenVLA/OFT + π0.5；
- [ ] `T_sem`：Stable Diffusion、PixArt-α、PixArt-Σ、Playground-v2.5 在 calibration split 上选 best-single；
- [ ] `T_retention`：DB-style feature baseline 的 teacher、层、projector seed 和版本；
- [ ] 权重下载路径、许可证、commit/SHA256 记录到 experiment manifest。

### Gate C：数据与动作合同

- [ ] Flickr30k Entities：train/calibration/test manifest；
- [ ] RefCOCOg：官方 split 记录；
- [ ] LIBERO：任务、环境版本、demo 版本、seed；
- [ ] DROID：episode-level split、动作坐标系、频率、单位、预训练重叠审计；
- [ ] 所有 VLA 原生动作与 7D delta EEF adapter 分开记录，不混成同一指标。

### Gate D：小样本接口审计

- [ ] VLM：token grid、phrase score、teacher map、raw/ratio/gradient/occlusion 图；
- [ ] VLA：action token/flow 输出、`A_act-output` 与 `A_act-loss`、固定 noise/time、梯度是否可算；
- [ ] DB-style feature teacher：projector 参数是否在 optimizer、teacher preprocess 与 student grid 是否对应；
- [ ] D0：object-only、object+EEF、随机 support 的 soft leakage 小样本检查；
- [ ] 每项 audit 写入 `outputs/interface_audit.md`，通过后才开始训练。

P1 的 20 个固定样本只用于检查接口；不是缩小正式实验。P1 后的 teacher calibration、VLM V0–V4 和 VLA B0–B4 均使用各自冻结的完整训练/测试 split，具体边界见 `experiment_workspace/README.md`。

### Gate E：训练 protocol 先于结果

- [ ] 写 `experiments/<id>/protocol.md`：假设、数据、配置、primary metric、停止条件；
- [ ] 先 `git commit` protocol；
- [ ] 试运行 20–50 step，无 NaN、数据正确、loss 可回传、checkpoint 可恢复；
- [ ] 再提交正式训练。

## 6. 本项目的训练顺序

```text
P4 VLM：V0/V1/V2/V3/V4
  V0 checkpoint-only
  V1 SFT
  V2 DB-style feature retention
  V3 best-single T_sem → A_lang
  V4 V2 + V3
          ↓ 只有 phrase grounding 成立
P6 VLA：B0/B1/B2/B3/B4
  B0 native BC/SFT
  B1 DB-style feature retention
  B2 L_sem
  B3 B1 + B2
  B4 B3 + D0 soft action-evidence constraint
          ↓ 只有 B4 有独立动作增量
D1 phase / B WAM-VAM / RoboTwin 扩展
```

每个阶段的 Git protocol、metric 和停止条件见 `doc/03_execution_plan.md` 与 `paper_draft/05_experiments_and_expected_conclusions.md`。

## 7. 火山 8 卡提交的安全模板

当前 101 未发现 `volc`。在管理员安装 CLI、配置账号并确认队列权限前，以下命令只作为模板，不能执行：

```bash
ssh -p 27219 root@115.190.90.101
command -v volc
volc configure              # 仅由账号持有者配置；不得把凭据写进项目 YAML

cd /vepfs-mlp2/c20250405/400040/transfer/vla_attention/repo
volc ml_task submit -c server/jobs/<experiment>.yaml
```

每个 `<experiment>.yaml` 必须由本项目创建，至少显式记录：

```text
TaskName / experiment ID
Git commit
container image
ResourceQueueName
8-GPU flavor
VEPFS mount
entrypoint
data / checkpoint / cache / result paths
model + teacher + seed + dataset manifest
```

不得复制别的项目 YAML 中的环境变量、恢复 checkpoint、W&B 配置、凭据或挂载路径。先提交一个 20–50 step smoke job；确认数据、环境、日志和 checkpoint 路径后，才提交正式训练。

## 8. 101 上的只读日志检查

正式任务提交后，将任务 ID、提交 YAML 的 SHA256 和日志目录写到 `experiments/<id>/protocol.md`。日志检查模板：

```bash
ssh -p 27219 root@115.190.90.101 \
  'ls -lt /vepfs-mlp2/c20250405/400040/transfer/vla_attention/runs | head'

ssh -p 27219 root@115.190.90.101 \
  'tail -n 100 /vepfs-mlp2/c20250405/400040/transfer/vla_attention/runs/<experiment>/train.log'
```

只在确认路径属于本项目时执行 `tail`。遇到 CUDA OOM、NaN、数据 schema 或 teacher grid 异常，先停止扩大训练，记录失败分析和 commit。

## 9. 当前阻塞项

- [ ] `volc` CLI 与火山队列授权尚未确认；
- [ ] VEPFS 项目目录及写权限尚未由管理员确认；
- [ ] 101 `/root` 只剩约 206 MB，不能用作项目目录或 cache；
- [ ] 首轮模型/数据/teacher 的具体权重与版本尚未下载；
- [ ] P1/P4/P6 小样本接口 audit 未实现。

在上述项完成前，连接 101 只用于读环境、准备协议和查看已有资料，不启动正式训练。
