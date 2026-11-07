# 02 技术栈：数据、远端环境与审计工具

## 1. 本地工作区

```text
V=/home/bubble/类脑计算/VLM终局
```

主文档：`doc/vla_next/65_story_method_selection_20261107.md`。

本目录只保存文档、审计脚本、测试、契约和结果摘要；不保存 checkpoint、视频或大数据。

## 2. 远端工作区

SSH：

```bash
ssh -p 27219 root@115.190.90.101
```

关键路径：

```text
W=/vepfs-mlp2/c20250405/400040/transfer/vla_attention
R=$W/vla_workspace/repo/starvla_55_20261030/repo
D=$W/vla_workspace/data/jesbu1--libero_90_rlds/libero_90_openvla_processed/1.0.0
C1=$W/vla_workspace/data/jgate_c1
G13=$W/results/star105_goal13_20261107
```

远端 repo HEAD 记录为 `4507931a625c844404c4a76128fb116536d8ca7c`，但远端存在大量未提交文件；不得 reset，不得直接覆盖远端源码。新增脚本先在本地测试、commit，再同步到远端临时工具目录或结果目录执行。

## 3. 数据

### 3.1 RLDS 源数据

`D` 下是 256 个 TFRecord shard，约 19G。当前 Stage 0 需要从中审计：

- C1 三条训练 prompt 的 episode 数；
- `white_right` 训练计数；
- carrier `put the yellow and white mug on the right plate`；
- 可选 `put the white bowl to the right of the plate`；
- 每条 episode 的 shard / record index / payload hash。

### 3.2 C1 split manifest

```text
$C1/c1_split_manifest_v1.json
```

已知计数：

| split | episodes |
|---|---:|
| train | 96 |
| validation | 15 |
| rollout | 15 |

按 task：

| task | train | validation | rollout | source |
|---|---:|---:|---:|---:|
| task65 red-left | 27 | 5 | 5 | 37 |
| task66 red-right | 34 | 5 | 5 | 44 |
| task67 white-left | 35 | 5 | 5 | 45 |

### 3.3 已转换 LeRobot 数据

```text
$W/results/star105_goal10p5_20261105/data/LEROBOT_LIBERO_DATA/libero_90_gate_lerobot
```

当前 `meta/info.json`：`96 episodes / 12302 frames / 3 tasks`。该数据只含 C1 三 prompt，不含 held-out `white_right`。

### 3.4 HDF5 / sidecar

当前只有：

```text
$W/vla_workspace/data/libero_90_hdf5/task65_67
```

没有找到 carrier 的 HDF5 或 GT mask。Stage 1 若使用 `I2`，只能先走 P1 pseudo point，并标记 `weak_sidecar=true`；不能把 pseudo point 说成 GT。

## 4. Python 环境

远端当前检查到的 Python：

```text
/root/starvla_cu124/bin/python  # Python 3.11, numpy/PIL/yaml/omegaconf，无 tensorflow/tfrecord
```

因此 Stage 0 审计不能依赖 TensorFlow。实现策略：

1. 用纯 Python 读取 TFRecord framing（length + payload；可选 CRC32C 校验）；
2. 对每条 payload 做已知 prompt 的字节级匹配，统计 episode 级语言；
3. 用 payload SHA256 作为临时 episode_id；
4. 与 `c1_split_manifest_v1.json` 的 episode_id/语言做交叉核验；
5. 不解析图像和动作张量，不改变原始数据。

这样可以避免安装 TensorFlow，也避免在远端产生新环境依赖。

## 5. 计划新增工具

```text
scripts/audit_goal13_rlds.py       # 只读扫描 RLDS，输出语言计数和 episode 清单
scripts/build_goal13_contract.py   # 合并 audit + split manifest，生成 Run Contract
scripts/build_goal13_g0_visual_sample.py  # 100 帧 contact sheet 与最终位置抽检
scripts/check_goal13_checkpoint_health.py  # safetensors SHA + finite-value 证据
scripts/verify_goal13_postrun.py   # 日志/config/summary/checkpoint 的 postrun contract
tests/test_goal13_contract.py      # 纯 CPU 单元测试
tests/test_goal13_postrun.py       # postrun 解析与 fail-closed 单元测试
configs/vla/goal13_data_arms.json  # 可选；固定 D0/D1/D2 数据臂定义
```

工具只依赖 Python 标准库；不导入 torch、TensorFlow 或远端训练代码。

## 6. Run Contract

契约至少包含：

```json
{
  "schema": "goal13-run-contract-v1",
  "contract_status": "PASS|INVALID",
  "arm_status": "PENDING_GATE|PASS|FAIL|INVALID",
  "data": {
    "data_arm": "D0|D1|D2",
    "train_language_counts": {},
    "heldout_prompt_train_count": 0,
    "train_validation_rollout_overlap": 0,
    "carrier_episode_count": 48,
    "sampling_weights": {}
  },
  "runtime": {
    "effective_batch": 4,
    "expected_grad_accum": 1,
    "seed": 17
  }
}
```

`INVALID` 不是方法失败，而是配置/数据/评测不满足契约，必须修复后重跑。

## 7. Gate 与资源

- GPU：只用 GPU1；
- 串行：训练期间不启动评测；
- 先 offline `limit=2` smoke，再 pilot，再正式 30/50；
- 15 条 pilot 只用于筛选，不用于最终结论；
- 正式结论需要 30/50 + 3 seeds；
- 所有结果写入 `$G13/`，不得覆盖 Goal10.5/11/12 归档。

## 8. 外部方法

AC-VLA、AnySlot、CRAFT、CounterAlign、CAST 等方法只有在完成 `papers/{slug}.pdf`、`papers/{slug}.txt`、官方代码仓库 pin commit 和 `references/paper_reading/{slug}.md` 后才能进入训练。未完成者标记 `INVALID`。
