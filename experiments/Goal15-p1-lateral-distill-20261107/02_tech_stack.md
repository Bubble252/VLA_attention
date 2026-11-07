# P1 技术栈与数据契约

**日期**：2026-11-07
**状态**：`PROTOCOL_FROZEN_NOT_STARTED`

## 1. 已有资产

### 1.1 P0 教师证据

远端：

```text
$W/results/K4_lite/T4b/p0_authority_release_15init_20261107/
```

必读：

- `audit_p0_authority_release_v12.json`；
- `audit_p0_authority_release_v12.md`；
- `traces/A10/*.jsonl`。

教师控制器：

- arm：`A10`；
- fine cap：`0.10`；
- release：`xy=0.03`、`z=0.05`、streak `3`、settle `4`；
- 只改 x/y；z/rotation 保持 base policy；
- 特权 object/plate 位置只在教师标签生成时可用。

### 1.2 训练数据

已有 cache：

```text
$W/results/K4_lite/T4_residual/cache_train_v2
$W/results/K4_lite/T4_residual/cache_validation_v2
```

`cache_train_v2` 的已知契约：

- 11,630 transitions；
- 96 episodes；
- 3 tasks；
- route features `4096` 维；
- `base_action` / `target_action`：`[N, 8, 7]`；
- `proprio`：`[N, 8]`；
- optional `geometry`：16x16 route geometry；
- episode split 来自 `c1_split_manifest_v1.json`；
- checkpoint fingerprint 与 P0 的 J2 一致。

`white_right` 不进入训练或 validation loss。P0 的 15 个 rollout init 与训练/验证 episode 必须零重叠。

### 1.3 Backbone

```text
$W/models/openvla--openvla-7b
$W/results/JGATE_first_batch_20261023T0737cache/...--J2--12500_chkpt
```

J2 fingerprint：

```text
747a28690383ca45e795518121e9aed3f4610265c14618a69f203922b4238e20
```

### 1.4 旧 point head 的边界

Goal13 的 point head：

```text
$W/results/star105_goal10p5_20261105/point_head/pilot_5000/point_head.pt
```

它是 Qwen/starVLA 资产，不是 J2 的 point head。P1-A 必须先决定：

- 训练一个 J2-compatible point head；或
- 改用 route centroid evidence，并把 arm 改名为 `R2-ROUTE`。

不能直接把旧 Qwen point head 接进 J2。

## 2. 需要新增的组件

P1-A：

- `scripts/build_p1_evidence_adapter.py`（可命名不同）；
- point/region adapter preflight；
- point-in-mask、L1、覆盖率、split overlap 检查。

P1-B：

- `scripts/build_p1_teacher_cache.py`；
- 输入：96 个 training episodes + `A10` 教师；
- 输出：`cache_p1_teacher_v1`，包含 `teacher_delta_xy`、`teacher_release`、base action、route/point evidence、proprio、split；
- 不使用 `white_right`；不写入 15 个 rollout init。

P1-C：

- `scripts/train_p1_lateral_distill.py`；
- 复用 `EvidenceResidual` 的零初始化 gate 和 xy-only bound；
- 新增 release head；
- 只有 `xy` 维有梯度；z/rotation 输出严格为 0。

P1-D：

- `scripts/audit_p1_lateral_distill.py`；
- 复用 P0 `run_episode` 和 evaluator；
- arms：`R0/R1-CAP/R2-PRED/R3-GEO`；
- 15-init 先 pilot，不直接扩 50。

## 3. 训练配置建议

| 字段 | 初始值 |
|---|---|
| max translation delta | `0.10` |
| max z delta | `0.0` |
| max rotation delta | `0.0` |
| action horizon | `8` |
| hidden dim | `128` |
| dropout | `0.1` |
| batch size | `64` |
| learning rate | `5e-4` |
| weight decay | `0.01` |
| grad clip | `1.0` |
| action loss | xy L1 |
| margin loss | xy route-pair margin，权重 `0.25–0.5` |
| release loss | BCE，正样本加权；阈值在 validation 上冻结 |
| seed | `17` |
| smoke | `500` steps |
| pilot | `2000` steps |

这些值只是 P1 的初始冻结建议；P1-C 启动前必须写入 config 并记录 SHA。

## 4. 运行环境

远端：

```bash
W=/vepfs-mlp2/c20250405/400040/transfer/vla_attention/vla_workspace
R=/vepfs-mlp2/c20250405/400040/transfer/vla_attention/repo/VLA_attention
export PYTHONPATH="$R:$R/src:$W/repos/openvla-oft:$W/repos/libero"
export LIBERO_CONFIG_PATH="$W/libero_config"
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=1
```

解释器：

```text
$W/envs/oft/bin/python
```

不能使用 base `python`；base 没有 `tensorflow`。只使用 GPU1，训练/eval/rollout 串行。

## 5. 输出

建议输出根目录：

```text
$W/results/K4_lite/T4b/p1_lateral_distill_20261107/
```

必须保留：

- adapter preflight JSON；
- teacher cache manifest + SHA；
- 每个 arm 的 config SHA、checkpoint SHA、训练日志；
- offline eval JSON；
- 15-init 闭环 JSON/MD；
- 失败日志不删除、不覆盖。

不提交 checkpoint、视频、safetensors、pt 或大日志。
