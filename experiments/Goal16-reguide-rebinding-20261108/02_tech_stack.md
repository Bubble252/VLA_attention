# Goal16 技术栈与数据契约

**日期**：2026-11-08<br>
**状态**：`PROTOCOL_FROZEN_BEFORE_MEASUREMENT`

## 1. 冻结策略与评测链

复用 Goal14 的冻结策略 rollout 链，不另起一套 evaluator：

```text
configs/vla/k4_lite_t4b_v12_p0_authority_release.json
scripts/audit_k4_lite_t4b_v8_release_settle.py
scripts/audit_k4_lite_t4b_v12_p0_authority_release.py
src/vla_attention/t4b_v8_control.py
```

冻结资产：

```text
checkpoint:
$W/results/JGATE_first_batch_20261023T0737cache/openvla--openvla-7b+libero_90_gate+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--JGATE_formal_20261023T0737cache--J2--12500_chkpt

checkpoint_fingerprint:
747a28690383ca45e795518121e9aed3f4610265c14618a69f203922b4238e20

manifest:
$W/data/jgate_c1/counterfactual_20261027/manifest.jsonl

manifest_sha256:
e9564ca9aef800e618e3fdb5d144892eabe9471815bbd24bae3140583946f45e

statistics_sha256:
6d5ebb47bc5132df581aae0481d529d04a0821c331859c05a9818d78156c9538
```

远端环境：

```bash
W=/vepfs-mlp2/c20250405/400040/transfer/vla_attention/vla_workspace
R=/vepfs-mlp2/c20250405/400040/transfer/vla_attention/repo/VLA_attention
export PYTHONPATH="$R:$R/src:$W/repos/openvla-oft:$W/repos/libero"
export LIBERO_CONFIG_PATH="$W/libero_config"
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=1
```

只使用 GPU1。GPU0 禁止使用。

## 2. `R-GEO`：oracle 语义 + 3D pre-place

`R-GEO` 与 A10 的明确区别：

| 项 | A10（Goal14） | `R-GEO`（Goal16） |
|---|---|---|
| 目的地来源 | GT `plate_2` | GT `plate_2` |
| 触发 | first grasp + 2 steps | first grasp + 2 steps |
| 目标 | 当前物体 xy 与 `plate_2` xy 的差 | `plate_2` 上方的 3D pre-place 配置 |
| 控制轴 | 只改 x/y | x/y/z；旋转保持 base |
| 行为 | 连续横向 authority | bounded transport → hand-back |
| 结束 | release gate + settle | hand-back 后 completion certificate / re-arm / release |
| 诊断身份 | 特权上界 | 特权语义+几何 rebinding |

第一版参数（可 calibration，不可 confirmation 调参）：

```text
approach_height_m = 0.10       # held-object bottom above destination surface
handback_xy_m = 0.05
handback_z_m = 0.06
transport_cap_xy = 0.20
transport_cap_z = 0.15
transport_gain_xy = 2.0
transport_gain_z = 2.0
deadband_m = 0.003
max_transport_steps = 240
release_xy_m = 0.03
release_z_m = 0.05
release_streak_steps = 3
settle_steps = 4
max_transports_per_stage = 2
```

`R-GEO` 只在冻结策略之外接管动作，之后必须回到冻结策略。它不写新的 checkpoint。

## 3. `R-NAIVE`：oracle 目的地的 naive transport

`R-NAIVE` 使用同一个 oracle `plate_2` xy，但：

- 不建立 3D pre-place 高度；
- 不检查 semantic mismatch；
- 不做 hand-back / re-arm；
- 只持续把 held object 的 xy 向 `plate_2` xy 伺服，z 保持 base action；
- 使用与 `R-GEO` 相同的 release fallback，以保证 released success 可比。

它回答：**如果只要把物体送到正确 xy，效果是否已经出现？**

## 4. `R-PRED`：可部署 predicted grounding

`R-PRED` 使用现有 C1 point head 作为外部 grounding module：

```text
$W/results/star105_goal10p5_20261105/point_head/pilot_5000/point_head.pt
```

该 point head 的既有 Gate P1 结果：

- point-in-mask `0.943503`；
- destination L1 `1.8756 px`；
- destination correct-vs-wrong margin `0.9449`。

边界：

- 它不是 J2 point head，不进入 J2 动作前向；
- 它只在 rollout 前对初始观测做一次外部 grounding；
- 不使用 GT mask/object id 作为 runtime 输入；
- 预测的 2D 点通过固定相机标定投回桌面平面；
- 如果 point head 对 `white_right` 的预测与 `plate_2` 偏差超过 `0.15 m`，该 init 标为 `grounding_miss`，不得用 oracle 替换。

`R-PRED` 的配置和 sidecar SHA 必须写入 Run Contract。

## 5. `R-SHUF`：目的地负控

`R-SHUF` 与 `R-GEO` 共用全部几何和 hand-back 逻辑，只把目的地从 `plate_2` 换成 `plate_1`。

- 如果 `R-SHUF` 成功率高，说明 wrapper 没有真正绑定 instruction；
- 如果 `R-SHUF` 接近 `B0`，负控成立；
- `R-SHUF` 的失败不应被当成 wrapper 方法失败。

## 6. 数据契约

| 项 | 要求 |
|---|---|
| manifest | 冻结 15-init `white_right` |
| train count of `white_right` | `0` |
| train/validation/rollout overlap | `0` |
| calibration | `init_0e94a4e6b9622c6b`、`init_2edf49999eb35566`、`init_3500df19a4d3b5a1` |
| confirmation | 其余 12 个 |
| episode seed | `7` |
| open-loop steps | `8` |
| wait steps | `10` |
| image size | `224` |
| resolution | `256` |

## 7. 必须记录的指标

每个 episode 同时记录：

- `official_success`；
- `strict_destination`（目标物体最终在 `plate_2` 的 official radius 内且 gripper open）；
- `released_success`；
- `hold_only_success`；
- `wrong_destination`；
- `first_destination`；
- `target_object_correct`；
- `action_change_max`、`x/y/z_action_change_max`、`gripper_action_change_max`；
- transport / hand-back / re-arm / release 的次数与步数；
- `object_retained`、`object_flew_out`、`nan_count`、`error`。

## 8. 输出目录

```text
$W/results/K4_lite/T4b/goal16_reguide_rebinding_20261108/
```

不得覆盖 Goal14、v7–v12 的任何目录。预测 grounding sidecar 放在：

```text
$W/results/K4_lite/T4b/goal16_reguide_rebinding_20261108/grounding/
```
