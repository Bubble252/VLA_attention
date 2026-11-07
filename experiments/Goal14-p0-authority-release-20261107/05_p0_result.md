# P0 结果：横向 authority + release/settle

**日期**：2026-11-07
**状态**：`P0 PASS`，但属于刚过线，不是压倒性正结果
**完整结果**：远端 `$W/results/K4_lite/T4b/p0_authority_release_15init_20261107/audit_p0_authority_release_v12.json`
**本地摘要**：`p0_summary.json`、`p0_summary.md`

## 1. 结论

P0 通过预注册 Gate，`H-AUTH` 获得支持：

> 在 `white_right` 的 15 个冻结 init 上，`A10` 产生了 `4/15` released success，正式超过 `B0` 和 `A05` 的负控，并且 12 个 confirmation init 中有 `6/12` official success。

但必须保留三个限定：

- `A10` 的总 success 为 `7/15`，只比 Gate 下限 `5/15` 高 2 个；
- `A10` 相对 `A05` 的优势恰好是 `2/15`，没有再高；
- `A20` 与 `A10` 总量相同，都是 `7/15`，没有证明更大 cap 更好。

因此这不是“residual 方法已经成功”，而是“横向 authority 值得进入 P1 蒸馏”。

## 2. 冻结契约

| 项 | 实际值 |
|---|---|
| manifest SHA256 | `e9564ca9aef800e618e3fdb5d144892eabe9471815bbd24bae3140583946f45e` |
| checkpoint fingerprint | `747a28690383ca45e795518121e9aed3f4610265c14618a69f203922b4238e20` |
| dataset statistics SHA256 | `6d5ebb47bc5132df581aae0481d529d04a0821c331859c05a9818d78156c9538` |
| config SHA256 | `ecb501f86b4b795d94429cc17ce21bb2461014b64a54b3b006f2c9d344cef52d` |
| metric amendment SHA256 | `053cc6e1fd9bfe84ec7c4fb6de760a8083f201264466e5a6046272d3fbbd8e52` |
| init | 15；3 calibration + 12 confirmation |
| seed | 7 |
| release | `xy=0.03`、`z=0.05`、streak `3`、settle `4` |
| 控制轴 | 只改 x/y；z 和旋转保持 base policy |
| 训练 | 无；无 backward、无 optimizer、无新 checkpoint |

## 3. 主表

| arm | fine cap | total success | calibration | confirmation | released success | hold-only success | released placement | wrong destination | object retained | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0` | 0.00 | 0/15 | 0/3 | 0/12 | 0/15 | 0/15 | 0/15 | 14/15 | 15/15 | 0 |
| `A05` | 0.05 | 5/15 | 1/3 | 4/12 | 1/15 | 4/15 | 2/15 | 3/15 | 15/15 | 0 |
| `A10` | 0.10 | 7/15 | 1/3 | 6/12 | 4/15 | 3/15 | 6/15 | 3/15 | 15/15 | 0 |
| `A20` | 0.20 | 7/15 | 1/3 | 6/12 | 4/15 | 3/15 | 7/15 | 3/15 | 15/15 | 0 |

稳定性：所有 arms `0 error`、`0 NaN`、`0 object flew_out`，object retained 均为 `15/15`。

## 4. Gate

| check | 结果 |
|---|---|
| `b0_baseline_reproduced` | `true`（0/15） |
| `a10_total_success` | `true`（7/15 >= 5/15） |
| `a10_confirmation_success` | `true`（6/12 >= 2/12） |
| `a10_released_success` | `true`（4/15 >= 4/15） |
| `a10_over_a05` | `true`（7 - 5 = 2/15） |
| `a20_overshoot_control` | `true`（7 <= 7 + 1） |
| `stability` | `true` |

**正式状态：`P0 PASS`。**

## 5. 逐 episode 解释

### 5.1 `B0`

- `0/15` official success；
- `14/15` wrong destination，`1/15` 没有可用 destination；
- 与既有 `white_right` 基线一致：base policy 主要把物体带到错误盘子，而不是差一点到达正确盘子。

### 5.2 `A05`

- `5/15` total success，但只有 `1/15` released success；
- `4/15` 是 hold-only success，不能算 released success；
- 说明 cap `0.05` 可以偶尔进入成功半径，但大多数时候没有完成有效 release/settle；
- 相对 `A10` 少 `2/15`，负控成立，但优势刚好达到 Gate 下限。

### 5.3 `A10`

- `7/15` total success，`4/15` released success；
- `3/15` hold-only success，不能计入 released success；
- calibration 只有 `1/3`，confirmation 为 `6/12`；因此结果主要来自 confirmation，而不是重复 v10 的 3 个 calibration init；
- 4 个 released success 的 release z 在 `0.012–0.024m`；另有 1 个 release 在 `z=0.049m` 但未成功，说明成功不依赖极窄的 z gate，但 z 接近上限也不保证成功；
- 仍有 2 个 `released placement` 但 official success 失败，属于 release 后毫米级/判定边界问题；
- 2 个 episode 的 final xy 仍在 `0.47–0.49m`，是 base policy 没有把物体带到目标附近，横向 authority 无法修复这类失败；
- 其余失败集中在 `0.030–0.036m`，属于刚好在 success 半径外的近失。

### 5.4 `A20`

- total success 与 `A10` 相同，都是 `7/15`；released success 也都是 `4/15`；
- 没有出现 Gate 级别的过冲，但也没有证明 cap `0.20` 比 `0.10` 更好；
- 个别 init 上 `A20` 更快/更近，另一些 init 上反而把 `A10` 的成功变成失败，说明更大 authority 不是单调收益；
- 因 `A20` 没有总量优势，P1 应采用更保守的 `0.10` 作为教师 authority，而不是 `0.20`。

## 6. 对假设的含义

### 6.1 支持 `H-AUTH` 的部分

- `B0` 没有任何 released success；
- `A10` 有 `4/15` released success；
- confirmation split 有 `6/12` official success；
- 所有 episode 稳定，没有 NaN/error/飞出。

因此“低 z 横向 authority 不足”是一个真实瓶颈，至少不是纯接口问题。

### 6.2 没有支持的部分

- `A10` 只比 `A05` 多 `2/15`，而 `A05` 也有 `5/15` total success；
- `A20` 没有优于 `A10`；
- `A10` 仍有 3 个 hold-only success，说明 release 触发条件仍会让一部分成功停在 hold-only；
- 两个 catastrophic failure 与横向 authority 无关，说明 base policy 的初始抓取/搬运轨迹仍是独立失败源。

## 7. 裁决与下一步

**P0 PASS，允许写 P1 protocol，但不允许直接训练。**

P1 必须按 `66` 的要求先写协议并 commit：

- 教师：`A10`；
- 学生输入：图像、语言、proprio、base action、predicted point；
- 学生输出：bounded `delta_xy` + release logit，z/rotation 为 0；
- arms：`R0`、`R1-CAP`、`R2-PRED`、`R3-GEO`；
- 离线 Gate：cosine `>=0.50`、dominant-axis sign `>=0.75`、magnitude ratio `0.70–1.30`；
- 闭环 Gate：`R2` official `>=3/15`、confirmation `>=2/12`、比 `R1-CAP` 至少多 `2/15`、released success `>=2/15`、seen group-any `>=10/15`、wrong-destination 不超过 `B0`。

在 P1 protocol 和 Run Contract 写好之前，不启动训练。

## 8. 复现命令

远端正式运行脚本：

```bash
ssh -p 27219 root@115.190.90.101
cd /vepfs-mlp2/c20250405/400040/transfer/vla_attention/repo/VLA_attention
bash server/run_k4_lite_t4b_v12_p0.sh
```

本地测试：

```bash
cd /home/bubble/类脑计算/VLM终局
PYTHONPATH=. pytest -q tests/test_k4_lite_t4b_v12.py
```
