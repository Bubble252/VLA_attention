# P0 执行计划

**日期**：2026-11-07
**状态**：预注册执行计划；当前不训练

## 1. 冻结决策

本实验只回答一个问题：横向 authority 与 release/settle 组合后，是否有真实 released success。

### 1.1 Arms

| arm | 横向 authority | release | settle | 作用 |
|---|---|---:|---:|---|
| `B0` | 无；base policy 原样通过 | 否 | 0 | 复现 `0/15` 基线 |
| `A05` | fine cap `0.05` | 到目标后 release | 4 步 | 低于阈值的负控 |
| `A10` | fine cap `0.10` | 到目标后 release | 4 步 | 主上界臂 |
| `A20` | fine cap `0.20` | 到目标后 release | 4 步 | 过冲/上限控制 |

### 1.2 冻结的 release 口径

P0 统一使用：

```text
release_xy_m = 0.03
release_z_m = 0.05
release_streak_steps = 3
settle_steps = 4
gain = 2.0
deadband_m = 0.003
stop_xy_m = 0.005
```

冻结理由：

- v10 已证明 `z<=0.011` 不是成功必要条件；
- v11 已证明 z/姿态控制会损害横向 authority；
- P0 只测横向 authority，不能让过窄 z gate 再次阻塞 release；
- 一旦开始测量，不得看结果后改 cap、gain、release 半径、streak 或 settle 步数。

### 1.3 控制器边界

- 只写 `x/y`；
- z 和旋转保持 base policy；
- `B0` 不激活横向控制，不 release，不 hold gripper；
- 特权的物体/盘子位置只作诊断触发，不作为部署前向输入。

## 2. 预注册 Gate

| 检查 | 通过条件 |
|---|---|
| `B0` 基线复现 | official success `0/15`；非零则先审计，不能直接判方法有效 |
| `A10` 总量 | official success `>=5/15` |
| `A10` confirmation | 12 个未参与 v10 的 init 中 official success `>=2/12` |
| `A10` 真实释放 | released success `>=4/15` |
| `A05` 负控 | `A10 - A05 >=2/15` |
| `A20` 过冲控制 | `A20 <= A10 +1/15` |
| 稳定性 | 所有 arms 0 error、0 NaN、0 飞出、object retained `>=14/15` |
| 契约 | manifest、checkpoint、statistics、config、metric amendment SHA 全部一致 |

状态：

- `PASS`：所有检查通过；允许写 P1 protocol，仍不允许直接训练；
- `FAIL`：配置正确、测量完成，但任一 Gate 不通过；
- `INVALID`：数据、配置、评测或 SHA 不满足契约，不能判方法失败。

## 3. 执行步骤

### 阶段 A：文档和冻结协议

- [ ] 写 `01_background.md`；
- [ ] 写 `02_tech_stack.md`；
- [ ] 写 `03_execution_plan.md`；
- [ ] 写 `04_run_contract.md`；
- [ ] 确认本文件中的 arms、release/settle 和 Gate 不再改动。

提交：

```bash
cd /home/bubble/类脑计算/VLM终局
git add experiments/Goal14-p0-authority-release-20261107/01_background.md \
        experiments/Goal14-p0-authority-release-20261107/02_tech_stack.md \
        experiments/Goal14-p0-authority-release-20261107/03_execution_plan.md \
        experiments/Goal14-p0-authority-release-20261107/04_run_contract.md
git diff --cached --check
git commit -m "docs(vla): freeze p0 authority release protocol"
```

推送（只在用户明确要求时）：

```bash
git push origin main
```

### 阶段 B：配置、runner 和本地测试

- [ ] 新增 `configs/vla/k4_lite_t4b_v12_p0_authority_release.json`；
- [ ] 计算 config SHA 并写入 runner 常量；
- [ ] 新增 `scripts/audit_k4_lite_t4b_v12_p0_authority_release.py`；
- [ ] 新增 `tests/test_k4_lite_t4b_v12.py`；
- [ ] `pytest -q tests/test_k4_lite_t4b_v12.py` 通过；
- [ ] `python -m py_compile` 通过；
- [ ] 检查 config 不允许被静默篡改。

提交：

```bash
cd /home/bubble/类脑计算/VLM终局
git add configs/vla/k4_lite_t4b_v12_p0_authority_release.json \
        scripts/audit_k4_lite_t4b_v12_p0_authority_release.py \
        tests/test_k4_lite_t4b_v12.py
git diff --cached --check
git commit -m "feat(vla): add p0 authority release runner"
```

### 阶段 C：远端只读 preflight

- [ ] SSH 可连接；
- [ ] GPU1 空闲，GPU0 不被本实验使用；
- [ ] manifest SHA 一致；
- [ ] J2 checkpoint fingerprint 一致；
- [ ] statistics SHA 一致；
- [ ] config SHA 一致；
- [ ] 输出目录不存在；
- [ ] 远端代码与本地提交一致；
- [ ] 不覆盖 v7–v11 结果。

只有全部通过才允许进入阶段 D。

### 阶段 D：远端运行 P0

```bash
ssh -p 27219 root@115.190.90.101
export W=/vepfs-mlp2/c20250405/400040/transfer/vla_attention/vla_workspace
export R=/vepfs-mlp2/c20250405/400040/transfer/vla_attention/repo/VLA_attention
export OUT=$W/results/K4_lite/T4b/p0_authority_release_15init_20261107
export CUDA_VISIBLE_DEVICES=1
cd "$R"
python scripts/audit_k4_lite_t4b_v12_p0_authority_release.py \
  --model "$W/models/openvla--openvla-7b" \
  --checkpoint "$W/results/JGATE_first_batch_20261023T0737cache/openvla--openvla-7b+libero_90_gate+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--JGATE_formal_20261023T0737cache--J2--12500_chkpt" \
  --manifest "$W/data/jgate_c1/counterfactual_20261027/manifest.jsonl" \
  --gate-config configs/vla/k4_lite_t4b_v12_p0_authority_release.json \
  --expected-checkpoint-fingerprint 747a28690383ca45e795518121e9aed3f4610265c14618a69f203922b4238e20 \
  --output "$OUT" \
  --max-inits 15 \
  --seed 7
```

- [ ] 4 arms x 15 init 全部跑完；
- [ ] 无 NaN、error、飞出；
- [ ] 结果 JSON/MD 和 traces 已落盘；
- [ ] 不产生 optimizer checkpoint。

### 阶段 E：回传与判 Gate

- [ ] 回传 `audit_p0_authority_release_v12.json` 和 `.md`；
- [ ] 写 `05_p0_result.md`；
- [ ] 按 Gate 判 `PASS/FAIL/INVALID`；
- [ ] 如果是 `PASS`，只写 P1 protocol，不直接训练；
- [ ] 如果是 `FAIL`，写 `H-AUTH` 否证证据和 caddy/PIVOT 建议。

提交：

```bash
cd /home/bubble/类脑计算/VLM终局
git add experiments/Goal14-p0-authority-release-20261107/05_p0_result.md \
        experiments/Goal14-p0-authority-release-20261107/p0_summary.json \
        experiments/Goal14-p0-authority-release-20261107/p0_summary.md
git diff --cached --check
git commit -m "results(vla): report p0 authority release gate"
```

推送（只在用户明确要求时）：

```bash
git push origin main
```

## 4. 停止条件

- `B0` 不是 `0/15` 且无法解释：停止，先审计评测或环境漂移；
- manifest/checkpoint/config/evaluator SHA 任一不一致：`INVALID`，停止；
- 出现 NaN、error、飞出或物体丢失：停止，先审计控制；
- `A20` 超过 `A10 + 1/15`：先审计过冲和 release 失败类型，不得直接扩大 cap；
- P0 `FAIL`：停止 P1，转 caddy headroom audit 或重写 PIVOT；
- 达到 4 arms x 15 init 预算：停止，不再自动加 arm。

## 5. Git 约束

- 只 `git add` 本实验相关文件；
- 不使用 `git add -A`；
- 不提交 checkpoint、视频、safetensors、`pt` 或大日志；
- 每次提交前执行 `git diff --cached --check`；
- 不自动 push，除非用户明确要求。
