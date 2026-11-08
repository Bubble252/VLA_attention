# Goal16 执行计划

**日期**：2026-11-08<br>
**状态**：`PROTOCOL_FROZEN_BEFORE_MEASUREMENT`

## 1. 总顺序

```text
Step 0  Run Contract / SHA / split / GPU 预检
Step 1  实现 ReGuide wrapper 与单元测试
Step 2  生成 R-PRED grounding sidecar
Step 3  1-episode 机械 smoke（只检查接口，不看 efficacy）
Step 4  B0 15-init 复现
Step 5  R-GEO / R-NAIVE 15-init
Step 6  R-PRED / R-SHUF 15-init
Step 7  G1/G2 Gate 判读
Step 8  结果归档、commit、push
```

## 2. Step 0：预检

- [x] 校验 manifest SHA256 = `e9564ca9...f45e`；
- [x] 校验 checkpoint fingerprint = `747a2869...8e20`；
- [x] 校验 statistics SHA256 = `6d5ebb47...9538`；
- [x] 校验 config SHA256 与 runner 中的常量一致；
- [x] 校验 `white_right` 训练计数 = `0`；
- [x] 校验 train/validation/rollout overlap = `0`；
- [x] 校验 15 个 init = 3 calibration + 12 confirmation；
- [x] 校验 GPU1 空闲 `>=25000 MiB`；
- [x] 生成 `run_contract.json`，状态为 `PASS`（grounding predictor 单列为 INVALID）。

## 3. Step 1：实现

新增：

```text
src/vla_attention/reguide_rebinding.py
scripts/audit_goal16_reguide_rebinding.py
scripts/build_goal16_grounding_sidecar.py
tests/test_goal16_reguide_rebinding.py
server/run_goal16_reguide_rebinding.sh
configs/vla/goal16_reguide_rebinding.json
```

- [x] 纯函数单元测试：坐标、目标选择、cap、deadband、hand-back、shuffle；
- [x] `python3 -m py_compile`；
- [x] `pytest -q tests/test_goal16_reguide_rebinding.py`（6 passed）；
- [x] 1-episode smoke：B0 0/1、R-GEO 1/1，transport + hand-back 正常；
- [x] smoke 不写入论文结果，不根据成功率改 confirmation 参数。

## 4. Step 2：R-PRED sidecar

- [x] 渲染 15 个 init 的 policy 224 agentview + wrist 图；
- [x] 用冻结 point head 预测 object/destination 2D 点；
- [x] 用固定相机标定把 destination 点投到桌面平面；
- [x] 记录预测点、原始点、oracle 距离、SHA；
- [x] point head 加载正常，但 15/15 超过 `0.15 m`；`R-PRED` 标 `INVALID`，未使用 oracle fallback。

## 5. Step 3：机械 smoke

只跑一个 calibration init，检查：

- [x] env reset / set_init_state 正常；
- [x] action queue 和 hand-back 正常；
- [x] `R-GEO` 的 z 动作没有 NaN / 飞出；
- [x] release fallback 正常；
- [x] 记录 `transport_start_step`、`handback_step`、`release_step`。

## 6. Step 4–6：15-init pilot

严格顺序：

1. `B0`；
2. `R-GEO`；
3. `R-NAIVE`；
4. `R-PRED`；
5. `R-SHUF`。

每个 arm 使用相同 15 个 init、相同 seed、相同 evaluator、相同 release/settle 口径。不得因为前一个 arm 的结果修改后一个 arm 的 confirmation 参数。

## 7. Gate

### G1：机械有效性

- [x] 五个 arm 都有 15 行；
- [x] `errors=0`、`nan_count=0`、`object_flew_out=0`；
- [x] `R-GEO` transport 成功且 hand-back 成功 `>=12/15`（15/15 + 14/15）；
- [x] `R-SHUF` 不显著优于 `B0`（0 vs 0）；
- [x] 所有 arm 的 action magnitude 都有记录。

G1 不通过：`INVALID` 或 `FAIL`，不进入 G2。

### G2：pilot 效果

- [x] `R-GEO` official success `>=1/15`（2/15）；
- [x] `R-PRED` official success `>=1/15`（FAIL，0/15；grounding INVALID）；
- [x] `R-PRED` 相对 `B0` 至少 `+1/15`（FAIL，0）；
- [x] `R-GEO` released success 有记录（2/15）；
- [x] `R-GEO` wrong-destination 不高于 `B0`（3 vs 14）。

G2 PASS 才允许扩 30/50 和 3 seeds。

### G3：扩展

- [ ] 30/50 init 上仍为正；
- [ ] 3 seeds 方向一致；
- [ ] 才考虑写“ReGuide-style wrapper 在 C1 上有效”。

## 8. 失败分支

| 观察 | 判读 | 下一步 |
|---|---|---|
| `R-GEO` 也 `0/15` | 不是 destination binding 单独问题 | 停止 wrapper 主线，转 APT 或任务重选 |
| `R-GEO` 正、`R-PRED` 负 | grounding 预测是瓶颈 | 修 point head / grounding，不训练新动作头 |
| `R-GEO` 与 `R-NAIVE` 同效 | 增益来自机械 transport | 重新定义方法贡献，不能写成语义 rebinding |
| `R-SHUF` 成功率接近 `R-GEO` | 语义负控失败 | 结果判 `INVALID`，检查 wrapper 选择逻辑 |

## 9. Git 流程

每次只 add 本阶段文件，禁止 `git add -A`。

### 文档提交

```bash
cd /home/bubble/类脑计算/VLM终局
git add doc/vla_next/67_dependency_repair_and_reguide_pivot_20261108.md \
        experiments/Goal16-reguide-rebinding-20261108/ \
        references/paper_reading/reguide_2609.38616.md
git diff --cached --check
git commit -m "docs(vla): freeze goal16 reguide rebinding protocol"
```

### 实现提交

```bash
git add src/vla_attention/reguide_rebinding.py \
        scripts/audit_goal16_reguide_rebinding.py \
        scripts/build_goal16_grounding_sidecar.py \
        tests/test_goal16_reguide_rebinding.py \
        server/run_goal16_reguide_rebinding.sh \
        configs/vla/goal16_reguide_rebinding.json
git diff --cached --check
git commit -m "feat(vla): add goal16 reguide rebinding wrapper"
```

### 结果提交

```bash
git add experiments/Goal16-reguide-rebinding-20261108/05_pilot_result.md \
        experiments/Goal16-reguide-rebinding-20261108/pilot_summary.json \
        experiments/Goal16-reguide-rebinding-20261108/run_contract.json
git diff --cached --check
git commit -m "results(vla): report goal16 reguide rebinding pilot"
git push
```

## 10. 当前检查点

- [x] 冻结原有 P1-C 结论：不再加 residual/route 容量；
- [x] 确认 B0 的失败模式是 first destination 错到 `plate_1`；
- [x] 明确 R-GEO 与 A10 的区别；
- [x] 明确 R-PRED 使用外部 point head，不使用 oracle fallback；
- [x] 明确 R-SHUF 使用 `plate_1` 负控；
- [x] 创建 Run Contract JSON（PASS；grounding predictor 为 INVALID）；
- [x] 实现 wrapper；
- [x] 生成 grounding sidecar；
- [x] 运行 15-init pilot；
- [x] Gate 判读和结果归档（`05_pilot_result.md`）。
