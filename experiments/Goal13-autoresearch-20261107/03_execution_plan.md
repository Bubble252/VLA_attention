# 03 执行计划：Stage 0 -> D/I/A

## 0. 本阶段目标

先证明数据契约干净，再允许 D0/D1；先证明接口层有信号，再允许 A2；只有 A2 通过后才允许合并 F0。任何阶段失败先查契约，不先改方法。

## 1. Stage 0：数据审计与 Run Contract

### 0a 文档与代码边界

- [x] 建立 Goal13 目录；
- [x] 写背景、技术栈、执行计划和 Run Contract；
- [x] 实现纯 Python RLDS 审计脚本；
- [x] 实现契约构建器；
- [x] 写并通过单元测试；
- [x] 提交本地 git；
- [x] 同步审计脚本到远端临时目录（不覆盖远端 repo）。

### 0b 远端只读审计

- [x] 扫描 `D` 下全部 256 个 shard；
- [x] 输出语言计数和 episode 清单；
- [x] 证明 `white_right` 训练计数 = 0；
- [x] 证明 C1 train/validation/rollout 交集 = 0；
- [x] 记录 carrier 和 `white_bowl_right` 计数；
- [x] 检查 carrier HDF5/GT mask 是否存在；
- [x] 产生 `goal13_audit.json`。
- [x] 生成 100 帧 contact sheet 和最终位置一致性检查。

建议输出目录：

```text
$W/results/star105_goal13_20261107/audit/
  goal13_audit.json
  goal13_audit.log
```

### 0c Run Contract

- [x] 读取 audit + `c1_split_manifest_v1.json`；
- [x] 写 `D0` arm contract（PASS）；
- [ ] 写 `D1`、`D2` arm contract（等待 carrier LeRobot 转换后的 observed contract）；
- [x] 检查 `train_language_counts`、`heldout_prompt_train_count`、交集、carrier 数量（D0）；
- [x] 检查实际 sampling weights（从训练 config/launcher 读取，不从 YAML 猜测）（D0）；
- [x] 检查 effective batch、grad accumulation、seed、code/config SHA（D0）；
- [ ] 任一失败标 `contract_status=INVALID` 并停止；
- [x] 全通过才标 `PASS`，进入 G0（D0）。

### G0 数据 Gate

- [x] C1 三 prompt 计数与历史一致（37 / 44 / 45）；
- [x] `white_right` 训练计数 = 0；
- [x] carrier 至少 40 episodes；
- [x] 100 帧人工抽检语言与画面一致；
- [x] train/validation/rollout init 交集 = 0；
- [x] 实际采样权重与计划一致（D0 = 1.0）；
- [x] HDF5/GT mask 状态明确，缺失时 `weak_sidecar=true`。
- [x] 100 帧语言与画面抽检 PASS（red_left / red_right 最终红杯位置方向相反）。

## 2. Stage 1a：D0 vs D1

- [x] D0 2-step training smoke 跑通；
- [x] D0 smoke postrun contract PASS（见 [05_d0_smoke_result.md](05_d0_smoke_result.md)）；
- [x] 先跑 D0 offline `limit=2` smoke（[06_d0_offline_smoke_result.md](06_d0_offline_smoke_result.md)）；
- [x] 跑 D0 matched 2.5K training，并重新过 postrun contract；
- [x] 跑 D0 matched 完整 offline eval（118 groups，78+78 pairs）；
- [ ] D0 matched 复现 `white_right = 0/15` 或解释差异；
- [ ] D0 seen group-any / wrong-destination 与历史一致；
- [ ] D1 加入 carrier，重新过 Run Contract；
- [ ] D1 offline smoke 检查 NaN、动作幅度、gripper；
- [ ] D0/D1 pilot 使用同一 evaluator、batch、seed；
- [ ] 只取 best data arm 进入接口层；
- [ ] 不在没有 D0 的情况下宣称 D1 有效。

## 3. Stage 1b：I0/I1/I2/I3

- [ ] 选 best data arm；
- [ ] `I0` 无 marker；
- [ ] `I1` oracle marker，只作诊断；
- [ ] `I2` P1 predicted marker；
- [ ] `I3` shuffle/wrong marker 负控；
- [ ] `I1 > I0` 且 non-gripper MAE 改善 >= 10% 才继续；
- [ ] `I2 > I3` 才说明 marker 语义有效；
- [ ] 动作幅度、rotation、gripper 不塌缩；
- [ ] 没有 `I1/I2` 信号时不上 A2。

## 4. Stage 2：A0/A1/A2

- [ ] 固定 best interface arm；
- [ ] `A0` 普通 BC；
- [ ] `A1` counterfactual action ranking；
- [ ] `A2` sufficiency/necessity；
- [ ] `A1 > A0` 后才允许 A2；
- [ ] 报告 correct-vs-wrong action margin；
- [ ] 报告 wrong-destination，不允许用 no-destination 换分；
- [ ] 记录 slot/swap/suff/nec 与 action 的梯度范数比；
- [ ] 至少 2 seeds 同向才进入 F0。

## 5. Stage 3：F0 与论文级验证

- [ ] 只有 A2 PASS 后才合并 `F0 = D1 + I2 + A2`；
- [ ] 跑 `F0/F1/F2/F3/F4` 消融；
- [ ] `F0` 优于至少两个消融臂；
- [ ] `F4` 证明 phase/AC-VLA-lite 有贡献后才升级论文近似版；
- [ ] 30/50 条确认 + 3 seeds；
- [ ] caddy 或 LIBERO-CF 至少一个同向；
- [ ] 未完成论文/代码复现检查的组件标记 `INVALID`。

## 6. 远端执行顺序

```bash
ssh -p 27219 root@115.190.90.101
W=/vepfs-mlp2/c20250405/400040/transfer/vla_attention
G13=$W/results/star105_goal13_20261107
mkdir -p "$G13/audit" "$G13/contracts" "$G13/logs"
```

审计脚本先只读运行，不启动训练：

```bash
/root/starvla_cu124/bin/python /tmp/goal13/audit_goal13_rlds.py \
  --data "$W/vla_workspace/data/jesbu1--libero_90_rlds/libero_90_openvla_processed/1.0.0" \
  --split-manifest "$W/vla_workspace/data/jgate_c1/c1_split_manifest_v1.json" \
  --output "$G13/audit/goal13_audit.json" \
  --progress "$G13/logs/goal13_audit_progress.json"
```

契约构建：

```bash
/root/starvla_cu124/bin/python /tmp/goal13/build_goal13_contract.py \
  --audit "$G13/audit/goal13_audit.json" \
  --split-manifest "$W/vla_workspace/data/jgate_c1/c1_split_manifest_v1.json" \
  --data-arm D0 \
  --output "$G13/contracts/D0_contract.json" \
  --code-commit "$(cd "$W/vla_workspace/repo/starvla_55_20261030/repo" && git rev-parse HEAD)" \
  --per-device-batch 4 --grad-accum 1 --num-processes 1 --seed 17
```

## 7. Git 流程

每次有实质进展只提交本轮文档/脚本，不提交无关 modified 文件：

```bash
cd /home/bubble/类脑计算/VLM终局
git status --short
git add experiments/Goal13-autoresearch-20261107
git add scripts/audit_goal13_rlds.py scripts/build_goal13_contract.py tests/test_goal13_contract.py
git diff --cached --check
git commit -m "goal13: add stage0 data audit and run contract"
```

若某个文件尚未准备好，不得为了“先提交”把它加入暂存区。远端只同步已提交脚本；不同步未审查改动，不 push。

## 8. 停止条件

- 审计发现 `white_right` 训练计数 > 0：停止，修 split；
- train/validation/rollout 交集 > 0：停止，修 split；
- audit/contract 脚本依赖 TensorFlow 或无法在 CPU 环境复现：停止，先修工具；
- carrier 少于 40 episodes 或没有可用 sidecar：标记 `weak_sidecar`，不进入 I2；
- C1 数据不可用或 G0 持续 INVALID：按 65 的触发条件转向 caddy；
- 任何 `INVALID` 都不能被解释成“方法失败”。
