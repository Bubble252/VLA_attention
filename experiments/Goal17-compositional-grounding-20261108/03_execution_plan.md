# Goal17 执行计划

**日期**：2026-11-08<br>
**状态**：`COMPLETED_PASS`（见 `06_result.md`）

## 1. 总顺序

```text
Step 0  Run Contract
Step 1  构造 2×2 counterfactual 训练特征
Step 2  训练 no-margin baseline
Step 3  训练 margin 版
Step 4  Grounding Gate（Track D / Track C）
Step 5  oracle destination + 简化 xy/release wrapper
Step 6  predicted destination（R-PRED）
Step 7  Gate 判读、归档、commit/push
```

## 2. Step 0：Run Contract

- [x] manifest / checkpoint / statistics SHA；
- [x] 15 个 eval init 不在 train；
- [x] white_right eval 不进入训练；
- [x] 4 prompt 覆盖检查；
- [x] no oracle fallback；
- [x] GPU1；
- [x] 无 residual/route/action training。

## 3. Step 1：特征

- [x] 从 train split 采样帧；
- [x] 每帧生成 4 条 prompt；
- [x] 提取 Qwen hidden states；
- [x] 写入 `index.jsonl` + `features/*.npz`；
- [x] 记录 mask、object/destination 标签、image_id、prompt_id。

## 4. Step 2–3：训练

- [x] no-margin baseline；
- [x] destination margin；
- [x] object margin；
- [ ] 3 seeds（本轮只完成 1 seed pilot；多 seed 留到下一轮）
- [x] 保存 checkpoint、metrics、config SHA。

## 5. Step 4：Grounding Gate

Track D：

- [x] destination error `<=0.15 m` 至少 `12/15`（G4-MARGIN 15/15）；
- [x] object in-mask 至少 `12/15`（15/15）；
- [x] left/right margin 方向正确至少 `12/15`（14/15）；
- [x] red/white object margin 方向正确至少 `12/15`（12/15）。

Track C：

- [x] 只报告，不参与主 Gate；Track C 仍 0/15。

## 6. Step 5–6：执行

- [x] `E1`：oracle destination + 简化 xy/release wrapper；
- [x] `E1`：predicted destination（R-PRED）；settle=4 为 0/15，settle=32 为 2/15；
- [x] `E2`：predicted destination + full 3D hand-back；3/15；
- [x] 15 init，seed 7；
- [x] 记录 official success、released success、wrong-destination、动作幅度；
- [x] 报告 `E1` vs `E2`：E2 数值最好，但只多 1/15，单 seed 不宣称显著。

## 7. Gate

- [x] `R-PRED success >=1/15`（E1-SETTLE32 2/15；E2-PRED 3/15）；
- [x] `R-PRED - B0 >= +1/15`（+2 / +3）；
- [x] `R-PRED released success >=1/15`（2/15）；
- [x] wrong-destination 不高于 `B0`（3 vs 14）；
- [x] 0 error / 0 NaN / 0 flew-out。

## 8. Git

```bash
cd /home/bubble/类脑计算/VLM终局
git add doc/vla_next/68_goal17_compositional_grounding_20261108.md \
        experiments/Goal17-compositional-grounding-20261108/ \
        configs/vla/goal17_grounding.json \
        scripts/audit_goal17_rpred.py \
        scripts/select_goal17_execution.py \
        server/run_goal17_grounding.sh \
        research/LEDGER.jsonl
git diff --cached --check
git commit -m "docs(vla): plan goal17 compositional grounding repair"
git push
```

禁止 `git add -A`。
