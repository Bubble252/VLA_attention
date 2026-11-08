# Goal17 执行计划

**日期**：2026-11-08<br>
**状态**：`PLANNED`

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

- [ ] manifest / checkpoint / statistics SHA；
- [ ] 15 个 eval init 不在 train；
- [ ] white_right eval 不进入训练；
- [ ] 4 prompt 覆盖检查；
- [ ] no oracle fallback；
- [ ] GPU1；
- [ ] 无 residual/route/action training。

## 3. Step 1：特征

- [ ] 从 train split 采样帧；
- [ ] 每帧生成 4 条 prompt；
- [ ] 提取 Qwen hidden states；
- [ ] 写入 `index.jsonl` + `features/*.npz`；
- [ ] 记录 mask、object/destination 标签、image_id、prompt_id。

## 4. Step 2–3：训练

- [ ] no-margin baseline；
- [ ] destination margin；
- [ ] object margin；
- [ ] 3 seeds（至少 1 seed pilot + 3 seeds confirmation）；
- [ ] 保存 checkpoint、metrics、config SHA。

## 5. Step 4：Grounding Gate

Track D：

- [ ] destination error `<=0.15 m` 至少 `12/15`；
- [ ] object in-mask 至少 `12/15`；
- [ ] left/right margin 方向正确至少 `12/15`；
- [ ] red/white object margin 方向正确至少 `12/15`。

Track C：

- [ ] 只报告，不参与主 Gate。

## 6. Step 5–6：执行

- [ ] oracle destination + 简化 xy/release wrapper；
- [ ] predicted destination（R-PRED）；
- [ ] 15 init，seed 7；
- [ ] 记录 official success、released success、wrong-destination、动作幅度。

## 7. Gate

- [ ] `R-PRED success >=1/15`；
- [ ] `R-PRED - B0 >= +1/15`；
- [ ] `R-PRED released success >=1/15`；
- [ ] wrong-destination 不高于 `B0`；
- [ ] 0 error / 0 NaN / 0 flew-out。

## 8. Git

```bash
cd /home/bubble/类脑计算/VLM终局
git add doc/vla_next/68_goal17_compositional_grounding_20261108.md \
        experiments/Goal17-compositional-grounding-20261108/
git diff --cached --check
git commit -m "docs(vla): plan goal17 compositional grounding repair"
git push
```

禁止 `git add -A`。
