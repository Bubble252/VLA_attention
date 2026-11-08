# Goal18 执行计划

**日期**：2026-11-08<br>
**状态**：`COMPLETED_FAIL`（G-P FAIL；见 `05_gate.md` 和 `06_result.md`）

## 0. Preflight

- [x] 读取 `AGENTS.md`、`research/README.md`、`research/STATE.md`；
- [x] 读取 Goal17 result 和 evidence；
- [x] 写 `research/blockers/2026-11-08-goal18-destination-precision.md`；
- [x] 写 `references/paper_reading/goal18_precision_survey.md`；
- [x] 写 Run Contract 并通过；
- [x] 确认 15 eval init、validation 不进入训练；
- [x] 确认 GPU 可用或排队；
- [x] 确认只训练 destination calibration/residual，不训练完整动作头。

## 1. 数据与特征

- [x] 读取 Goal17 `features/train/index.jsonl`、`features/val/index.jsonl`；
- [x] 只使用 train split 拟合 offset/residual；
- [x] 生成 train/val base predictions；
- [x] 生成 C1 global / per-prompt / per-relation offset；
- [x] 生成 C2 residual head 训练集；
- [x] 记录 feature SHA、point head SHA、camera SHA。

## 2. C0 baseline

- [x] 复现 Goal17 `E1-PRED` 15 init，settle=32；
- [x] 记录 `2/15` baseline 和 per-init failure labels；
- [x] 记录 `E1-ORACLE 7/15` 作为上界，不作为 deployable arm。

## 3. C1 offset calibration

- [x] train-only global offset；
- [x] train-only per-prompt offset（主 arm）；
- [x] train-only per-relation offset（诊断）；
- [x] 在 15 init 上评测 G-P（FAIL）；
- [ ] 若 G-P 通过，跑 E1；
- [ ] 若 G-E 不过，按 H2 只做 release XY residual。

## 4. C2 residual head

- [x] 训练小 MLP；
- [x] 保存 checkpoint / train-val metrics；
- [x] 在 15 init 上评测 G-P（FAIL）；
- [ ] 若 G-P 通过，跑 E1；
- [x] 不把 C2 做成完整 action head。

## 5. Gates

- [x] G-P：mean `<=0.015 m`，max `<=0.030 m`，`>=12/15` 在 3 cm 内，object in-mask `>=12/15`；结果 FAIL。
- [ ] G-E：E1 success `>=4/15`，比 baseline `+2`，correct-destination subset `>=3/10`，wrong dest `<=3`；
- [ ] G-S：最终 arm 3 seeds；
- [ ] G-C：Track C 单独报告，若 0/15 不写组合泛化。

## 6. 结果与 Git

- [x] 写 `06_result.md`；
- [x] 更新 `research/LEDGER.jsonl`；
- [x] 运行 `check_research_ledger.py --strict --require-result`；
- [x] 运行相关 pytest / py_compile；
- [x] 只 add Goal18 相关文件；
- [x] `git diff --cached --check`；
- [x] commit；
- [x] push。
