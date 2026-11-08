# Goal19：Target-aligned destination grounding

**日期**：2026-11-09<br>
**上游**：[Goal18](69_goal18_destination_precision_20261108.md)、[Goal18 result](../../experiments/Goal18-destination-precision-20261108/06_result.md)<br>
**状态**：`PROTOCOL_DRAFTED_BEFORE_MEASUREMENT`

## Research Contract

```yaml
autoresearch:
  required: true
  triggers:
    - gate18_gp_fail
    - target_label_mismatch
    - no_metric_movement_after_3_fixes
  ledger: research/LEDGER.jsonl
  blockers_dir: research/blockers/
  decisions_dir: research/decisions/
  reading_dir: references/paper_reading/
  max_outer_loops: 3
  check_command: python3 scripts/check_research_ledger.py --goal Goal19 --goal-doc doc/vla_next/70_goal19_goal18_failure_research_loop_20261109.md --strict --require-result
```

Goal19 不是继续调 2D offset，而是修 Goal18 已经定位到的上游问题：
**训练标签与成功评测目标不一致**。

## 1. 已有事实

Goal18 的五个 arm 全部 `FAIL`，3 cm 内都只有 `6/15`：

| arm | 3 cm 内 | mean error | max error |
|---|---:|---:|---:|
| C0 | 6/15 | 0.03372 m | 0.07806 m |
| C1-GLOBAL | 6/15 | 0.03379 m | 0.07819 m |
| C1-PROMPT | 6/15 | 0.03394 m | 0.07836 m |
| C1-RELATION | 6/15 | 0.03387 m | 0.07831 m |
| C2-RESIDUAL | 6/15 | 0.03310 m | 0.07811 m |

C2 train MAE `0.46 mm`、val `8.07 mm`，是过拟合；C1 offset 约 `1e-4 m`，没有修正空间。

Goal18 的失败后诊断：

| 目标点 | mean error to oracle | max | 3 cm 内 |
|---|---:|---:|---:|
| predicted point | 33.72 mm | 78.06 mm | 6/15 |
| visible mask centroid | 22.27 mm | 31.83 mm | 0/15 |
| projected object center | 10.64 mm | 13.26 mm | 15/15 |

训练 manifest 使用 `visible_surface_centroid`；成功 oracle 使用 receptacle object center。
predicted point 到 projected object center 平均 `(-11.26, -3.85) px`，到 visible mask centroid 平均 `(-4.38, -1.50) px`。
15/15 的 world dx、dy 都为正，符合系统性标签偏移。

## 2. 假设

### H1：target-aligned label 足以修复 G-P

用 projected object center（或经过审核的等价区域目标）替换 visible surface centroid 后，
point/region head 应达到：

- mean destination error `<=0.015 m`；
- max `<=0.030 m`；
- 至少 `12/15` 在 3 cm 内；
- object in mask `>=12/15`。

最便宜证伪：先重建 train/validation 标签，只重训同一小 head；不增加新模块。

### H2：标签修复后仍有区域/接触点误差

如果 target-aligned label 把 mask centroid 的 `22.27 mm` 上限修到 `10.64 mm`，但模型仍达不到 G-P，
则剩余问题才是 region/contact-point 表达或 head capacity；此时才上 region head 或 TAG/AnySlot。

最便宜证伪：比较 target-aligned head 的 predicted-vs-label pixel error 与 label-vs-oracle error。

### H3：G-P 修好后 E1 仍然失败

如果 G-P 通过但 E1 不涨，说明 execution mapping 仍占主导；此时只做 release/transport 校准，
不继续扩大 grounding 模块。

最便宜证伪：在 3 个 calibration init 上，用同一 initial state 比较 target-aligned predicted 与 oracle 的
transport/release/settle 失败分解。

## 3. 数据与隔离

- 训练标签只来自 train split 的 sidecar；
- `projected_centroid_xy_256` 只作为训练标签，不作为部署输入；
- 3 个 calibration init 用于开发诊断；
- 12 个 confirmation init 在方法冻结前不得用于选标签、超参、seed 或架构；
- validation 只用于离线报告，不进入训练梯度；
- 禁止 oracle fallback 冒充 deployable arm；
- 所有 manifest、config、checkpoint 记录 SHA。

## 4. 执行计划

### Phase 0：Preflight

- [ ] 读取 Goal18 `05_gate.md`、`06_result.md`、`evidence/white_right_pixel_alignment.csv`；
- [ ] 写 `01_background.md`、`02_tech_stack.md`、`03_execution_plan.md`、`04_run_contract.md`；
- [ ] 在 Run Contract 中写清 label source、train/val/eval 隔离和 no oracle fallback；
- [ ] 运行 `python3 scripts/check_research_ledger.py --goal Goal18 ... --strict --require-result`，确认 Goal18 已关闭。

### Phase 1：Target-aligned label

- [ ] 写新的 label builder，保留旧 `visible_surface_centroid` 作为诊断，不覆盖旧证据；
- [ ] 用 `projected_centroid_xy_256` 生成新的 destination label，并核对 crop/resize 坐标；
- [ ] 对每个 train/val image 生成 4-prompt counterfactual 标签；
- [ ] 检查 train/validation 标签覆盖率、SHA、无 eval 泄漏；
- [ ] 先做 label-only ceiling：predicted point 换成 projected centroid 时 G-P 是否可达；
- [ ] 不训练大模型。

### Phase 2：Point/region head

- [ ] 冻结 VLM 特征和相机协议，只替换 destination label；
- [ ] 重训同一 PointQueryHead 或同一轻量 region head；
- [ ] 先在 3 个 calibration init 上评估 G-P；
- [ ] 如果 G-P 通过，再在 12 个 confirmation init 上做一次冻结评估；
- [ ] 不在 confirmation 上选 epoch、threshold 或 seed。

### Phase 3：Execution

- [ ] 只有 G-P 通过后才跑 E1；
- [ ] G-E：success `>=4/15`，比 Goal17 的 `2/15` 至少 +2；correct destination subset `>=3/10`；
- [ ] 只有 G-P 过、G-E 不过才做 release/transport 校准；
- [ ] 不扩大到完整 3D/6-DoF，除非有独立 ablation。

## 5. Gate

- G-P：mean `<=0.015 m`；max `<=0.030 m`；`>=12/15` 在 3 cm 内；object in mask `>=12/15`；0 NaN。
- G-E：E1 success `>=4/15`，且比 `2/15` 至少 +2；correct destination subset `>=3/10`；wrong dest `<=3`。
- G-S：最终 arm 至少 3 seeds，或明确写“单 seed 机制信号，不宣称显著”。
- G-C：Track C 单独报告；0/15 时禁止写“组合泛化已解决”。

## 6. 失败裁决

| 观察 | 判读 | 下一步 |
|---|---|---|
| target-aligned G-P 通过 | 标签是主因 | 进入 E1，不堆新模块 |
| target-aligned G-P 不过、label ceiling 仍差 | 区域/接触点定义仍错 | 上 region/contact-point head |
| target-aligned G-P 不过、label ceiling 够 | point head 容量/特征不足 | 才考虑 TAG/AnySlot/open-vocab |
| G-P 过、E1 不动 | execution mapping 主导 | 只做 release/transport 校准 |
| 只用 confirmation 调参 | 结果污染 | 标 INVALID，不判方法失败 |

## 7. 文献与 code

Goal18 已记录：

- ReGuide `2609.38616`：有 PDF/TXT 和 reading note，官方 code 未找到；
- AnySlot `2604.10432`：有 PDF/TXT 和 reading note，官方 code 未找到；
- TAG `2603.24584`：有 PDF/TXT 和 reading note，官方 code 未找到；
- R-VLA OpenReview：challenge 阻止下载，未使用。

Goal19 先用 target-aligned label；如果使用上述外部方法进入代码，必须重新检查 code、pin commit 或写
`official_code_missing=true`。没有 code 和 reading note 的方法不能进训练。

## 8. Git

禁止 `git add -A`。只 add Goal19 相关文件：

```bash
git add doc/vla_next/70_goal19_goal18_failure_research_loop_20261109.md \
        experiments/Goal19-target-aligned-20261109/ \
        configs/vla/goal19_target_aligned.json \
        scripts/*goal19* tests/*goal19* server/run_goal19* \
        references/paper_reading/ research/STATE.md research/LEDGER.jsonl
git diff --cached --check
git commit -m "feat(vla): target-aligned destination grounding"
git push
```

## 9. Goal19 提示词

```text
你现在执行 Goal19：Target-aligned destination grounding。

先完整阅读：
- /home/bubble/类脑计算/AGENTS.md
- VLM终局/AGENTS.md
- VLM终局/research/README.md
- VLM终局/research/STATE.md
- VLM终局/research/LEDGER.jsonl
- VLM终局/doc/vla_next/69_goal18_destination_precision_20261108.md
- VLM终局/experiments/Goal18-destination-precision-20261108/05_gate.md
- VLM终局/experiments/Goal18-destination-precision-20261108/06_result.md
- VLM终局/references/paper_reading/goal18_precision_survey.md

autoresearch: required

Goal18 已关闭：C0/C1/C2 全部 G-P FAIL，3 cm 内 6/15。失败后诊断发现
训练标签是 visible_surface_centroid，成功 oracle 是 receptacle object center。
mask centroid 到 oracle mean 22.27 mm；projected object center 到 oracle mean 10.64 mm；
所以先修标签，不继续调 2D offset，也不先堆 TAG/AnySlot。

硬约束：
1. 先写 Goal19 的 01_background.md、02_tech_stack.md、03_execution_plan.md、04_run_contract.md；
2. 训练标签只使用 train split 的 projected object center 或经过审核的等价区域目标；
3. 3 个 calibration init 只做开发诊断；12 个 confirmation init 在方法冻结前不得用于调参；
4. validation 不参与训练或选超参；禁止 oracle fallback；
5. 只重训同一小 head，不先引入新大模块；
6. 不覆盖 Goal18 的旧标签和旧证据，新 label 另存并记录 SHA；
7. 任何外部方法进入代码前：下载 PDF/TXT、查 code、clone/pin 或写
   official_code_missing=true，并写 references/paper_reading/ note。

执行顺序：
1. Phase 1：生成 projected object center 的新 destination label；
   核对 224/256、crop、mask 和 4-prompt counterfactual 坐标；
2. 先做 label-only ceiling：predicted point 换成 projected centroid 时 G-P 是否可达；
3. Phase 2：重训同一 PointQueryHead；先在 3 个 calibration init 上评估 G-P；
4. G-P 通过后，才在 12 个 confirmation init 上做一次冻结评估；
5. G-P 通过后才跑 E1；G-E 不通过只做 release/transport，不扩大 3D/6-DoF；
6. 如果 target-aligned G-P 仍失败，先判断是区域定义还是 head 容量，
   再决定 TAG/AnySlot/open-vocab；不得直接堆模块。

Gate：
- G-P：mean <=0.015 m；max <=0.030 m；12/15 在 3 cm 内；object in mask >=12/15；0 NaN；
- G-E：success >=4/15，且比 2/15 至少 +2；correct destination subset >=3/10；wrong dest <=3；
- G-S：最终 arm 至少 3 seeds，或明确写单 seed 机制信号；
- G-C：Track C 单独报告，0/15 时禁止写“组合泛化已解决”。

完成时：
- 写 06_result.md；
- 更新 research/STATE.md、research/LEDGER.jsonl、blockers/ 和必要的 decisions/；
- 运行：
  python3 scripts/check_research_ledger.py --goal Goal19 \
    --goal-doc doc/vla_next/70_goal19_goal18_failure_research_loop_20261109.md \
    --strict --require-result
- 只 add 本次相关文件，禁止 git add -A；git diff --cached --check；commit；push。
```
