# Goal18：Destination precision + residual correction（precision-first）

**日期**：2026-11-08<br>
**上游**：[Goal17](68_goal17_compositional_grounding_20261108.md)、[Goal17 result](../../experiments/Goal17-compositional-grounding-20261108/06_result.md)<br>
**状态**：`COMPLETED_FAIL`（见 [06_result.md](../../experiments/Goal18-destination-precision-20261108/06_result.md)）

## Research Contract

```yaml
autoresearch:
  required: true
  triggers:
    - gate_fail_twice
    - no_metric_movement_after_3_fixes
    - heldout_collapse
    - seed_noise_ambiguous
  ledger: research/LEDGER.jsonl
  blockers_dir: research/blockers/
  decisions_dir: research/decisions/
  reading_dir: references/paper_reading/
  max_outer_loops: 3
  check_command: python scripts/check_research_ledger.py --goal Goal18 --goal-doc doc/vla_next/69_goal18_destination_precision_20261108.md --strict
```

Goal18 是精度修复型 goal。预注册 Gate 已记录为 `FAIL`；失败后已进入 autoresearch outer loop，并把 decision 写成 `pivot`。结果、Gate 和证据见实验目录。

## 1. 背景

Goal17 已经修好 deployable binding：

- `G4-MARGIN` / `G4-NO-MARGIN` 的 `white_right` destination 都是 `15/15`；
- 但 destination mean error 仍是 `0.0337 m` / `0.0320 m`，max 约 `0.078 m`；
- `E1-ORACLE` 为 `7/15`，`E1-PRED` 为 `2/15`，`E2-PRED` 为 `3/15`。

逐 init 复核显示：

- `E1-PRED` 有 10/15 到达正确盘，但只有 1 个 official success；
- 同样这 10 个 init 换成 oracle destination 后有 6 个成功；
- 失败主要是 `predicate_failure`，说明物体到了盘附近，但最终落点不在官方 success 的 3 cm 内；
- 在正确目的地的 10 个 init 中，destination error 与最终距离的相关系数约 `0.57`。

结论：**当前最便宜、最有信息量的下一步不是加完整 3D/6-DoF residual，而是把 predicted destination 的 2D 误差压下去。**

## 2. 假设

### H1：2D bias 是主要误差

只使用 train split 的 mask 标签，学习 global / per-prompt / per-relation 的 2D destination offset，能把 15 个 held-out init 的 mean destination error 压到 `<=0.015 m`，并把 `E1` success 从 `2/15` 提到至少 `4/15`。

- 成立：G-P 通过，且 `E1` 至少 `4/15`；
- 不成立：G-P 不通过，或 error 降了但 success 不动；
- 最便宜证伪：先做 global offset，再做 per-prompt offset，不训练大模型。

### H2：剩余误差是 release/execution offset

如果 2D destination calibration 后 G-P 通过、但 G-E 不过，剩余误差主要来自 release/execution 的固定 XY offset；应只学习 release XY residual，而不是完整 6-DoF action residual。

- 成立：校正 destination 后仍出现稳定方向的最终落点偏移；
- 不成立：校正后 final distance 已进入 3 cm，失败来自 wrong destination / target not reached；
- 最便宜证伪：比较 corrected destination 与 release 点的 XY 误差，不训练动作头。

### H3：E2 的 1/15 优势是 seed noise

`E2-PRED 3/15` 与 `E1-PRED 2/15` 的差异可能只是 seed noise；必须用 3 seeds 判断 full 3D hand-back 是否真的更好。

- 成立：3 seeds 的均值差异小于 seed 方差，或方向不稳定；
- 不成立：E2 在 3 seeds 上都稳定高至少 2/15；
- 最便宜证伪：只对最终候选跑 3 seeds，不扩展 3D 模块。

## 3. 数据与隔离

- 训练帧只来自 train split；
- 训练标签只来自已有 mask / 几何，不使用 simulator success、future frame、eval image；
- 15 个 eval init、validation、rollout 不得进入训练；
- 15 个 eval init 中 calibration 3 个、confirmation 12 个，继续冻结；
- 禁止 oracle fallback；`E1-ORACLE` 只作为上界诊断，不是 deployable arm；
- 禁止用 eval/validation 选择 offset、residual 超参数或 seed。

## 4. 方法

### C0：复现 E1-PRED

使用 Goal17 的 `G4-MARGIN` point head、`settle_steps=32`、15 init 复现 `2/15`。这是 baseline，不重新调参。

### C1：train-only 2D offset calibration

1. 用 frozen `G4-MARGIN` point head 对 train split 的 4-prompt 特征做预测；
2. 对每条样本计算 `label_destination - predicted_destination`；
3. 在 train split 上计算 global offset、per-prompt offset、per-relation offset；
4. 将 offset 加到 eval 的 predicted destination 上，不改变 point head 权重；
5. 主 arm 预注册为 per-prompt offset；global 和 per-relation 作为诊断。

### C2：轻量 destination residual head

输入：

- masked-mean visual token；
- masked-mean language token；
- frozen point head 的 object/destination point；
- prompt id embedding。

输出：

- destination 的 `dx, dy` 修正量（米，桌面平面 XY）。

训练：

- 只用 train split 的特征和 mask 标签；
- 目标为 `true_destination_xy - predicted_destination_xy`；
- 小 MLP，固定步数，不使用 eval；
- 保存 checkpoint、训练/验证误差、输入 feature SHA。

### C3：release XY residual（条件执行）

只有 C1/C2 通过 G-P、但 G-E 不过时才允许做。只学习 release 点的 XY 修正，不扩大到 z、姿态或完整 6-DoF。

### E2：对照，不扩展

`E2-PRED` 只保留为 full 3D 对照。不能因为 `3/15 > 2/15` 就提前扩大 3D 模块；H3 需要 3 seeds。

## 5. Gate

### G-P：Grounding precision

| check | 阈值 |
|---|---:|
| mean destination error | `<=0.015 m` |
| max destination error | `<=0.030 m` |
| destination within `0.03 m` | `>=12/15` |
| object in mask | `>=12/15` |
| no NaN | 15/15 |

### G-E：Execution

| check | 阈值 |
|---|---:|
| E1 success | `>=4/15` |
| E1 success - current `2/15` | `>=+2` |
| correct-destination subset success | `>=3/10` |
| wrong destination | `<=3` |
| errors / NaN / flew-out | `0 / 0 / 0` |

### G-S：Stability

- 对通过 G-P/G-E 的最终 arm 跑 3 seeds；
- 报告 mean / std；
- 1/15 的差异不得写成显著。

### G-C：Compositional diagnostic

- Track C 仍单独报告；
- 如果 Track C 仍为 `0/15`，只能写 deployable precision，不得写组合泛化已解决。

## 6. 失败裁决

| 观察 | 判读 | 下一步 |
|---|---|---|
| G-P 不过 | 现有 point head + 2D 校准不够 | 停止 point-head 精度线，转 TAG/AnySlot/open-vocab detector；先写 reading note 和 code 检查 |
| G-P 过、G-E 不过 | 剩余是 release/execution offset | 只做 release XY residual，不做完整动作 residual |
| G-P/G-E 过、3 seeds 不稳定 | 单 seed 机制信号 | 写成不稳定，不扩方法 |
| C1/C2 都不过 val | offset 不是主要矛盾 | 回到逐 init 误差分解，检查 mask / 相机 / 坐标系 |

## 7. 研究要求

- 任何外部论文方法进入代码前，必须下载 PDF/TXT、查官方 code、clone/pin 到 `references/` 或记录 `official_code_missing=true`，并写 `references/paper_reading/` note；
- 不使用外部方法时，在 result 中写 `no_external_method_used`；
- 结果必须回答：是否触发 autoresearch、触发了哪个 blocker、读了哪些论文、是否有下载和 reading note、如果没有触发为什么；
- 完成前运行：

```bash
python scripts/check_research_ledger.py --goal Goal18 \
  --goal-doc doc/vla_next/69_goal18_destination_precision_20261108.md --strict
```

## 8. Git

禁止 `git add -A`。只提交 Goal18 相关文件，提交前运行 `git diff --cached --check`。

```bash
git add doc/vla_next/69_goal18_destination_precision_20261108.md \
        experiments/Goal18-destination-precision-20261108/ \
        configs/vla/goal18_precision.json \
        scripts/*goal18* server/run_goal18_precision.sh tests/*goal18* \
        references/paper_reading/goal18_precision_survey.md \
        research/STATE.md research/LEDGER.jsonl research/blockers/2026-11-08-goal18-destination-precision.md
git diff --cached --check
git commit -m "feat(vla): add goal18 destination precision protocol and tools"
git push
```

## 9. Goal18 提示词

```text
你现在执行 Goal18：Destination precision + residual correction（precision-first）。

先完整阅读：
- /home/bubble/类脑计算/AGENTS.md
- VLM终局/AGENTS.md
- VLM终局/research/README.md
- VLM终局/research/STATE.md
- VLM终局/research/LEDGER.jsonl
- VLM终局/doc/vla_next/69_goal18_destination_precision_20261108.md
- VLM终局/experiments/Goal17-compositional-grounding-20261108/06_result.md
- VLM终局/references/paper_reading/goal18_precision_survey.md

autoresearch: required

这次不要把“Gate 第一次通过”当成不需要 research 的理由。Goal18 完成时必须至少有一条可检查的 research 记录；如果出现以下任一 hard blocker，必须完整执行 outer loop，而不是换参数继续试：
- 同一个 Gate 连续失败 2 次；
- 连续 3 次修复后指标没有移动；
- held-out / OOD 出现 100% 系统性错误；
- 两个 arm 的差异小于 seed 噪声；
- 现有 hypothesis 无法解释失败，且本地 ablation 无法区分。

完整 loop 的最低产物：
1. research/blockers/<date>-<slug>.md：现象、证据、已排除项、至少 2 个可证伪 hypothesis；
2. 按“数据 → 标签 → 评测 → baseline → 架构 → 文献”的顺序排查；
3. 用 paper-search / deep-research 检索，记录查询词、来源、日期和结论；
4. 如果使用外部方法：下载 PDF/TXT，查找并记录官方 code；能 clone 就 clone 并 pin commit，不能 clone 就写 official_code_missing=true 和理由；写 references/paper_reading/ note；
5. 跑最便宜的 falsification experiment；
6. 在 research/LEDGER.jsonl 写 entry，包含 decision（continue / pivot / stop）和 next；
7. 只有 decision 为 pivot/stop 时才允许改主线，且写 research/decisions/。

研究目标：
当前 binding 已修好，但 destination mean error 约 3.4 cm，E1-PRED 只有 2/15。先验证 2D bias 假设，不要一开始上完整 3D/6-DoF residual。

执行顺序：
1. 写 Run Contract，状态只能是 PASS / FAIL / INVALID；INVALID 不能判方法失败。
2. C0：复现 Goal17 G4-MARGIN + settle=32 的 E1-PRED 2/15，不调参。
3. C1：只用 train split 的 mask 标签计算 global / per-prompt / per-relation 2D offset；主 arm 是 per-prompt，global 和 per-relation 只作诊断。
4. 先跑 C1 的 G-P；如果 G-P 不过，停止 point-head 精度线，按失败裁决进入 outer loop，调研 TAG / AnySlot / open-vocab detector，不得偷偷改 G-P 阈值。
5. G-P 通过后跑 C2 轻量 destination residual head；输入只用 train 特征，输出桌面平面 dx/dy，禁止使用 eval/validation 调参。
6. 重新跑 E1；只有 G-P 通过但 G-E 不过时，才允许做 C3 release XY residual；不要扩大到 z、姿态或完整 6-DoF。
7. E2-PRED 只保留 full 3D 对照；3/15 对 2/15 的差异必须按 H3 跑 3 seeds，单 seed 不得宣称显著。

硬约束：
- 15 个 eval init、validation、rollout 不得进入训练；
- 禁止 oracle fallback；E1-ORACLE 只作上界诊断；
- 禁止用 eval/validation 选择 offset、residual 超参数或 seed；
- Track C 仍单独报告；Track C 0/15 时不能写“组合泛化已解决”；
- 所有 checkpoint、config、manifest、feature index 记录 SHA；
- 0 error / 0 NaN / 0 flew-out；
- GPU 按远端实际空闲情况选择，默认只用 GPU1，不占用 GPU0。

Gate：
- G-P：mean destination error <=0.015 m；max <=0.030 m；至少 12/15 在 0.03 m 内；object in mask >=12/15；无 NaN。
- G-E：E1 success >=4/15；比当前 2/15 至少 +2；correct-destination subset success >=3/10；wrong destination <=3；error/NaN/flew-out 全为 0。
- G-S：最终候选 arm 跑 3 seeds，报告 mean/std；不得把 1/15 差异写成显著。

完整做完后：
- 更新 experiments/Goal18-destination-precision-20261108/06_result.md；
- result 必须明确回答：是否触发 autoresearch、触发哪个 blocker、读了哪些论文、是否下载、是否有 reading note、是否使用外部方法、没有触发时为什么；
- 更新 research/STATE.md 和 research/LEDGER.jsonl；
- 运行：
  python scripts/check_research_ledger.py --goal Goal18 \
    --goal-doc doc/vla_next/69_goal18_destination_precision_20261108.md \
    --strict --require-result
- 只 add Goal18 相关文件，禁止 git add -A；运行 git diff --cached --check；commit 并 push。
```
