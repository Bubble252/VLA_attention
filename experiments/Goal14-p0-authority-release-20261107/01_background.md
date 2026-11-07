# Goal 14 P0：横向 authority + release/settle 闭环上界

**日期**：2026-11-07
**状态**：预注册；先复现闭环上界，不训练
**上游依据**：[65](../../doc/vla_next/65_story_method_selection_20261107.md)、[66](../../doc/vla_next/66_outer_loop_pivot_after_interface_failure_20261107.md)

## 1. 为什么现在只做 P0

Goal13 已经关闭 `W-I`：common-mode、cross-attention、query-token 三种接口注入都没有得到 `I1 > I0` 或 `I2 > I3`。因此继续加 marker、slot 或 action-query 模块不会回答真正的问题。

当前剩余的两个假设是：

- `H-AUTH`：低 z 段的横向 authority 不足，是 `white_right` 闭环失败的主要剩余瓶颈；
- `H-DISTILL`：如果 `H-AUTH` 成立，特权教师生成的横向修正能否被可部署学生网络蒸馏出来。

P0 只检验 `H-AUTH`，不检验 `H-DISTILL`，也不训练任何网络。

## 2. 已有证据和本实验要补的空白

| 已有证据 | 已知结论 | 还缺什么 |
|---|---|---|
| v8 `D3` | release/settle 可执行，但仍有毫米级横向误差 | 没有与 v10 的更强横向 authority 组合 |
| v8.1 | 延迟释放 4 步只改变约 `0.1mm` | 不是“释放太早”单独造成的 |
| v9 | 继续控制 142 步只改善 `0.08mm` | 低 z 段可能进入横向平台期 |
| v10 | fine cap `0.10` 在 3 个 calibration init 上 hold-only `3/3` | 只有 hold-only，不是 released success |
| v11 | 加 z/姿态控制后 `XYZ10` 从 `3/3` 退化到 `0/3` | z/姿态不能替代横向 authority |

因此 P0 的唯一问题是：

> 把 v10 验证过的横向 authority 与 v8/v8.1 的 release/settle 组合后，`white_right` 能否出现真实的 released success，而不只是 hold-only success？

## 3. 实验边界

- 任务：C1 `LIVING_ROOM_SCENE5`；
- prompt：`put the white mug on the right plate`；
- 冻结 15-init manifest；
- 3 个 v10 calibration init 与 12 个未参与 v10 的 confirmation init 分开报告；
- seed `7`；
- arms：`B0`、`A05`、`A10`、`A20`；
- 控制器只改 `x/y`，z 和旋转保持 base policy；
- 不训练、不 backward、不产生 optimizer checkpoint；
- 特权物体/盘子位置只用于诊断控制，不能写成可部署方法结果。

## 4. 预注册假设与判据

### 4.1 `H-AUTH` 预测

- `A10` 的 official success 明显高于 `B0`；
- `A05` 因低于有效 authority 阈值，success 明显低于 `A10`；
- `A20` 可能更快，但会出现过冲，不能稳定优于 `A10`；
- `A10` 的 success 必须包含 released success，不能重复 v10 的 hold-only。

### 4.2 否证条件

- `A10` official success `<5/15`；
- `A10` 在 12 个 confirmation init 中 `<2/12`；
- `A10` released success `<4/15`；
- `A10` 相对 `A05` 的优势 `<2/15`；
- `A20` official success 超过 `A10 + 1/15`，且审计无法排除过冲/飞出；
- 出现 error、NaN、物体飞出，或 object retained `<14/15`。

## 5. 不做的事

- 不做接口注入变体；
- 不启动 `A2/A3/F0`；
- 不升级 CompoSlot、AC-VLA-lite 或 factor loss；
- 不把 P0 的 oracle authority 写成最终方法；
- 不追已经接近 ceiling 的 `14/15`；
- P0 未 PASS 前不写 P1 训练代码、不启动 P1。

## 6. 结果解释

- **PASS**：`H-AUTH` 获得强支持；下一步只允许写 P1 protocol，仍不直接训练；
- **FAIL**：停止 residual 容量路线，转 caddy zero-shot headroom audit 或重写 outer-loop PIVOT；
- **INVALID**：manifest、checkpoint、config、evaluator SHA 或数据 split 不满足契约，不能判方法失败。
