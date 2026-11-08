# Goal18：Destination precision + residual correction

**日期**：2026-11-08<br>
**上游**：Goal17 `06_result.md`、doc 69

## 1. 问题

Goal17 修好了 binding，但没有修好执行成功率：

| arm | success | released | target dest correct | wrong dest |
|---|---:|---:|---:|---:|
| `B0` | 0/15 | 0/15 | 0/15 | 14/15 |
| `E1-ORACLE` | 7/15 | 4/15 | 10/15 | 3/15 |
| `E1-PRED` | 2/15 | 2/15 | 10/15 | 3/15 |
| `E2-PRED` | 3/15 | 2/15 | 10/15 | 3/15 |

逐 init 看，`E1-PRED` 在 10 个正确目的地 init 中只有 1 个 success；同样 10 个 init 换成 oracle destination 后有 6 个 success。失败主要是 `predicate_failure`，最终距离中位数约 4.7 cm，oracle 约 2.9 cm。

## 2. 本实验要回答

> 只用 train split 的 mask 标签，学习 2D destination offset / residual，能否把 held-out destination error 压到 1.5 cm，并把 E1 success 从 2/15 提到至少 4/15？

## 3. 不做

- 不训练完整 6-DoF action residual；
- 不扩大 full 3D hand-back；
- 不用 eval/validation 调参；
- 不用 oracle fallback；
- 不改 15-init eval 和 split。
