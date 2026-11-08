# Research Ledger

这个目录把“遇到难关自己调研”外化成可检查的文件，而不是依赖模型记忆。

## 文件

| 文件 | 作用 |
|---|---|
| `STATE.md` | 当前 goal、blocker、hypothesis、next experiment |
| `LEDGER.jsonl` | 每次 research 的决策记录，一行一条 |
| `blockers/` | 单个 blocker 的现象、证据、排除项 |
| `decisions/` | pivot / stop 的决策记录 |

## 触发

见 `/home/bubble/类脑计算/AGENTS.md` 和 `VLM终局/AGENTS.md`。核心触发是：

- 同一 Gate 连续失败 2 次；
- 连续 3 次修复后指标不动；
- held-out / OOD 系统性错误；
- 方法差异小于 seed 噪声；
- 现有 hypothesis 无法解释失败。

## LEDGER 字段

每条 JSONL 必须包含：

```text
date, goal, trigger, evidence, hypotheses, experiment, result, decision, next
```

`decision` 只能是 `continue` / `pivot` / `stop`。`hypotheses` 至少 2 条。

## 检查

```bash
python scripts/check_research_ledger.py --goal Goal17 --goal-doc doc/vla_next/68_goal17_compositional_grounding_20261108.md --strict
```
