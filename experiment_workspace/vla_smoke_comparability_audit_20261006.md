# B0–B4 smoke 可比性审计（2026-10-06）

B1–B4 均使用相同的 train manifest SHA `695681afb03012bd18c34d09c9a4fbf15ca489605d4463cbf891cd7213bca921`、eval manifest SHA `c93ae5afeb919de8fb148bfa1f935173d04f3ac0502b85e3b6916afb15e9dd0b`、statistics SHA `21692284b4cbf9cdbdb2b3e2060ac4e62924eabf2b6db1881292e47f3b830397` 和 seed 17。

| 实验 | steps | 主要 teacher/loss | 状态 |
|---|---:|---|---|
| B0 | 50 offline preflight | action loss | passed；9 held-out episode，L1 mean `0.82927` |
| B1 | 20 interface smoke | C-RADIO retention，λ=`0.05` | passed |
| B2 | 20 interface smoke | semantic attribution，λ=`0.10` | passed |
| B3 | 20 interface smoke | semantic + containment，λ=`0.10/0.02` | passed |
| B4 | 20 interface smoke | retention + semantic + containment | passed |

这些运行只能证明接口、teacher 读取、forward/backward、finite 检查和结果落盘一致；由于步数、采样轨迹和评测目标不同，不能比较 B1–B4 的 action loss 均值来声称优劣。正式比较必须使用相同初始化、相同 step budget、相同 episode 顺序和 held-out evaluator。
