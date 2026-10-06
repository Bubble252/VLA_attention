# VLA 结果入口索引（2026-10-06）

远端服务器：`vla101`，hostname=`di-20260613120619-84fwm`。

| 阶段 | 结果路径 | 当前证据 |
|---|---|---|
| B0 offline | `results/P6_vla_b0_b4_preflight_20261006/B0_offline_50_seed17/offline_eval.json` | 50-step offline preflight；审计 `vla_b0_offline_preflight_audit_20261006.json` 为 passed |
| B0 restore | `results/P6_vla_b0_b4_preflight_20261006/B0_restore_50_seed17/restore_report.json` | exact prediction equality；continuation 审计 passed |
| B1 | `results/P6_vla_b0_b4_preflight_20261006/B1_current_radio_20_seed17/report.json` | 20-step passed；radio manifest `2c472004...` |
| B2 | `results/P6_vla_b0_b4/B2_semantic_interface_seed17/report.json` | 20-step passed；semantic manifest `d7660c9c...` |
| B3 | `results/P6_vla_b0_b4/B3_dynamic_action_interface_seed17/report.json` | 20-step passed；semantic manifest `d7660c9c...` |
| B4 | `results/P6_vla_b0_b4/B4_combined_interface_seed17/report.json` | 20-step passed；semantic manifest `d7660c9c...` |

这些文件证明接口、有限步训练和 checkpoint 恢复可运行，不证明正式 LIBERO 成功率、OOD 或方法优越性。正式 rollout 仍受 demo provenance/disjointness 审计限制。
