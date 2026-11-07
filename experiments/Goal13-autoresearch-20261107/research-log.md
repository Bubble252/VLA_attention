# Goal13 Research Log

## 2026-11-07

- D0 固定为 best data arm；D1 契约 PASS 但方向为负，W-D 关闭。
- 定义 I0/I1/I2/I3 接口层，并加入 matched launcher、evaluator、postrun verifier。
- 启动并完成 I0 500-step action-only continuation。
- 修复 `verify_goal13_postrun.py` 对 `data.expected_raw_transitions` 的嵌套字段读取问题。
- I0 postrun `PASS`；I0 limit=2 offline smoke `SMOKE_PASS`，动作幅度未塌缩。
- I2 500-step 完成，projector 已训练（约 0.43M 可训练参数）；postrun `PASS`。
- 修复 preflight 的 `--point-checkpoint-sha256` 参数与 P1 SHA 校验。
- 修复 evaluator 的 GT point 注入：从 validation point manifest 逐键合并 354 行，并对 manifest SHA fail-closed。
- I1/I2/I3 limit=2 smoke 全部 `SMOKE_PASS`，三臂均产生 `point_results`。
- 2-group 早期信号为负：predicted - shuffle = -0.000690，oracle - none = -0.001014。
- I1/I2/I3 完整 118 groups 完成；四臂 contract PASS，`gate_status=DESCRIPTIVE`。
- 完整结果：I0 destination MAE `0.144959`；I1 `0.148985`；I2 `0.149226`；I3 `0.148852`。
- 接口 Gate FAIL：I1 的 non-gripper MAE 比 I0 差约 `+2.16%`；I2 比 I3 差约 `+0.35%`。
- 完美 oracle marker 的 MAE 仍比 none 差 `+0.000384`；predicted 的 MAE 比 shuffle 高 `0.000237`（正数表示更差；JSON 字段 `predicted_minus_shuffle` 的方向相反）。
- point head 质量不是主因：predicted destination L1 `2.645 px`、object L1 `7.807 px`、point-in-mask `0.8008`。
- 裁决：W-I 关闭；A2/A3/F0 冻结；下一步只做一次 `I2-XATTN` 注入修复，不增加 loss、不延长训练。
- 完成 `I2-XATTN` 2-step smoke：preflight PASS、postrun `SMOKE_PASS`、7 个 cross-attention 参数进入 checkpoint、`all_finite=true`。
- 完成 `I2-XATTN` 500-step matched pilot 及完整 118 groups：`I1` non-gripper MAE 比 `I0` 差 `+3.162%`，`I2` 比 `I3` 差约 `0.188%`，`I2` destination margin 也低于 `I3`；动作幅度未塌缩。
- 裁决：`I2-XATTN` FAIL；W-I 保持关闭；按协议只允许最后一次 `evidence_token=true` 注入修复，随后无论结果如何都不再增加第三个注入变体。
