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
- 下一步：跑 I1/I2/I3 完整 118 groups，再判接口 Gate。
