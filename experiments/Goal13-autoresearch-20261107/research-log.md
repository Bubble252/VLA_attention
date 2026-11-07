# Goal13 Research Log

## 2026-11-07

- D0 固定为 best data arm；D1 契约 PASS 但方向为负，W-D 关闭。
- 定义 I0/I1/I2/I3 接口层，并加入 matched launcher、evaluator、postrun verifier。
- 启动并完成 I0 500-step action-only continuation。
- 修复 `verify_goal13_postrun.py` 对 `data.expected_raw_transitions` 的嵌套字段读取问题。
- I0 postrun `PASS`；I0 limit=2 offline smoke `SMOKE_PASS`，动作幅度未塌缩。
- 下一步：完成 I2 500-step，再跑 I1/I2/I3 smoke。
