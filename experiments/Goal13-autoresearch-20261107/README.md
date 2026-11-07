# Goal13：Evidence-to-Control Stage 0

本目录执行 [65_story_method_selection_20261107.md](../../doc/vla_next/65_story_method_selection_20261107.md) 的 `Data -> Interface -> Control` 三级阶梯。

当前状态：**Stage 0 数据审计、G0 视觉抽检、D0 Run Contract、D0 2-step smoke 和 offline `limit=2` smoke 已 PASS；D0 matched 2.5K/pilot、D1/D2 仍待完成。**

阅读顺序：

1. [01_background.md](01_background.md)：任务、已有证据、假设与边界。
2. [02_tech_stack.md](02_tech_stack.md)：数据、远端环境、脚本与契约。
3. [03_execution_plan.md](03_execution_plan.md)：阶段、Gate、命令和 git 流程。
4. [03b_run_contract.md](03b_run_contract.md)：每个 arm 启动前必须满足的契约。
5. [04_stage0_result.md](04_stage0_result.md)：Stage 0 审计、D0 contract 和 sidecar 状态。
6. [05_d0_smoke_result.md](05_d0_smoke_result.md)：D0 2-step smoke 的 postrun contract 和 checkpoint 有限性证据。
7. [06_d0_offline_smoke_result.md](06_d0_offline_smoke_result.md)：D0 offline `limit=2` evaluator smoke；只验证链路，不作方法判断。

一句话目标：先证明 C1 `white_right` 的数据和评测契约没有泄漏、没有配置幻觉，再决定是否值得跑 `D0/D1` 和后续接口/控制实验。

硬约束：

- `white_right` 训练计数必须为 0；
- train / validation / rollout 的 episode 交集必须为 0；
- 没有 Run Contract PASS，不启动训练；
- 没有本层 Gate，不升级论文级组件；
- 只使用 GPU1，训练和评测串行。
