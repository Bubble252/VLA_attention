# P1：模型接口审计 protocol

**状态**：未运行。  
**Git 前置提交**：本文件必须先于任何 P1 结果提交。  
**目标**：不是比较性能，而是确认四个首轮模型能产生坐标正确、目标明确、可微且可重复的归因接口。

## 假设

| ID | 模型 | 假设 | 通过条件 |
|---|---|---|---|
| H-P1-1 | Prismatic-7B | phrase score 可对视觉 token 求梯度 | 20 个固定样本均有有限梯度和有效 grid |
| H-P1-2 | Qwen2.5-VL-7B | 动态分辨率 token 能恢复到原图 | 每图 crop/grid/token 顺序有可逆 provenance |
| H-P1-3 | OpenVLA/OFT | action target 可对视觉 token 求归因 | action token/log-prob 或 adapter target 明确且 action shape 有记录 |
| H-P1-4 | π0.5 | fixed flow condition 下动作归因可复现 | 固定 seed/noise/time/chunk 后 map 有限且重复稳定 |

## 固定输入

- 20 个 immutable sample ID；
- 每个样本保存 image SHA256、instruction、phrase、original size、preprocess metadata；
- 不训练、不更新权重、不下载整套数据；
- 一次只审计一个模型，外部代码仓保持只读；
- 所有结果写入 `results/P1-interface-audit/<model>/`，大图和 tensor 放服务器 VEPFS，Git 只保存 `metrics.json` 和 `analysis.md`。

## 每个样本必须保存

```text
model_id / revision / adapter commit
target type / target scalar definition
image SHA256 / phrase token span / prompt template
visual token count / grid / crop / view / token order
seed / noise / flow time / action chunk（若适用）
raw map / normalized map / validity flags
```

## 停止条件

- grid 无法恢复；
- 目标标量没有到视觉 token 的可微路径；
- 归因随相同输入/seed 非确定地漂移；
- native action 和 7D adapter 的定义不一致；
- π0.5 的 checkpoint、许可证或接口不能确认。

出现任一项时，不用更多训练步数掩盖问题；记录失败，并切换到下一候选模型或更换 target 定义。
