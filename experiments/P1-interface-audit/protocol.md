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

## 机器可验证的结果合同

每个 native adapter 写一个不含 tensor 的 `report.json`；实际 `.npy/.pt` 图仅以 VEPFS 的绝对路径出现在 `map_path`。格式由 `src/vla_attention/audit_io.py` 读取，最低限度含：

```text
capability: model_id / revision / visual-token support / declared targets / coordinate notes
measurements[]: sample_id / target / scalar_definition / seed / map_path /
                grid(height,width,token_indices,view,crop metadata) /
                gradient_finite / repeatability / action_shape（动作模型）
```

模型运行结束后先校验，不能通过就不能作为 P4/P6 的候选：

```bash
PYTHONPATH=src /root/starvla_cu124/bin/python scripts/validate_p1_report.py \
  /vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/P1-interface-audit/<model>/report.json
```

校验器默认要求 20 个唯一样本、全部有限梯度，以及出现 repeatability 时不低于 0.90。阈值改变必须写入 protocol 和 Git commit，不能只在命令行临时改。

## 每个样本必须保存

```text
model_id / revision / adapter commit
target type / target scalar definition
image SHA256 / phrase token span / prompt template
visual token count / grid / crop / view / token order
seed / noise / flow time / action chunk（若适用）
raw map / normalized map / validity flags
```

### Qwen2.5-VL 的额外坐标约束

`image_grid_thw` 是视觉 encoder 的 pre-merge patch 几何，不能直接把 `H×W` 当作语言模型中的 visual-token grid。对单张图，P1 从 checkpoint config 读取 `spatial_merge_size`，并固定记录：

```text
pre-merge grid: T,H,W = image_grid_thw
post-merge map: H / merge, W / merge
visual token order: 处理器输出的 image token positions
```

`src/vla_attention/adapters/qwen25.py` 只接受 `T=1` 来产生二维 P1 图；视频或多帧输入必须先声明 frame selection，禁止隐式铺平到一张图。测试以非方形 `12×20 → 6×10` grid 排除“由 token 总数猜平方网格”的错误。

## 停止条件

- grid 无法恢复；
- 目标标量没有到视觉 token 的可微路径；
- 归因随相同输入/seed 非确定地漂移；
- native action 和 7D adapter 的定义不一致；
- π0.5 的 checkpoint、许可证或接口不能确认。

出现任一项时，不用更多训练步数掩盖问题；记录失败，并切换到下一候选模型或更换 target 定义。
