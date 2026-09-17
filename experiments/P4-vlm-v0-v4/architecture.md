# V0--V4 共享代码架构

## 不变的运行骨架

```text
Flickr manifest + image/phrase/box
        │
        ├── Qwen adapter ───────────────► SFT / phrase-score attribution
        ├── frozen retention encoder ───► L_retention（仅 V2/V4）
        └── frozen best-single diffusion ► T_sem map cache ─► L_sem（仅 V3/V4）
                                                              │
                                              evaluator: grounding / OOD / retention
```

所有组共用同一 Qwen revision、同一 train manifest、同一 batch schedule、同一 seed 集、同一 evaluation manifest；差异只能来自下表启用的 loss，不能偷偷改变数据或预算。

**确认的首轮实现**：Qwen2.5-VL-7B-Instruct 采用 BF16 LoRA；视觉 encoder 冻结。`T_retention` 固定为 DINOv2 ViT-L/14，`T_sem` 为四个 diffusion 候选在 val calibration 后选择的 best-single，且 semantic map 在训练前离线缓存。Prismatic 不进入本轮 Qwen 主表。

LoRA 配置在 `configs/qwen_lora_v1.json`：rank 16、alpha 32、dropout 0.05、AdamW 2e-4、3% cosine warmup、global grad clip 1.0。`target_modules` 不是常见的 suffix list，而是完整路径正则，仅匹配 `model.language_model.layers.*` 的 attention/MLP projection；这从配置层排除名称相同的 `model.visual.blocks.*` 模块。所有 V1--V4 必须复用此文件的内容，训练前用真实 PEFT 匹配审计确认没有 visual LoRA parameter。

首轮 `train manifest` 的每行是 **image + 固定 caption prompt + 原始 Flickr30k caption**，而非 phrase-box/referring 标签。这样 diffusion map 的 caption token 与学生的 SFT target 一一对应。Entities phrase/box 留在 calibration/evaluation manifest；任何 referring prompt 训练都必须建立主表之外的独立 `R*` 消融。

| 组 | 可训练参数 | loss | 禁止项 |
|---|---|---|---|
| V0 | 无 | 无，仅 checkpoint eval | 训练、teacher 图 |
| V1 | Qwen SFT adapter | `L_sft` | retention、semantic teacher |
| V2 | 与 V1 相同 | `L_sft + λ_ret L_retention` | diffusion teacher |
| V3 | 与 V1 相同 | `L_sft + λ_sem L_sem` | retention teacher |
| V4 | 与 V1 相同 | `L_sft + λ_ret L_retention + λ_sem L_sem` | 更换数据或预算 |

## 将实现的模块

| 模块 | 职责 | 不承担的职责 |
|---|---|---|
| `data/flickr_entities.py` | 读取冻结 JSONL、加载图片/phrase/boxes、验证 SHA256 | 不重划 split |
| `adapters/qwen25_runtime.py` | processor、visual token positions、phrase/answer score、hidden-state hook | 不猜测网格或重排 token |
| `teachers/retention.py` | 冻结 feature encoder、student/teacher patch coordinate bridge | 不作为 `T_sem` |
| `teachers/diffusion_runtime.py` | 冻结 best-single diffusion，离线生成/缓存词图 | 不在 test set 选超参 |
| `losses/vlm.py` | `L_sft`、`L_retention`、`L_sem`，逐项记录数值 | 不隐藏 loss 开关 |
| `train/vlm_runner.py` | 固定 manifest、resume、20--50 step smoke、训练 | 不修改 teacher 或 split |
| `evaluation/grounding.py` | pointing、mass-in-box、IoU、wrong-word/image/random controls | 不把训练图当测试指标 |

## 依赖顺序

1. Qwen P1：运行时 adapter 和 token grid 已验证；
2. teacher calibration：选择并冻结一个 `T_sem`，离线 cache 只覆盖 train/calibration；
3. retention teacher、层、projector seed 固定；
4. 生成 V0--V4 五份不可变 manifest；
5. 每组先 20--50 step smoke，检查数据、loss、checkpoint restore；
6. smoke 均通过后才提交正式训练。

`V3/V4` 不能在 best-single 未冻结时实现为“任意扩散模型都能替换”的运行时开关；候选比较发生在 teacher calibration，不发生在 test 或主训练中。

### DINOv2 ViT-L/14 retention bridge

已冻结 `facebook/dinov2-large@47b73eefe95e8d44ec3623f8890bd894b6ea2d6c`（Apache-2.0）。其 smoke 对 224×224 输入返回 `1 + 16×16` 个 token、1024 维；CLS 不进入 `L_retention`。Qwen 的 `image_grid_thw` 依图变化，例如 P1 样本是 pre-merge `1×34×36`，post-merge 为 `17×18`。因此 `L_retention` 不能按 token index 直接相减：先删除 DINO CLS，再用以 token center 为定义的 normalized-image bilinear bridge 把 DINO feature grid 采样到 Qwen post-merge grid，最后经固定 projector 比较。

`src/vla_attention/spatial.py` 的 bridge 单测要求每个目标 token 权重和为 1；任何 crop/pad/resize 未记录时停止运行，而不是默认视为相同原图坐标。

V1--V4 将统一实例化相同维度的 DINO-to-Qwen projector，确保模型参数结构不因组别改变；V1/V3 的 `λ_ret=0`，V2/V4 启用 `L_retention=mean(1-cos(projector(bridge(DINO_patch)), Qwen_visual_token))`。DINO CLS 一律删除，DINO 参数始终 `requires_grad=False`。运行时必须断言 teacher feature 的 source grid 与 Qwen target grid 和 bridge 完全一致。
