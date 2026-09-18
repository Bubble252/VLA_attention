# F0v2 idea validation protocol

- Held-out manifest: 64 image-phrase records from official Entities test, image-disjoint from F0 train, seed 23.
- Evaluate V0--V4 on the same records with pointing, mass-in-box, IoU and map validity.
- Compare V3 vs V1, V4 vs V2, and correct teacher vs wrong-word/wrong-image/random maps.
- This is directional evidence only; pass requires all trends before F1-10k.

当前 held-out manifest 已生成于 101：64 条、seed 23、SHA256 `7b16778e…`。F0 smoke 原先只保存 loss JSON，没有 adapter checkpoint，因此下一步必须补跑带 checkpoint save/load 的短训练，才能进行 V0–V4 held-out 指标比较；不能把 smoke loss 当作 held-out 结果。

## 运行入口（已冻结）

`scripts/eval_qwen_heldout.py` 只读加载模型与 LoRA，不更新权重；`scripts/run_f0_heldout_all.sh` 使用同一 64 条图像独立样本批量评估 V0--V4。每条样本输出：`pointing`、`mass_in_box`，以及最高 20% 归因 token 外接框与 GT 框的 `top20_box_iou`。

先将新增脚本上传到 101（GitHub 暂不可达时可直接用 scp）：

```bash
scp scripts/run_qwen_v3_smoke.py scripts/run_qwen_v4_smoke.py \
    scripts/eval_qwen_heldout.py scripts/eval_teacher_map_controls.py \
    scripts/run_f0_heldout_all.sh vla101:/vepfs-mlp2/c20250405/400040/transfer/vla_attention/repo/VLA_attention/scripts/
```

服务器上补齐 V3/V4 checkpoint 后执行：

```bash
cd /vepfs-mlp2/c20250405/400040/transfer/vla_attention/repo/VLA_attention
MODEL=/vepfs-mlp2/c20250405/400040/transfer/vla_attention/models/Qwen2.5-VL-7B-Instruct \
ROOT=/vepfs-mlp2/c20250405/400040/transfer/vla_attention/data/flickr30k_entities \
MANIFEST=/vepfs-mlp2/c20250405/400040/transfer/vla_attention/repo/VLA_attention/experiment_workspace/manifests/flickr30k_entities/F0v2_heldout_test_64.jsonl \
CHECKPOINT_ROOT=/vepfs-mlp2/c20250405/400040/transfer/vla_attention/checkpoints/F0v2_idea_validation \
OUT=/vepfs-mlp2/c20250405/400040/transfer/vla_attention/results/F0v2_heldout_64 \
PYTHON=/vepfs-mlp2/c20250405/400040/transfer/vla_attention/envs/p1/bin/python \
bash scripts/run_f0_heldout_all.sh
```

教师负控随后使用 `scripts/eval_teacher_map_controls.py`。它在同一 held-out
manifest 上比较正确 SD phrase map、wrong-image（循环错配）、固定横向 shift 和 seeded random
map；若 held-out SD cache 尚未生成，先用 `run_sd_cache.py` 生成 64 条，不能把训练集 cache
冒充测试教师图。
