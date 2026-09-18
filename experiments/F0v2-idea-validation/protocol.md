# F0v2 idea validation protocol

- Held-out manifest: 64 image-phrase records from official Entities test, image-disjoint from F0 train, seed 23.
- Evaluate V0--V4 on the same records with pointing, mass-in-box, IoU and map validity.
- Compare V3 vs V1, V4 vs V2, and correct teacher vs wrong-word/wrong-image/random maps.
- This is directional evidence only; pass requires all trends before F1-10k.

当前 held-out manifest 已生成于 101：64 条、seed 23、SHA256 `7b16778e…`。F0 smoke 原先只保存 loss JSON，没有 adapter checkpoint，因此下一步必须补跑带 checkpoint save/load 的短训练，才能进行 V0–V4 held-out 指标比较；不能把 smoke loss 当作 held-out 结果。

## 运行入口（已冻结）

`scripts/eval_qwen_heldout.py` 只读加载模型与 LoRA，不更新权重；`scripts/run_f0_heldout_all.sh` 使用同一 64 条图像独立样本批量评估 V0--V4。每条样本输出：`pointing`、`mass_in_box`，以及最高 20% 归因 token 外接框与 GT 框的 `top20_box_iou`。

先将新增脚本上传到 101（GitHub 暂不可达时可直接用本机 SSH）：

```bash
scp scripts/run_qwen_v3_smoke.py scripts/run_qwen_v4_smoke.py \
    scripts/eval_qwen_heldout.py scripts/eval_teacher_map_controls.py \
    scripts/run_f0_heldout_all.sh vla101:/vepfs-mlp2/c20250405/400040/transfer/vla_attention/repo/VLA_attention/scripts/
```

为避免漏传或传错版本，推荐直接运行仓库内的校验脚本：

```bash
cd /home/bubble/类脑计算/VLM终局
bash server/upload_validation_scripts.sh
```

上传成功并确认 `SSH_OK` 后，在本机运行：

```bash
bash server/run_f0_checkpoint_smokes.sh
```

该入口会在 101 上检查 256 个 SD map、模型、manifest 和配置；拒绝覆盖已有
V3/V4 产物；顺序完成 V3→V4 各 20 step，并分别检查 adapter 文件和 V4 的
`dino_projector.pt`。最终标志为 `CHECKPOINT_SMOKES_OK`。

若某张卡显存被其他任务占用，可显式选择空闲卡，例如：

```bash
CUDA_VISIBLE_DEVICES=1 bash server/run_f0_checkpoint_smokes.sh
```

重试前必须执行 `ssh vla101 nvidia-smi`，确认目标卡上没有未知任务；不要直接终止 PID。

V3/V4 smoke 成功后，运行 64 条 held-out 评估：

```bash
CUDA_VISIBLE_DEVICES=1 bash server/run_f0_heldout_remote.sh
```

该入口在 101 的 P1 环境依次评估 V0、V1、V2、V3、V4，拒绝覆盖已有
`report.json`，最终标志为 `HELDOUT_MODELS_OK`。它只生成归因图和指标，不更新模型。

它会上传 5 个验证脚本并在 101 端逐个执行 SHA256 校验；只有看到
`UPLOAD_VERIFY_OK` 才继续 checkpoint 或 held-out 评估。若本机 SSH alias 不是
`vla101`，可显式指定：

```bash
REMOTE_HOST='root@115.190.90.101' bash server/upload_validation_scripts.sh
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

评估结束后可查看只读状态与初步趋势：

```bash
bash server/summarize_f0_heldout.sh
```

`MODEL_HELDOUT_READY` 只表示五组报告齐全；最终 idea 判定还必须加入 teacher
correct、错配、shift 和 random 负控。

20-step 若未出现趋势，只能作为工程 smoke，不能直接宣判 idea 失败。为区分训练步数不足与方法无效，使用同一 256 条训练 manifest、同一 seed 和超参运行 100-step 四组对照：

```bash
F0_STEPS=100 F0_TAG=100 CUDA_VISIBLE_DEVICES=1 bash server/run_f0_100step_train.sh
```

成功标志为 `LONG_TRAIN_100_OK`；该阶段仍是小规模方向性验证，不替代后续 F1-10k。

100-step checkpoint 完成后，在同一 held-out 上复评：

```bash
CUDA_VISIBLE_DEVICES=1 bash server/run_f0_100step_heldout.sh
```

最终标志为 `HELDOUT_100_MODELS_OK`。

若 100-step 仍为部分支持，使用同一入口进行 500-step 复核：

```bash
F0_STEPS=500 F0_TAG=500 CUDA_VISIBLE_DEVICES=1 bash server/run_f0_100step_train.sh
CUDA_VISIBLE_DEVICES=1 bash server/run_f0_500step_heldout.sh
```

如需一次性完成模型评估、held-out SD cache 和负控，可运行：

```bash
CUDA_VISIBLE_DEVICES=1 bash server/run_f0_full_validation_remote.sh
```

它会复用已有报告和 map，不覆盖 checkpoint；最终标志为
`F0_FULL_VALIDATION_OK`。
