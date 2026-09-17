# P4：VLM V0–V4 protocol

未运行。P1 四个接口通过、teacher calibration 完成后才填写并提交。

```text
V0 checkpoint-only
V1 SFT
V2 DB-style feature retention
V3 best-single T_sem → A_lang
V4 V2 + V3
```

主 benchmark：Flickr30k Entities；外部指代：RefCOCOg；视觉 OOD：固定 Flickr task-preserving perturbations；保留诊断：VL-Think static QA。具体 data manifest、teacher configuration、预算和 metrics 在运行前填入 JSON 并 commit。

受控比较模板位于 `configs/experiments/P4_vlm_v0_v4_template.json`，由 `scripts/validate_v0_v4_template.py` 审计。它在 teacher 未选择、LoRA target modules/rank/LR 未经 runtime audit 冻结时故意不可运行；禁止为了“尽快启动”把这些占位字段默认为任意值。

`scripts/build_caption_sft_manifest.py` 从官方 Entities `train.txt` 与原始 Flickr caption CSV 的交集构建 train JSONL：每张图展开原始五条 caption，固定 prompt 为 *Describe the image in a single sentence as a caption.*。此 manifest 是 V1--V4 唯一 SFT 标签来源；Entities phrase/box 不进入主表训练。

### F0 cache 完成后的自动顺序

F0 的 SD cache job 完成后无需人工确认，按以下顺序自动推进：

1. 读取 256 条 cache metadata、`failures.json`、map 有限性和 cache manifest；
2. 固定有效样本交集；若有失败，所有 V1--V4 都切换到同一交集，并记录失败集合；
3. 运行 V3 20-step semantic smoke；
4. 运行 V4 20-step retention + semantic smoke；
5. 运行 checkpoint save/load、V0 evaluation、wrong-word/wrong-image/random-map 负控；
6. 写 F0 status table，明确每项为 flow-gate 结果而非论文主表。

任何步骤出现 NaN、cache/grid mismatch、visual LoRA 非零、checkpoint 无法恢复或负控未生效时，停止后续扩大运行，记录失败与日志路径；不静默跳过。
