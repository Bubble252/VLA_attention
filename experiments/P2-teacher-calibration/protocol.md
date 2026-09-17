# P2：扩散语义教师校准 protocol

**状态**：P1 已通过；候选模型已下载；尚未选择 best-single。  
**目的**：从固定四个 `T_sem` 候选中选择一个对真实图像上的 phrase-region map 最可靠的教师；不是比较图像生成质量。

## 输入与固定性

- 校准数据：Flickr30k Entities 的完整官方 `val.txt`，1000 张图；
- 每图恰选一个有非空 box 的 phrase，按 `caption_index → phrase.lower() → sample_id` 的确定性字典序，不根据模型输出筛样本；
- image、phrase、GT boxes、image SHA256、候选 revision 和所有 map 写入 record；
- 候选：SD1.5、PixArt-alpha、PixArt-Sigma、Playground-v2.5，见 `configs/teachers/semantic_teacher_candidates.json`；
- 图像 inversion 方法、attention layer/step、token span、resolution、seed、CFG、归一化必须在首次全量运行前写入 candidate config 并 commit。

## 允许的校准结论

每个候选仅在相同 1000 个 phrase-image pair 上报告：pointing accuracy、mass-in-box、box IoU、无效 map 比例、跨 seed map cosine。选择 best-single 的规则必须在运行前明确，例如按 primary pointing、再按 invalid-rate、再按 seed-stability 的字典序。

无论单个模型是否生成漂亮图像，都不能作为 `T_sem` 选择依据。任何只在 text-to-image 随机初始 latent 上得到的 attention，不能标为“真实图像教师图”；它最多是 hook smoke。

## 验收和后续

- [ ] 所有候选完成固定 val manifest；
- [ ] 每图都有 map、配置、失败原因和 GT 指标；
- [ ] 冻结唯一 best-single 与全部抽取参数；
- [ ] 对训练集离线生成该教师 map cache；
- [ ] 将 selected teacher 以新的 commit 写入 V3/V4 manifest。

## 当前实现证据（不等于校准通过）

2026-09-17：已用现代 Diffusers 0.38 + 本地 SD1.5 成功在一条固定 real-image latent / phrase 输入上捕获 conditional cross-attention：64px query grid 5 层、32px 5 层、16px 5 层、8px 1 层。运行时 FlashAttention 不能加载，自动回退 PyTorch attention。该结果只证明 hook 与模型接口可用；它没有 DDIM/null-text inversion，因此不能填入候选的 pointing/IoU 表，不能成为 `T_sem` 图或训练 cache。
