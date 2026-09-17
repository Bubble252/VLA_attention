# P2：扩散语义教师校准 protocol

**状态**：P1 已通过；候选模型已下载；尚未选择 best-single。  
**目的**：从固定四个 `T_sem` 候选中选择一个对真实图像上的 phrase-region map 最可靠的教师；不是比较图像生成质量。

## 输入与固定性

- 校准数据：Flickr30k Entities 的完整官方 `val.txt`，1000 张图；
- 每图恰选一个有非空 box 的 phrase，按 `caption_index → phrase.lower() → sample_id` 的确定性字典序，不根据模型输出筛样本；teacher prompt 是同一行去除 Entities 标记后的**完整原始 caption**，目标是其中该 phrase 的 token span；
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

同日 DDIM smoke 已在完整 caption 中定位目标 phrase 的 CLIP token span `[4,5,6,7]`，以 5-step conditional DDIM inversion 聚合 25 个 16×16 attention tensors，说明 image-conditioned trajectory 与 token-span bridge 可运行。此结果仍是 `ddim_conditional_no_nulltext`：它可作为实现诊断，不可替代 Lavender 的 null-text inversion baseline，也不能用于候选排名。

### null-text runner 的正式单图验收

`scripts/run_sd_nulltext_map.py` 已在本地单测通过，待同步 101 后先运行 `steps=5, inner_steps=1` 的接口 smoke，再运行 `steps=20, inner_steps=10` 的单图正式配置。后者必须同时满足：

- [ ] DDIM trajectory、20 个 optimized unconditional embeddings、20 个 forward reconstruction steps 均完成；
- [ ] 每个 timestep 的 MSE 都有限，保存 mean MSE 和逐 timestep loss；
- [ ] 目标 phrase 能在完整 caption 的 CLIP tokenization 中找到 span；
- [ ] 指定空间 resolution 有非空二维 map，map 有限、非负、归一化；
- [ ] 记录 Diffusers/PyTorch attention fallback、seed、steps、inner steps、CFG、model revision；
- [ ] map 与对应 image/phrase/GT box 的 record 写入 calibration output。

只有该单图正式配置通过，才能扩大到 1000 图 SD1.5 calibration；PixArt/Playground 的 adapter 也必须各自满足等价的“真实图像条件 + token span + map”验收，不能复用 SD hook 名称或假定其 attention 张量一致。

**单图验收结果（2026-09-17）**：FP16 `5×1` null-text 的 mean MSE 为 NaN，判失败；改为 FP32 后 `5×1` MSE `0.09803`，`20×10` MSE `0.03884`，并捕获 100 个 16×16 conditional attention tensors。SD1.5 升级到 10 图 pilot；pilot 必须记录逐样本失败、map 指标、耗时和 GPU memory，不能直接跳到 1000 图。
