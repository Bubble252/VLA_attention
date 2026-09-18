# 训练前准备进度

本记录与 `doc/03_execution_plan.md` 的 P1 章节保持一致；这里保留实际运行环境和产物路径，避免把服务器状态混入论文叙事。

## 运行中检查点格式

每次处理量达到 10/25/50/75/100%，以及每次 smoke、checkpoint、失败或配置变更后，新增如下条目：

```text
时间 / experiment ID / Git commit / 101 PID 或 job ID
输入 manifest + SHA256 / total-success-failed
model + teacher + revision + seed + steps
partial metrics / VEPFS result + log 路径
是否 non-final / 下一步 / 是否允许扩大
```

所有 partial 数字必须显式标 `non-final`；处理进度、失败集合和有效样本交集变化时，同步更新对应 protocol。

## 2026-09-17

- [x] 101 直连：`root@115.190.90.101:27219` 可进行无交互 SSH；连接凭据不写入本文件。
- [x] 本机稳定入口：`vla101` SSH alias 固定地址/端口并启用严格 host-key 校验；项目受限操作由 `server/vla101.sh` 执行。
- [x] VEPFS 项目根已建立：`/vepfs-mlp2/c20250405/400040/transfer/vla_attention/`。
- [x] 建立空目录：`repo`、`envs`、`hf_cache`、`models`、`data`、`teacher_maps`、`runs`、`checkpoints`、`results`、`jobs`。
- [x] 存储检查：VEPFS 有约 620 TB 可用；101 `/root` 仅约 207 MB 可用，因此禁止将 cache、虚拟环境、模型或数据写到 `/root`。
- [x] 依赖基线：`/root/starvla_cu124/bin/python` 可发现 torch、transformers、huggingface_hub、datasets、diffusers、accelerate、peft。
- [x] 在 VEPFS 创建 `envs/p1` 并生成 `envs/p1/versions.txt`；环境只读继承共享包，不修改 `/root/starvla_cu124`。
- [x] Qwen checkpoint 配置审计：`Qwen2_5_VLForConditionalGeneration`，vision depth 32、`fullatt_block_indexes=[7,15,23,31]`、patch size 14、`spatial_merge_size=2`、window size 112；运行环境为 Transformers 4.57.1，`qwen_vl_utils` 已存在。P1 必须记录 post-merge token 网格，不可仅据 token 数猜方格。
- [x] DINOv2 ViT-L/14 retention teacher：`facebook/dinov2-large@47b73eefe95e8d44ec3623f8890bd894b6ea2d6c` 已下载、SHA256 通过（约 2.3GB；Apache-2.0、非 gated）。GPU1 单图 smoke 成功：224×224、`last_hidden_state=(1,257,1024)`、patch size 14；`use_fast=False` 将在正式 runner 固定。DINO 16×16 patch 与 Qwen 动态 post-merge grid 通过 normalized-coordinate bridge，不直接按 token index 对齐。
- [ ] 将当前 Git commit 传到 VEPFS `repo/VLA_attention`，记录 commit 与文件 SHA256。
- [ ] 在 VEPFS 创建 `envs/p1`；只读继承共享依赖或按锁定 requirements 安装，绝不修改 `/root/starvla_cu124`。
- [ ] 记录网络和代理检测；下载使用 7897 代理，若服务器无该代理则先记录失败后再选已验证镜像。
- [ ] 下载并核验 Qwen2.5-VL-7B、Flickr30k Images、Flickr30k Entities。2026-09-17 Qwen 已下载到 VEPFS（16 个文件、约 16 GB；`Qwen2_5_VLForConditionalGeneration`，checkpoint 声明 Transformers 4.41.2）。下载使用 `hf-mirror.com`，原因是 101 对 Hugging Face 直连超时；`SOURCE.txt` 已记录模型 repository 与实际端点。首次 SHA256 自校验发现清单把自身纳入输入，产生一个预期的自引用 mismatch；脚本已修复为排除 `SHA256SUMS`，待重建清单后才将 Qwen 勾为已核验。
- [ ] 审计 `nlphuji/flickr30k` 的实际文件与 revision，并探测 Flickr30k Entities 的原始 annotation archive；必须确认 image 与 phrase-region 标注可对齐，才建立全量 manifest。
- [x] 数据源审计：`nlphuji/flickr30k` 提供完整 image zip 和 caption CSV；原作者 `BryanPlummer/flickr30k_entities` 提供 `annotations.zip`（约 29 MB）。原作者 README 明确该 archive 含 `Sentences/` 的 phrase/coreference chain 和 `Annotations/` 的 XML boxes，且含 train/val/test split；图片遵循 Flickr Terms，仅非商业研究/教学使用。旧 Plummer 网页与 Oxford VGG archive URL 均为 404，故不使用它们。
- [ ] 运行 `download-flickr`，验 archive、解压、SHA256 与 JPEG/Sentences/XML 文件计数；通过前不建立 split 或运行 P1。
- [x] Flickr 图片与 caption 下载：完整图片 archive 和 `flickr_annotations_30k.csv` 已从 `hf-mirror.com` 下载到 VEPFS。原作者 annotation 的 `raw.githubusercontent.com` 直传在 120 秒内仅约 1 MB，随后 shallow Git clone 约 15 分钟仅到 16 MB，均不适合作为可靠传输。GitHub API blob 的 JSON 传输也在约 458KB 中断；已验证 API raw media 支持该 exact blob 的 ZIP Range 读取（返回 `PK` magic），故改用 4MB、字节校验、可重试的 versioned range 下载。最终 `SOURCE.txt` 会记录 transport、commit、blob SHA 与 SHA256；不是不明第三方镜像。
- [ ] Entities archive 当前为**部分下载**：第一个 4MB range 已通过长度检查；第二段在 HTTP/2 `CANCEL` 后未通过，临时段未被追加。下载器改为从已验证 archive 大小续传，使用 1MB + HTTP/1.1 + 每段字节检查 + 最多五次重试。`SOURCE.txt` 在 archive 完整、`unzip -t` 通过前不得作为完整来源证据。
- [x] 代理探测：`ghproxy.net` 与 `gh-proxy.com` 均能返回原作者精确 revision 的 `annotations.zip` ZIP magic；`mirror.ghproxy.com` 超时。为避免代理替换内容，下载器会从 `ghproxy.net` 取得 candidate 文件，并计算 Git `blob <size>\\0<payload>` SHA1，只有等于官方 API 的 blob `95513f3315f98068d0aeb558649ebe09fbeffe95` 且 `unzip -t` 通过时，才替换部分 archive。
- [x] 镜像传输修复：发现 `curl --retry` 会在连接重置后覆盖 candidate；已停止唯一受影响的本项目任务，保留当时约 1.18MB candidate，并改为 `--continue-at -`。后续重试从已有字节续传；最终仍以官方 Git blob SHA 和 ZIP test 为唯一准入条件。
- [ ] 建立全量 manifest，随后从 calibration split 固定 P1 的 20 个 sample ID。
- [x] Manifest 规则冻结：原作者 `train.txt` 完整用于训练、`val.txt` 用于独立 teacher calibration、`test.txt` 用于最终 grounding；`build_flickr_entities_manifest.py` 将从 val 确定性抽取 20 张 P1 图并为其保存 image SHA256。该规则不从 train 偷取 calibration 样本，也不让 P1 缩小正式实验规模。
- [x] Qwen P1 report：2026-09-17 在 GPU1 对固定 20 个 val 样本运行真实 teacher-forced phrase-score gradient×activation 审计；`validate_p1_report.py` 通过，`PASS qwen2.5-vl-7b@Qwen2.5-VL-7B-Instruct: 20 measurements / 20 samples`。report/map 位于 VEPFS `results/P1-interface-audit/qwen2.5-vl-7b/`；此通过只允许进入 DINO/teacher calibration，不等于 V0--V4 训练已完成。
- [x] Qwen P1 前向 smoke（GPU1）：冻结 P1 manifest 首样本 `1321949151:c0:p0:e8556` 成功加载图像与 Qwen，`image_grid_thw=[[1,34,36]]`，logits shape `(1,338,152064)`，返回 29 个 hidden-state 层。Transformers 提示 fast processor 默认行为会改变输出；首轮 P1/V0--V4 将显式固定 `AutoProcessor(..., use_fast=False)` 并记录版本，避免 processor 漂移。
- [x] Qwen LoRA PEFT runtime audit：从 `qwen_lora_v1.json` 读取完整路径正则后成功匹配 language model 第 0--27 层，共 392 个可训练 LoRA parameter tensors，`visual_lora_tensors=0`。命令行手写 regex 出现双重转义会导致零模块匹配；正式 runner 必须从 JSON decode 后直接传入 PEFT，并在启动时断言 visual trainable count 为 0。
- [x] Prismatic 来源审计：官方 `TRI-ML/prismatic-vlms` README 将 `prism-dinosiglip+7b` 作为空间理解/定位首选；代码为 MIT，但该 checkpoint 继承 Llama-2 许可。101 的 `hf auth whoami` 返回未登录，因此不得下载该 gated checkpoint。待合法 HF token 登录并确认已接受 Llama-2 条款后，再下载并运行 Prismatic P1。
- [x] T_sem 候选冻结前 registry：Stable Diffusion v1.5、PixArt-alpha、PixArt-Sigma、Playground-v2.5 均经 `hf-mirror.com` metadata 查询为非 gated；准确 repo/revision/license 已写入 `configs/teachers/semantic_teacher_candidates.json`。这只是下载与校准候选列表，不能替代在独立 val split 上按 pointing/IoU、无效词率和跨 seed 一致性选 best-single。
- [x] SD1.5 attention hook smoke：新版 Diffusers 0.38 对真实 calibration 图像 latent 与 phrase `a blue hard hat` 成功捕获 cross-attention，分辨率/层计数为 64:5、32:5、16:5、8:1。FlashAttention 导入不兼容，已回退 PyTorch attention；该运行环境行为需记录。该 smoke 明确为 `formal_calibration_eligible=false`，因为尚未执行 DDIM/null-text inversion，不能用于 T_sem 选择或 V3/V4 cache。
- [x] SD1.5 DDIM image-conditioned map smoke：完整 caption `A man in a blue hard hat ...` 内目标 phrase 成功定位到 CLIP token positions `[4,5,6,7]`；5-step conditional DDIM inversion 生成 25 个 16×16 cross-attention tensors 和归一化 map。metadata 固定为 `method=ddim_conditional_no_nulltext`、`formal_calibration_eligible=false`；下一 gate 是 per-timestep null-text optimization 与原始 Lavender legacy baseline 对照。
- [ ] null-text runner：本地实现与 15 个测试已提交（`ad46d0a`）；待在 101 VEPFS 同步后依次执行 `5×1` 接口 smoke、`20×10` 单图正式验收。尚未运行，不能称为 Lavender-equivalent map 或用于候选教师排名。
- [x] null-text runner GPU1 验收：FP16 `5×1` 出现 NaN，已保留为失败证据并改用新增 FP32-v2 runner；FP32 `5×1` 通过（mean MSE `0.09803`），FP32 `20×10` 正式单图通过（mean MSE `0.03884`、100 个 16×16 attention tensors、目标 span `[4,5,6,7]`）。SD1.5 可以进入 10 图 calibration pilot；该单图通过不等于四教师 best-single 已选择。
- [x] SD1.5 pilot10：固定 calibration manifest 前十图全部成功（10/10），`20×10`、seed 17、FP32 null-text。唯一 summary `summary_sd1_5_20x10_seed17.json`：pointing `0.50`、mean mass-in-box `0.4754`、mean MSE `0.03207`（min `0.01317` / max `0.06615`）。这证明 SD pipeline/map/metric 可运行；样本太小且尚未与 PixArt/Playground 相比，严禁据此选 best-single 或生成 V3/V4 train cache。
- [x] PixArt-alpha runtime audit：P1 环境补充 `sentencepiece==0.2.2` 与 `tiktoken==0.14.0` 后成功加载本地 pipeline；T5Tokenizer、DPMSolverMultistepScheduler、56 个 transformer attention processors。初始化告警仅为 unused `caption_projection.y_embedding`；PixArt 需独立 transformer-token attention adapter，不能复用 SD UNet/null-text runner。
- [ ] PixArt-alpha phrase-map adapter：pipeline 与 tokenizer已加载，但 Diffusers 0.38 的 PixArt transformer 内部模块路径和预期名称不一致；当前只确认 processor count，尚未捕获真实 image-conditioned token attention。此兼容性 gate 不影响 SD1.5 pilot，也禁止将 SD map 用作 PixArt map。
- [ ] V1 caption manifest/GPU smoke：101 上已确认 builder、runner、LoRA config 存在且 `caption_sft_train.jsonl` 初始缺失；构建命令随后出现 SSH 无输出异常，连只读 `echo`/process 查询亦未回显。未假定 manifest 已完成，未启动 V1 smoke，避免重复创建或训练；恢复可靠远端观察后先检查文件 SHA256/行数与是否有活跃 builder。
- [ ] 快速主线 10k：全量 caption manifest 后续已确认有 145,355 条（SHA256 `deb2d4…`），10k seed17 子集与 metadata 已确认存在。V1 20-step GPU1 smoke 已提交到唯一新输出路径，但会话句柄和后续 GPU/process 查询无输出，状态必须回收结果 JSON 后才能判定；禁止重启第二个 V1 smoke。
- [ ] F0-256：从 10k seed17 manifest 抽取固定 256 pair 的构建已提交至此前不存在的目标路径；101 未回传输出且随后的最小存在性检查无输出，尚不能确认 F0 manifest 是否写成。禁止重抽样、生成 F0 SD cache 或启动 F0 V1--V4，直到可读取唯一文件的 SHA256。
- [x] F0v2 shared manifest：发现 caption-only F0 不含目标 phrase，停止了尚未产出 map 的错误全句 cache job。改从 Entities train 生成 256 条 image-caption-phrase-box 对齐记录，并显式补固定 caption prompt；最终训练输入 `F0_aligned_caption_phrase_256_seed19_v2.jsonl`，SHA256 `2a37c49…`。F0v2 V1/V2 20-step smoke 均通过；V2 DINO retention 从约 0.995 降至 0.862，visual LoRA=0、DINO frozen。
- [ ] F0v2 SD cache：已提交后台 job PID `1554962`，路径 `teacher_maps/F0v2_sd1_5_nulltext_phrase_256_seed17`；每条使用完整 caption 条件、同一行 Entities phrase token target、20×10 FP32 null-text、seed17、sample_id 唯一键。完成后须核对 256 map/metadata、failures 和有效交集，才可开始 V3/V4。
- [x] F0v2 V1--V4 smoke：正确对齐 manifest 和 SD phrase cache（256 maps、`failures=[]`）下，V1/V2/V3/V4 均完成 20-step，所有记录的 loss 有限。V3 使用 eager attention 以支持二阶 phrase-score attribution；visual encoder parameter 仅为 attribution 计算开启 grad，不在 optimizer，visual LoRA=0。`F0v2_status_nonfinal.json` 是工程 gate 汇总，不是论文结果。剩余 F0：checkpoint restore、V0 eval、三种负控与状态表。 
- [x] F0v2 idea-validation checkpoint gate 开始：V1 LoRA adapter 已保存到 `checkpoints/F0v2_idea_validation/V1`（约 161MB），V2 LoRA adapter + `dino_projector.pt` 已保存到 `.../V2`；held-out manifest 已生成 64 条、SHA256 `7b16778e…`。下一步保存 V3/V4 checkpoint 后评测 64 条 held-out 和三类负控。
- [x] held-out evaluator 已实现：`scripts/eval_qwen_heldout.py` 固定 teacher-forced phrase-score gradient×activation，输出 pointing、mass-in-box、top20 attribution box IoU；`scripts/run_f0_heldout_all.sh` 冻结 V0--V4 同一 manifest 批量入口。V3/V4 checkpoint 导出与视觉参数排除修正提交 `339bbb3`，评估器提交 `cf7a7d4`，批量入口提交 `9426f19`。
- [x] 教师负控脚本已实现：`scripts/eval_teacher_map_controls.py` 对 held-out SD phrase cache 计算 correct、wrong-image、shifted、random，commit `0fb77a1`。尚未执行，因为 101 SSH 当前不可用，且 64 条 held-out teacher cache 尚未确认存在。
- [x] 评估器修正：PEFT inference 会冻结 Qwen base；`eval_qwen_heldout.py` 现在只重新打开 visual path 的 `requires_grad` 以取得 phrase-score attribution，仍不创建 optimizer、不更新权重，commit `d756212`。
- [x] 上传工具已加入：`server/upload_validation_scripts.sh` 会从用户本机上传 5 个验证脚本并在 101 端逐个 SHA256 校验，成功标志为 `UPLOAD_VERIFY_OK`。由于 Codex 执行沙盒的 socket 限制，该脚本应在用户自己的 SSH 可用终端运行。
- [x] 用户已确认本机 `ssh vla101 'echo SSH_OK'` 成功；下一步入口为 `server/run_f0_checkpoint_smokes.sh`，顺序生成 V3/V4 checkpoint，拒绝覆盖旧产物，成功标志为 `CHECKPOINT_SMOKES_OK`。
- [ ] 2026-09-18 checkpoint smoke 首次启动在 V3 forward 阶段因 GPU0 显存耗尽失败：PID `291422` 占用约 59.86 GiB、PID `2093781` 占用约 19.27 GiB，仅剩约 5.5 MiB；未生成有效 V3 checkpoint。未杀进程，避免影响其他任务。启动器已改为显式传递 `CUDA_VISIBLE_DEVICES`，默认改用 GPU1；重试前需由 `nvidia-smi` 确认 GPU1 空闲或选择明确空闲卡。
- [x] 2026-09-18 用户在 GPU1 成功完成 V3/V4 checkpoint smoke：两个结果 JSON、V3/V4 adapter 均生成，V4 `dino_projector.pt` 存在，远端输出 `CHECKPOINT_SMOKES_OK`。下一步为同一 64 条 image-disjoint held-out 的 V0--V4 attribution evaluation。
- [x] held-out 远端入口已加入：`server/run_f0_heldout_remote.sh` 在 101 P1 环境运行五组评估，拒绝覆盖已有报告，成功标志 `HELDOUT_MODELS_OK`。
- [x] 2026-09-18 15:20（UTC+8）V0--V4 held-out attribution 已完成 64/64，远端输出 `HELDOUT_MODELS_OK`。初步结果（20-step checkpoint，non-final）：V0 pointing/mass/IoU=`0.2813/0.3095/0.2693`；V1=`0.3594/0.3183/0.2692`；V2=`0.3594/0.3313/0.2696`；V3=`0.2969/0.3197/0.2683`；V4=`0.3594/0.3210/0.2682`。因此当前短 smoke 未显示 `V3>V1` 或 `V4>V2` 的一致提升；不能据此否定 idea，需先完成 teacher 负控并明确这是工程 gate 还是训练不足。
- [x] held-out evaluator 修复记录：远端缺少 `evaluation/spatial.py`（已同步并通过 `SPATIAL_IMPORT_OK`）；PEFT wrapper 层级差异与 phrase BPE 直接匹配问题已修复（本地 commits `6129934`、`4dc2e14`）。
- [ ] 2026-09-18 SD held-out cache 首次尝试因工作目录未设置导致 64 条 `ModuleNotFoundError: scripts`，未生成有效 map；已切换到 repo 工作目录并设置 `PYTHONPATH` 重跑，当前第 5/64 条已成功，使用 FP32 null-text `20x10`、seed23。
- [x] 2026-09-18 held-out SD cache 已完成 64/64，`failures=0`；teacher controls 已运行：correct pointing/mass/IoU=`0.6563/0.4088/0.3253`，wrong-image=`0.2969/0.3234/0.2701`，shifted=`0.3750/0.3567/0.3255`，random=`0.2969/0.3033/0.2676`。correct 对 pointing/mass 和 wrong-image/random 的三项均明显更好；shifted 的 IoU 与 correct 几乎持平，故 IoU 单项负控不完全通过。
- [ ] 20-step held-out 已完成但只是 smoke：V3/V1、V4/V2 没有一致提升；已增加 `server/run_f0_100step_train.sh` 进行公平 100-step 学习曲线，避免把短 smoke 误判为最终科研结论。
- [x] 2026-09-18 100-step V1--V4 训练已完成，远端输出 `LONG_TRAIN_100_OK`；四个 adapter checkpoint 和 JSON 均存在，下一步在同一 64 条 held-out 上评估，入口 `server/run_f0_100step_heldout.sh`。
- [x] held-out 只读汇总入口已加入：`server/summarize_f0_heldout.sh` 检查 V0--V4 报告、打印三项指标和 V3/V1、V4/V2 初步趋势；`MODEL_HELDOUT_READY` 不等价于最终 idea 通过，仍需 teacher 负控。
- [x] 一键 F0 入口已加入：`server/run_f0_full_validation_remote.sh` 复用或生成 V0--V4 held-out 报告、64 条 SD teacher cache，并运行四类 teacher map controls；拒绝覆盖 checkpoint/report，最终标志 `F0_FULL_VALIDATION_OK`。尚待用户终端执行。
- [ ] 2026-09-18 当前执行窗口无法建立 `vla101` SSH socket（`Operation not permitted`）；未假定远端状态、未重复启动任务。待连接恢复后按 protocol 先同步 runner，再补 V3/V4 checkpoint 和 64 条 held-out 指标。此为环境阻塞，不是实验失败。

## 下一条可接受的证据

`results/P1-interface-audit/qwen2.5-vl-7b/report.json` 应由 native adapter 生成，并通过：

```bash
PYTHONPATH=src python scripts/validate_p1_report.py \
  <VEPFS>/results/P1-interface-audit/qwen2.5-vl-7b/report.json
```

它必须覆盖 20 个固定样本，保存动态 token grid、phrase target 标量、梯度有限性、seed、图路径及模型 revision。只有该文件通过，才把 Qwen 记为 P1 通过。
