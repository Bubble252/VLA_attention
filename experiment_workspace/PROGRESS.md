# 训练前准备进度

本记录与 `doc/03_execution_plan.md` 的 P1 章节保持一致；这里保留实际运行环境和产物路径，避免把服务器状态混入论文叙事。

## 2026-09-17

- [x] 101 直连：`root@115.190.90.101:27219` 可进行无交互 SSH；连接凭据不写入本文件。
- [x] 本机稳定入口：`vla101` SSH alias 固定地址/端口并启用严格 host-key 校验；项目受限操作由 `server/vla101.sh` 执行。
- [x] VEPFS 项目根已建立：`/vepfs-mlp2/c20250405/400040/transfer/vla_attention/`。
- [x] 建立空目录：`repo`、`envs`、`hf_cache`、`models`、`data`、`teacher_maps`、`runs`、`checkpoints`、`results`、`jobs`。
- [x] 存储检查：VEPFS 有约 620 TB 可用；101 `/root` 仅约 207 MB 可用，因此禁止将 cache、虚拟环境、模型或数据写到 `/root`。
- [x] 依赖基线：`/root/starvla_cu124/bin/python` 可发现 torch、transformers、huggingface_hub、datasets、diffusers、accelerate、peft。
- [x] 在 VEPFS 创建 `envs/p1` 并生成 `envs/p1/versions.txt`；环境只读继承共享包，不修改 `/root/starvla_cu124`。
- [x] Qwen checkpoint 配置审计：`Qwen2_5_VLForConditionalGeneration`，vision depth 32、`fullatt_block_indexes=[7,15,23,31]`、patch size 14、`spatial_merge_size=2`、window size 112；运行环境为 Transformers 4.57.1，`qwen_vl_utils` 已存在。P1 必须记录 post-merge token 网格，不可仅据 token 数猜方格。
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
- [ ] 建立全量 manifest，随后从 calibration split 固定 P1 的 20 个 sample ID。
- [x] Manifest 规则冻结：原作者 `train.txt` 完整用于训练、`val.txt` 用于独立 teacher calibration、`test.txt` 用于最终 grounding；`build_flickr_entities_manifest.py` 将从 val 确定性抽取 20 张 P1 图并为其保存 image SHA256。该规则不从 train 偷取 calibration 样本，也不让 P1 缩小正式实验规模。
- [ ] 输出 Qwen P1 报告；通过后才下载/审计 Prismatic。
- [x] Prismatic 来源审计：官方 `TRI-ML/prismatic-vlms` README 将 `prism-dinosiglip+7b` 作为空间理解/定位首选；代码为 MIT，但该 checkpoint 继承 Llama-2 许可。101 的 `hf auth whoami` 返回未登录，因此不得下载该 gated checkpoint。待合法 HF token 登录并确认已接受 Llama-2 条款后，再下载并运行 Prismatic P1。

## 下一条可接受的证据

`results/P1-interface-audit/qwen2.5-vl-7b/report.json` 应由 native adapter 生成，并通过：

```bash
PYTHONPATH=src python scripts/validate_p1_report.py \
  <VEPFS>/results/P1-interface-audit/qwen2.5-vl-7b/report.json
```

它必须覆盖 20 个固定样本，保存动态 token grid、phrase target 标量、梯度有限性、seed、图路径及模型 revision。只有该文件通过，才把 Qwen 记为 P1 通过。
