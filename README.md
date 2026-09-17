# VLM 终局：语言条件空间归因对齐

本仓库记录最终研究方案、文献归档和后续实现入口。核心方法不限定于某一种 VLM 的 attention 结构，而是把不同 VLM/VLA 的内部视觉证据统一成**语言条件空间归因图**：

> Lavender / Stable Diffusion 提供 `word → image region` 教师图；Prismatic-7B、Qwen2.5-VL-7B、InternVL3.5、Ovis2.5、LLaVA、OpenVLA、π0、π0.5、MolmoAct2 等候选通过各自可用的 hidden-state gradient、attention 或 action-conditioned attribution 产生学生归因图；二者统一到图像坐标后对齐。

实验顺序是 VLM grounding → LIBERO 闭环及 OOD → DROID 离线泛化。主线研究语言—动作空间归因约束，VLA 候选包括 OpenVLA/OFT、π0、π0.5、MolmoAct2 和 LingBot-VLA，具体组合待用户筛选。VLM 候选包括 Prismatic-7B、Qwen2.5-VL-7B、InternVL3.5、Ovis2.5 和 LLaVA。SpikingBrain 已确定后置，有空再做，不作为主线或验收依赖。

模型角色按实验区分：π0、LingBot 等可以作为待选学生骨干；若作为 B 路线外部教师，则只用于相关性细化。学生选择与教师选择分开。

52 份 PDF 的逐篇摘要、重合与筛选建议见 [文献归纳总览](references/abstract_review/README.md)。推荐不等于冻结配置，所有效果均须实验验证。

当前更倾向的 VLA 主方法是 **Structure-Native Action Attribution Consistency**：先用扩散图锚定模型内部语言归因 `A_lang`，再约束动作 query/action head 的归因 `A_act` 不偏离语言相关区域。Action-Relevance Refiner 和 LIBERO 状态投影保留为扩展与诊断，不作为主创新。

先读：

1. [`doc/01_project_background.md`](doc/01_project_background.md)
2. [`doc/02_technical_stack.md`](doc/02_technical_stack.md)
3. [`doc/03_execution_plan.md`](doc/03_execution_plan.md)
4. [`references/README.md`](references/README.md)
5. [`references/download_manifest.md`](references/download_manifest.md)

仓库不提交模型权重、LIBERO 数据集、运行日志和大规模输出。GPU 由服务器环境提供；本项目只固定软件、数据、模型版本和实验配置。

论文和官方代码仓库的下载命令为：

```bash
cd /home/bubble/类脑计算/VLM终局
bash scripts/download_references.sh
```

脚本强制使用 `http://127.0.0.1:7897`，并生成下载失败日志、论文 SHA256 和仓库 commit 清单。

下载后运行下面的校验命令：

```bash
python3 scripts/verify_references.py
```

## 在新电脑恢复工作区

GitHub 仓库只保存文档、代码、配置、引用元数据和阅读归纳；不保存 PDF、参考代码快照、模型权重、LIBERO/DROID 数据、实验输出或飞书凭据。

### 1. 恢复 Git 文档与计划

```bash
git clone https://github.com/Bubble252/VLA_attention.git
cd VLA_attention
git status
git log -1 --oneline
```

日后在另一台电脑继续时：

```bash
git pull --ff-only
```

### 2. 恢复参考资料

先运行已有归档脚本；它只恢复脚本中登记的资料：

```bash
bash scripts/download_references.sh
python3 scripts/verify_references.py
```

新近候选的 URL、角色和期望 commit 记录在 `references/repositories.yaml`，例如 BlindVLA、PixArt、InternVL、Ovis。它们若尚未由下载脚本覆盖，应按 YAML 中的官方 URL 单独 clone，并记录实际 commit：

```bash
git -c http.proxy=http://127.0.0.1:7897 clone --depth 1 \
  https://github.com/CognitiveAISystems/BlindVLA.git references/repos/blindvla
```

本地 PDF 不随 Git 恢复；论文归纳仍可从 `references/abstract_review/` 阅读。若需要重建 PDF 页码证据，先补回 `references/papers/` 中的 PDF，再运行：

```bash
/usr/bin/python3.10 scripts/index_pdf_abstracts.py
/usr/bin/python3.10 scripts/finalize_pdf_review_index.py
```

### 3. 恢复实验环境、权重和数据

模型权重与数据必须按最终冻结的模型组合重新下载。先审计动作接口、许可、checkpoint 版本和数据字段，再下载 OpenVLA/OFT、π0、π0.5、MolmoAct2、VLM 候选、LIBERO 或 DROID。不要将权重、数据或运行输出 `git add`。

### 4. 可选恢复飞书同步

飞书凭据不在本仓库。新电脑需要单独取得 `doc-sync-main`，创建本地且被忽略的 `sync_config.json`，填入自己的 App ID、Secret、User Access Token 和文件夹 Token；不要从 Git 恢复或提交该文件。完成后运行：

```bash
cd /path/to/doc-sync-main
/usr/bin/python3.10 main.py sync --force
```

同步前先检查云端是否存在人工编辑，避免 `--force` 覆盖它们。详细同步状态见 `references/abstract_review/06_delivery_status.md`。

## 飞书实时同步

`doc/` 下的三个核心 Markdown 已同步到飞书。后续修改文件后，需保持 DocSync 实时同步进程运行：

```bash
cd /home/bubble/类脑计算/doc-sync-main
/usr/bin/python3.10 main.py live --config sync_config.json --poll-interval 3
```

该进程会监听本地文档变化并同步到目标飞书文件夹，同时轮询云端修改。凭据保存在 `doc-sync-main/sync_config.json`，不会提交到本仓库。
