# 文档真源与同步规则

**日期**：2026-10-23  
**目的**：避免本地与 101 服务器各有一份文档、各自被修改后分叉。

## 唯一真源

本地 git 仓库：

```text
/home/bubble/类脑计算/VLM终局
```

它是唯一可编辑的文档真源。101 上的：

```text
/vepfs-mlp2/c20250405/400040/transfer/vla_attention/repo/VLA_attention
```

是部署快照，不是 git 仓库；**只读，不就地编辑**。

## 权威文档位置

- `doc/03_execution_plan.md`：主执行计划与 J-gate 决策冻结；
- `doc/vla_next/`：下一阶段的技术栈、决策、不确定性、part-level 数据计划、J0 runbook；
- `experiment_workspace/PROGRESS.md`：运行历史与 gate 记录；
- `experiments/*/protocol.md`：各实验协议；
- `paper_draft/`：论文稿。

服务器上同路径文件只作为部署副本，内容必须由本地同步。

## 工作流

1. 只在本地编辑文档；
2. `git diff --check` 后检查差异；
3. 需要时 commit；
4. 运行 `DRY_RUN=1 server/sync_docs_to_101.sh` 预览，再运行 `server/sync_docs_to_101.sh` 同步；
5. 在 101 上核对 `md5sum` 或 SHA256；
6. 代码仍走 `server/upload_vla_bundle.sh`。

同步范围包括 `doc/`、`experiments/`、`paper_draft/`、`references/`（排除第三方克隆目录 `references/repos/`）、`server/`、`tests/` 下的 Markdown，以及根目录 `README.md`、`experiment_workspace/PROGRESS.md`、`configs/vla/README.md`。同步按 Markdown 镜像执行，不会删除各目录下的代码、结果、数据或日志。

## 禁止

- 不在 101 上直接编辑 `.md`；
- 不用 101 的旧快照覆盖本地；
- 若发现 101 有本地没有的内容，先拉回本地合并，再同步回去；
- 不把 `results/`、`wandb/`、checkpoint、数据或日志当作文档同步。

## 2026-10-23 首次对账

- 101 的 `doc/03_execution_plan.md`（159,305 B）比本地新，含 2026-10-20 J-gate 决策冻结；已拉回本地；
- `doc/04_P4_formal_v0_v4_results.md` 本地缺失，已从 101 拉回；
- 101 的 `experiment_workspace/PROGRESS.md` 中 2026-10-19/20 条目已合并进本地完整历史；
- 本地 `README.md` 的 `doc/vla_next/README.md` 链接保留（本地更新）；
- 对账后两端文档由本地单向同步。

## 2026-10-23 全量 Markdown 对账

- 101 共 69 个 Markdown，本地共 315 个；57 个内容相同，247 个仅本地存在，属于后续新增文档和 `references/repos/` 第三方镜像。
- 11 个同名文件内容不同：`experiments/P2-teacher-calibration/protocol.md`、`experiments/P4-vlm-v0-v4/architecture.md`、`experiments/P4-vlm-v0-v4/protocol.md`、`experiments/P6-vla-b0-b4/protocol.md`、`experiments/P7-vla-lite/protocol.md`、`paper_draft/03_related_work.md`、`paper_draft/04_system_and_method.md`、`paper_draft/05_experiments_and_expected_conclusions.md`、`paper_draft/08_model_baseline_benchmark_options.md`、`references/attribution_literature/README.md`、`server/01_vla_attention_training.md`。
- 上述 11 个文件均为 101 旧快照；远端相对本地的新增内容只有已过期的状态行或章节标题，没有需要回流的独有信息，因此保留本地版本。
- 101 独有的 `tests/protocol.md` 是本地 `experiments/P7-vla-lite/protocol.md` 的旧副本，已完成内容归档到 `doc/archive/tests_protocol_remote_20261014.md`；归档副本不再参与正式协议维护。
- 对账后由更新后的 `server/sync_docs_to_101.sh` 单向覆盖 101，并删除 101 上已被本地归档取代的旧 Markdown。
