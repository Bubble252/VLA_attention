# VLA instruction–evidence 最终选题审计状态

日期：2026-10-16。范围：文献、源码、数据结构、任务可行性和实验识别；不启动 VLA/WAM/Copper 训练，不修改既有数据、cache、checkpoint 或 manifest。

## 最终交付 gate

- [x] 近邻创新性审计：scoop-check 七步完成，7 篇原文核验，Level 2 verdict 与 reviewer delta
- [x] LIBERO 数据可构造性：130 task/BDDL/init/demo/action provenance 完成，结论为 Diagnostic-only intervention
- [x] 方法识别审计：因子协议、替代解释、matched budget 和 falsification gate 完成
- [x] idea-evaluator 最终评审：冻结 **B — diagnostic/mechanistic paper**
- [x] 只在 `experiment_workspace/PROGRESS.md` 追加一条索引与结论
- [x] Git 仅提交本目录轻量产物及索引；PDF/全文缓存忽略
- [x] 飞书文献目录同步并核验

## 已冻结边界

- 新候选：在相同或严格匹配的场景与状态下，联合改变语言目标和对应目标视觉证据，检验/训练正确动作重定向，即 `Δinstruction × Δtarget evidence → Δaction`。
- 语言图只标识 source/target；策略仍读取完整视觉、夹爪、接触与 proprioception。
- 旧 action-distance triplet 候选已经是 Level 2 — High Overlap，只作 baseline/诊断。
- 禁止“原图 + 错指令 + 原动作”作为新目标的正确动作标签；遮挡不自动等于因果干预。
- 101 上正式 VLM 作业保持独立；本审计不停止、不重启、不占用 GPU。

## Scoop-check 七步任务

- [x] Step 1：候选四轴拆解并冻结
- [x] Step 2：三组 paper-search、多源错误和去重记录
- [x] Step 3：35 条去重记录摘要筛选
- [x] Step 4：7 篇高风险候选选择
- [x] Step 5：候选 introduction/method/experiments/limitations 原文核验
- [x] Step 6：四轴逐论文比较与 level
- [x] Step 7：总 verdict 与 reviewer delta

## 运行记录

- 2026-10-16：读取 goal 附件、scoop-check、paper-search、idea-evaluator 及其四份 reference rubric；读取既有 novelty audit、三近邻精读、VLA 颗粒度、P6 protocol、论文 method/experiments 与 PROGRESS。
- 2026-10-16：本地确认 LIBERO 源码含 130 个 BDDL；LIBERO-90 中存在同 scene 的对象/容器/位置变化任务族。远端只读确认 RLDS inventory 与 rollout init inventory 存在；尚未据此断言任务间 init state 或 action label 可配对。
- 2026-10-16：paper-search 首轮使用 7897 代理时六源 TLS EOF，保留失败 JSON/log；直连重试得到 arXiv=15、Crossref=15，DBLP/OpenAlex/OpenReview/Semantic Scholar 失败如实保留。加入 LIT、Anchor-Align、Robust Skills、Pointing-VLA、RoboIRG 强制召回；新增 ECL 原文核验。
- 2026-10-16：解析 130 个 BDDL、init files 及远端 432 条 Spatial RLDS episode；10 个候选 task pair 均无 exact shared init，且当前无 pair 两端合法动作 inventory，判定 Diagnostic-only。
- 2026-10-16：完成 factorial protocol、matched-budget matrix 与 falsification gate；idea-evaluator 选择 gate B。训练机制因 ECL/LIT 等强近邻不冻结，交互量作为诊断测量。

## Git 规则

每个审计阶段完成后只暂存本目录对应产物；运行 `git diff --cached --check` 后提交。大 PDF、提取全文、模型与数据只作为本地审计缓存，不加入 Git。提交后记录 commit SHA。

## 完成核验

- `audit_libero_constructibility.py` 已通过只读复核；输出为 130 个 task、10 个候选 pair、0 个严格 trainable pair。
- `step2_search_records.json` 含 3 组检索的去重结果；7897 代理 TLS 失败、直连部分源失败和 Semantic Scholar 429 均保留，未把失败当作无相关工作。
- 7 篇候选完成 introduction/method/experiment/limitation 范围核验；最终选题冻结为 B，不冻结训练 loss。
- 当前唯一未完成项是飞书同步；同步成功后在此记录文档链接和校验结果。

## 飞书同步记录

- 2026-10-16：审计目录下 14 个 Markdown 文档均已成功上传/更新到文献调研文件夹 `DAB9w31AUiuZJAkN5B7cukoGn2g`。
- 代表性文档：`STATUS.md` → `WCDJdqbtooFIm4xVzBwctJxtnPh`；`final_topic_verdict.md` → `OZzTdrayZoxIBAxb0HMcJF6Znsd`；`step3_abstract_triage.md` → `H7R4dHBpVo2zi6xLDtScYPYnnBf`。
- 同步器同时发现云端已有 8 个历史文档而本地审计目录没有对应文件，因而报告 8 个任务失败；这些是云端反向下载任务，不影响本轮 14 个本地审计 Markdown 的上传，也没有删除或覆盖本地文件。
