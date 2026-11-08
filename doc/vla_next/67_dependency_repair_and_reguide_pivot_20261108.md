# 67 调研工具链修复与 ReGuide 式重绑定 Pivot

**日期**：2026-11-08<br>
**状态**：依赖与检索通路已修复；本轮只做调研、预注册和实验设计，不启动训练<br>
**输入**：`65`、`66`、Goal15 P1-C 结果、`papers/` 下的本地论文、2026 年 VLA 近邻工作<br>
**一句话**：P1-C 已经否证“raw route evidence 蒸馏能稳定改善 residual”；下一步不应继续在 residual/route 容量上加模块，而应先用 training-free 的语义+几何重绑定，确认 0/15 是否主要是 referent binding / 执行恢复问题。

---

## 1. 这次先修了什么

### 1.1 依赖与自检

使用的虚拟环境：

```text
/home/bubble/.cc-switch/skills/paper-search/.venv
Python 3.10.18
```

已补齐：

```text
beautifulsoup4 lxml PyYAML tenacity feedparser httpx pydantic
```

自检结果：

- [x] `selftest_runtime.py`：16/16 通过
- [x] `selftest_postprocess.py`：15/15 通过
- [x] 代理 `http://127.0.0.1:7897` 下 arXiv / OpenReview / OpenAlex / Crossref 可真实返回结果

复现命令：

```bash
cd /home/bubble/.cc-switch/skills/paper-search/scripts
HTTPS_PROXY=http://127.0.0.1:7897 \
HTTP_PROXY=http://127.0.0.1:7897 \
ALL_PROXY=http://127.0.0.1:7897 \
NO_PROXY=localhost,127.0.0.1 \
../.venv/bin/python selftest_runtime.py

HTTPS_PROXY=http://127.0.0.1:7897 \
HTTP_PROXY=http://127.0.0.1:7897 \
ALL_PROXY=http://127.0.0.1:7897 \
NO_PROXY=localhost,127.0.0.1 \
../.venv/bin/python selftest_postprocess.py
```

### 1.2 数据源状态

| 数据源 | 状态 | 说明 |
|---|---|---|
| arXiv | 可用 | API 与 PDF 下载均正常 |
| OpenReview | 可用 | 已改为优先走官方 `search_notes`，并规范化作者/过滤评审 note |
| OpenAlex | 可用 | 偶发 429，但有重试 |
| Crossref | 可用 | 偶发 429，但有重试 |
| DBLP | 当前不可用 | `dblp.org`/镜像返回 Anubis 人机验证 HTML，不是网络断开 |
| Semantic Scholar | 当前不可用 | 无 API key 时持续 429，不能阻塞调研 |

因此后续调研使用 **arXiv + OpenReview + OpenAlex + Crossref**，DBLP/S2 的失败在报告里明确标注，不静默忽略。

### 1.3 OpenReview 修复

原连接器会先拉取各会议全部 note 再本地过滤，容易出现“网络可达但结果为零”的假阴性。现改为：

1. 优先调用 `OpenReviewClient.search_notes(term=query, limit=...)`；
2. 过滤没有标题、作者或 venue 的评审 note；
3. 保留旧 `_fetch_venue_notes` 作为 SDK 不支持搜索时的兼容回退；
4. 作者字段同时兼容字符串和 `{fullname, username}` 对象。

已验证：

```text
OpenVLA -> 3 条结果，第一条为 OpenVLA: An Open-Source Vision-Language-Action Model
vision language action model -> 5 条结果，包含 ICLR/NeurIPS/CoRL 论文
```

---

## 2. 调研裁决

### 2.1 先给结论

**继续 residual/route 容量路线，不应成为下一步主实验。**

P1-C 的关键事实不是“residual 完全没用”，而是：

- `R1-CAP` 只吃 base action/proprio，已经在 cosine、sign、MAE 上优于 `R2-ROUTE`；
- 3 个 seed 上 `R2-ROUTE` 都更差，说明 raw route token 没有提供可复用的方向增益；
- `R3-GEO` 的 cosine/sign 更好，说明 6D 几何方向有信息，但 magnitude 和 seed 稳定性没有过 Gate；
- 因此当前问题更像是：**动作专家没有把“正确 referent/geometry”变成稳定的执行配置**，而不是“再加一点 residual 容量”。

### 2.2 2026 年近邻论文给出什么信号

| 论文 | 核心结论 | 对本项目的含义 |
|---|---|---|
| **Encoded but Not in Control**（`papers/encoded_2610.06235`） | 语言 target 可以被线性 probe 解码，但 attention/action 仍偏向场景习惯目标；任务成功不等于 instruction following | 我们的 route probe 可读但动作不用，和这篇的诊断一致；不能再用“probe 准”当方法证据 |
| **ReGuide**（`papers/reguide_2609.38616`） | training-free wrapper：用 grounding 得到的 object pose + demonstration statistics 做 semantic/geometric rebinding，在冻结策略上把组合成功率提升 55.5–56.8 pt；真实机器人从 15% 到 90% | **最接近“先出效果”的路线**：不改 backbone，先验证 referent binding/执行恢复是否是主瓶颈 |
| **APT**（`papers/apt_2606.12366`） | continuous action expert 从随机初始化开始会被语言不平衡拖坏；先做 vision-action prior，再用 gated fusion 注入语言，OOD/组合指令更好 | 如果 ReGuide 证明执行链可用但策略仍不会跟随语言，下一层应做 action expert pretraining，而不是继续加 residual |
| **CRAFT**（`papers/craft_2610.00524`） | 用 same-observation / changed-instruction 反事实 pair，对齐可复用的 skill representation；在 3 个 VLA、2 个 benchmark 上提升未演示组合 | 和我们的 C1 四 prompt 结构高度契合；但需要 skill 分段或至少 phase 标签，成本高于 ReGuide |
| **AnySlot**（`papers/anyslot_2604.10432`） | 把语言转成显式 spatial visual goal，再用 goal-conditioned VLA 执行；在 SlotBench 上显著优于 flat/modular baseline | 说明“显式目标接口”有效，但它依赖强 image generator 和 goal-conditioned 低层策略；我们当前更适合先做几何重绑定 |
| **CounterAlign / CofactVLA / CAST** | 用错指令配对或反事实标签提供负监督，提升 instruction following / robustness | 可作为 ReGuide 后的学习式升级；但若 referent 本身没有被正确重绑定，单加 ranking loss 仍可能只学到关键词 |
| **World Action Verifier / DiVeR / tau0-VLA / Plan-in-Latent** | 通过候选采样 + verifier/world model reranking 提升 test-time success | 需要可靠 candidate generator 和较强 world model；当前底座闭环 0/15 时，reranking 很可能没有可排序的正样本，暂不作为第一枪 |
| **Scale Up Strategically**（`papers/scaleup_2607.21582`） | factor bias 可以用 FDR/FDH 诊断；按欠覆盖因子补数据，半量演示即可超过 baseline | 数据覆盖仍是必要 baseline，但 Goal13 的 D1 已显示“只补 carrier”不够，不能单独当主方法 |

### 2.3 为什么不是“直接上全方法”

“全方法”通常同时改数据、接口和监督目标。当前已有证据是：

- 数据层：D1 没有解决 held-out；
- 接口层：marker / cross-attention / evidence token 都没有稳定收益；
- control 层：P1-C raw route 蒸馏没有超过 capacity control。

如果此时直接把 CompoSlot + AC-VLA + counterfactual + WAM 全叠上去，即使失败也无法判断是哪一层坏了；如果成功，也无法说明是哪一层贡献。**ReGuide 是一个低成本的“断点定位器”**：如果它在 oracle geometry 下能把 0/15 拉起来，说明失败主要是 referent binding/执行恢复；如果 oracle 也拉不起来，说明动作底座或任务本身还有更硬的瓶颈，应该先做 APT/任务重选，而不是继续叠方法。

---

## 3. 下一阶段：ReGuide-style rebinding pilot

### 3.1 目标

在 **不训练新主干** 的前提下，回答：

> 当前 C1 `white_right` 的 0/15，是否主要是因为冻结策略把执行绑定到习惯 referent，而不是因为它完全没有局部操作能力？

### 3.2 方法骨架

```text
当前观测 + 指令
  -> grounding module: 物体/目标盘 pose（oracle 或 P1 predicted）
  -> semantic binding: 选择指令指定的 object / destination
  -> geometric rebinding: 把 end-effector 送到 demonstration-derived pre-contact / pre-place 配置
  -> 冻结策略 hand-back，继续执行
  -> 运动包络 / 释放条件 / 失败恢复
```

第一版只实现三个必要组件：

1. **referent monitor**：判断当前执行是否绑定到错误物体/错误盘子；
2. **configuration transfer**：用目标物体/盘子的 pose 和演示统计，把末端送到可达的预接触/预放置配置；
3. **hand-back + bounded execution**：回到冻结策略，限制动作幅度和高度，避免 wrapper 自己变成手工控制器。

### 3.3 实验臂

| arm | grounding | 作用 | 是否可部署 |
|---|---|---|---|
| `B0` | 无 | 现有冻结策略闭环基线 | 是 |
| `R-GEO` | GT object/destination pose | 诊断上界：如果 oracle 也不行，停止 wrapper 路线 | 否 |
| `R-PRED` | P1 point head / predicted pose | 主候选 | 是 |
| `R-NAIVE` | GT pose，仅做 naive transport | 最低成本对照；ReGuide 论文中 naive transport 已恢复约一半增益 | 否 |
| `R-SHUF` | GT pose，但目标随机打乱 | 负控，排除“只要移动末端就变好” | 否 |

### 3.4 评估与 Gate

第一轮只跑冻结的 15-init manifest；不允许用 `white_right` 训练或调参。

**G1：可执行性**

- [ ] `R-GEO` 无 NaN、无越界、无飞出；
- [ ] `R-GEO` 的 transport/hand-back 成功率 > 80%；
- [ ] `R-SHUF` 不显著优于 `B0`；
- [ ] seen prompt 的 group-any 不低于 `B0 - 1/15`。

**G2：held-out 正信号**

- [ ] `R-GEO` official success `>=1/15`；
- [ ] `R-PRED` official success `>=1/15`；
- [ ] `R-PRED` 相对 `B0` 至少多 `1/15`；
- [ ] wrong-destination 不高于 `B0`；
- [ ] 至少一个机制指标同向：strict destination、released placement、first-target 或 correct-vs-wrong action margin。

**G3：扩展**

只有 G2 通过才跑 30/50 和 3 seeds：

- [ ] `R-PRED` 在 30/50 上仍有正增益；
- [ ] 3 seeds 方向一致；
- [ ] 不牺牲 seen 任务。

### 3.5 失败分支

| 观察 | 解释 | 下一步 |
|---|---|---|
| `R-GEO` 也 0/15 | 不是 referent binding 单独问题；动作底座/任务执行链仍有硬瓶颈 | 停止 ReGuide 主线，转 APT 或任务重选 |
| `R-GEO` 有效、`R-PRED` 无效 | grounding 预测是瓶颈 | 先升级 point/pose predictor，不训练新动作头 |
| `R-PRED` 有效但 `R-NAIVE` 同样有效 | 增益来自“把末端送到目标附近”，不是语义重绑定 | 把 naive transport 作为强 baseline，重新定义方法贡献 |
| `R-PRED` 只在 seen 有效 | 仍是关键词/场景捷径 | 检查 factor swap 和负控，不进入论文级结论 |

---

## 4. 后续两级升级

只有 ReGuide pilot 出现正信号，才进入学习式方法：

### 4.1 第一升级：APT-style action expert pretraining

- Stage 1：冻结 VLM，用 vision-action pairs 预训练 action expert，建立 VA prior；
- Stage 2：用 gated fusion 注入语言，不直接让随机 action expert 从零学语言；
- 对照：同参数、同数据、同 step 的 joint training；
- 目标：解决 APT 论文指出的“continuous action expert 从随机初始化开始被语言不平衡拖坏”。

### 4.2 第二升级：CRAFT-style counterfactual skill representation

- 同图、换指令构造 counterfactual pair；
- 用 skill representation + state representation 分离“做什么”和“在哪里做”；
- 用 action-prediction error 做 same-skill / different-skill 对比；
- 与现有 `counterfactual_binding.py` 的 rank loss 对接，但不把 rank loss 单独当方法。

### 4.3 暂不作为第一枪

- WAM / world-model reranking：需要可靠 candidate generator，当前底座 0/15 时没有足够正样本；
- CompoSlot/AC-VLA 全量组合：作为后续组件的对照，不先叠满；
- 继续 residual/route 容量：P1-C 已给出负结果，停止。

---

## 5. Run Contract（启动前必须填）

| 字段 | 要求 |
|---|---|
| `code_commit` | 与实际执行一致，禁止远端漂移 |
| `config_sha256` | 与实际加载 config 一致 |
| `manifest_sha256` | 与冻结 15-init manifest 一致 |
| `train_language_counts` | `white_right` 必须为 0 |
| `train_validation_rollout_overlap` | 必须为 0 |
| `grounding_arm` | `oracle` / `predicted` / `shuffled` 明确 |
| `pose_source` | GT simulator state 或 P1 point head，不能混写 |
| `transport_bounds` | 逐项记录 xy/z/rotation 上限 |
| `release_mode` / `settle_steps` | 口径冻结 |
| `gate_status` | `PASS / FAIL / INVALID` |

`INVALID` 表示契约不满足，不能用来判方法失败。

---

## 6. 外部论文/代码准入清单

在把任何外部方法写进训练代码前，必须完成：

- [ ] 下载论文 PDF 与 TXT 到 `papers/`；
- [ ] 检查正文、脚注、附录、OpenReview 页面中的 code/project 链接；
- [ ] 若有仓库：clone 到 `references/repos/<slug>`，记录 commit，不直接复制未知版本；
- [ ] 写 `references/paper_reading/<slug>.md`，至少包含方法、输入输出、数据要求、可复现风险；
- [ ] 在实验配置中标注“论文近似版”还是“原论文实现”；
- [ ] 不把论文报告的数值当作本项目结果。

当前已下载：

```text
papers/reguide_2609.38616.pdf/.txt
papers/apt_2606.12366.pdf/.txt
papers/counteralign_2608.21740.pdf/.txt
papers/cofactvla_2608.04396.pdf/.txt
papers/diver_2610.04933.pdf/.txt
```

ReGuide/APT 暂未检索到公开代码入口；若后续使用其仓库，必须先完成上面的 clone + pin commit + reading note。

---

## 7. 执行步骤与 Git

### Step 0：预检（不训练）

- [ ] 复核 15-init manifest 与 `white_right` 训练计数为 0；
- [ ] 复核 P1 point head/pose 输出与仿真 GT 的对齐误差；
- [ ] 复核当前冻结策略的 `B0` 基线仍是 0/15；
- [ ] 写 `run_contract.json`，状态必须为 `PASS` 才能进入 Step 1。

### Step 1：实现 wrapper（只读现有代码，不改主干）

- [ ] 新增 `scripts/reguide_rebinding.py`（或等价模块）；
- [ ] 先只实现 `R-GEO` 与 `R-NAIVE`；
- [ ] 加单元测试：坐标系、目标选择、边界、hand-back、负控；
- [ ] 跑 `pytest -q` 与一个 1-episode smoke。

### Step 2：15-init pilot

- [ ] 先 `R-GEO` + `R-NAIVE` + `B0`；
- [ ] 再 `R-PRED` + `R-SHUF`；
- [ ] 记录 official success、strict destination、released success、wrong-destination、动作幅度；
- [ ] 按 G1/G2 写 `result.md`，状态只能是 `PASS/FAIL/INVALID`。

### Git 规则

不要 `git add -A`；只添加本次相关文件。示例：

```bash
git status --short
git diff -- doc/vla_next/67_dependency_repair_and_reguide_pivot_20261108.md
git add doc/vla_next/67_dependency_repair_and_reguide_pivot_20261108.md
git commit -m "docs: preregister reguide rebinding pivot after p1c"
git push
```

如果需要提交实现，再单独：

```bash
git add scripts/reguide_rebinding.py tests/test_reguide_rebinding.py
git commit -m "feat: add reguide-style rebinding diagnostic"
git push
```

---

## 8. 下一阶段 Goal 提示词

```text
你现在执行 Goal16：ReGuide-style referent rebinding pilot。

先读：
- doc/vla_next/67_dependency_repair_and_reguide_pivot_20261108.md
- doc/vla_next/65_story_method_selection_20261107.md
- doc/vla_next/66_outer_loop_pivot_after_interface_failure_20261107.md
- experiments/Goal15-p1-lateral-distill-20261107/07_p1c_result.md
- experiments/Goal15-p1-lateral-distill-20261107/09_p1c_seed_robustness_result.md

硬约束：
1. 不启动新的 residual/route 容量训练；P1-C 已经 FAIL。
2. white_right 训练计数必须为 0；15-init manifest 不得改动。
3. 先写 Run Contract，状态 PASS 才能跑；否则判 INVALID。
4. 只做冻结策略上的 training-free wrapper：B0、R-GEO、R-NAIVE、R-PRED、R-SHUF。
5. 所有结果必须同时报告 official success、strict destination、released success、wrong-destination 和动作幅度。
6. 如果 R-GEO 也 0/15，停止 wrapper 主线并写失败裁决，不继续加模块。
7. 如果 R-GEO 有效而 R-PRED 无效，先查 grounding 预测，不训练新动作头。
8. 允许自主迭代实现细节，但不允许改 held-out、改数据 split、用 validation 调参来刷 Gate。
9. 外部论文方法必须先下载 PDF/TXT、查 code 链接、写 paper_reading note，再进代码。

执行顺序：
- 先跑 B0，确认 0/15；
- 再跑 R-GEO 与 R-NAIVE，判断是否存在 headroom；
- 再跑 R-PRED 与 R-SHUF；
- 按 G1/G2/G3 做 Gate；
- 只有 G2 PASS 才扩 30/50 和 3 seeds；
- 最后更新 result md、提交并 push。
```

---

## 9. 一句话总结

**依赖问题已经解决；调研结论是停止 residual/route 容量线，先用 ReGuide 式语义+几何重绑定确认“看对但用不对”是否能在不训练主干的情况下被修复。** 这是当前最便宜、最有信息量、也最可能先出正结果的一步。

---

## 10. 执行结果（2026-11-08）

Goal16 已跑完 15-init 五臂。Run Contract `PASS`；G1 `PASS`；G2 因 R-PRED grounding `INVALID` 未通过，正式状态 `INVALID`。

关键数字：

| arm | success | released | target dest correct | wrong dest |
|---|---:|---:|---:|---:|
| `B0` | 0/15 | 0/15 | 0/15 | 14/15 |
| `R-GEO` | 2/15 | 2/15 | 10/15 | 3/15 |
| `R-NAIVE` | 4/15 | 1/15 | 10/15 | 3/15 |
| `R-PRED` | 0/15 | 0/15 | 0/15 | 14/15 |
| `R-SHUF` | 0/15 | 0/15 | 0/15 | 14/15 |

结论：

- 动作底座有 headroom：把物体送到正确 `plate_2` 后，`R-GEO`/`R-NAIVE` 都能产生成功；
- 当前 full 3D hand-back 不如简化 xy transport：`R-GEO` 2/15 < `R-NAIVE` 4/15；
- deployable 瓶颈是 grounding：point head 对 15 个 held-out `white_right` 全部预测到 `plate_1`（mean 0.603 m，15/15 miss）；
- 下一步不是继续加 hand-back 模块，而是先修 compositional grounding，再用简化 xy + release wrapper 重跑 `R-PRED`。

详见 `experiments/Goal16-reguide-rebinding-20261108/05_pilot_result.md`。
