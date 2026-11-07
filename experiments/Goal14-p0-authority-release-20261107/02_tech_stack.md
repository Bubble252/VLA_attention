# P0 技术栈与运行环境

**日期**：2026-11-07
**状态**：预注册；只读 preflight 通过后才运行

## 1. 本地仓库

- 项目根：`/home/bubble/类脑计算/VLM终局`
- 分支：`main`
- 本地代码是实验协议和 runner 的权威来源；
- 远端 `repo/VLA_attention` 是该目录的同步副本，不是带 `.git` 的独立克隆。

## 2. 远端环境

SSH：

```bash
ssh -p 27219 root@115.190.90.101
```

路径：

```bash
W=/vepfs-mlp2/c20250405/400040/transfer/vla_attention/vla_workspace
R=/vepfs-mlp2/c20250405/400040/transfer/vla_attention/repo/VLA_attention
```

GPU：

- GPU0 已被其他任务占用，禁止使用；
- GPU1 空闲，本实验只使用 `CUDA_VISIBLE_DEVICES=1`；
- 所有 episode 串行执行，不并行抢占显存。

## 3. 冻结资产

J2 checkpoint：

```text
$W/results/JGATE_first_batch_20261023T0737cache/openvla--openvla-7b+libero_90_gate+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--JGATE_formal_20261023T0737cache--J2--12500_chkpt
```

基座模型：

```text
$W/models/openvla--openvla-7b
```

预期 checkpoint fingerprint：

```text
747a28690383ca45e795518121e9aed3f4610265c14618a69f203922b4238e20
```

J2 `dataset_statistics.json` SHA256：

```text
6d5ebb47bc5132df581aae0481d529d04a0821c331859c05a9818d78156c9538
```

冻结 manifest：

```text
$W/data/jgate_c1/counterfactual_20261027/manifest.jsonl
```

manifest 预期 SHA256：

```text
e9564ca9aef800e618e3fdb5d144892eabe9471815bbd24bae3140583946f45e
```

manifest 为 60 行，即 15 init x 4 prompt；本实验只选 `white_right` 的 15 行。

## 4. 数据 split

v10 已使用的 3 个 calibration init：

```text
init_0e94a4e6b9622c6b
init_2edf49999eb35566
init_3500df19a4d3b5a1
```

其余 12 个 init 是 confirmation。报告中必须分开写，不能把 3 个 calibration 重复算成独立 confirmation。

## 5. 代码组件

新增：

- `configs/vla/k4_lite_t4b_v12_p0_authority_release.json`：冻结 arms、release/settle、Gate；
- `scripts/audit_k4_lite_t4b_v12_p0_authority_release.py`：选择 15 init、运行 4 arms、分 split 汇总和判 Gate；
- `tests/test_k4_lite_t4b_v12.py`：SHA、配置篡改、Gate 和 split 测试。

复用：

- `scripts/audit_k4_lite_t4b_v8_release_settle.py`：episode runner、arm 汇总、metric amendment 校验；
- `src/vla_attention/t4b_v8_control.py`：横向控制、release/settle 状态机；
- `scripts/eval_oft_single_checkpoint.py`、`scripts/eval_j0_rollout.py`：checkpoint 和策略加载。

## 6. 输出

远端输出目录：

```text
$W/results/K4_lite/T4b/p0_authority_release_15init_20261107
```

必须产生：

- `audit_p0_authority_release_v12.json`；
- `audit_p0_authority_release_v12.md`；
- `traces/{B0,A05,A10,A20}/{row_id}.jsonl`。

不产生 checkpoint、视频、safetensors 或 `pt` 文件。

## 7. 环境检查

运行前必须确认：

- `nvidia-smi` 显示 GPU1 空闲，GPU0 不被本实验使用；
- manifest SHA 与冻结值一致；
- J2 fingerprint 与冻结值一致；
- `dataset_statistics.json` SHA 与冻结值一致；
- config SHA 与 runner 常量一致；
- 输出目录不存在；
- 远端代码与本地提交一致。
