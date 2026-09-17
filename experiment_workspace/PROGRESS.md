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
- [ ] 将当前 Git commit 传到 VEPFS `repo/VLA_attention`，记录 commit 与文件 SHA256。
- [ ] 在 VEPFS 创建 `envs/p1`；只读继承共享依赖或按锁定 requirements 安装，绝不修改 `/root/starvla_cu124`。
- [ ] 记录网络和代理检测；下载使用 7897 代理，若服务器无该代理则先记录失败后再选已验证镜像。
- [ ] 下载并核验 Qwen2.5-VL-7B、Flickr30k Images、Flickr30k Entities。2026-09-17 首次 Qwen 下载已到达服务器端 CLI，但旧 `huggingface-cli` 被环境禁用；受限脚本已切换为已安装的 `hf download`，仍只使用官方源与 7897 代理。
- [ ] 建立全量 manifest，随后从 calibration split 固定 P1 的 20 个 sample ID。
- [ ] 输出 Qwen P1 报告；通过后才下载/审计 Prismatic。

## 下一条可接受的证据

`results/P1-interface-audit/qwen2.5-vl-7b/report.json` 应由 native adapter 生成，并通过：

```bash
PYTHONPATH=src python scripts/validate_p1_report.py \
  <VEPFS>/results/P1-interface-audit/qwen2.5-vl-7b/report.json
```

它必须覆盖 20 个固定样本，保存动态 token grid、phrase target 标量、梯度有限性、seed、图路径及模型 revision。只有该文件通过，才把 Qwen 记为 P1 通过。
