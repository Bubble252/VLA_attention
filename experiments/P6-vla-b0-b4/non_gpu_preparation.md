# VLA 非 GPU 准备与 B0 验证

2026-09-28；依据用户新增 goal。原 VLM 全量训练/验证目标继续保留。

## 工作边界

远端独立根目录为 `/vepfs-mlp2/c20250405/400040/transfer/vla_attention/vla_workspace`。所有准备命令设置 `CUDA_VISIBLE_DEVICES=''`；不改变当前 VLM 环境、进程、cache。GPU 验证等待 cache 验收后另排独占窗口。

## 执行与验收

- [x] 核对新目标文件和实际 cache 进程；本次起点为 614/9952，运行中。
- [x] 读取官方 OFT LIBERO 文档：原生 chunk=8、action dim=7、proprio dim=8、训练 L1 loss；这些是静态事实，仍需真实 P1。
- [ ] 固定官方 OpenVLA-OFT、LIBERO、Transformers fork、dlimp 和 RADIO 来源/commit/许可，生成 source lock。
- [ ] 创建隔离 Python 3.10 环境并安装匹配官方的 torch 2.2.0/torchvision 0.17.0 与 OFT 专用依赖；安装日志及最终 freeze 留档。
- [ ] 下载 OpenVLA 基座、OFT LIBERO-spatial 评测权重、C-RADIOv3-L、官方 LIBERO 数据；固定 revision 和 SHA。评测权重不得作为干净 task-unseen 初始化。
- [ ] CPU 审计任务/episode/camera/action；由真实数量生成 split，禁止复制演示凑数。
- [ ] 实现真实 adapter/B0 train/restore/evaluator；不得把现有 schema holder 当成可训练模型。
- [ ] 准备三类 VLA cache 的输入合同：semantic、C-RADIO feature、冻结 VLM 的 SAEB evidence/latent；暂不运行 GPU 提取。
- [ ] 当前 cache 验收后：P1 → B0 20–50 updates → 新进程恢复 → held-out prediction → 少量官方 rollout。保留完整 8 步输出，首步只单独诊断。

## 重要解释边界

单任务基线不能被要求达到官方 150k steps、全 suite 的论文分数；smoke 证明工程可用性。LIBERO-Plus 只做后续 OOD 评测；不进入训练或校准。ALN/SAEB、B1–B4 长训练不在此次新增授权内。

GPU job 独立于下载安装；只检查一次运行事实，不高频轮询。依赖冲突、下载失败保留日志，可修复就继续。在线来源失败可使用镜像或本地 7897 代理搬运，但不冒充已下载。

## Git

仅提交本次脚本、source lock、审计和进度；不提交环境/权重/数据/cache。

```bash
git add server/prepare_vla_cpu.sh experiments/P6-vla-b0-b4/non_gpu_preparation.md experiments/P6-vla-b0-b4/oft_libero_static_audit.md
git commit -m "feat: prepare isolated OFT and LIBERO runtime"
git push origin main
```

推进状态记录在 `experiment_workspace/PROGRESS.md`；准备脚本退出成功也不等于 B0 通过。
