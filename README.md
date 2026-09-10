# VLM 终局：语言条件空间归因对齐

本仓库记录最终研究方案、文献归档和后续实现入口。核心方法不限定于某一种 VLM 的 attention 结构，而是把不同 VLM/VLA 的内部视觉证据统一成**语言条件空间归因图**：

> Lavender / Stable Diffusion 提供 `word → image region` 教师图；SpikingBrain、Qwen、DeepSeek、OpenVLA 等模型通过各自可用的 attention、hidden-state gradient 或 action-conditioned attribution 产生学生归因图；二者统一到图像坐标后对齐。

实验顺序是：先在 VLM grounding 上证明这种归因对齐优于普通 fine-tuning 或 naive attention 对齐，再在 LIBERO 中验证它是否能改善 7D delta EEF 的 VLA 控制。SpikingBrain 是主实验骨干，因为它的 full/window/GLA 结构适合做结构感知消融；OpenVLA/OFT 用于后续跨 VLA 骨干验证。

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
