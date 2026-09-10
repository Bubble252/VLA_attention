# 参考文献与模型归档

本目录用于保存最终方案所依据的论文、官方仓库版本和校验信息。PDF 默认被 `.gitignore` 排除，仓库只提交本索引、BibTeX 和 SHA256；如果某个 PDF 必须随仓库分发，需要单独说明许可证和文件大小。

## 文件登记表

| 本地文件 | 类型 | 来源 | 在项目中的角色 | 状态 |
|---|---|---|---|---|
| `papers/lavender_2502.06814.pdf` | 论文 | `https://arxiv.org/abs/2502.06814` | 词级扩散教师图 | 待下载 |
| `papers/spikingbrain_2509.05276.pdf` | 论文 | `https://arxiv.org/abs/2509.05276` | 主干层级结构 | 待下载 |
| `papers/qwen25_vl_2409.12191.pdf` | 论文 | `https://arxiv.org/abs/2409.12191` | VLM 结构参照 | 待下载 |
| `papers/deepseek_vl2_2412.10302.pdf` | 论文 | `https://arxiv.org/abs/2412.10302` | 高分辨率/token 路由参照 | 待下载 |
| `papers/openvla_2406.09246.pdf` | 论文 | `https://arxiv.org/abs/2406.09246` | LIBERO VLA 基线 | 待下载 |
| `papers/pi0_2410.24164.pdf` | 论文 | `https://arxiv.org/abs/2410.24164` | 连续动作专家参照 | 待下载 |
| `papers/libero_2306.03310.pdf` | 论文 | `https://arxiv.org/abs/2306.03310` | 仿真环境 | 待下载 |
| `papers/diffusion_policy_2303.04137.pdf` | 论文 | `https://arxiv.org/abs/2303.04137` | 动作建模参照 | 待下载 |
| `papers/gla_2312.06635.pdf` | 论文 | `https://arxiv.org/abs/2312.06635` | GLA 理论参照 | 待下载 |
| `papers/attention_sinks_2309.17453.pdf` | 论文 | `https://arxiv.org/abs/2309.17453` | 注意力异常/稳定性参照 | 待下载 |

## 官方仓库索引

- Qwen2.5-VL：`https://github.com/QwenLM/Qwen2.5-VL`
- Qwen3-VL：`https://github.com/QwenLM/Qwen3-VL`
- DeepSeek-VL2：`https://github.com/deepseek-ai/DeepSeek-VL2`
- OpenVLA：`https://github.com/openvla/openvla`
- π0/openpi：`https://github.com/Physical-Intelligence/openpi`
- LIBERO：`https://github.com/Lifelong-Robot-Learning/LIBERO`
- Lavender 本地代码：`/home/bubble/类脑计算/参考/vlm`，commit `58fc71b`
- SpikingBrain 本地代码：`/home/bubble/类脑计算/参考/spikingbrain仓库/SpikingBrain-7B`，commit `ef99987`

LingBot 相关工作需在下载前再次确认具体论文标题、版本和官方代码仓库；当前只将其作为视频/世界模型/VLA 的架构参照，不把未经核对的条目当作主实验依赖。

## 下载与校验约定

优先使用本地代理 `http://127.0.0.1:7897`。每次下载后执行：

```bash
sha256sum references/papers/*.pdf > references/checksums.sha256
```

并在 Git commit 中记录下载日期、源 URL 和失败重试情况。论文 PDF 只用于研究阅读，引用时以正式论文版本和官方仓库为准。

