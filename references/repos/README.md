# 官方模型仓库归档

本目录保存与最终方案相关的模型/基线官方代码仓库。实际仓库目录被 `.gitignore` 排除，不随 `VLM终局` 主仓库提交；主仓库只提交本索引和 `references/repositories.yaml`。

## 已在本机归档的仓库

| 目录 | 来源 | 当前 commit | 说明 |
|---|---|---|---|
| `lavender/` | `https://github.com/AstraZeneca/vlm.git` | `58fc71b7b225326fd6f49014b14323bfa4e22590` | 本项目的 Lavender 教师注意力图来源 |
| `spikingbrain-7b/` | `https://github.com/BICLab/SpikingBrain-7B.git` | `ef99987167cf7386ab3348312c8c7aa00a6696ee` | 主干模型与层级结构来源 |
| `openvla-oft/` | `https://github.com/moojink/openvla-oft.git` | `e4287e94541f459edc4feabc4e181f537cd569a8` | LIBERO VLA/OFT 参考实现，来自本机已有副本 |

## 待通过 7897 补齐的仓库

| 目录 | URL | 用途 |
|---|---|---|
| `qwen2.5-vl/` | `https://github.com/QwenLM/Qwen2.5-VL.git` | Qwen2.5-VL 表征和视觉 token 对照 |
| `qwen3-vl/` | `https://github.com/QwenLM/Qwen3-VL.git` | Qwen3-VL 表征和 DeepStack/Interleaved-MRoPE 对照 |
| `deepseek-vl2/` | `https://github.com/deepseek-ai/DeepSeek-VL2.git` | DeepSeek-VL2 视觉语言结构对照 |
| `openvla/` | `https://github.com/openvla/openvla.git` | OpenVLA 官方基线 |
| `openpi/` | `https://github.com/Physical-Intelligence/openpi.git` | π0 / continuous action 参考 |
| `libero/` | `https://github.com/Lifelong-Robot-Learning/LIBERO.git` | LIBERO 仿真环境 |
| `diffusion_policy/` | `https://github.com/real-stanford/diffusion_policy.git` | 视觉运动 diffusion policy 参考 |
| `lingbot-va/` | `https://github.com/Robbyant/lingbot-va.git` | LingBot-VA 世界模型路线 |
| `lingbot-vla/` | `https://github.com/Robbyant/lingbot-vla.git` | LingBot-VLA 路线 |
| `lingbot-vla-v2/` | `https://github.com/Robbyant/lingbot-vla-v2.git` | LingBot-VLA 2.0 路线 |
| `qwen-robotmanip/` | `https://github.com/QwenLM/Qwen-RobotManip.git` | Qwen-VL 机器人操作 VLA |

执行：

```bash
bash scripts/download_references.sh
```

脚本会强制设置 `http_proxy`、`https_proxy`、Git HTTP proxy 为 `http://127.0.0.1:7897`。
