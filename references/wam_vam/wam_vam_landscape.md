# WAM / VAM 参考路线

本清单用于扩展 D 方法的后续 world-model / video-action refiner 路线。它们不是 D0/D1 的必要依赖；首轮仍先完成 VLM，再在 LIBERO pick-place 上验证 VLA。

| 路线 | 代表工作/代码 | 对本项目的启发 | 首轮处理 |
|---|---|---|---|
| World Action Model | Genie / Genie 2（DeepMind，官方项目页） | 由视频学习可交互世界状态，适合作为未来状态一致性教师 | 只做概念参照 |
| Video World Model | Sora、Cosmos（NVIDIA） | 视频预测可提供短期未来区域变化，支持 `R_future` | 后续 B 路线 |
| Video-Action Model | GR00T N1、GR00T N1.5（NVIDIA） | 将视觉、语言、动作轨迹联合建模，提示动作相关归因可从动作 token/隐藏状态获得 | VLA 结构参照 |
| Vision-Language-Action | OpenVLA / OpenVLA-OFT | 开放 VLA baseline，便于验证跨骨干归因适配器 | 作为普适性骨干 |
| VLA flow/diffusion | π0 / openpi | 连续动作生成与视觉语言条件结合，适合 delta-EEF 输出对照 | 作为动作头参照 |
| Video-action policy | DreamZero | 用视频预训练迁移到机器人控制，说明视频表征可作为动作先验 | 后续对照 |
| 轨迹世界模型 | UniPi、RoboDreamer 等 | 将语言条件、未来观测和动作轨迹联系起来，可定义未来归因一致性 | 只读论文与代码 |

## 与 D 方法的接口

首轮不把 world model 的预测误差加入训练损失。后续可从冻结的 VAM/WAM 得到短、中、长时域未来相关性 `R_future^h`，构造

`T_AR^h = Normalize(T_sem ⊙ R_future^h)`，

再与模型自身的动作归因 `A_act(t)` 比较。这样 world model 只提供未来相关性细化，核心创新仍是结构原生的语言条件空间归因一致性。

## 下载状态

候选项目的 URL、版本和下载命令维护在 `wam_vam_download_targets.md` 与 `scripts/download_references.sh`。由于当前 7897 代理端口不可连接，新增项目先登记为 `pending_download`，不得宣称已归档。
