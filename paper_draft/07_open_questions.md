# 7. 写作与实验前仍需确认的问题

这些问题不阻塞论文骨架，但会影响最终实验表和方法措辞：

- DROID 首轮具体子集、episode 数和许可/存储路径；
- DROID 原生动作字段到 7D delta EEF 的转换细节；
- VLM 首轮使用 Flickr30k Entities 的哪一版划分，以及 Lavender attention map 的可复用范围；
- 主线 VLA 选择 OpenVLA/OFT、π0 系或其他候选的最终冻结；SpikingBrain 是否加入扩展实验；
- `L_contain` 是对 union map 使用 max、sum 还是 soft OR；
- D1 是否有足够稳定的 grasp/near-target 事件；
- WAM/VAM 未来目标 `S^k` 采用目标距离、接触状态还是视频 latent 变化；
- 论文主结果是否使用 OpenVLA/OFT + π0 系，还是 OpenVLA/OFT + LingBot-VLA；Qwen/LLaVA 作为 VLM 侧普适性结果；
- 若 DROID 只能离线评估，论文标题是否避免“closed-loop control”表述。

每个问题都应在对应实验开始前记录决定、理由和 commit。
