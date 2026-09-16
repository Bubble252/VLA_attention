# 8. 模型、Baseline 与 Benchmark 筛选备选

本文件专门给研究者筛选，不把所有候选都默认塞进主实验。核心标准来自 *Breaking the Vision–Action Shortcut*：模型覆盖结构差异；baseline 必须 paired；benchmark 要同时检验 ID 能力、task-preserving OOD 和机制解释。

## 8.1 主线候选

| 候选 | 结构/动作接口 | 优势 | 主要风险 | 适合位置 |
|---|---|---|---|---|
| OpenVLA/OFT | 开放 VLA；action token 或 action adapter | 代码、LIBERO 生态和社区对照较成熟 | 原生动作接口与 7D delta EEF 需核对 | 首选 VLA 主线 |
| π0/π0.5 | VLM 与 flow/action expert；连续动作 | 代表连续动作生成和较强开放世界泛化 | 归因路径和权重工程较复杂 | 第二 VLA 主线 |
| MolmoAct2 | layer-wise cross-attention action expert | 可作为显式 cross-attention 对照 | 代码/权重和数据配置需要单独核验 | 有资源时加入 |
| LingBot-VLA | 视频/多 embodiment VLA | 能连接视频动作和多 embodiment | 训练接口重，可能引入额外变量 | 后期扩展 |
| SpikingBrain | full/window/GLA、脉冲稀疏 | 类脑和异构结构分析 | 会重新占据论文叙事，接口审计成本高 | 有空再做 |

### 推荐冻结

- 最小交付：OpenVLA/OFT 单 VLA + Qwen/LLaVA VLM；
- 推荐论文主线：OpenVLA/OFT + π0 系；
- 扩展：MolmoAct2 或 LingBot-VLA；
- 附加：SpikingBrain，不阻塞任何主线里程碑。

## 8.2 Paired baseline 设计

每个主线模型建立独立 baseline，固定：

- 同一 pretrained backbone 和 action expert 初始化策略；
- 同一 demonstrations、数据划分、图像预处理和训练步数；
- 同一原生 action representation 与 action objective；
- 同一 rollout 数量、seed 和评估环境；
- 只改变是否加入我们的 attribution module。

### 必选 baseline 层级

1. `BC/native baseline`：原始行为克隆或模型官方训练方式；
2. `L_sem only`：只加入 Lavender→`A_lang`；
3. `D0`：加入 `L_contain`；
4. `D1`：再加入 phase soft target；
5. `D + C`：加入 LIBERO state-rule refiner；
6. `D + B`：加入 WAM/VAM future refiner；
7. `B/C without D`：检验外部 refiner 是否能独立解释收益；
8. 错词、错图、随机教师和随机 phase；
9. 只做 attention/gradient/latent aggregation 的替代方案。

## 8.3 Benchmark 设计选项

| 选项 | ID | OOD | 优点 | 局限 |
|---|---|---|---|---|
| LIBERO only | 一个或多个 suite | 无 | 环境可控、可做闭环和 phase | 不能支撑真实视觉泛化 |
| LIBERO + task-preserving perturbation | LIBERO | camera、lighting、background、object layout、robot init、language、sensor noise | 最适合证明 shortcut 缓解 | 仍是仿真 |
| LIBERO + DROID offline | LIBERO rollout | DROID scene/object/camera/operator split | 机制与真实视觉分工清楚 | DROID 首轮没有闭环环境 |
| DROID offline only | DROID train/val/test | scene/object/operator split | 直接检验真实数据分布 | 难以解释闭环成功原因 |
| DROID + real robot | DROID demonstrations | camera/lighting/distractor | 最完整的现实验证 | 工程和安全成本最高 |

### 推荐协议

```text
LIBERO：单 pick-place 做机制和闭环 smoke
DROID：pick-place 对应子集做离线 action prediction 和跨场景泛化
真机：只有 DROID 与 LIBERO 结果稳定后再做
```

## 8.4 评价指标必须分表

- VLM grounding：pointing accuracy、phrase IoU、entropy、augmentation consistency；
- VLA offline：7D delta EEF MAE/RMSE、gripper accuracy、occlusion agreement；
- VLA closed-loop：success rate、接近/抓取/放置子阶段成功率、轨迹平滑度；
- OOD：按 camera/object/scene/operator 分组，而不是只报总平均；
- 机制：`A_act` 越界质量、语言/state 反事实变化、raw/sink-free 差异。

## 8.5 筛选规则

- 若某模型无法稳定取得视觉 token 和输出条件归因，不能作为主线模型；
- 若只能得到 attention 但不能做 intervention，最多作为可视化对照；
- 若模型原生 action space 无法可靠转成 7D delta EEF，保留原生结果，不混入统一动作表；
- 若一个方法只在 LIBERO ID 提升而 OOD 不提升，不把它写成泛化方法；
- 若 D0 在两个接口不同的 VLA 上成立，D1/B 才进入扩展；
- 若 SpikingBrain 加入需要重写主线叙事，则延期，不为凑“类脑”而改变论文范围。

## 8.6 待研究者确认

- [ ] 推荐主线是否冻结为 OpenVLA/OFT + π0 系；
- [ ] π0 系接口若在规定时间内跑不通，是否自动退回 OpenVLA/OFT 单骨干；
- [ ] DROID 是否只做 offline action prediction；
- [ ] LIBERO 是否固定为机制验证，DROID 固定为泛化验证；
- [ ] SpikingBrain 是否完全移出主论文，仅保留 future work/appendix。
