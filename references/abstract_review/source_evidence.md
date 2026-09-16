# 本地 PDF 来源与摘要页证据

本文件由脚本提取，保留页面阅读顺序以便复核；双栏文本可能交错。不是自动翻译或已完成精读的声明。

## P01 — 3D-hamster.pdf

PDF 第 1 页；共 8 页；SHA256 `eb5777f02a6ada3f63f310892b540142878521fbedf1f8042852074f4a1b6271`。

```text
                                                                     This is a preprint - Published in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2026




                                         3D HAMSTER: Bridging Planning and Control in Hierarchical Vision
                                             Language Action Models through 3D Trajectory Guidance
                                                Dongyoon Hwang1∗ , Byungkun Lee1∗ , Dongjin Kim1∗ , Hyojin Jang1 , Hoiyeong Jin1 , Jueun Mun2 ,
                                                             Minho Park1 , Hojoon Lee3 , Hyunseung Kim1,4 , and Jaegul Choo1†

                                                                                                                                                 Task Instruction : Pick up the square ring on the red peg




                                           Abstract— Hierarchical Vision-Language-Action (VLA) mod-                                       System 2: 2D Planner                                                       System 2: 3D Planner




                                        els decouple high-level planning from low-level control to                                                                        2D Traj.


                                                                                                                                                                                                                                                 3D Traj.



                                        improve generalization in robot manipulation. Recent work in                                      2D to 3D Unprojection




                                        this paradigm uses 2D end-effector trajectories predicted by
                                        a Vision-Language Model (VLM) as explicit guidance for a
arXiv:2606.31329v2 [cs.RO] 1 Jul 2026




                                        downstream policy. However, state-of-the-art low-level policies
                                        operate in 3D metric space on point clouds, and feeding them 2D
                                        guidance that lacks depth forces each waypoint to be assigned
                                        the depth of whatever scene surface lies beneath it, producing                                                                              Misaligned
Guidance
                                                                                                                                                                                                                                        Metrically Aligned Guidance 




                                        geometrically distorted trajectories. We propose 3D HAMSTER,
                                        a hierarchical framework that closes this gap by having the
                                        planner directly output metrically reliable 3D trajectories.                                           Graffiti
Effect



                                        We augment a VLM with a dedicated depth encoder and a
                                        dense depth reconstruction objective to predict 3D waypoint                                                                                                                             3D Trajectory 
with Scene Point Clouds
                                                                                                                                                                 2D Trajectory
with Scene Point Clouds


                                        sequences, which are directly integrated into a pointcloud-
                                        based low-level policy. Across 3D trajectory prediction, simula-
                                        tion, and real-world manipulation, 3D HAMSTER consistently                                        System 1:
3D Policy                                             System 1:
3D Policy




                                        outperforms proprietary VLMs and 2D-guided baselines, with
                                        the largest gains under appearance-altering shifts and unseen
                                        language, spatial, and visual conditions. The project page                                                                                   Task Failure                                                  Task Success




                                        is available at https://davian-robotics.github.io/                                                                   (a) HAMSTER                                       (b) 3D HAMSTER (Ours)




                                        3D_HAMSTER/.                                                                                Fig. 1. Comparison of 2D and 3D guidance in hierarchical VLAs.
                                                                                                                                    (a) 2D planners create representational misalignment: unprojecting
                                                                  I. INTRODUCTION                                                   2D plans yields flawed 3D guidance, leading to brittle execution. (b)
                                           A long-standing challenge in robot manipulation is bridg-                                Our 3D-aware planner generates metrically reliable 3D guidance,
                                        ing the gap between high-level semantic reasoning and low-                                  establishing a shared metric space for robust manipulation.
                                        level motor control. The strong semantic understanding ca-
                                                                                                                                       To better leverage the generalization capability of VLMs,
                                        pabilities of Vision-Language Models (VLMs) [1]–[3] have
                                                                                                                                    hierarchical VLA frameworks [9]–[11] have been proposed
                                        inspired the development of end-to-end Vision-Language-
                                                                                                                                    as a compelling alternative that explicitly decouples semantic
                                        Action (VLA) architectures that directly map visual observa-
                                                                                                                                    reasoning from low-level motor control. Specifically, they
                                        tions and language instructions to continuous actions [4]–[7].
                                                                                                                                    utilize a VLM as a high-level planner that produces 2D
                                        However, end-to-end VLA architectures, also referred to as
                                                                                                                                    keypoints on a camera image (System 2), while a low-level
                                        monolithic models [5]–[7], have shown limited performance
                                                                                                                                    controller receives this guidance to generate motor com-
                                        due to the scarcity of robot demonstration data, which
                                                                                                                                    mands (System 1). This decoupling provides a critical scaling
                                        remains costly to collect at scale on physical hardware. Fine-
                                                                                                                                    advantage. Because the planner predicts visual targets rather
                                        tuning on this limited data erodes the broad generalizability
                                                                                                                                    than robot-specific actions, it can be trained on abundant non-
                                        of the underlying VLM, leaving these models vulnerable to
                                                                                                                                    robot data encompassing spatial reasoning, visual grounding,
                                        out-of-distribution (OOD) visual shifts across novel objects,
                                                                                                                                    2D bounding boxes, and general VQA, preserving the broad
                                        viewpoints, and environments [8].
                                                                                                                                    generalizability of the underlying VLM.
                                          ∗ Equal  contribution, † Corresponding author.                                               While this 2D formulation is a natural fit for VLM-based
                                           This research was supported by a grant from KRAFTON AI and the                           planners, recent research on low-level policies has increas-
                                        “Advanced GPU Utilization Support Program” funded by the Government
                                        of the Republic of Korea (Ministry of Science and ICT).                                     ingly favored 3D-native architectures that operate on point
                                           1 Dongyoon Hwang, Byungkun Lee, Dongjin Kim, Hyojin Jang,                                clouds, consistently outperforming 2D alternatives in spatial
                                        Hoiyeong Jin, Minho Park, Hyunseung Kim, and Jaegul Choo are with                           precision and robustness to viewpoint changes [12]–[15].
                                        the Kim Jaechul Graduate School of AI, KAIST, Seoul, Republic of Korea
                                        (e-mail: godnpeter@kaist.ac.kr).
                                                                                                                                    This creates a fundamental representational misalignment in
                                           2 Jueun Mun is with the Graduate School of Artificial Intelligence,                      current hierarchical VLAs: the planner would reason in 2D
                                        POSTECH, Pohang, Republic of Korea.                                                         pixel coordinates, while the controller operates in 3D space.
                                           3 Hojoon Lee is with Holiday Robotics, Seoul, Republic of Korea.
                                           4 Hyunseung Kim is with KRAFTON AI, Seoul, Republic of Korea.                               For instance, HAMSTER [9], pairs a 2D VLM planner
                                           Links: GitHub Code | HF Models | Project Page                                            with a pointcloud-based 3D controller (Fig. 1a). To bridge

                                        ©2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing
                                        this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of
                                        this work in other works.
```

## P02 — AVA-VLA.pdf

PDF 第 1 页；共 11 页；SHA256 `2a65ff76ac4b0c1662d50cd0d96f17ba3049572a3073a1d3b9305820602eff5c`。

```text
                        This CVPR paper is the Open Access version, provided by the Computer Vision Foundation.
                                     Except for this watermark, it is identical to the accepted version;
                               the final published version of the proceedings is available on IEEE Xplore.




               AVA-VLA: Improving Vision-Language-Action models with Active
                                    Visual Attention

                                              ∗                                                                        †
                                Lei Xiao1,  Jifeng Li1, * Juntao Gao1,2 Feiyang Ye1,
                                                                                     †
                        Yan Jin1 Jingjing Qian3 Jing Zhang2 Yong Wu1 Xiaoyuan Yu1,
 1                        2                                             3
     LiAuto Inc.              Beijing University of Technology              The Chinese University of Hong Kong, Shenzhen


                                Abstract                                Vanilla
                                                                                   Action T-1                  Action T

    Vision-Language-Action (VLA) models have shown re-
markable progress in embodied tasks recently, but most
methods process visual observations independently at each                                 VLA model                 VLA model

timestep. This history-agnostic design treats robot manip-
ulation as a Markov Decision Process, even though real-                                   Observation               Observation
                                                                                             T-1                        T
world robotic control is inherently partially observable and
requires reasoning over past interactions. To address this
                                                                            Ours
mismatch, we reformulate VLA policy learning from a Par-                           Action T-1                  Action T
tially Observable Markov Decision Process perspective and                     Recurrent                 Recurrent                 Recurrent
propose AVA-VLA, a framework that conditions action gen-                        State                     State                     State

eration on a recurrent state that serves as a neural ap-                                  VLA model                 VLA model
proximation to the agent’s belief over task history. Built
on this recurrent state, we introduce Active Visual Atten-
                                                                                          Observation               Observation
tion (AVA), which dynamically reweights visual tokens in the                                 T-1                        T
current observation to focus on regions most relevant given
both the instruction and execution history. Extensive exper-          Figure 1. (a) Visualized comparison of the proposed AVA-VLA
iments show that AVA-VLA achieves state-of-the-art perfor-            framework and vanilla VLAs. (b) Qualitative comparison of vi-
mance on standard robotic benchmarks, including LIBERO                sual focus from two viewpoints while executing the task “turn
and CALVIN, and transfers effectively to real-world dual-             on the stove and put the moka pot on it.” The vanilla OpenVLA-
arm manipulation tasks. These results demonstrate the ef-             OFT [20] baseline fails to locate the task-critical “stove” switch,
fectiveness of temporally grounded active visual process-             whereas AVA-VLA exhibits more stable focus by leveraging his-
ing for improving VLA performance in robotic sequential               torical context.
decision-making. The project page is available at https:
//liauto-dsr.github.io/AVA-VLA-Page.                                  exhibit strong understanding and generalization abilities af-
                                                                      ter being fine-tuned for downstream scenarios.
                                                                          To adopt the ability to understand diverse scenes, ob-
1. Introduction                                                       jects, and language instructions, most VLA models are
                                                                      built upon pretrained Vision-Language Models (VLMs)
Recent advances in robotic manipulation have demonstrated             [9, 18, 29]. Such models typically extend VLM architec-
impressive progress in training robot action policies that            tures with modules such as action tokenization [19, 45] or
can act across diverse real-world tasks. One transformative           specialized action experts [22, 57] to enable action-oriented
paradigm is Vision-Language-Action (VLA) models [2–                   outputs. Based on this architectural inheritance, these VLA
4, 7, 19, 22, 36, 65], which integrate visual perception, nat-        models typically process visual inputs as isolated temporal
ural language understanding, and action generation within a           frames, treating each frame independently. This implicitly
unified neural architecture. These models, which are capa-            formulates robot manipulation as a Markov Decision Pro-
ble of instruction following and robotic action generation,           cess (MDP) [16, 31], where actions are generated from the
     * Equal contribution.                                            current visual observation, assumed to represent the com-
     † Corresponding author.                                          plete world state. In realistic robotic manipulation, how-



                                                                13453
```

## P03 — Attention Distillation.pdf

PDF 第 1 页；共 11 页；SHA256 `a2a4f740c134eba88c4be4f2b3bad9aec2c32b06525a0598b03c69879fe862ae`。

```text
                              This CVPR paper is the Open Access version, provided by the Computer Vision Foundation.
                                           Except for this watermark, it is identical to the accepted version;
                                     the final published version of the proceedings is available on IEEE Xplore.




  Attention Distillation: A Uniﬁed Approach to Visual Characteristics Transfer

                                Yang Zhou      Xu Gao      Zichong Chen       Hui Huang*
                               Visual Computing Research Center, CSSE, Shenzhen University

                              Artistic Style Transfer                       Appearance Transfer                Style-specific T2I Generation
              Content                                                     Structure



                                                                          Appearance


                                                                                                           A deer     A butterfly surrounded by flowers   A dragon

                                                                          Structure
                                                                                                                Controlled Texture Synthesis
   Content                    Content                   Content                                   Texture                               Texture



    Style                      Style                     Style            Appearance              Layout                               Annotation




                                                                  Texture Expansion
    Example             Result (512x4096)




Figure 1. Given a reference image, our approach can faithfully reproduce its visual characteristics in synthesis, providing a uniﬁed
framework for a wide range of example-based image synthesis applications, such as artistic style transfer, appearance transfer, style-
speciﬁc text-to-image generation, and various texture synthesis tasks.

                                Abstract                                       1. Introduction

Recent advances in generative diffusion models have shown                      Synthesizing new images with visual elements, such as the
a notable inherent understanding of image style and se-                        style or texture, of an example image, is a long-standing yet
mantics. In this paper, we leverage the self-attention fea-                    challenging problem in computer graphics and vision. The
tures from pretrained diffusion networks to transfer the vi-                   key challenge lies in properly representing images’ texture
sual characteristics from a reference to generated images.                     or style features. Traditional methods [6, 11, 15, 34, 37,
Unlike previous work that uses these features as plug-and-                     38, 42, 67] usually deﬁne textures as repeated local patterns
play attributes, we propose a novel attention distillation                     and synthesize new textures by copying local patches from
loss calculated between the ideal and current stylization                      the source image. When it comes to style, an extensive yet
results, based on which we optimize the synthesized im-                        more abstract visual characteristic than texture, new repre-
age via backpropagation in latent space. Next, we pro-                         sentations are required.
pose an improved Classiﬁer Guidance that integrates atten-                        Thanks to the deep learning revolution, neural repre-
tion distillation loss into the denoising sampling process,                    sentations of visual features have emerged. One group of
further accelerating the synthesis and enabling a broad                        approaches performs texture or style-speciﬁc synthesis by
range of image generation applications. Extensive exper-                       matching the global distribution of deep features between
iments have demonstrated the extraordinary performance                         the reference and the output. For example, the seminal work
of our approach in transferring the examples’ style, ap-                       Gram loss [21, 22] regards the feature maps’ statistics as
pearance, and texture to new images in synthesis. Code                         the texture/style representation. Some other work optimizes
is available at https://github.com/xugao97/                                    deep features by minimizing Wassertein distance [24] or ad-
AttentionDistillation.                                                         versarial discrimination loss [51, 52, 68]. However, match-
                                                                               ing global distributions lacks local perception, usually lead-
  * Corresponding   author.                                                    ing to conspicuous detail artifacts. Another group retakes



                                                                       18270
```

## P04 — Breaking the Vision-Action Shortcut.pdf

PDF 第 1 页；共 10 页；SHA256 `90555f1ad97f731f9fe89b6d030c34654e8d58b874e684d50b602f67c4228149`。

```text
                                             Breaking the Vision–Action Shortcut: Latent Interface Training for
                                                        Generalizable Robotics Foundation Models
                                                                        Jianman Lin1,∗ , Shailesh Shailesh2,∗ , Zhongyi Luo3 , Jiafei Duan2


                                            Abstract— Robot foundation models achieve strong in-                         Under visual conditioning, action experts may learn
                                         distribution performance but often degrade under visual dis-                 vision–action shortcuts: reliance on task-irrelevant visual
                                         tribution shifts. When learning to generate actions from                     cues that correlate with demonstrated actions during training
                                         pretrained visual representations, models may exploit task-
                                         irrelevant visual cues that correlate with demonstrated ac-                  but become unreliable under distribution shifts [7], [8].
                                         tions within the training distribution. Such vision–action short-            Limited visual diversity in robot demonstrations can fur-
arXiv:2609.12641v1 [cs.RO] 11 Sep 2026




                                         cuts can undermine generalization when these correlations                    ther encourage this reliance, undermining generalization to
                                         change under distribution shifts. Mitigating these shortcuts                 changes in viewpoint, appearance, or sensing conditions.
                                         requires constraining how visual information is used for action              Yet visual representations also encode task-relevant spatial
                                         generation while preserving task-relevant spatial information.
                                         We propose Latent Interface Training (LIT), a framework-                     information essential for action generation. Mitigating these
                                         agnostic two-stage strategy that first establishes a spatial-                shortcuts therefore requires reducing sensitivity to task-
                                         goal-conditioned action prior without images, then constrains                irrelevant visual variations while preserving responsiveness
                                         visual conditioning through a pose-supervised latent interface.              to task-relevant changes that require different actions.
                                         Stage 1 trains the action expert to generate action chunks                      Existing efforts improve generalization through repre-
                                         conditioned on language, robot state, and each demonstrated
                                         chunk’s terminal SE(3) end-effector pose, learning goal-directed             sentation enhancement and stage-wise action pretraining.
                                         action generation independently of visual cues. Stage 2 intro-               Representation-enhanced methods introduce structured spa-
                                         duces a latent interface that aggregates visual and semantic                 tial or motion information to ground action generation in
                                         representations and serves as the pretrained action expert’s                 task-relevant geometry [9], [10], [11]. However, enrich-
                                         only visual conditioning pathway. The interface is supervised                ing the available information does not directly constrain
                                         to reconstruct the terminal pose previously used to condition
                                         Stage 1, encouraging it to retain the goal-relevant spatial                  how the action expert uses it, leaving room for vision–
                                         information needed for action generation. Across four vision–                action shortcuts. Stage-wise approaches first learn language-
                                         language–action and world–action architectures—π0.5 , Mol-                   conditioned action priors without images, then use them to
                                         moAct2, FAST-WAM, and ImageWAM—LIT improves overall                          initialize visual policy training [12], [13]. However, these
                                         LIBERO-Plus success by 3.87–10.70 percentage points while                    approaches lack an explicit spatial goal for each action chunk
                                         preserving or improving average LIBERO success. Real-world
                                         evaluations show 13.30–16.70 percentage-point gains in success               during pretraining, and subsequent training introduces visual
                                         aggregated across three tasks under unseen camera config-                    conditioning without explicitly constraining its use. Image-
                                         urations, lighting variations, and distractors. Project page:                free pretraining alone therefore leaves the policy susceptible
                                         magiclab-nus.github.io/LIT                                                   to shortcuts once visual conditioning is introduced. These
                                                                                                                      limitations motivate a central question: How can action
                                                                I. INTRODUCTION                                       learning and visual conditioning be structured to mitigate
                                                                                                                      vision–action shortcuts while preserving the task-relevant
                                            Recent robot foundation models, including representa-
                                                                                                                      spatial information needed for action generation?
                                         tive vision–language–action (VLA) and world–action model
                                                                                                                         To address this challenge, we propose Latent Interface
                                         (WAM) architectures, combine pretrained vision–language
                                                                                                                      Training (LIT), a model-agnostic two-stage strategy that
                                         or video backbones with action experts, achieving strong
                                                                                                                      first learns a spatial-goal-conditioned action prior without
                                         in-distribution manipulation performance [1], [2], [3], [4].
                                                                                                                      images, then introduces visual conditioning through a pose-
                                         Visual representations condition action generation alongside
                                                                                                                      supervised latent interface (Fig. 1, right). In the first stage,
                                         language and robot-state information [5], [6], as illustrated in
                                                                                                                      the action expert is conditioned on language and robot-state
                                         Fig. 1 (left). Robust generalization under visual distribution
                                                                                                                      representations only from a frozen pretrained backbone, to-
                                         shifts, however, remains a challenge [7].
                                                                                                                      gether with an encoding of each demonstrated action chunk’s
                                           ∗ Equal co-first contribution.
                                                                                                                      terminal SE(3) end-effector pose. This establishes a spatial-
                                           1 Jianman    Lin is with the School of            Future Technology,       goal-conditioned action prior without visual inputs. In the
                                         South    China University of Technology,            Guangzhou, China         second stage, learnable latent tokens cross-attend to visual
                                         linjianmancjx@gmail.com                                                      and semantic backbone representations and provide layer-
                                           2 Shailesh Shailesh is with the National University of Singapore, Singa-
                                         pore shailesh.xml@nus.edu.sg                                                 wise conditioning to the pretrained action expert, serving as
                                           3 Zhongyi Luo is with Nanyang Technological University, Singapore
                                                                                                                      its only visual conditioning pathway. A pose-reconstruction
                                         LUOZ0031@e.ntu.edu.sg                                                        objective supervises these tokens to recover the same termi-
                                           2 Jiafei Duan is with the School of Computing, National University of
                                         Singapore, Singapore duanj1@nus.edu.sg                                       nal pose used in Stage 1, encouraging them to retain goal-
                                           Jiafei Duan is the corresponding author.                                   relevant spatial information. The shared spatial target thus
```

## P05 — BridgeVLA.pdf

PDF 第 1 页；共 39 页；SHA256 `be473fbb35ef1b9d367e78b050c5579617e8d2fd5dd53c8aa003d851711d540d`。

```text
BridgeVLA: Input-Output Alignment for Efficient 3D
Manipulation Learning with Vision-Language Models


       Peiyan Li1,2,3,∗, Yixiang Chen1,3 , Hongtao Wu2,∗,† , Xiao Ma2,∗ , Xiangnan Wu1
                Yan Huang1,3,4,†, Liang Wang1,3 , Tao Kong2 , Tieniu Tan1,3,5
                           1
                             New Laboratory of Pattern Recognition (NLPR),
                         Institute of Automation, Chinese Academy of Sciences
    2
      ByteDance Seed 3 School of Artificial Intelligence, University of Chinese Academy of Sciences
                                     4
                                       FiveAges 5 Nanjing University



                                                 Abstract

             Recently, leveraging pre-trained vision-language models (VLMs) for building
             vision-language-action (VLA) models has emerged as a promising approach to
             effective robot manipulation learning. However, only few methods incorporate
             3D signals into VLMs for action prediction, and they do not fully leverage the
             spatial structure inherent in 3D data, leading to low data efficiency. In this paper,
             we introduce a new paradigm for constructing 3D VLAs. Specifically, we first
             pre-train the VLM backbone to take 2D images as input and produce 2D heatmaps
             as output. Using this pre-trained VLM as the backbone, we then fine-tune the
             entire VLA model while maintaining alignment between inputs and outputs by:
             (1) projecting raw point cloud inputs into multi-view images, and (2) predicting
             heatmaps before generating the final action. Extensive experiments show that
             the resulting model, BridgeVLA, can learn 3D manipulation both efficiently and
             effectively. BridgeVLA outperforms state-of-the-art baselines across three simula-
             tion benchmarks. In RLBench, it improves the average success rate from 81.4%
             to 88.2%. In COLOSSEUM, it demonstrates significantly better performance in
             challenging generalization settings, boosting the average success rate from 56.7%
             to 64.0%. In GemBench, it surpasses all the comparing baseline methods in terms
             of average success rate. In real-robot experiments, BridgeVLA outperforms a state-
             of-the-art baseline method by 32% on average. It generalizes robustly in multiple
             out-of-distribution settings, including visual disturbances and unseen instructions.
             Remarkably, it is able to achieve a success rate of 95.4% on 10+ tasks with only 3
             trajectories per task, while other VLA methods such as π0 fail completely. Project
             Website: https://bridgevla.github.io/.


1        Introduction

Leveraging pre-trained vision-language models (VLMs) [3, 43, 2, 24] for developing large vision-
language-action (VLA) models has become a promising method for learning generalizable and robust
manipulation policies [26, 4, 17, 31, 7]. However, most VLA models only incorporate 2D image
inputs and require extensive efforts on data collection. On the other hand, 3D robot policies leverage
3D structural priors in model design and demonstrate exceptional sample efficiency in learning
complex 3D robot manipulation tasks [39, 25, 13–15]. Can we develop a unified 3D VLA model
which combines the effectiveness of VLA models with the efficiency from 3D policies?
     ∗
         Project lead
     †
         Corresponding author


39th Conference on Neural Information Processing Systems (NeurIPS 2025).
```

## P06 — Don't Blind Your VLA.pdf

PDF 第 1 页；共 13 页；SHA256 `276ea96810526617bd35e818c6337eff82f88691562de1d1c98dd244c3ecb059`。

```text
                                                                    Don’t Blind Your VLA:
                                                   Aligning Visual Representations for OOD Generalization
                                                        Nikita Kachaev                                 Mikhail Kolosov                            Daniil Zelezetsky
                                                         Cognitive AI Lab                                  IAI MIPT                                    IAI MIPT
                                                         Moscow, Russia                                  Moscow, Russia                              Moscow, Russia

                                                                             Alexey K. Kovalev                            Aleksandr I. Panov
                                                                          Cognitive AI Lab, IAI MIPT                   Cognitive AI Lab, IAI MIPT
                                                                               Moscow, Russia                               Moscow, Russia
                                         ABSTRACT
arXiv:2510.25616v1 [cs.LG] 29 Oct 2025




                                         The growing success of Vision-Language-Action (VLA) models
                                         stems from the promise that pretrained Vision-Language Models
                                         (VLMs) can endow agents with transferable world knowledge and
                                         vision-language (VL) grounding, laying a foundation for action mod-
                                         els with broader generalization. Yet when these VLMs are adapted
                                         to the action modality, it remains unclear to what extent their orig-
                                         inal VL representations and knowledge are preserved. In this work,
                                         we conduct a systematic study of representation retention during
                                         VLA fine-tuning, showing that naive action fine-tuning leads to
                                         degradation of visual representations. To characterize and measure
                                         these effects, we probe VLA’s hidden representations and analyze at-
                                         tention maps, further, we design a set of targeted tasks and methods
                                         that contrast VLA models with their counterpart VLMs, isolating
                                         changes in VL capabilities induced by action fine-tuning. We further
                                         evaluate a range of strategies for aligning visual representations
                                         and introduce a simple yet effective method that mitigates degra-         Figure 1: Visual alignment method overview. Mid-level VLA features
                                         dation and yields improved generalization to out-of-distribution          are projected onto a normalized sphere and aligned with teacher
                                         (OOD) scenarios. Taken together, our analysis clarifies the trade-off     embeddings, preserving visual semantics and improving OOD gener-
                                         between action fine-tuning and the degradation of VL representa-          alization. Bottom plots show comparison with standard SFT across
                                         tions and highlights practical approaches to recover inherited VL         three generalization axes on the Simpler-based benchmark [33].
                                         capabilities. Code is publicly available: blind-vla-paper.github.io
                                                                                                                   studies [11, 15, 32, 36, 40] have shown that current VLA models
                                         1    INTRODUCTION                                                         struggle to maintain generalization in visually and linguistically
                                         Vision–Language Models (VLMs) have demonstrated remarkable                complex tasks, raising questions about whether strong VL capa-
                                         success due to their ability to integrate large-scale multimodal          bilities of VLMs truly transfer to embodied settings. This issue
                                         datasets, thereby acquiring semantic grounding and generalizable          becomes the most evident during task-specific fine-tuning, where
                                         visual-language (VL) representations [2, 3, 5, 16, 37, 48]. When ex-      limited data diversity and datasets frequently lead to overfitting
                                         posed to novel visual or linguistic contexts, such models exhibit         [13, 14, 40, 42, 54].
                                         robust cross-modal understanding and compositional perception                 During large-scale robotic pretraining, recent works have at-
                                         – properties that underpin their strong zero and few-shot gener-          tempted to mitigate this degradation by preserving multimodal un-
                                         alization beyond the training distribution. These advancements            derstanding capabilities. Prior strategies include incorporating aux-
                                         have naturally inspired the extension of VLMs toward embodied             iliary reasoning objectives [10], applying multimodal co-training
                                         domains.                                                                  on web-scale data [52], or freezing pretrained visual–language
                                            Vision–Language–Action (VLA) models represent a prominent              backbones to preserve VL representations and improve instruc-
                                         direction in this research trajectory. They adapt pretrained VLMs         tion following [15, 38]. While these approaches help retain vi-
                                         to action prediction tasks in robotic settings, with the goal of lever-   sion–language knowledge and improve generalization, they often
                                         aging the semantic priors and cognition abilities inherited from          depend on heavy supervision, high computational cost, or con-
                                         large-scale vision–language pretraining. The underlying hypothe-          strained model architecture. Yet, despite these advances at the
                                         sis is that, if appropriately adapted, VLA models can transfer the        pretraining stage, there remain no effective methods to address
                                         visual–semantic representations of their initial VLM to the action        representation degradation during task-specific supervised fine-
                                         domain, enabling generalization to previously unseen scenes, in-          tuning (SFT) – the critical phase where VLA models must adapt to
                                         structions, and scenarios. However, in practice, adapting VLMs to         certain robotic domains without losing their semantic grounding
                                         the action modality often introduces new challenges. Several recent       and VL abilities.
                                         Under review
```

## P07 — FAST-WAM.pdf

PDF 第 1 页；共 13 页；SHA256 `4eea24883dcc4d8a5c0f760870f501baabb07db4eb65e5fd6c3b4b500601be8d`。

```text
                                                   Fast-WAM: Do World Action Models Need
                                                         Test-time Future Imagination?


                                                         Tianyuan Yuan1,2 , Zibin Dong1,2 , Yicheng Liu1,2 , Hang Zhao1,2
                                                                   1
                                                                     IIIS, Tsinghua University 2 Galaxea AI
                                                             https://yuantianyuan01.github.io/FastWAM/
arXiv:2603.16666v2 [cs.CV] 23 Mar 2026




                                                                                       Abstract

                                                  World Action Models (WAMs) have emerged as a promising alternative to Vision-
                                                  Language-Action (VLA) models for embodied control because they explicitly
                                                  model how visual observations may evolve under action. Most existing WAMs
                                                  follow an imagine-then-execute paradigm, incurring substantial test-time latency
                                                  from iterative video denoising, yet it remains unclear whether explicit future
                                                  imagination is actually necessary for strong action performance.
                                                  In this paper, we ask whether WAMs need explicit future imagination at test time,
                                                  or whether their benefit comes primarily from video modeling during training.
                                                  We disentangle the role of video modeling during training from explicit future
                                                  generation during inference by proposing Fast-WAM, a WAM architecture that
                                                  retains video co-training during training but skips future prediction at test time. We
                                                  further instantiate several Fast-WAM variants to enable a controlled comparison of
                                                  these two factors. Across these variants, we find that Fast-WAM remains competi-
                                                  tive with imagine-then-execute variants, while removing video co-training causes
                                                  a much larger performance drop. Empirically, Fast-WAM achieves competitive
                                                  results with state-of-the-art methods both on simulation benchmarks (LIBERO and
                                                  RoboTwin) and real-world tasks, without embodied pretraining. It runs in real time
                                                  with 190 ms latency, over 4× faster than existing imagine-then-execute WAMs.
                                                  These results suggest that the main value of video prediction in WAMs may lie
                                                  in improving world representations during training rather than generating future
                                                  observations at test time.


                                         1   Introduction

                                         Building general-purpose embodied agents requires policies that can not only map visual observations
                                         to actions, but also reason about how the physical world evolves under interaction. This has motivated
                                         growing interest in World Action Models (WAMs), which combine future visual prediction and action
                                         modeling in a unified framework. Compared with standard Vision-Language-Action (VLA) models,
                                         WAMs are appealing because modeling future observations may help capture physical dynamics and
                                         task-relevant temporal structure.
                                         Most existing WAMs follow an imagine-then-execute paradigm: they first generate future observa-
                                         tions, then predict actions conditioned on the imagined future. While intuitive, this design incurs
                                         substantial test-time latency due to iterative video denoising [1, 2, 3, 4, 5]. More fundamentally, it re-
                                         mains unclear whether explicit future imagination is actually necessary for strong action performance.
                                         The effectiveness of WAMs may stem from two distinct sources: (1) the video prediction objective
                                         during training, which may help the model acquire stronger physical priors and action-conditioned
                                         representations, and (2) explicit future generation during inference, which may provide additional
                                         foresight for action prediction. Existing WAM systems typically entangle these two factors, making
                                         it difficult to determine which one is actually responsible for the observed gains.
```

## P08 — FullFlow.pdf

PDF 第 1 页；共 36 页；SHA256 `5df6bd9647dce1052597ffd7661c22ed48c8e2d58b6dafc49ebe44b33cdae616`。

```text
                                          FullFlow: Upgrading Text-to-Image Flow Matching
                                         Models for Bidirectional Vision–Language Generation


                                                     Eric Tillmann Bill1     Enis Simsar1 Alessio Tonioni2          Thomas Hofmann1
                                                                                 1
                                                                                   ETH Zurich 2 Google
                                                                        https://ericbill21.github.io/fullflow/
arXiv:2605.20316v1 [cs.CV] 19 May 2026




                                                                                         Abstract
                                                     Modern text-to-image diffusion models encode rich visual priors, but expose
                                                     them only through one-way text-conditioned generation. Existing unified vision–
                                                     language models derived from them recover bidirectional capability through large-
                                                     scale joint pretraining or substantial retraining of the text pathway, discarding the
                                                     strong image prior the text-to-image backbone already encodes. We introduce
                                                     FullFlow, a parameter-efficient recipe that upgrades a pretrained rectified-flow
                                                     text-to-image model into a bidirectional vision–language generator by training only
                                                     LoRA adapters and lightweight text heads. FullFlow keeps images in their native
                                                     continuous flow and adds a discrete insertion process for text. Separate image
                                                     and text timesteps turn inference into trajectory selection in a two-dimensional
                                                     generative space, enabling text→image, image→text, joint sampling, and partial-
                                                     text prediction with a single backbone. On Stable Diffusion 3 (SD3) under an
                                                     identical trainable-parameter count and matched LoRA rank, FullFlow improves
                                                     text→image FID from 62.7 to 31.6 and image→text CIDEr from 2.0 to 99.4 over
                                                     a LoRA equivalent following the previous SOTA formulation (Dual Diffusion) at
                                                     matched wall-clock training time, while reducing peak VRAM from ∼84 GB to
                                                     ∼38 GB and raising throughput by ∼8× on two RTX A5000 GPUs in under 24
                                                     hours, training only ∼5% of the backbone parameters. The same recipe transfers to
                                                     FLUX.1-dev and supports downstream VQA through partial-text generation. These
                                                     results show that strong bidirectional vision–language capability can be unlocked
                                                     from pretrained text-to-image flow models without full multimodal pretraining.


                                         1    Introduction
                                         Pretrained text-to-image flow models are strong image generators, yet they work in only one direction:
                                         text in, image out. Diffusion established the paradigm for high-fidelity synthesis [20, 45], and rectified-
                                         flow transformers further improved scalability, prompt fidelity, and compositional generation [43, 12,
                                         32, 35, 25]. Their internal representations are deeply structured: attention localizes entities [8, 24],
                                         token interventions affect attribute binding [18, 6, 40], and benchmarks confirm increasingly rich
                                         object-relation modeling [22]. A backbone that has learned how words, objects, and spatial layouts
                                         co-vary should, in principle, also support the inverse direction: describing an image, answering
                                         questions about it, or sampling coherent image–text pairs.
                                         Exploiting this latent capability is not straightforward. Most vision–language models (VLMs) are
                                         built in the opposite direction: a pretrained language model is given a visual encoder, making vision
                                         a conditioning signal for autoregressive text generation [2, 28, 34, 62, 11]. Unified generators that
                                         combine autoregressive text with diffusion-based images [61, 38] are similarly language-first. These
                                         designs discard the strong image prior the text-to-image backbone already encodes, and rebuilding it
                                         requires expensive joint pretraining. This raises a natural question: can a pretrained text-to-image
                                         flow model be made bidirectional without retraining from scratch?

                                         Preprint.
```

## P09 — GenRL.pdf

PDF 第 1 页；共 27 页；SHA256 `cfb3ccccdbe30110e1b5349273fade643a8efa01c994b1e13456e802e4b17698`。

```text
         GenRL: Multimodal-foundation world models
            for generalization in embodied agents


          Pietro Mazzaglia∗               Tim Verbelen                      Bart Dhoedt
       IDLab, Ghent University        VERSES AI Research Lab            IDLab, Ghent University

                        Aaron Courville                             Sai Rajeswar
                  Mila, University of Montreal                  ServiceNow Research




Figure 1: Multimodal-foundation world models connect and align the video-language space of a
foundation model with the latent space of a generative world model for reinforcement learning,
requiring vision-only data. Our GenRL framework turns visual and/or language prompts into latent
targets and learns to realize the corresponding behaviors by training in the world model’s imagination.

                                              Abstract
           Learning generalist embodied agents, able to solve multitudes of tasks in different
           domains is a long-standing problem. Reinforcement learning (RL) is hard to scale
           up as it requires a complex reward design for each task. In contrast, language can
           specify tasks in a more natural way. Current foundation vision-language models
           (VLMs) generally require fine-tuning or other adaptations to be adopted in embod-
           ied contexts, due to the significant domain gap. However, the lack of multimodal
           data in such domains represents an obstacle to developing foundation models for
           embodied applications. In this work, we overcome these problems by presenting
           multimodal-foundation world models, able to connect and align the representation
           of foundation VLMs with the latent space of generative world models for RL,
           without any language annotations. The resulting agent learning framework, GenRL,
           allows one to specify tasks through vision and/or language prompts, ground them
           in the embodied domain’s dynamics, and learn the corresponding behaviors in
           imagination. As assessed through large-scale multi-task benchmarking in locomo-
           tion and manipulation domains, GenRL enables multi-task generalization from
           language and visual prompts. Furthermore, by introducing a data-free policy learn-
           ing strategy, our approach lays the groundwork for foundational policy learning
           using generative world models.
                        Website, code and data: mazpie.github.io/genrl
   ∗
       Work done while interning at Mila/ServiceNow Research. Email: pietro.mazzaglia@ugent.be


38th Conference on Neural Information Processing Systems (NeurIPS 2024).
```

## P10 — Generalizable VLA Finetuning.pdf

PDF 第 1 页；共 39 页；SHA256 `2ef5a4aec9eec6913b8f0a75d5ae6876fe48760554e56c60440f282343799a1f`。

```text
                                             Generalizable VLA Finetuning via Representation
                                               Anchoring and Language-Action Alignment
                                                    Dwip Dalal1 , Shivansh Patel1 , Chahit Jain1 , Jeonghwan Kim1 , Utkarsh Mishra2 ,
                                                     Alex Baratian1 , Hyeonjeong Ha1 , Heng Ji1 , Svetlana Lazebnik1* , Unnat Jain3∗
                                                           1
                                                               University of Illinois Urbana-Champaign 2 Texas A&M University
                                                                                  3
                                                                                    University of California, Irvine

                                                      Abstract:
                                                      Finetuning a pretrained vision-language model (VLM) on robot demonstrations
                                                      via behavior cloning (BC) has become the standard recipe for vision-language-
                                                      action (VLA) policies. However, BC finetuning progressively overwrites the
                                                      pretrained representations that support visual and semantic generalization. Co-
                                                      training on web image-text data, a common remedy, does not prevent this; it
                                                      applies language and action losses to separate observations, leaving VLAs with
arXiv:2607.13429v1 [cs.RO] 15 Jul 2026




                                                      language-action misalignment that standard manipulation benchmarks do not ex-
                                                      pose. We propose Anchor-Align, which augments BC with two objectives: Vision-
                                                      Language Anchoring distills layer-wise representations from a frozen VLM copy
                                                      to prevent this drift, while Language-Action Alignment converts each action tar-
                                                      get into a discrete motion-direction label and jointly trains language and action
                                                      prediction on the same robot observation. On a physical xArm7 robot, across
                                                      two widely used VLA architectures, Anchor-Align improves real-robot success
                                                      on both (28% → 54% and 37% → 60%). At scale in simulation, we demonstrate
                                                      consistent improvements on OOD perturbations, perceptual robustness, and long-
                                                      horizon control across LIBERO-PRO, LIBERO-Plus, and CALVIN, respectively,
                                                      suggesting that preserving pretrained representations and effective action learning
                                                      are not fundamentally at odds. Project page: anchoralignvla.github.io

                                                      Keywords: Vision-Language-Action Models, Robot Manipulation, Catastrophic
                                                      Forgetting, Language-Action Alignment, Out-of-Distribution Generalization

                                         1       Introduction
                                         Vision-Language-Action (VLA) models have become a popular approach for learning robot manip-
                                         ulation policies [3, 4, 6, 18, 20, 36, 37, 40, 43, 57, 61, 70, 72, 73]. VLAs are typically trained by
                                         finetuning a pretrained vision-language model (VLM) on expert demonstrations via supervised ac-
                                         tion prediction, known as behavior cloning (BC). This can be done through direct regression [37, 72]
                                         or flow-matching and diffusion [3, 4, 69]. The premise is that such finetuning should transfer the
                                         VLM’s semantic priors (understanding of spatial layout, directions, color, shape, etc.) to the result-
                                         ing control policy.
                                         Consider a VLA policy finetuned to “pick up the green mug and place it on the plate” in a scene
                                         containing both a green and pink mug (Fig. 1). Since the two mugs share the same shape and manip-
                                         ulation affordances, the policy should generalize to “pick up the pink mug” if finetuning preserves
                                         the VLM’s color understanding. However, with existing methods, it does not. In fact, BC finetuning
                                         corrupts the very prior that makes VLMs worth adapting, leading to two failure modes that persist
                                         even under good training practices.
                                         First, standard BC optimizes only the action prediction loss, with nothing protecting the VLM’s
                                         pretrained representations from being overwritten. Over the course of finetuning, these updates
                                         progressively erase the visuolinguistic and spatial concepts the VLM acquired during internet-scale
                                             ∗
                                                 Equal advising
```

## P11 — ImageWAM.pdf

PDF 第 1 页；共 19 页；SHA256 `0bf354364a18a2e504b46dcefd895ea923a63f04bc2f4ac12d9e8ea57ea80e2a`。

```text
                                             ImageWAM: Do World Action Models Really Need Video
                                             Generation, or Just Image Editing?
                                             Yuyang Zhang 123∗ , Wenyao Zhang 123∗† , Zekun Qi 4 , He Zhang 3 , Haitao Lin 3 , Jingbo Zhang 3 ,
                                             Yao Mu 1 , Xiaokang Yang 1 , Wenjun Zeng 2 , Xin Jin 25B
                                             1
                                               Shanghai Jiao Tong University, 2 Eastern Institute of Technology, 3 Tencent Robotics X, 4 Tsinghua
                                             University, 5 Zhongguancun Academy
                                             ∗
                                               Equal contribution, † Project Lead, B Corresponding author


                                             World Action Models (WAMs) commonly rely on video generation to bridge visual world modeling and robot
arXiv:2606.19531v1 [cs.CV] 17 Jun 2026




                                             control. However, video-based WAMs face three coupled limitations: dense multi-frame future tokens make
                                             inference costly, full video prediction spends capacity on action-irrelevant temporal and appearance details,
                                             and long-horizon future imagination may introduce errors that mislead action prediction. These issues raise a
                                             simple question: Does world action model really need video generation? We propose ImageWAM, a simple
                                             WAM framework that repurposes pretrained image editing models for robot action prediction. In contrast
                                             to video generation, image editing provides a better-matched prior: it only needs to model a target-frame
                                             transformation, focuses on action-relevant current-to-target visual differences, and grounds task instructions
                                             to localized visual changes through edit pretraining. In practice, ImageWAM does not decode the target
                                             frame at inference time; instead, it conditions a flow-matching action expert on the KV caches produced by
                                             image-editing denoising, using them as a compact world-action context. ImageWAM outperforms standard
                                             VLA baselines and matching competitive WAMs without additional policy pretraining across different simulator
                                             and real-world experiments. It also reduces FLOPs to 1/6 and latency to 1/4 of video-based WAMs. Attention
                                             analysis further shows that editing caches focus on task-relevant change regions, supporting image editing
                                             as an effective alternative to video-based world-action modeling.

                                             Date: June 19, 2026
                                             Project Page: https://zhangwenyao1.github.io/ImageWAM/
                                             Github: https://github.com/yuyangalin/ImageWAM



                                         1       Introduction
                                         Recent robot policy learning has increasingly explored video generation models as world-action backbones.
                                         This direction is appealing because video pretraining exposes models to rich visual dynamics, such as object
                                         motion, temporal continuity, physical interaction, and scene evolution [1–5]. It also supports a reason-before-
                                         act paradigm: a policy may first imagine how the scene will change, and then use this imagined future to guide
                                         action prediction [6–8]. Together with the scalability of generative pretraining on large and heterogeneous
                                         video data [9–12], video models provide an intuitive bridge between visual world modeling and robot control.
                                         However, this bridge also reveals a mismatch as shown in Figure 1(a). Video generation models are trained
                                         to synthesize complete future videos. To do so, they must model appearance details, background changes,
                                         camera motion, temporal smoothness, and many other factors that may be only weakly related to the
                                         robot’s next action [13–15]. Generating many spatio-temporal tokens across multiple frames makes inference
                                         costly for real-time robot control [2, 3]. Moreover, generating a physically consistent video is a hard proxy
                                         task [16–18]. This is especially true for fine-grained manipulation, where small contact events, slight object
                                         displacements, or subtle configuration changes can determine success, but are difficult to predict reliably over
                                         multiple frames. If the imagined video is wrong, the downstream action predictor may be misled. These
                                         issues raise a simple question: Does the world action model really require video generation?
                                         We argue that image editing models offer a more direct visual generative prior for language-conditioned
                                         manipulation. Instead of predicting how an entire scene evolves over time, image editing models are trained
                                         to transform a source image according to a language instruction. This objective matches a key requirement of
```

## P12 — LaWAM.pdf

PDF 第 1 页；共 23 页；SHA256 `07aa676e56ab20a5de037fe3444ea7009af5e54c4a21271c20aef35418e54f7f`。

```text
                                             LaWAM: Latent World Action Models for Efficient
                                                   Dynamics-Aware Robot Policies
                                             Jialei Chen♦,▼,⋆ Kai Wang▶,▼ Kang Chen♣,▼ Shuaihang Chen■,▼ Feng Gao♠,⋆ Wenhao Tang♠
                                                       Zhiyuan Li♢ Weilin Liu♢ Zhuyu Yao♢ Boxun Li♢ Yuanbo Xu♦† Chao Yu♠†
                                                        ♠
                                                          Tsinghua University ♦ Jilin University ▶ Nankai University ♣ Peking University
                                                    ■
                                                      Harbin Institute of Technology ▼ Zhongguancun Academy ⋆ Striding.AI ♢ Infinigence AI
arXiv:2606.15768v1 [cs.RO] 14 Jun 2026




                                                  Abstract: Vision-Language-Action models (VLAs) leverage large-scale vision-
                                                  language pretraining for semantic robot control, but often lack explicit foresight
                                                  into how robot actions change the scene. World-Action Models (WAMs) address
                                                  this limitation by conditioning policies on predicted futures, yet existing approaches
                                                  typically rely on computationally expensive video generation with substantial pixel-
                                                  level redundancy. We present LaWAM, a Latent World Action Model that exposes
                                                  predictive dynamics to robot policies through compact latent visual subgoals instead
                                                  of reconstructed future video. At the core of LaWAM is a latent-action-conditioned
                                                  Latent World Model (LaWM). We obtain LaWM by training a latent action model
                                                  in the latent space of a pretrained vision foundation model and repurposing its
                                                  forward decoder to predict future observation features for scene evolution. LaWAM
                                                  then conditions action generation on these predicted latent visual subgoals to enable
                                                  dynamics-aware robot control. LaWAM achieves state-of-the-art or competitive
                                                  success rates (SRs) across LIBERO (98.6% SR), RoboTwin (91.22% SR), and
                                                  real-world manipulation tasks while retaining low-latency inference. LaWAM runs
                                                  in 187 ms per action-chunk prediction and achieves up to 24× lower wall-clock
                                                  latency than pixel-space WAMs.

                                                  Keywords: Robot Manipulation, World Action Models, Latent World Models

                                         1    Introduction
                                         Vision-Language-Action models (VLAs) [1, 2, 3,
                                         4, 5] have recently shown strong performance on
                                                                                                                  24.0× lower latency
                                         robotic manipulation by transferring large-scale
                                         vision-language pretraining into action generation.
                                         However, most current VLAs predict actions pri-
                                         marily from the current visual-language context,
                                         without explicitly modeling how the scene evolves
                                         under candidate actions [6, 7].
                                         World-Action Models (WAMs) [8, 9, 10, 11, 12,
                                         13, 14] offer a natural way to introduce temporal
                                         dynamics by augmenting policies with predicted
                                         future observations or states as additional context.
                                         However, current WAMs remain inefficient for Figure 1: Latency–success trade-off on LIBERO.
                                         manipulation policies. First, many methods pre- Latency for 10 denoising steps on an A100 GPU ver-
                                                                                               sus LIBERO success rate. The marker area denotes
                                         dict future images or videos, allocating substantial model size; the pink sector denotes world-modeling
                                         modeling capacity to pixel-level synthesis rather parameters.
                                         than compact action-relevant dynamics. Second,
                                         iterative future generation introduces considerable inference latency; under the same evaluation setup
                                         used in Fig. 1, LingBot-VA [12] requires 4482 ms for a single policy inference, whereas the represen-
                                         tative VLA π0.5 [15] requires only 220 ms. Third, effective future prediction for manipulation should
```

## P13 — Latent-WAM.pdf

PDF 第 1 页；共 29 页；SHA256 `b02d50f25f82296cc364859aa883aadd5edce0156a175f168b26112a18253429`。

```text
                                           Latent-WAM: Latent World Action Modeling for
                                                  End-to-End Autonomous Driving

                                          Linbo Wang1,2,5⋆ , Yupeng Zheng1⋆⋆ , Qiang Chen2 , Shiwei Li2 , Yichen Zhang1 ,
                                                 Zebin Xing1,3 , Qichao Zhang1,3⋆ ⋆ ⋆ , Xiang Li4 , Deheng Qian2 ,
                                              Pengxuan Yang1 , Yihang Dong5 , Ce Hao5 , Xiaoqing Ye2 , Junyu Han2 ,
                                                               Yifeng Pan2 , and Dongbin Zhao1,3,5
arXiv:2603.24581v1 [cs.CV] 25 Mar 2026




                                                            1
                                                              Institute of Automation, Chinese Academy of Sciences
                                                                  2
                                                                     Chongqing Chang’an Technology Co., Ltd
                                                3
                                                    School of Artificial Intelligence, University of Chinese Academy of Sciences
                                                                       4
                                                                          College of AI, Tsinghua University
                                                                             5
                                                                                Zhongguancun Academy



                                                    Abstract. We introduce Latent-WAM, an efficient end-to-end autonomous
                                                    driving framework that achieves strong trajectory planning through spatially-
                                                    aware and dynamics-informed latent world representations. Existing world-
                                                    model-based planners suffer from inadequately compressed representa-
                                                    tions, limited spatial understanding, and underutilized temporal dynam-
                                                    ics, resulting in sub-optimal planning under constrained data and com-
                                                    pute budgets. Latent-WAM addresses these limitations with two core
                                                    modules: a Spatial-Aware Compressive World Encoder (SCWE) that
                                                    distills geometric knowledge from a foundation model and compresses
                                                    multi-view images into compact scene tokens via learnable queries, and
                                                    a Dynamic Latent World Model (DLWM) that employs a causal Trans-
                                                    former to autoregressively predict future world status conditioned on
                                                    historical visual and motion representations. Extensive experiments on
                                                    NAVSIM v2 and HUGSIM demonstrate new state-of-the-art results: 89.3
                                                    EPDMS on NAVSIM v2 and 28.9 HD-Score on HUGSIM, surpassing the
                                                    best prior perception-free method by 3.2 EPDMS with significantly less
                                                    training data and a compact 104M-parameter model.


                                          Keywords: Autonomous Driving · Latent World Action Model · Scene Repre-
                                          sentation


                                          1      Introduction
                                          End-to-end autonomous driving has attracted considerable attention due to its
                                          data-driven nature and scalability. Prior methods integrate perception and pre-
                                          diction into a unified differentiable network, extracting planning-relevant repre-
                                          sentations for end-to-end trajectory prediction [13, 15]. However, these methods
                                           ⋆
                                               Work done during internship at Chongqing Chang’an Technology Co., Ltd.
                                          ⋆⋆
                                               Project Leader & Equal Contribution
                                         ⋆⋆⋆
                                               Corresponding author.
```

## P14 — MV-WAM.pdf

PDF 第 1 页；共 20 页；SHA256 `8360e2e4fabb4fed2c537f61da2d3ebb08759caed47869f63839f089bdc8b463`。

```text
                                              MV-WAM: Manifold-Aware World Action Model
                                                      with Value Augmentation


                                              Jintao Chen1,2 ∗ , Peidong Jia1,2,∗,† , Qingpo Wuwu1,2,∗ , Jiaming Liu2 , Mengfei Du2 ,
                                               Chun-Kai Fan1,2 , Xiaowei Chi1,2 , Hao Chen2 , Chengyu Bai1,2 , Zezhong Qian1,2 ,
                                            Hao Wang1,2 , Jiajun Cao1,2 , Weishi Mi2 , Xiaozhu Ju2 , Jian Tang2 , Shanghang Zhang1 ,B
arXiv:2606.21088v1 [cs.RO] 19 Jun 2026




                                                              a) Previous works                                                                               c) Simulation
                                                                                                                                           RoboTwin 2.0 - Clean                RoboTwin 2.0 - Random
                                                                                                  rollouts                                    S.R. (50 tasks)                      S.R. (50 tasks)
                                                                    Action Generation                                             90                               84     60                               55.7
                                                                                                                                                     80.5
                                                                                                                                  80                        75.3          50
                                                                                                  rollouts                        70                                      40
                                                                                                                                                                                           26.4
                                                                                                                                  60          52.9                        30                      20.9
                                                                    𝑎       𝑎     ··· 𝑎                                                46.4                                    16.3 15.2
                                             instruction                                                                          50                                      20
                                                                                                                                  40                                      10
                                                              b) Our MV-WAM                                                            𝜋         UP-VLA            HALO         BagelVLA          MV-WAM

                                                                        Video Expert
                                                                                                  rollouts                                                   d) Real-World
                                                                                                                                                                                   Real World S.R
                                                                                          helps                                                                           90                             77.5
                                                                                                  rollouts                                                                70
                                                                   Action-Value Expert                                                                                           42.5
                                                                                                                                                                          50
                                             instruction                                                                                                                                   32.5
                                                                                                                                                                          30
                                                                                                                       𝑎′Worse!
                                               Action Token    𝑎        𝑎       ··· 𝑎        𝑉                                                                            10
                                                                                                                 𝑎 Better!
                                                                                                                Action Manifold
                                               Value Token         resample             rollout                                        𝜋                                RDT                       MV-WAM
                                                                                                     Manifold-aware loss target


                                         Figure 1: Overview of MV-WAM. We introduce asymmetric video-action experts with a causal
                                         mask to condition actions on visual dynamics. By coupling world modeling, action prediction, and
                                         progress-value estimation, it achieves strong performance in (c) simulation and (d) real-world tasks.

                                                                                                                  Abstract
                                                       Achieving robust and generalizable manipulation across diverse environments re-
                                                       mains a fundamental challenge in embodied robotics. Recent world action models
                                                       achieve strong in-domain performance, yet their gains do not extend proportion-
                                                       ally to out-of-distribution scenarios. We attribute this to a structural mismatch be-
                                                       tween visual and action modalities, whose intrinsically heterogeneous manifolds
                                                       cause joint optimization to disproportionately degrade action robustness under dis-
                                                       tribution shift. To address this, we propose MV-WAM, a novel end-to-end frame-
                                                       work that jointly models visual prediction, action generation, and value estimation
                                                       designed to effectively leverage video priors during both training and inference for
                                                       enhanced action generalization. Key to this unification is a cross-modality causal
                                                       mask that hierarchically grounds actions in predicted video frames and value func-
                                                       tion tokens in both modalities. To further narrow the generalization gap, MV-
                                                       WAM adopts a manifold-aware optimization scheme that explicitly accounts for
                                                       the structural heterogeneity across modalities. Finally, MV-WAM introduces a
                                                       progress-value regulation mechanism that estimates task completion and detects
                                                       misalignment between predicted frames and generated actions, enabling the policy
                                                       to autonomously identify execution deviations and recover through value-guided
                                                       rollback. On the RoboTwin simulation, MV-WAM achieves a 55.7% mean suc-
                                                       cess rate on random scenarios without any randomized action supervision, out-
                                            ∗ Equal Contribution. † Project Leader. 1 State Key Laboratory of Multimedia Information Processing, School of Com-

                                         puter Science, Peking University. 2 Beijing Innovation Center of Humanoid Robotics. B Corresponding author.


                                         Preprint.
```

## P15 — MolmoAct2.pdf

PDF 第 1 页；共 51 页；SHA256 `3029bc601e4d71ebc2e45ce996cbe35649b24eeb26b5925a73c83cb83e7fadad`。

```text
                                        MolmoAct2
                                        Action Reasoning Models for Real-World Deployment
                                        Haoquan Fang               Jiafei Duan
                                                          ♥1,2∗                  ♥1,2,3∗


                                        Donovan Clay              Sam Wang          Shuo Liu     Weikai Huang                     Xiang Fan           Wei-Chuan Tsai
                                                         ♥1,2              ♥1,4                 ♥2                    ♥1,2                 ♥1,2                        ♥2

                                        Shirui Chen             Yi Ru Wang         Shanli Xing
                                                    ♥1,2                  ♥1,2                ♥2


                                        JaeminCho            JaeSungPark AinazEftekhar    PeterSushko KarenFarley AngadWadhwa
                                                     1,2,5                  1                        1,2                      1                   1                     2

                                        Cole Harrison        Winson Han Ying-Chun Lee Eli VanderBilt Rose Hendrix Suveen Ellawela
                                                      6                1             2              1            1                7

                                        Lucas Ngoo
                                                    7
arXiv:2605.02881v2 [cs.RO] 8 May 2026




                                        Joyce Chai       Zhongzheng Ren                    Ali Farhadi          Dieter Fox           Ranjay Krishna
                                                     8                       ♥1,2,9                      ♥1,2                ♥1,2                        ♥1,2



                                        1                           2                                3                                            4
                                         Allen Institute for AI, University of Washington, National University of Singapore, University of Penn-
                                                 5                            6        7          8                         9
                                        sylvania, Johns Hopkins University, Amazon, Cortex AI, University of Michigan, University of North
                                        Carolina at Chapel Hill

                                        * denotes equal contribution in no particular order. ♥ marks core contributors. See full author contributions here.

                                            MolmoAct2: MolmoAct2-Finetune              MolmoAct2         MolmoAct2-Think          MolmoAct2-Pretrain       Molmo2-ER
                                            MolmoAct2 Data: MolmoAct2-BimanualYAM                 MolmoAct2-SO100/101             MolmoAct2-DROID       Molmo2-ER
                                            Code: allenai/molmoact2
                                            Blog: allenai.org/blog/molmoact2




                                            Abstract
                                            Vision-Language-Action (VLA) models aim to provide a single generalist controller for robots, but
                                            today’s systems fall short for real-world deployment. Frontier models are closed; open-weight alterna-
                                            tives are tied to expensive hardware; reasoning-augmented policies pay prohibitive latency for their
                                            grounding; and fine-tuned success rates remain below the threshold for dependable use. We present
                                            MolmoAct2, a fully open action reasoning model built for practical deployment, advancing its
                                            predecessor, MolmoAct along five axes. (1) MolmoAct2 is built on top of our new Molmo2-ER, a
                                            VLM backbone specialized for spatial and embodied reasoning, trained on a 3.3M-sample corpus with
                                            a specialize-then-rehearse recipe. (2) We release three new robot datasets spanning low-to-medium
                                            cost platforms: MolmoAct2-BimanualYAM Dataset, 720 hours of teleoperated bimanual trajec-
                                            tories that constitute the largest open bimanual dataset to date; MolmoAct2-DROID Dataset, a
                                            quality-filtered Franka subset of DROID; and MolmoAct2-SO100/101 Dataset, a quality-filtered
                                            SO-100/101 subset. (3) We train and release MolmoAct2-FAST Tokenizer, an open-weight,
                                            open-data action tokenizer trained on millions of trajectories across five embodiments. (4) We design a
                                            new VLA architecture to graft the discrete-token VLM into the flow-matching continuous-action expert
                                            via per-layer key-value (KV) conditioning. (5) we propose MolmoAct2-Think, an adaptive-depth
                                            reasoning variant that re-predicts depth tokens only for scene regions that change between timesteps,
                                            retaining geometric grounding at a fraction of prior latency. In the most extensive empirical study of
                                            any open VLA to date, spanning 7 simulation and real-world benchmarks, MolmoAct2 outperforms
                                            strong baselines including π0.5 , while Molmo2-ER surpasses GPT-5 and Gemini Robotics ER-1.5
                                            across 13 embodied-reasoning benchmarks. We release model weights, training code, and complete
                                            training data.




                                                                                                         1
```

## P16 — MotionEnhancer.pdf

PDF 第 1 页；共 26 页；SHA256 `44591e42257efe7ec2e40474fbba07f17682e6be72de7828b4ccc414ea9d5266`。

```text
                                                  MotionEnhancer: Leveraging Video Diffusion for Motion-Enhanced
                                                                    Vision-Language Models

                                               Yifan Xu1,2 , Chao Zhang2 *, Ruifei Ma2 , Fei Gao2 , Zhifei Yang3 , Jiaxing Qi1 , Zhipeng Chen4
                                                            1
                                                              School of Computer Science and Engineering, Beihang University
                                                                   2
                                                                     Beijing Digital Native Digital City Research Center
                                                                      3
                                                                        School of Computer Science, Peking University
                                                 4
                                                   School of Artificial Intelligence, Beijing University of Posts and Telecommunications
arXiv:2606.06853v1 [cs.CV] 5 Jun 2026




                                                                          yifan xu@buaa.edu.cn, ariczhang2009@gmail.com
                                                                          https://motion-enhancer.github.io/


                                                                     Abstract
                                                                                                          A man is kneeling                                                      Can you describe
                                                                                                          on a surfboard.                                                        this video?
                                            The new era has witnessed a remarkable capability to
                                                                                                                                          MotionEnhancer
                                        extend Vision-Language Models (VLMs) for tackling tasks                                            Motion-centric
                                                                                                                  VDM                   Attention Refinement                      VLM
                                        of video understanding. While current VLMs excel at                                                                      Guide
                                        event- or story-level understanding, their ability to cap-                                                         Motion Prior
                                        ture fine-grained motion details remains limited, primar-                                                            + SFT

                                        ily due to their focus on high-level static semantic struc-
                                        tures and macro-event logic. In contrast, Video Diffusion                                                               Standard
                                                                                                                                                               SFT Result
                                        Models (VDMs) are adept at modeling dynamic motion pat-
                                                                                                            VDM T2V Attention               Motion Prior                       VLM T2V Attention
                                        terns, benefiting from large-scale video data and the intrin-
                                                                                                                           (A) High-level Overview of MotionEnhancer
                                        sic requirement of temporal generation. In this paper, we
                                        introduce MotionEnhancer, a novel approach that lever-                                                    VDM
                                                                                                                  Distinctions Between                            Distinctions Between
                                                                                                                    Different Heads                              Different Text Tokens
                                        ages motion priors distilled from a powerful video diffu-
                                        sion model as auxiliary supervision to enhance the mo-
                                        tion understanding capability of a VLM via attention align-
                                                                                                          head1    head2        head3     head4            man            is     kneeling surfboard
                                        ment. MotionEnhancer comprises two simple parameter-
                                                                                                                                 (B) Observation on VDM Attention
                                        free modules, Motion-sensitive Head Selection (MHS) and
                                        Motion-salient Text Token Identification (MTTI), to directly    Figure 1. (A) High-level overview of MotionEnhancer, which
                                        extract and optimize motion-related attentions from the         incorporates motion priors from the VDM as guidance during su-
                                        VDM in a computation-only manner. MotionEnhancer pro-           pervised fine-tuning of the VLM for improved motion understand-
                                        vides a scalable solution for motion understanding with-        ing. (B) Observation of VDM attention. We observe distinct
                                        out additional training parameters, modifications to ex-        patterns in the attention maps across different transformer heads
                                        isting architectures, or tool calling. Extensive experi-        and text tokens in the VDM, which motivates our refinement of
                                        ments demonstrate that MotionEnhancer can achieve con-          motion-centric attention.
                                        sistent improvements over state-of-the-art VLMs on two
                                        motion-level video understanding benchmarks, especially
                                        on motion-related metrics.
                                                                                                        advancing tasks like video captioning and question answer-
                                                                                                        ing through multimodal alignment and semantic reasoning
                                                                                                        [1, 4, 5, 12, 27, 34]. Unlike static images, videos contain se-
                                        1. Introduction                                                 quential frames that reflect scene dynamics over time. The
                                        In recent years, Vision-Language Models (VLMs) have be-         temporal relationships among these frames reveal how ob-
                                        come the mainstream framework for video understanding,          jects move, interact, and transform over time. Effectively
                                                                                                        modeling these temporal relationships is crucial for cap-
                                          * Corresponding   author                                      turing object movement and interaction. Therefore, VLMs
```

## P17 — OREO.pdf

PDF 第 1 页；共 8 页；SHA256 `ddbebe28ff10cc52dd9d1bbb9d20eedd14f57cde5c48c9308d9e6df81d0bf305`。

```text
                                         OREOS: Oriented Recognition of 3D Point Clouds in Outdoor Scenarios
                                                       Lukas Schaupp1 , Mathias Bürki1,2 , Renaud Dubé1,2 , Roland Siegwart1 , Cesar Cadena1


                                            Abstract— We introduce a novel method for oriented place
                                         recognition with 3D LiDAR scans. A Convolutional Neural
                                         Network is trained to extract compact descriptors from single
                                         3D LiDAR scans. These can be used both to retrieve near-by
                                         place candidates from a map, and to estimate the yaw dis-
                                         crepancy needed for bootstrapping local registration methods.
arXiv:1903.07918v1 [cs.RO] 19 Mar 2019




                                         We employ a triplet loss function for training and use a hard-
                                         negative mining strategy to further increase the performance
                                         of our descriptor extractor. In an evaluation on the NCLT and
                                         KITTI datasets, we demonstrate that our method outperforms
                                         related state-of-the-art approaches based on both data-driven
                                         and handcrafted data representation in challenging long-term
                                         outdoor conditions.

                                                                  I. I NTRODUCTION
                                            Global localization constitutes a pivotal component for
                                         many autonomous mobile robotics applications. It is a re-
                                         quirement for bootstrapping local localization algorithms                     Fig. 1: We aim at accurately localizing our vehicle in a map build from
                                                                                                                       previously collected LiDAR point clouds. A query point cloud scan is fed
                                         and for re-localizing robots after temporarily leaving the                    through the OREOS pipeline, yielding a compact descriptor that allows to
                                         mapped area. Global localization can furthermore be used                      retrieve near-by place candidates from the map, and estimate the yaw angle
                                         for mitigating pose estimation drift through loop-closure                     discrepancy. With this information, a local registration method, such as ICP,
                                                                                                                       can be bootstrapped and used for subsequent high accuracy localization
                                         detection and for merging mapping data collected during                       along the traversal.
                                         different sessions. Prior-free localization is especially chal-
                                         lenging for autonomous vehicles in urban environments, as                     clouds which allows for long-term 3 DoF metric global lo-
                                         GNSS-based localization systems fail to provide reliable and                  calization in outdoor environments. Specifically, our method
                                         precise localization near buildings due to multi-path effects,                allows us to estimate the relative orientation between scans.
                                         or in tunnels or parking garages due to a lack of satellite                   Our novel data-driven metric global localization descriptor
                                         signal reception. Due to their rich and descriptive information               is fast to compute and robust with respect to long-term
                                         content, camera images have been of great interest for place                  appearance changes of the environment, and shows similar
                                         recognition, with mature and efficient data representations                   place recognition performance compared to other state-of-
                                         and feature descriptors evolving in recent years. However,                    the-art LiDAR place recognition approaches. Additionally
                                         visual place recognition algorithms struggle to cope with                     our architecture provides orientation descriptors capable of
                                         strong appearance changes that commonly occur during long-                    predicting a yaw angle estimation between two point clouds
                                         term applications in outdoor environments, and fail under                     realizations of the same place. Our contributions can be
                                         certain ill-lighted conditions [1]. In contrast to that, active               summarized as follows:
                                         sensing modalities, such as LiDAR sensors, are mainly un-
                                                                                                                          •   We present OREOS: an efficient data-driven architecture
                                         affected by appearance change [2]. Efficient and descriptive
                                                                                                                              for extracting a point cloud descriptor that can be used
                                         data representations for place recognition using LiDAR point
                                                                                                                              both for place recognition purposes and for regressing
                                         clouds remain, however, an open research question [3]–[5].
                                                                                                                              the relative orientation between point clouds.
                                         In contrast to our work, typical place recognition methods do
                                                                                                                          •   In an evaluation using two public dataset collections, we
                                         not always explicitly deal with the full problem of estimating
                                                                                                                              demonstrate the capability of our approach to reliably
                                         a 3 DoF transformation [6] [7] [8].
                                                                                                                              localize in challenging outdoor environments across
                                            This paper addresses the aforementioned issue by pre-
                                                                                                                              seasonal and weather changes over the course of more
                                         senting a data-driven descriptor for sparse 3D LiDAR point
                                                                                                                              than a year. We show that our approach works even
                                                                                                                              under strong point cloud misalignment, allowing the
                                           This research has received funding from the EU H2020 research project
                                         under grant agreement No 688652, the Swiss State Secretariat for Education,          arbitrary positioning of a robot.
                                         Research and Innovation (SERI) No 15.0284                                        •   A computational performance analysis showing that our
                                           1 Autonomous Systems Lab (ASL), ETH Zürich, Switzerland
                                                                                                                              proposed algorithm exhibits real-time capabilities and
                                         {firstname.lastname}@ethz.ch
                                           2 Sevensense            Robotics            AG,              Switzerland           performs similarly to other state-of-the-art approaches
                                         {firstname.lastname}@sevensense.ch                                                   in place recognition performance while providing ro-
```

## P18 — PosA-VLA.pdf

PDF 第 1 页；共 19 页；SHA256 `19d4ebceb3baa2b9e8e4e5734e2992932f4c12ecb5231ad63d1be5ad638e5b3f`。

```text
                                              PosA-VLA: Enhancing Action Generation via Pose-Conditioned Anchor
                                                                         Attention

                                           Ziwen Li1    Xin Wang2   Hanlue Zhang1    Runnan Chen3                                         Runqi Lin3       Xiao He2
                                                            2                 2
                                                   Han Huang      Yandong Guo      Fakhri Karray1                                         Tongliang Liu1,3

                                                                           Mingming Gong1,4
                                               1                2                    3                                       4
                                                   MBZUAI           AI2 Robotics         The University of Sydney                The University of Melbourne
arXiv:2512.03724v2 [cs.CV] 8 Dec 2025




                                                                Abstract

                                            The Vision-Language-Action (VLA) models have demon-
                                        strated remarkable performance on embodied tasks and
                                        shown promising potential for real-world applications.                          (a) Start frame                (b) GT end frame
                                        However, current VLAs fail to exhibit consistent and pre-
                                        cise target-oriented actions, as they often generate redun-
                                        dant motions along trajectories, limiting their applicabil-
                                        ity in time-sensitive scenarios. In this work, we attribute
                                        the redundant actions to the spatially uniform perception
                                        field of VLAs, which leads them to be distracted by target-
                                        irrelevant objects, particularly in complex environments. To
                                        this end, we propose an efficient PosA-VLA framework,
                                        which anchors visual attention via pose-conditioned super-
                                        vision, consistently guiding the model’s perception toward
                                        task-relevant regions. The pose-conditioned anchor atten-
                                        tion mechanism enables the model to better align instruc-           Figure 1. Quantitative analysis of the grasping task (pick up the
                                        tion semantics with actionable visual cues, thereby enhanc-         bread). Top: the initial scene (left) and the ground-truth grasp-
                                        ing action generation precision and efficiency. Moreover,           ing moment captured from human teleoperation (right). Bottom:
                                                                                                            distance between the robot end-effector and the ground-truth grasp
                                        our framework is built upon a lightweight architecture and
                                                                                                            point over time; the light-blue area denotes the successful grasping
                                        requires no auxiliary perception modules (e.g., segmenta-
                                                                                                            range. Our PosA-VLA reaches the grasping region faster and more
                                        tion or grounding networks), ensuring efficient inference.          accurately, while DexGraspVLA and π0 eventually succeed but
                                        Extensive experiments verify that our method performs em-           require longer execution time. In contrast, OpenVLA and Smol-
                                        bodied tasks with precise and time-efficient execution across       VLA fail to reach the successful grasping range.
                                        diverse robotic manipulation benchmarks and shows robust
                                                                                                            lation tasks. By equipping vision-language understanding
                                        generalization in various environments.
                                                                                                            models with the ability to act within physical environments,
                                                                                                            VLAs represent one of the most promising pathways toward
                                                                                                            achieving embodied AI.
                                        1. Introduction
                                                                                                                However, these models still struggle to generate consis-
                                        Despite remarkable progress in artificial intelligence (AI)         tent and precise actions in real-world applications, which
                                        across various domains, a considerable gap still exists be-         hinders their ability to handle time-sensitive and accuracy-
                                        tween its computational intelligence and real-world phys-           critical tasks. In practice, their motion trajectories often
                                        ical interaction.    Recently, the emergence of vision-             exhibit redundant or even unstable behaviors, showing in-
                                        language-action (VLA) models [4, 6, 7, 13, 17, 22, 25–              consistencies in action generation and insufficient control
                                        27, 33, 49, 54, 58, 59, 62] has aimed to bridge this gap by         precision during execution.
                                        enabling robots to perceive visual scenes, interpret natural            As illustrated in Figure 1, current VLA models [4, 23,
                                        language instructions, and execute corresponding manipu-            44, 59] exhibit considerable fluctuating distance curves be-


                                                                                                        1
```

## P19 — ReVLA.pdf

PDF 第 1 页；共 7 页；SHA256 `70e1d28ee886dfb0e7c78481e9008f34339d9a17c0bcfc210c07d59b4507ed33`。

```text
                                                                  ReVLA: Reverting Visual Domain Limitation
                                                                       of Robotic Foundation Models
                                                       Sombit Dey1 , Jan-Nico Zaech1 , Nikolay Nikolov1 , Luc Van Gool1 , Danda Pani Paudel1

                                                                                                                         0.6                                                      0.6
                                                                                                                                              openVLA                                                     openVLA      ReVLA
                                             Abstract— Recent progress in large language models and
                                         access to large-scale robotic datasets has sparked a paradigm




                                                                                                              Success Rate




                                                                                                                                                                       Success Rate
                                                                                                                         0.4                                                      0.4
                                         shift in robotics models transforming them into generalists
                                         able to adapt to various tasks, scenes, and robot modalities.                   0.2                                                          0.2
                                         A large step for the community are open Vision Language
arXiv:2409.15250v3 [cs.CV] 20 May 2025




                                         Action models which showcase strong performance in a wide                       0.0                                                          0.0
                                                                                                                               Indomain    OOD Objects Distractors                          Indomain   OOD Objects     Distractors
                                         variety of tasks. In this work, we study the visual generalization
                                         capabilities of three existing robotic foundation models, and
                                         propose a corresponding evaluation framework.
                                             Our study shows that the existing models do not exhibit
                                                                                                                                DIN   O-v2       Sig    LIP                                     DINO-v2             SigLIP
                                         robustness to visual out-of-domain scenarios. This is poten-
                                         tially caused by limited variations in the training data and/or                                                         Visual Encoder
                                                                                                                                          Llama 7B                  Reversal                              Llama 7B
                                         catastrophic forgetting, leading to domain limitations in the
                                         vision foundation models. We further explore OpenVLA, which
                                         uses two pre-trained vision foundation models and is, therefore,     Fig. 1: The OpenVLA model tuned on fractal data struggles
                                         expected to generalize to out-of-domain experiments. However,        with out-of-domain objects and in the presence of distractors
                                         we showcase catastrophic forgetting by DINO-v2 in OpenVLA            (left), due to catastrophic forgetting in its DINO-v2 and
                                         through its failure to fulfill the task of depth regression.         SigLIP vision encoders. Our ReVLA (DS gradual) model
                                             To overcome the aforementioned issue of visual catastrophic
                                         forgetting, we propose a gradual backbone reversal approach          addresses this by reverting the vision encoders to their
                                         founded on model merging. This enables OpenVLA – which               original pre-trained weights, leading to improved overall
                                         requires the adaptation of the visual backbones during initial       performance across all three settings (right).
                                         training – to regain its visual generalization ability. Regaining
                                         this capability enables our ReVLA model to improve over
                                         OpenVLA by a factor of 77% and 66% for grasping and
                                         lifting in visual OOD tasks. Comprehensive evaluations, episode      as those in OpenVLA [5] require adapting not only the
                                         rollouts and model weights are available on the ReVLA Page           LLM-based reasoning model but also the vision backbones to
                                                                                                              achieve high performance on in-domain tasks. Although pre-
                                                              I. INTRODUCTION
                                                                                                              trained vision models themselves are known for generalizing
                                            Generalist robot foundation models based on vision-               well across diverse settings, their adaptation during the
                                         language models (VLMs) offer a promising path toward                 training of robotic foundation models can lead to visual
                                         developing dexterous policies that generalize across tasks,          domain limitations due to catastrophic forgetting.
                                         embodiments, and environments. This expectation is pri-                 To quantify the impact of this training approach and assess
                                         marily based on the capabilities of the underlying large             the overall generalization capability of robotic foundation
                                         language models (LLMs) like Llama [1] that have been                 models, we develop a realistic out-of-domain evaluation
                                         demonstrated in a range of real-world applications. VLMs             framework based on the SIMPLER [6] environment. This
                                         equip LLMs with foundational vision encoders, such as                evaluation framework includes challenging visual tasks using
                                         DINO [2], CLIP [3] or SigLIP [4], enabling them to perform           OOD objects and visual distractors. We use it to evaluate
                                         complex tasks that require perception and reasoning. Train-          the visual generalization performance of three recent open-
                                         ing of robot foundation models – which output the robot’s            source robotic foundation models: (1) RT-1 [7], which pi-
                                         action trajectories – requires training on large amounts of          oneered end-to-end generalist policies and employs an Effi-
                                         robot-specific data. Doing so adequately is a delicate task,         cientNet backbone alongside a transformer model for predict-
                                         which we study in this paper while focusing on the visual            ing low-level robot control commands; (2) Octo [8], which
                                         generalization capabilities in the out-of-domain (OOD) sce-          is built on a diffusion policy integrated with a transformer
                                         narios.                                                              model; and (3) OpenVLA [5], the first publicly available
                                            We begin by observing that most demonstrations used for           vision-language-action model for robot control, built on the
                                         training robotic foundation models are acquired in controlled        Prismatic [9] VLM that combines Llama [1] with DINO-v2
                                         lab settings and prioritize trajectory variability rather than       and SigLIP. In all three cases, the vision encoders are either
                                         comprehensive coverage of all possible interaction objects           trained or fine-tuned on visually limited robotics data.
                                         and environments. Simultaneously, training protocols such               In our evaluations, we observe that all three models
                                           1 INSAIT,Sofia University “St. Kliment Ohridski”, Bulgaria         exhibit limited visual generalization performance. However,
                                           Corresponding Author: jan-nico.zaech@insait.ai                     OpenVLA shows strong potential due to its powerful VLM
```

## P20 — Robust-WAM.pdf

PDF 第 1 页；共 13 页；SHA256 `3fb998b1986c4e2fa8fc175de63a33f982d51ca85df8b71692219284ca114651`。

```text
                                            Robust-WAM: Bridging Generative Pretraining and
                                            Semantic Foresight in World-Action Models
                                            Haodong Yan1,∗,† , Junfeng Li1,∗ , Junjie He1,∗ , Zhide Zhong1 , MingMing Yu2 , Wenxuan Song1 , Jiaguan Zhu1 ,
                                            Yangyang Zheng1 , Yuqiao Du1 , Jiadi You1 , Yingjie CAI3 , Xu Yan3 , Guanyi Zhao3 , Bingbing Liu3 , Haoang Li1,‡
                                            1
                                              The Hong Kong University of Science and Technology (Guangzhou), Guangzhou, China, 2 Beihang
                                            University, Beijing, China, 3 Huawei Foundation Model Department
                                            ∗
                                              Equal contribution, † Project Leader, ‡ Corresponding author.

                                            Mainstream World-Action Models (WAMs) adapt pretrained video generation models (VGMs) for
                                            robot control, transferring their learned dynamics prior for action prediction. These VGMs are
                                            typically trained in a variational autoencoder (VAE) latent space. However, the VAE latent space
                                            is optimized for pixel reconstruction, which rewards fine appearance detail and leaves the action
                                            prediction fragile under visual shifts. Recent works build WAMs in semantic latent space, which
                                            are more robust to appearance shifts. However, these models cannot leverage the large-scale VGM
                                            pretraining that exists only in VAE space. To overcome this dilemma, we propose Robust-WAM,
arXiv:2608.05903v2 [cs.CV] 7 Aug 2026




                                            a general post-training method for video-generation-based WAMs that preserves the VAE-based
                                            generative path and adds a lightweight semantic foresight alignment objective on the action stream.
                                            This retains the large-scale VGM pretraining while grounding actions in appearance-invariant dynamics
                                            that stay reliable under illumination shifts and other visual out-of-distribution conditions. Specifically,
                                            we employ learnable query tokens to bring future-scene semantics into the action stream by aligning
                                            their output hidden states with the semantic foresight of future ground-truth frames. To establish the
                                            temporal correspondence between each query and the future step it describes, we give it the positional
                                            encoding of the matching action tokens. Experiments on out-of-distribution generalization simulation
                                            benchmarks and a real-robot setup show that our Robust-WAM consistently improves the success
                                            rates of multiple WAM baselines without sacrificing in-distribution performance.

                                            Date: August 2026
                                            Project Page: https://haodong-yan.github.io/robust-wam-project-page/




                                        1   Introduction
                                        World-Action Models (WAMs) have recently emerged as a powerful paradigm for robot manipulation (Wu
                                        et al., 2024; Cheang et al., 2024; Guo et al., 2024; Bi et al., 2026; Li et al., 2026; Kim et al., 2026; Ye et al.,
                                        2026b; Yuan et al., 2026; Zhong et al., 2026, 2025; Song et al., 2025; Yan et al., 2026, 2025). They harness the
                                        dynamics priors of pretrained video generation models (VGMs) to ground action generation. These priors,
                                        learned from web-scale video, capture how the world evolves: how objects move, make contact, and interact
                                        over time (Team Wan et al., 2025; NVIDIA et al., 2025).
                                        Although the dynamics priors from large-scale pretrained VGMs are a great help to action generation, they are
                                        learned in a variational-autoencoder (VAE) latent space trained for pixel reconstruction. This objective forces
                                        the space to preserve realistic appearance, such as texture and illumination. A large share of the priors is
                                        therefore devoted to appearance detail that is irrelevant to actions. Inheriting this appearance bias, the action
                                        stream is easily disrupted under visual out-of-distribution conditions, such as photometric or hue changes (Fei
                                        et al., 2026; Zhang et al., 2026). To overcome this limitation, recent methods (Zhou et al., 2025; Lyu et al.,
                                        2026; Chen et al., 2026a) rebuild visual generation in a semantic latent space, such as DINO (Siméoni et al.,
                                        2025) or V-JEPA (Assran et al., 2025). These models are visually robust, but they abandon the strong
                                        dynamics priors of large-scale pretrained VGMs; regaining that knowledge requires costly re-pretraining on
                                        large-scale video data (Assran et al., 2025). This leaves WAMs facing a fundamental trade-off (illustrated in
                                        Figure 1): VAE-based WAMs inherit the strong dynamics priors of large-scale VGM pretraining but remain
                                        fragile to appearance changes, whereas semantic-latent WAMs are robust to appearance changes but cannot
                                        leverage that large-scale pretraining.
                                                                                                1
```

## P21 — Shortcut Learning in Generalist Robot Policies.pdf

PDF 第 1 页；共 29 页；SHA256 `cfee466085db8fc63c6dd1b23f6c5dbf69589c27ab0195dc50ff877c9827f323`。

```text
                                            Shortcut Learning in Generalist Robot Policies: The
                                               Role of Dataset Diversity and Fragmentation
                                            Youguang Xing1∗ , Xu Luo1∗ , Junlin Xie1 , Lianli Gao1 , Hengtao Shen2 , Jingkuan Song2†
                                                                          1
                                                                            UESTC, 2 Tongji University
                                                        ygxing@std.uestc.edu.cn, frank.luox@outlook.com



                                                    Abstract: Generalist robot policies trained on large-scale datasets such as Open
                                                    X-Embodiment (OXE) demonstrate strong performance across a wide range of
arXiv:2508.06426v1 [cs.RO] 8 Aug 2025




                                                    tasks. However, they often struggle to generalize beyond the distribution of their
                                                    training data. In this paper, we investigate the underlying cause of this limited
                                                    generalization capability. We identify shortcut learning—the reliance on task-
                                                    irrelevant features—as a key impediment to generalization. Through comprehen-
                                                    sive theoretical and empirical analysis, we uncover two primary contributors to
                                                    shortcut learning: (1) limited diversity within individual sub-datasets, and (2)
                                                    significant distributional disparities across sub-datasets, leading to dataset frag-
                                                    mentation. These issues arise from the inherent structure of large-scale datasets
                                                    like OXE, which are typically composed of multiple sub-datasets collected inde-
                                                    pendently across varied environments and embodiments. Our findings provide
                                                    critical insights into dataset collection strategies that can reduce shortcut learn-
                                                    ing and enhance the generalization ability of generalist robot policies. Moreover,
                                                    in scenarios where acquiring new large-scale data is impractical, we demonstrate
                                                    that carefully selected robotic data augmentation strategies can effectively reduce
                                                    shortcut learning in existing offline datasets, thereby improving generalization ca-
                                                    pabilities of generalist robot policies, e.g., π0 , in both simulation and real-world
                                                    environments. More information at our website1 .

                                                    Keywords: Generalist Robot Policies, Shortcut Learning, Large-Scale Robot
                                                    Datasets

                                        1       Introduction
                                        The recent advancements in machine learning, particularly in domains such as computer vision and
                                        natural language processing, can be largely attributed to the scaling up of both data and model
                                        sizes. Notably, scaling laws in these domains [1, 2, 3] indicate a consistent trend of performance
                                        improvement and emergent generalization capabilities as the number of model parameters and the
                                        volume of data are increased.
                                        It is anticipated that analogous trends will emerge in the field of robotics. Consequently, re-
                                        cent research efforts in the field of robot learning have concentrated on the development of in-
                                        creasingly large-scale robot datasets [6, 5, 8, 9, 10, 11] and the training of high-capacity models
                                        [6, 12, 13, 14, 15, 7, 16, 17] on these datasets, which directly map observations to actions, e.g.,
                                        Vision-Language-Action (VLA) models [12]. The hope is that, by feeding abundant web and robot
                                        data, we can develop a generalist robot policy capable of addressing a wide spectrum of tasks and,
                                        more importantly, generalizing to novel tasks and environments out of box.
                                        Despite advancements in training models on large-scale datasets like Open X-Embodiment (OXE)
                                        [10], these models continue to demonstrate limited generalization capabilities across multiple axes,
                                                ∗
                                                 : denotes equal contribution. † : denotes corresponding author.
                                            1
                                                Poject page: https://lucky-light-sun.github.io/proj/shortcut-learning-in-grps/


                                        9th Conference on Robot Learning (CoRL 2025), Seoul, Korea.
```

## P22 — Spatial-Aware VLA Pretraining.pdf

PDF 第 1 页；共 12 页；SHA256 `fd5725b617b3e4a6a5b737463244b8c839c599506309567ea4787d95ea5b8cbf`。

```text
                      This CVPR paper is the Open Access version, provided by the Computer Vision Foundation.
                                   Except for this watermark, it is identical to the accepted version;
                             the final published version of the proceedings is available on IEEE Xplore.




                                  Spatial-Aware VLA Pretraining
                       through Visual-Physical Alignment from Human Videos

                 Yicheng Feng1,3 Wanpeng Zhang1,3 Ye Wang2,3 Hao Luo1,3
                         Haoqi Yuan1,3 Sipeng Zheng3 Zongqing Lu1,3†
  1
    School of Computer Science, Peking University 2 Renmin University of China 3 BeingBeyond


                            Abstract                                  demonstrated the potential to develop generalist robot poli-
                                                                      cies across a wide range of tasks[5, 6, 8, 9, 38].
   Vision-Language-Action (VLA) models provide a                          Nevertheless, current VLA models typically rely on 2D
promising paradigm for robot learning by integrating                  visual inputs to perceive the world while performing ac-
visual perception with language-guided policy learning.               tions in a 3D physical environment, leaving a substantial
However, most existing approaches rely on 2D visual                   gap between visual perception and embodied action. This
inputs to perform actions in 3D physical environments,                weak correspondence limits their ability to ground actions
creating a significant gap between perception and action              in physical space. For effective policy learning, an agent
grounding. To bridge this gap, we propose a Spatial-                  must not only interpret pixels but also understand how these
Aware VLA Pretraining paradigm that enables models                    visual cues map to 3D geometry and how physical actions
to acquire 3D spatial understanding before robot pol-                 interact with the surrounding environment. While humans
icy learning. Starting from pretrained vision-language                can infer 3D space from 2D visual signals, existing VLA
models, we leverage large-scale human demonstration                   models largely overlook this aspect, resulting in poor spa-
videos to extract 3D visual and 3D action annotations,                tial grounding and limited generalization.
forming a new source of supervision that aligns 2D visual                 To bridge this gap, we introduce a new Spatial-Aware
observations with 3D spatial reasoning. We instantiate                VLA Pretraining paradigm that enables models to acquire
this paradigm with VIPA-VLA, a dual-encoder architec-                 3D spatial awareness before learning robotic policies. Start-
ture that incorporates a 3D visual encoder to augment                 ing from pretrained VLMs, we leverage large-scale human
semantic visual representations with 3D-aware features,               demonstration videos as a rich source of supervision, where
and aligns the two through visual-physical alignment                  implicit correspondences between 2D visual observations
pretraining. When adapted to downstream robot tasks,                  and 3D physical actions naturally exist. Compared to robot
VIPA-VLA achieves significantly improved grounding                    data, human demonstrations are easier to obtain across di-
between 2D vision and 3D action, resulting in more robust             verse environments and naturally provide rich evidence of
and generalizable robotic policies. The project website:              how actions are carried out in the physical world under
https://beingbeyond.github.io/VIPA- VLA.                              diverse visual contexts. By extracting 3D cues such as
                                                                      hand-object relationships and motion trajectories from these
                                                                      videos, we construct pretraining tasks that teach the model
                                                                      to align 2D vision with 3D spatial understanding—forming
1. Introduction                                                       what we term visual-physical alignment. This spatially
                                                                      grounded pretraining provides a strong foundation for sub-
The rapid progress of large-scale vision-language models
                                                                      sequent VLA post-training, allowing the model to learn and
(VLMs) has showcased remarkable capabilities in learning
                                                                      generalize more effectively in robot manipulation tasks.
joint representations across modalities[1, 3, 22, 41, 50, 66,
                                                                          To instantiate this paradigm, we present VIPA-VLA
77, 86, 87, 94, 95, 98]. This progress has opened up new
                                                                      (Visual-Physical-Alignment-VLA), a dual-encoder archi-
opportunities for robot policy learning, where VLMs offer
                                                                      tecture that augments semantic visual representations with
a solid foundation for understanding both visual observa-
                                                                      explicit 3D spatial features. Alignment between vision, lan-
tions and language instructions, thereby guiding robot inter-
                                                                      guage, and 3D action is achieved through Spatial-Aware
action with the physical world. Building on this foundation,
                                                                      VLA Pretraining on our dataset Hand3D, which is con-
the vision-language-action (VLA) paradigm has recently
                                                                      structed from diverse human manipulation recordings with
  † Corresponding   to <zongqing.lu@pku.edu.cn>.                      annotated hand poses. From these videos, we derive two



                                                                712
```

## P23 — SpikingBrain_Report_Chi.pdf

PDF 第 1 页；共 38 页；SHA256 `c270adff9bb3f8c8ca5e7f4131a09e28a122c11c2d935aa9eb5e572ab71d3aa3`。

```text
           技术报告：原生国产自主可控类脑脉冲大模型
                                     SpikingBrain-瞬悉 1.0
           潘昱锜1,2,3 ，冯宇鹏1 ，庄景豪1 ，丁思宇1 ，徐涵1,4 ，刘泽昊1,5 ，孙博涵1 ，
                      侴雨宏1,5 ，邱雪睿1,6 ，邓岸林1 ，胡安杰1,6,7 ，汪树荣1,8 ，
              周芃9 ，姚满1,2,3 ，吴冀彬5 ，杨建10 ，孙国梁10 ，徐波1,2∗，李国齐1,2,3∗
          1：中国科学院自动化研究所； 2：中国科学院自动化所类脑通用智能大模型北京市重点实验室；
           3：脑认知与类脑智能全国重点实验室； 4：北京智源人工智能研究院； 5：香港理工大学；
6：北京中关村学院； 7：北京航空航天大学； 8：浙江大学； 9：陆兮科技； 10：沐曦集成电路有限公司




                                             摘要

              Transformer 的训练开销随序列长度呈平方级增长，推理显存占用随序列长度线性增加，这
          造成了基于 Transformer 的大模型资源消耗巨大、长序列处理能力受限。为应对这些问题，本
          项目借鉴大脑结构功能与信息处理机制，开发 7B 和 76B 规模的高效类脑脉冲大模型，整个
          训练与推理全流程在国产算力（沐曦科技1 曦云 C550）集群上进行。本报告涉及的核心技术点
          为：（1）在模型层面，基于脉冲神经元构建了具有新型线性或混合线性复杂度的基础模型架
          构；
           （2）在算法层面，构建了与现有大模型兼容的通用模型转换技术和高效训练范式，并配套
          开发了专用的脉冲化编码框架；
                       （3）在工程层面，在国产 GPU 集群开发了兼容的大模型训练
          框架、Triton/CUDA 算子库、模型并行策略以及集群通信原语。测试评估亮点包括：
                                                    （1）两
          款模型的长序列训练效率显著提升，能以极低的数据量实现与众多开源 Transformer 模型相
          媲美的通用语言建模性能（约为主流大模型的 2%）
                                 ；（2）推理阶段模型的低计算/存储复杂度，
          结合脉冲事件驱动特性，1M 长度下的 TTFT（生成第一个 Token 所需时间）加速可达 26.5
          倍，4M 长度上加速超过 100 倍，长序列处理上展现出数量级的效率和速度提升；
                                                 （3）面向国
          产算力集群的训练框架算子加速和通信适配，能保持百卡规模训练的数周稳定运行，7B 模型
          训练 MFU 超过 23.4%；（4）将压缩到 1B 的类脑模型部署到 CPU 手机端推理框架上，在
          64k-128k-256k 长度下较 Llama3.2 的 1B 模型解码速度分别提升 4.04×-7.52×-15.39×；（5）
          细粒度动态阈值脉冲化策略结合粗粒度的 MoE 方案，使得网络整体稀疏性超过 69.15%，为
          低功耗类脑大模型运行提供有力支撑。本次尝试为国产算力平台上的高效类脑脉冲大模型的研
          发扩展提供了有益探索和实践，也将启发下一代类脑芯片的设计。



1         引言
         自 2017 年 Transformer 架构 [1] 提出以来，依托 GPU 集群的大规模计算能力，人工
智能迈入大模型时代并取得巨大成功 [2, 3, 4, 5]。当前大模型的规模在依赖“数据-算力-算
法”的 Scaling Law[6] 驱动下变得越来越大。我们将此路径称为“基于外生复杂性的通用
智能模型”：通过增加网络规模、算力资源和数据量提升模型智能水平，但模型的基本计算
    ∗
        通讯作者
    1
        https://www.metax-tech.com



                                              1
```

## P24 — SpikingBrain_Report_Eng.pdf

PDF 第 1 页；共 40 页；SHA256 `d3e62937d995d036603b1d2ffb0925b4a087ff1eef60f208c06f21b195409808`。

```text
SpikingBrain: Spiking Brain-inspired Large Models

Yuqi Pan1,2,3 , Yupeng Feng1 , Jinghao Zhuang1 , Siyu Ding1 , Han Xu1,4 , Zehao Liu1,5 ,
Bohan Sun1 , Yuhong Chou1,5 , Xuerui Qiu1,6 , Anlin Deng1 , Anjie Hu1,6,7 , Shurong Wang1,8 ,
Peng Zhou9 , Man Yao1,2,3 , Jibin Wu5 , Jian Yang10 , Guoliang Sun10 , Bo Xu1,2∗ & Guoqi Li1,2,3∗

1
  Institute of Automation, Chinese Academy of Sciences
2
  Beijing Key Laboratory of Brain-Inspired General Intelligence Large Model
3
  Key Laboratory of Brain Cognition and Brain-inspired Intelligence Technology
4
  Beijing Academy of Artificial Intelligence 5 The Hong Kong Polytechnic University
6
  Zhongguancun Academy 7 Beihang University 8 Zhejiang University
9
  LuxiTech 10 MetaX Integrated Circuit Co., Ltd.



                                                          Abstract

            Mainstream Transformer-based large language models (LLMs) face significant efficiency
            bottlenecks: training computation scales quadratically with sequence length, and inference
            memory grows linearly. These constraints limit their ability to process long sequences
            effectively. In addition, building large models on non-NVIDIA computing platforms poses
            major challenges in achieving stable and efficient training and deployment. To address these
            issues, we introduce SpikingBrain, a new family of brain-inspired models designed for efficient
            long-context training and inference. SpikingBrain leverages the MetaX1 GPU cluster and
            focuses on three core aspects: i) Model Architecture: linear and hybrid-linear attention
            architectures with adaptive spiking neurons; ii) Algorithmic Optimizations: an efficient,
            conversion-based training pipeline compatible with existing LLMs, along with a dedicated
            spike coding framework; iii) System Engineering: customized training frameworks, op-
            erator libraries, and parallelism strategies tailored to the MetaX hardware. Using these
            techniques, we develop two models: SpikingBrain-7B, a linear LLM, and SpikingBrain-
            76B, a hybrid-linear MoE LLM. These models demonstrate the feasibility of large-scale LLM
            development on non-NVIDIA platforms, and our training framework supports weeks of stable
            training on hundreds of MetaX GPUs with Model FLOPs Utilization (MFU) at expected
            levels. SpikingBrain achieves performance comparable to open-source Transformer baselines
            while using exceptionally low data resources (continual pre-training of ∼150B tokens). Our
            models also significantly improve long-context efficiency and deliver inference with (partially)
            constant memory and event-driven spiking behavior. For example, SpikingBrain-7B achieves
            more than 100× speedup in Time to First Token (TTFT) for 4M-token sequences. Further-
            more, the proposed spiking scheme achieves 69.15% sparsity, enabling low-power operation.
            Overall, this work demonstrates the potential of brain-inspired mechanisms to drive the next
            generation of efficient and scalable large model design. 2


1     Introduction

Recent advances in large language models (LLMs) built on the Transformer architecture (Vaswani et al.,
2017) have been driven by the scaling law (Kaplan et al., 2020), which suggests that performance improves
with larger model sizes and more data (OpenAI, 2025; Google DeepMind, 2025; Anthropic, 2025). However,
this scale-driven approach comes with significant challenges: extremely high training costs, substantial energy
consumption, and complex deployment pipelines. Therefore, achieving high performance and energy efficiency
    ∗ Corresponding   authors: xubo@ia.ac.cn and guoqi.li@ia.ac.cn
    1 https://www.metax-tech.com/en
    2 The   code for this project is publicly available at https://github.com/BICLab/SpikingBrain-7B.


                                                                1
```

## P25 — WAM_survey.pdf

PDF 第 1 页；共 57 页；SHA256 `17816dc3628a0bf2276dad781e394fc077f5e8c464d98d50fd3d62e3ea9f42e7`。

```text
                                                                      World Action Models: A Survey
                                                                                     Dream Less, Act More
                                                            Qiuhong Shen, Shihua Zhang, Yue Liao, Qi Li, Zhenxiong Tan, Shizun Wang,
                                                                                Shuicheng Yan, Xinchao Wang†

                                                                                 National University of Singapore

                                            World Action Models (WAMs) are embodied predictive-action models that make a forecast of the
                                            future available to action. Recent WAMs repurpose large video generation models, and a parallel line
                                            relies on language or vision-language backbones without a video-generation core. This rapid expansion
                                            has blurred the boundary among broad world models, video generation models, action-grounded video
                                            world models, Vision-Language-Action policies, and WAMs. This survey gives the field a common
arXiv:2606.20781v1 [cs.RO] 18 Jun 2026




                                            account. It first clarifies these boundaries, then organizes existing works through two complementary
                                            views. The first view asks what each method is required to generate, spanning rendered futures, latent
                                            futures, and video-generation-free action reasoning. The second view decomposes each method by
                                            predictive substrate, backbone, action coupling, and deployment regime. This anatomy supports a
                                            unified discussion of interactability, causality, persistence, physical plausibility, and generalization,
                                            followed by data, evaluation, and open challenges. Across these axes, a consistent design pattern
                                            emerges: WAMs are not simply video generators with action heads, but predictive-action methods
                                            whose design choices trade representational richness against compute, memory, latency, and action-
                                            label cost. The field is moving toward methods that generate less of the future while preserving what
                                            control requires. The survey homepage is available at https://world-action-models.github.io/.




                                         Contents

                                         1 Introduction                                                                                                                                                        2

                                         2 The   Emergence of World Action Models                                                                                                                              4
                                           2.1   Vision-Language-Action Models: Policies from Vision-Language Pretraining . . . . . . . . . .                                                                  4
                                           2.2   World Models: Predictive Dynamics Beyond Video . . . . . . . . . . . . . . . . . . . . . . . .                                                                4
                                           2.3   World Action Models: Making the Future Action-Facing . . . . . . . . . . . . . . . . . . . . .                                                                5

                                         3 Three Design Philosophies of World Action Models                                                                                                                     6
                                           3.1 Render-and-Decode World Action Models . . . . . . .                     .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .    7
                                           3.2 Latent-Only World Action Models . . . . . . . . . . .                   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   10
                                           3.3 Video-Generation-Free World Action Models . . . . . .                   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   11
                                           3.4 Where the three philosophies meet the formal anatomy                    .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   12

                                         4 What Makes a World Action Model                                                                                                                                     12
                                           4.1 A Unified Notation for World Action Models .            .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   12
                                           4.2 Predictive Substrate: Where WAMs Dream . .              .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   14
                                               4.2.1 Pixel-grounded substrates . . . . . . . .         .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   14
                                               4.2.2 Feature substrates . . . . . . . . . . . .        .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   15
                                               4.2.3 Geometric primitive substrates . . . . .          .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   16
                                               4.2.4 Affordance map substrates . . . . . . . .         .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   16
                                               4.2.5 Substrate and coupling are distinct axes          .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   16
                                            Corresponding author: Xinchao Wang (xinchao@nus.edu.sg).




                                                                                                  1
```

## P26 — WordCon.pdf

PDF 第 2 页；共 12 页；SHA256 `e3a4a4b3f24087468b63ec61ebdb5ee7aabf38ca7249d30a1ead6f605fcf4f01`。

```text
2 •    Wenda Shi, Yiren Song, Zihan Rao, Dengming Zhang, Jiaming Liu, and Xingxing Zou∗


Achieving precise word-level typography control within generated images
remains a persistent challenge. To address it, we newly construct a word-
level controlled scene text dataset and introduce the Text-Image Alignment
(TIA) framework. This framework leverages cross-modal correspondence
between text and local image regions provided by grounding models to
enhance the Text-to-Image (T2I) model training. Furthermore, we propose
                                                                                                                                    Girl             Book                 Dog
WordCon, a hybrid parameter-efficient fine-tuning (PEFT) method. WordCon
reparameterizes selective key parameters, improving both efficiency and
portability. This allows seamless integration into diverse pipelines, including
artistic text rendering, text editing, and image-conditioned text rendering. To                      A girl holding a book and

                                                                                                     accompanied by a dog,
further enhance controllability, the masked loss at the latent level is applied                      the book title is \"Control

to guide the model to concentrate on learning the text region in the image,                          Target Words\".


and the joint-attention loss provides feature-level supervision to promote                                                         Control           Target              Words

disentanglement between different words. Both qualitative and quantitative
results demonstrate the superiority of our method to the state of the art. Our
project is available at https://wendashi.github.io/WordCon-Page/.
CCS Concepts: • Computing methodologies → Computer vision; Image
manipulation.
Additional Key Words and Phrases: Diffusion Model, Text Rendering, Ty-                                                              Boy              T-shirt              Cap

pography, Text-Guided Image Generation
ACM Reference Format:
Wenda Shi, Yiren Song, Zihan Rao, Dengming Zhang, Jiaming Liu, and Xingx-                            A boy wearing a T-shirt


ing Zou∗ . 2025. WordCon: Word-level Typography Control in Scene Text                                and a cap, the slogan on

                                                                                                     the T-shirt is \"Control

Rendering. ACM Trans. Graph. 1, 1 (June 2025), 12 pages. https://doi.org/10.                         Target Words\".


1145/nnnnnnn.nnnnnnn
                                                                                                                                   Control           Target              Words




1     INTRODUCTION                                                                                   Fig. 3. The green regions are the attention maps of each word in the prompt.
                                                                                                     Compared to words that refer to common objects (e.g., ‘Girl’, ‘Book’, ‘Dog’,
                                                                                                     ‘Boy’, ‘T-shirt’, ‘Cap’), attention map of words for text rendering is more likely
                                                          A large road sign at a countryside
                                                          intersection surrounded by golden wheat    to be misaligned. In this paper, we refer to it as word-level misalignment.
                                                          fields, presenting black Text: \"What a
                                                          trifle scares you!\", make 'scares' bold
                                                                                                     Wang et al. 2025c; Yang et al. 2024; Zhang et al. 2025b]. For ex-
                                                                                                     ample, Glyph-ByT5 [Liu et al. 2024a] improved text accuracy by
                                                                                                     introducing a new text encoder with contrastive learning, while
                                                                                                     TextDiffuser-2 [Chen et al. 2024b] enhanced layout planning by
                                                                                                     incorporating multiple large language models.
                                                                                                        Recent advances in diffusion transformer (DiT) models [bla 2024;
                                                                                                     Esser et al. 2024] greatly improved text accuracy by leveraging direct
                                                                                                     preference optimization (DPO) [Rafailov et al. 2023] and T5 [Raffel
       GPT4o Image                 Gemini2.5 Pro                           Ours                      et al. 2020] text encoder for better prompt understanding. Addition-
                                                                                                     ally, closed-source models [Google AI 2025; Ideogram 2024; OpenAI
Fig. 2. The challenge of text rendering. The SOTA T2I models excel at general
                                                                                                     2025; Recraft 2024] have demonstrated substantial improvements in
text rendering and controllability on common objects, however, they struggle
with precise word-level typography control.
                                                                                                     text rendering quality and general object control. However, these
                                                                                                     models still struggle with precise word-level typography control,
Scene text rendering, which involves the integration of text into                                    such as applying bold, italic, or underline to specific words, which is
images while maintaining visual coherence and realism, plays a                                       a capability essential for practical applications. As demonstrated in
crucial role in various applications such as advertising, branding,                                  Figure 2, even advanced commercial models like GPT4o-img [Ope-
and marketing [Bai et al. 2024; Chen et al. 2024b; Shi et al. 2025;                                  nAI 2025] and Gemini2 [Google AI 2025] fail to achieve the desired
Zou et al. 2025]. However, creating high-quality visual text is a                                    word-level control despite multiple attempts with various prompts.
challenging task that requires designers to carefully consider font                                     To understand the reason behind this limitation, we visualized
selection, typographic attributes, and overall visual harmony [Shi                                   the attention maps of Flux [bla 2024] in the same analysis method
et al. 2024b].                                                                                       as [Helbling et al. 2025; Hu et al. 2025]. As shown in Figure 3, atten-
   The emergence of diffusion text-to-image (T2I) models [Podell                                     tion map misalignment occurs more prominently in text rendering
et al. 2024; Rombach et al. 2022] has shown great potential in au-                                   than in common object generation. The decoupling between words
tomating this design process. However, these general-purpose mod-                                    used for text rendering is not as thorough as that between words for
els often struggle with text accuracy and controllability. To address                                common objects. This finding indicates that current methods lack ef-
these limitations, researchers have developed several UNet-based                                     fective mechanisms for precise text positioning, making it challenging
approaches [Chen et al. 2024a,b; Liu et al. 2024a,b; Tuo et al. 2024;                                to achieve controllability at the word level.

ACM Trans. Graph., Vol. 1, No. 1, Article . Publication date: June 2025.
```

## P27 — Zero-Shot Visual Generalization.pdf

PDF 第 1 页；共 21 页；SHA256 `2d3117be5048adefa7e29a9dec324d067c4b324c2cf22f62a476441d70be4e2f`。

```text
                                                    Zero-Shot Visual Generalization in Robot
                                                                 Manipulation
                                                            Sumeet Batra                                Gaurav S. Sukhatme
                                                   University of Southern California               University of Southern California
                                                             United States                                   United States
                                                         ssbatra@usc.edu                                  gaurav@usc.edu
arXiv:2505.11719v1 [cs.RO] 16 May 2025




                                                  Abstract: Training vision-based manipulation policies that are robust across di-
                                                  verse visual environments remains an important and unresolved challenge in robot
                                                  learning. Current approaches often sidestep the problem by relying on invariant
                                                  representations such as point clouds and depth, or by brute-forcing generalization
                                                  through visual domain randomization and/or large, visually diverse datasets. Dis-
                                                  entangled representation learning – especially when combined with principles of
                                                  associative memory – has recently shown promise in enabling vision-based re-
                                                  inforcement learning policies to be robust to visual distribution shifts. However,
                                                  these techniques have largely been constrained to simpler benchmarks and toy en-
                                                  vironments. In this work, we scale disentangled representation learning and asso-
                                                  ciative memory to more visually and dynamically complex manipulation tasks and
                                                  demonstrate zero-shot adaptability to visual perturbations in both simulation and
                                                  on real hardware. We further extend this approach to imitation learning, specifi-
                                                  cally Diffusion Policy, and empirically show significant gains in visual generaliza-
                                                  tion compared to state-of-the-art imitation learning methods. Finally, we introduce
                                                  a novel technique adapted from the model equivariance literature that transforms
                                                  any trained neural network policy into one invariant to 2D planar rotations, making
                                                  our policy not only visually robust but also resilient to certain camera perturba-
                                                  tions. We believe that this work marks a significant step towards manipulation
                                                  policies that are not only adaptable out of the box, but also robust to the complex-
                                                  ities and dynamical nature of real-world deployment. Supplementary videos are
                                                  available at https://sites.google.com/view/vis-gen-robotics/home.

                                                  Keywords: manipulation, representation learning, robot learning

                                         1   Introduction

                                         A key requirement of any generalist robot system deployed in the real-world is the ability to perform
                                         tasks across visually diverse environments. High-dimensional inputs like RGB images offer rich
                                         information but also introduce complexity due to the curse of dimensionality. Given the enormous
                                         diversity of real-world visual data, accounting for every possible variation within a fixed dataset is
                                         intractable. Extracting the underlying structural knowledge of the world from visual data while being
                                         robust to semantically irrelevant visual perturbations remains an open question. The robot learning
                                         field has largely relied on one of several trends, one of which is to train agents in simulation, where
                                         visual complexity can be controlled and large-scale synthetic and diverse data can be generated
                                         efficiently through GPU-accelerated simulators [1, 2, 3]. However, transferring policies trained in
                                         simulation to the real world is hindered by the ”Sim2Real” gap caused by mismatches in fidelity and
                                         unmodeled dynamics. Domain randomization is the leading strategy to close this gap by varying the
                                         simulation parameters such that real-world conditions fall within the distribution of the training data.
                                         Domain randomization has proven effective in both simulated benchmarks and real-world robotic
                                         tasks when the data diversity is sufficiently large [4, 5, 6]. A seemingly unrelated but conceptually
                                         similar approach to visual generalization in the age of foundation models has been to train large
```

## P28 — attention_sinks_2309.17453.pdf

PDF 第 1 页；共 21 页；SHA256 `e8a7de67c7c57642648362651b4cf64fd852467e18865a298916a36098314819`。

```text
                                        Published as a conference paper at ICLR 2024




                                        E FFICIENT S TREAMING L ANGUAGE M ODELS
                                        WITH ATTENTION S INKS

                                            Guangxuan Xiao1∗ Yuandong Tian2                   Beidi Chen3    Song Han1,4      Mike Lewis2
                                         1
                                          Massachusetts Institute of Technology 2 Meta AI
                                         3
                                          Carnegie Mellon University 4 NVIDIA
                                         https://github.com/mit-han-lab/streaming-llm
arXiv:2309.17453v4 [cs.CL] 7 Apr 2024




                                                                                           A BSTRACT

                                                      Deploying Large Language Models (LLMs) in streaming applications such as
                                                      multi-round dialogue, where long interactions are expected, is urgently needed but
                                                      poses two major challenges. Firstly, during the decoding stage, caching previous
                                                      tokens’ Key and Value states (KV) consumes extensive memory. Secondly, popular
                                                      LLMs cannot generalize to longer texts than the training sequence length. Window
                                                      attention, where only the most recent KVs are cached, is a natural approach — but
                                                      we show that it fails when the text length surpasses the cache size. We observe
                                                      an interesting phenomenon, namely attention sink, that keeping the KV of initial
                                                      tokens will largely recover the performance of window attention. In this paper, we
                                                      first demonstrate that the emergence of attention sink is due to the strong attention
                                                      scores towards initial tokens as a “sink” even if they are not semantically important.
                                                      Based on the above analysis, we introduce StreamingLLM, an efficient framework
                                                      that enables LLMs trained with a finite length attention window to generalize to
                                                      infinite sequence length without any fine-tuning. We show that StreamingLLM can
                                                      enable Llama-2, MPT, Falcon, and Pythia to perform stable and efficient language
                                                      modeling with up to 4 million tokens and more. In addition, we discover that
                                                      adding a placeholder token as a dedicated attention sink during pre-training can
                                                      further improve streaming deployment. In streaming settings, StreamingLLM
                                                      outperforms the sliding window recomputation baseline by up to 22.2× speedup.
                                                      Code and datasets are provided in the link.


                                        1        I NTRODUCTION

                                        Large Language Models (LLMs) (Radford et al., 2018; Brown et al., 2020; Zhang et al., 2022;
                                        OpenAI, 2023; Touvron et al., 2023a;b) are becoming ubiquitous, powering many natural language
                                        processing applications such as dialog systems (Schulman et al., 2022; Taori et al., 2023; Chiang et al.,
                                        2023), document summarization (Goyal & Durrett, 2020; Zhang et al., 2023a), code completion (Chen
                                        et al., 2021; Rozière et al., 2023) and question answering (Kamalloo et al., 2023). To unleash the
                                        full potential of pretrained LLMs, they should be able to efficiently and accurately perform long
                                        sequence generation. For example, an ideal ChatBot assistant can stably work over the content of
                                        recent day-long conversations. However, it is very challenging for LLM to generalize to longer
                                        sequence lengths than they have been pretrained on, e.g., 4K for Llama-2 Touvron et al. (2023b).
                                        The reason is that LLMs are constrained by the attention window during pre-training. Despite
                                        substantial efforts to expand this window size (Chen et al., 2023; kaiokendev, 2023; Peng et al., 2023)
                                        and improve training (Dao et al., 2022; Dao, 2023) and inference (Pope et al., 2022; Xiao et al., 2023;
                                        Anagnostidis et al., 2023; Wang et al., 2021; Zhang et al., 2023b) efficiency for lengthy inputs, the
                                        acceptable sequence length remains intrinsically finite, which doesn’t allow persistent deployments.
                                        In this paper, we first introduce the concept of LLM streaming applications and ask the question:

                                         Can we deploy an LLM for infinite-length inputs without sacrificing efficiency and performance?
                                             ∗
                                                 Part of the work done during an internship at Meta AI.


                                                                                                  1
```

## P29 — deepseek_vl2_2412.10302.pdf

PDF 第 1 页；共 28 页；SHA256 `cff2bd0ad1656769d7447f0e3cf3a1885902333894f123bdac27fce05c97459e`。

```text
                                          DeepSeek-VL2: Mixture-of-Experts Vision-Language Models
                                                  for Advanced Multimodal Understanding

                                            Zhiyu Wu∗ , Xiaokang Chen∗ , Zizheng Pan∗ , Xingchao Liu∗ , Wen Liu∗,† , Damai Dai, Huazuo Gao,
                                          Yiyang Ma, Chengyue Wu, Bingxuan Wang, Zhenda Xie, Yu Wu, Kai Hu, Jiawei Wang, Yaofeng Sun,
                                         Yukun Li, Yishi Piao, Kang Guan, Aixin Liu, Xin Xie, Yuxiang You, Kai Dong, Xingkai Yu, Haowei Zhang,
                                                                       Liang Zhao, Yisong Wang, Chong Ruan‡

                                                                                                                     DeepSeek-AI
arXiv:2412.10302v1 [cs.CV] 13 Dec 2024




                                                                                                                     Abstract

                                         We present DeepSeek-VL2, an advanced series of large Mixture-of-Experts (MoE) Vision-
                                         Language Models that significantly improves upon its predecessor, DeepSeek-VL, through two
                                         key major upgrades. For the vision component, we incorporate a dynamic tiling vision encoding
                                         strategy designed for processing high-resolution images with different aspect ratios. For the
                                         language component, we leverage DeepSeekMoE models with the Multi-head Latent Attention
                                         mechanism, which compresses Key-Value cache into latent vectors, to enable efficient inference
                                         and high throughput. Trained on an improved vision-language dataset, DeepSeek-VL2 demon-
                                         strates superior capabilities across various tasks, including but not limited to visual question
                                         answering, optical character recognition, document/table/chart understanding, and visual
                                         grounding. Our model series is composed of three variants: DeepSeek-VL2-Tiny, DeepSeek-VL2-
                                         Small and DeepSeek-VL2, with 1.0B, 2.8B and 4.5B activated parameters respectively. DeepSeek-
                                         VL2 achieves competitive or state-of-the-art performance with similar or fewer activated param-
                                         eters compared to existing open-source dense and MoE-based models. Codes and pre-trained
                                         models are publicly accessible at https://github.com/deepseek-ai/DeepSeek-VL2.

                                                                                           72                        DeepSeek-VL2            Qwen2-VL-7B
                                                                                                       DeepSeek-VL2-Small
                                                                     Average Performance




                                                                                                                                          InternVL2-8B
                                                                                           66
                                                                                                                  InternVL2-4B
                                                                                                DeepSeek-VL2-Tiny
                                                                                           60                 Qwen2-VL-2B
                                                                                                         InternVL2-2B
                                                                                                                     Phi-3.5-Vision
                                                                                           54                                             DeepSeek-VL2 Family
                                                                                                  InternVL2-1B                            InternVL2 Family
                                                                                                                                          Qwen2-VL Family

                                                                                           48 0             2             4           6       8             10
                                                                                                           Activated Parameters (Billions)
                                         Figure 1 | Average performance vs. activated parameters among different open-source models.
                                         We average the accuracy of MMBench v1.1, MMStar, MMMU (Val), MathVista (TestMini), AI2D
                                         (Test), and OCRBench. The scores of OCRBench are divided by 10 to scale them to [0, 100].


                                         ∗:   Core contributors. † : Project lead. ‡ : Corresponding author.
```

## P30 — diffusion_policy_2303.04137.pdf

PDF 第 1 页；共 22 页；SHA256 `b65c474b696a4802d8f1457d86b637ce2c5521412570d3aa928cd54563babc8f`。

```text
                                                                                                                                                                     .

                                         Diffusion Policy: Visuomotor Policy
                                         Learning via Action Diffusion
                                         Cheng Chi∗ 1 , Zhenjia Xu∗ 1 , Siyuan Feng2 , Eric Cousineau2 , Yilun Du3 , Benjamin Burchfiel2 ,
                                         Russ Tedrake 2,3 , Shuran Song1,4



                                         Abstract
                                         This paper introduces Diffusion Policy, a new way of generating robot behavior by representing a robot’s visuomotor
                                         policy as a conditional denoising diffusion process. We benchmark Diffusion Policy across 15 different tasks from 4
                                         different robot manipulation benchmarks and find that it consistently outperforms existing state-of-the-art robot learning
                                         methods with an average improvement of 46.9%. Diffusion Policy learns the gradient of the action-distribution score
                                         function and iteratively optimizes with respect to this gradient field during inference via a series of stochastic Langevin
                                         dynamics steps. We find that the diffusion formulation yields powerful advantages when used for robot policies, including
arXiv:2303.04137v5 [cs.RO] 14 Mar 2024




                                         gracefully handling multimodal action distributions, being suitable for high-dimensional action spaces, and exhibiting
                                         impressive training stability. To fully unlock the potential of diffusion models for visuomotor policy learning on physical
                                         robots, this paper presents a set of key technical contributions including the incorporation of receding horizon control,
                                         visual conditioning, and the time-series diffusion transformer. We hope this work will help motivate a new generation of
                                         policy learning techniques that are able to leverage the powerful generative modeling capabilities of diffusion models.
                                         Code, data, and training details is available diffusion-policy.cs.columbia.edu


                                         Keywords
                                         Imitation learning, visuomotor policy, manipulation



                                                                        Action
                                                                     Representation



                                                                                                                                              Diffusion Policy
                                                                  Scalar (Regression)    Implicit Policy
                                                Explicit Policy



                                                                  Mixture of Gaussians                                                                            iter




                                                                      Categorical


                                                      (a) Explicit Policy                              (b) Implicit Policy                                       (c) Diffusion Policy

                                         Figure 1. Policy Representations. a) Explicit policy with different types of action representations. b) Implicit policy learns an
                                         energy function conditioned on both action and observation and optimizes for actions that minimize the energy landscape c)
                                                                    p(a) into actions via a learned gradient field. This formulation provides stable training, allows the learned
                                         Diffusion policy refines noise
                                         policy to accurately model multimodal action distributions, and accommodates high-dimensional action sequences.

                                         1   Introduction                                                                    generates behavior via a “conditional denoising diffusion
                                                                                                                             process Ho et al. (2020) on robot action space”, Diffusion
                                         Policy learning from demonstration, in its simplest form, can
                                                                                                                             Policy. In this formulation, instead of directly outputting
                                         be formulated as the supervised regression task of learning to
                                                                                                                             an action, the policy infers the action-score gradient,
                                         map observations to actions. In practice however, the unique
                                                                                                                             conditioned on visual observations, for K denoising
                                         nature of predicting robot actions — such as the existence
                                                                                                                             iterations (Fig. 1 c). This formulation allows robot policies
                                         of multimodal distributions, sequential correlation, and the
                                                                                                                             to inherit several key properties from diffusion models –
                                         requirement of high precision — makes this task distinct and
                                                                                                                             significantly improving performance.
                                         challenging compared to other supervised learning problems.
                                            Prior work attempts to address this challenge by
                                         exploring different action representations (Fig 1 a) – using                            • Expressing multimodal action distributions. By
                                         mixtures of Gaussians Mandlekar et al. (2021), categorical                                learning the gradient of the action score function
                                         representations of quantized actions Shafiullah et al. (2022),                            Song and Ermon (2019) and performing Stochastic
                                         or by switching the the policy representation (Fig 1 b) – from                            Langevin Dynamics sampling on this gradient field,
                                         explicit to implicit to better capture multi-modal distributions                          Diffusion policy can express arbitrary normalizable
                                         Florence et al. (2021); Wu et al. (2020).                                                 distributions Neal et al. (2011), which includes mul-
                                            In this work, we seek to address this challenge by                                     timodal action distributions, a well-known challenge
                                         introducing a new form of robot visuomotor policy that                                    for policy learning.
```

## P31 — gla_2312.06635.pdf

PDF 第 1 页；共 23 页；SHA256 `9965bc4f590fb2ba35c2146f660e145fd730c7d9f8c6e7be10ba9a11518383d6`。

```text
                                                Gated Linear Attention Transformers with Hardware-Efficient Training


                                                            Songlin Yang 1 * Bailin Wang 1 * Yikang Shen 2 Rameswar Panda 2 Yoon Kim 1


                                                                  Abstract                                   1    Introduction
                                                                                                             Transformers with softmax attention (Vaswani et al., 2017)
                                               Transformers with linear attention allow for                  enjoy efficient parallel training but suffer from quadratic
arXiv:2312.06635v6 [cs.LG] 27 Aug 2024




                                               efficient parallel training but can simultaneously            (in sequence length) complexity, thus motivating more
                                               be formulated as an RNN with 2D (matrix-valued)               RNN-like models that allow for linear-time sequence
                                               hidden states, thus enjoying linear-time inference            modeling. Linear attention, which replaces the exponential
                                               complexity. However, linear attention generally               similarity function with a simple dot product over (possibly
                                               underperforms ordinary softmax attention. More-               transformed) key/query vectors, has emerged as a promising
                                               over, current implementations of linear attention             alternative to classic softmax attention (Katharopoulos et al.,
                                               lack I/O-awareness and are thus slower than highly            2020; Choromanski et al., 2021; Kasai et al., 2021; Peng
                                               optimized implementations of softmax attention.               et al., 2021). An attractive property of linear attention is that
                                               This work describes a hardware-efficient algo-                it admits a “recurrent form” in which it can be formulated
                                               rithm for linear attention that trades off memory             as a linear RNN with 2D hidden states (Katharopoulos et al.,
                                               movement against parallelizability. The resulting             2020), thus enabling linear-time inference. For training,
                                               implementation, dubbed F LASH L INEAR AT-                     linear attention also admits a subquadratic “chunkwise par-
                                               TENTION , is faster than F LASH ATTENTION -2                  allel form” which divides the sequence into non-overlapping
                                               (Dao, 2023) as a standalone layer even on short               chunks and performs (serial) inter-chunk recurrent computa-
                                               sequence lengths (e.g., 1K). We then generalize               tions followed by (parallel) intra-chunk computations (Hua
                                               this algorithm to a more expressive variant of                et al., 2022; Sun et al., 2023a; Lingle, 2023), thus (partially)
                                               linear attention with data-dependent gates. When              maintaining parallel training. However, existing algorithms
                                               used as a replacement for the standard attention              for linear attention are not I/O aware and thus, in practice,
                                               layer in Transformers, the resulting gated linear             slower than optimized implementations of softmax attention
                                               attention (GLA) Transformer is found to perform               (Dao et al., 2022b; Dao, 2023) on moderate sequence lengths.
                                               competitively against the LLaMA-architecture
                                               Transformer (Touvron et al., 2023) as well recent             From a performance standpoint, linear attention has gener-
                                               linear-time-inference baselines such as RetNet                ally been found to underperform ordinary softmax attention,
                                               (Sun et al., 2023a) and Mamba (Gu & Dao, 2023)                often by a significant margin in language modeling (Kasai
                                               on moderate-scale language modeling experi-                   et al., 2021). Recent variants of linear attention such as
                                               ments. GLA Transformer is especially effective                RetNet (Sun et al., 2023a) and TransNormerLLM (Qin et al.,
                                               at length generalization, enabling a model trained            2023b) obtain significant improvements by multiplying
                                               on 2K to generalize to sequences longer than 20K              the current hidden state with a decay factor before the
                                               without significant perplexity degradations. For              RNN update. However, these works use a global, data-
                                               training speed, the GLA Transformer has higher                independent decay factor, despite the fact that in 1D RNNs,
                                               throughput than a similarly-sized Mamba model.                a data-dependent gating mechanism has been shown to be
                                                                                                             crucial for performance (van der Westhuizen & Lasenby,
                                                https://github.com/sustcsonglin/fl                          2018; Qin et al., 2023c). And even with the decay factor,
                                               ash-linear-attention                                          linear attention Transformers underperform the strongest
                                                                                                             Transformer architectures when pretrained from scratch.
                                                                                                             This work develops a hardware-efficient algorithm for linear
                                           *
                                            Equal contribution 1 Massachusetts Institute of Technology       attention, and applies it to train a gated variant of linear
                                         2
                                           MIT-IBM Watson AI Lab. Correspondence to: Songlin Yang            attention that is competitive with softmax attention. We first
                                         <yangsl66@mit.edu>, Bailin Wang <bailinw@mit.edu>.                  discuss aspects of optimizing ordinary linear attention on
                                         Proceedings of the 41 st International Conference on Machine        modern GPUs and give two I/O-aware algorithms (tailored
                                         Learning, Vienna, Austria. PMLR 235, 2024. Copyright 2024           for different training settings) based on these principles (§3).
                                         by the author(s).                                                   Our implementation of the algorithm, called F LASH L IN -

                                                                                                         1
```

## P32 — hamster.pdf

PDF 第 1 页；共 29 页；SHA256 `c17f17e29601d9e0b43314b616607e3d55ac55428b0e6cb809edf92a49536ac4`。

```text
Published as a conference paper at ICLR 2025




HAMSTER: H IERARCHICAL ACTION M ODELS                                                            FOR
O PEN -W ORLD ROBOT M ANIPULATION
  Yi Li⋆‡1,2 , Yuquan Deng⋆2 , Jesse Zhang⋆1,3 , Joel Jang1,2 , Marius Memmel2 Caelan Garrett1 ,
 Fabio Ramos1 , Dieter Fox1,2 , Anqi Li†1 , Abhishek Gupta†1,2 , Ankit Goyal†1
 1
   NVIDIA 2 University of Washington 3 University of Southern California



                                                  A BSTRACT

           Large foundation models have shown strong open-world generalization to com-
           plex problems in vision and language, but similar levels of generalization have yet
           to be achieved in robotics. One fundamental challenge is the lack of robotic data,
           which are typically obtained through expensive on-robot operation. A promising
           remedy is to leverage cheaper, “off-domain” data such as action-free videos, hand-
           drawn sketches or simulation data. In this work, we posit that hierarchical vision-
           language-action (VLA) models can be more effective in utilizing off-domain data
           than standard monolithic VLA models that directly finetune vision-language mod-
           els (VLMs) to predict actions. In particular, we study a class of hierarchical VLA
           models, where the high-level VLM is finetuned to produce a coarse 2D path in-
           dicating the desired robot end-effector trajectory given an RGB image and a task
           description. The intermediate 2D path prediction is then served as guidance to
           the low-level, 3D-aware control policy capable of precise manipulation. Doing
           so alleviates the high-level VLM from fine-grained action prediction, while re-
           ducing the low-level policy’s burden on complex task-level reasoning. We show
           that, with the hierarchical design, the high-level VLM can transfer across signif-
           icant domain gaps between the off-domain finetuning data and real-robot testing
           scenarios, including differences on embodiments, dynamics, visual appearances
           and task semantics, etc. In the real-robot experiments, we observe an average of
           20% improvement in success rate across seven different axes of generalization
           over OpenVLA, representing a 50% relative gain. Visual results are provided at:
           https://hamster-robot.github.io/


1   I NTRODUCTION

Developing general robot manipulation policies has been notoriously difficult. With the advent of
large vision-language models (VLMs) that display compelling generalization capabilities, there is
optimism that the same recipe is directly applicable to robot manipulation. A line of prior work (Bro-
han et al., 2023a; Kim et al., 2024; Black et al., 2024) builds open-world vision-language-action
models (VLAs) by finetuning off-the-shelf pretrained VLMs to directly produce robot actions. These
VLA models, which we refer to in this work as monolithic VLA models, rely crucially on large
robotics datasets, complete with on-robot observations, e.g., images and proprioceptive states, and
actions. However, on-robot data is expensive, since end-to-end observation-action pairs are typically
collected on the robot hardware through, e.g., teleoperation. Despite recent community-wide efforts
in building large-scale robotics datasets (Collaboration et al., 2023; Khazatsky et al., 2024), the
size, quality, and diversity of existing robotics datasets are still limited, and monolithic VLA models
have yet to demonstrate emergent capability comparable to VLMs and LLMs in other domains of
study. Moreover, monolithic VLA models are constrained by their inference frequency to achieve
dexterous and dynamic manipulation tasks (Brohan et al., 2023a; Kim et al., 2024).
On the other hand, relatively small robot policy models have shown impressive dexterity and robust-
ness. Such models have demonstrated promise across a range of complex tasks involving contact-
rich manipulation and 3D reasoning, spanning domains from tabletop manipulation (Shridhar et al.,
    ⋆
        co-first authors ‡ project lead † equal advising


                                                           1
```

## P33 — int8.pdf

PDF 第 1 页；共 7 页；SHA256 `42f660ffffe92ae55e30928c28602d3082ae0580ce0eef8a0cc119021c41e28f`。

```text
                                             When Less is More: 8-bit Quantization Improves
                                             Continual Learning in Large Language Models


                                                Michael S. Zhang                 Rishi A. Ruia                   Arnav Kewalram
                                                   Algoverse                       Algoverse                        Algoverse
                                             mzhang3518@gmail.com            rishiru@outlook.com            arnav.kewalram@gmail.com
arXiv:2512.18934v2 [cs.LG] 30 Jun 2026




                                                    Saathvik Dharmapuram                              Utkarsh Sharma
                                                           Algoverse                                     Algoverse
                                                  saathvikd2686@gmail.com                    utkarsh@algoverseairesearch.org

                                                                                    Kevin Zhu
                                                                                    Algoverse
                                                                           kevin@algoverseacademy.com



                                                                                       Abstract
                                                  Catastrophic forgetting poses a fundamental challenge in continual learning, partic-
                                                  ularly when models are quantized for deployment efficiency. We systematically
                                                  investigate the interplay between quantization precision (FP16, INT8, INT4) and
                                                  replay buffer strategies in large language models, revealing unexpected dynamics.
                                                  While FP16 achieves superior initial task performance (74.44% on NLU), we ob-
                                                  serve a striking inversion on subsequent tasks: quantized models outperform FP16
                                                  by 8-15% on final task forward accuracy, with INT4 achieving nearly double FP16’s
                                                  performance on Code generation (40% vs 20%). Critically, even minimal replay
                                                  buffers (0.1%) dramatically improve retention—increasing NLU retention after
                                                  Math training from 45% to 65% across all precision levels—with INT8 consistently
                                                  achieving the optimal balance between learning plasticity and knowledge retention.
                                                  We hypothesize that quantization-induced noise acts as implicit regularization,
                                                  preventing the overfitting to new task gradients that plagues high-precision models.
                                                  These findings challenge the conventional wisdom that higher precision is always
                                                  preferable, suggesting instead that INT8 quantization offers both computational
                                                  efficiency and superior continual learning dynamics. Our results provide practical
                                                  guidelines for deploying compressed models in continual learning scenarios: small
                                                  replay buffers (1-2%) suffice for NLU tasks, while Math and Code benefit from
                                                  moderate buffers (5-10%, rising to 10-20% for INT8/INT4 models), with quantized
                                                  models requiring less replay than FP16 to achieve comparable retention. Code is
                                                  available at https://github.com/Festyve/LessIsMore.


                                         1   Introduction
                                         Although large language models (LLMs) have achieved state-of-the-art performance across a range
                                         of natural language and reasoning tasks, their ability to retain knowledge over time remains a key
                                         limitation, particularly when models must be continually updated with new data. Scaling alone does
                                         not address this challenge, as repeated fine-tuning often causes older capabilities to deteriorate, a
                                         phenomenon known as catastrophic forgetting [22, 3]. In real-world deployments, where models are
                                         expected to adapt continuously, this liability presents a major obstacle. To solve this, researchers use
                                         replay-based methods.

                                         39th Conference on Neural Information Processing Systems (NeurIPS 2025).
```

## P34 — lavender_2502.06814.pdf

PDF 第 1 页；共 38 页；SHA256 `22e95f65cb19d8bc8d69dae54679c17a6bb98a9dc9fdad04c5f6459c24bc9aac`。

```text
                                                                                                                    Diffusion Instruction Tuning


                                                                                       Chen Jin 1 Ryutaro Tanno 2 Amrutha Saseendran 1 Tom Diethe 1 Philip Teare 1

                                                                          12      Finetune MiniCPMv2.5
                                         Improvement after Finetune (%)




                                                                                  Finetune MiniCPMv2.5 + Lavender: Diffusion Instruction Tuning                                   ±10.6
                                                                                  Finetune Llama-3.2-11B
                                                                          10      Finetune Llama-3.2-11B + Lavender: Diffusion Instruction Tuning
arXiv:2502.06814v2 [cs.LG] 25 May 2025




                                                                           8                                                                  ±5.8
                                                                           6                                                                                                                                   ±3.9

                                                                           4
                                                                           2                          ±0.3                                                                                              ±2.5
                                                                                                                                     ±0.0                          ±1.7 ±13.3
                                                                           0   ±0.1 ±0.1                                                                   ±0.5                           ±0.1 ±0.2
                                                                                              ±1.1                 ±0.1 ±0.0
                                                                                   Real-World                            Hallucination                        Multi-Discipline                Chart/Doc QA
                                                                                    Ave. 2 tasks                           Ave. 2 tasks                           Ave. 10 tasks                 Ave. 6 tasks
                                                                                Figure 1. Average Performance on 20 Vision-Language Reasoning Benchmarks (Grouped into 4 Categories).


                                                                                                Abstract                                                     rate vision-language systems. Code, training data,
                                                                                                                                                             and models are available on the project page.
                                                                          We introduce Lavender, a simple supervised fine-
                                                                          tuning (SFT) method that boosts the performance
                                                                          of advanced vision-language models (VLMs) by
                                                                                                                                                                   Stable Diffusion                       Attention
                                                                          leveraging state-of-the-art image generation mod-                                         Model (frozen)                        Alignment
                                                                          els such as Stable Diffusion. Specifically, Laven-
                                                                          der aligns the text-vision attention in the VLM
                                                                          transformer with the equivalent used by Stable                                          Vision-Language
                                                                          Diffusion during SFT, instead of adapting separate                                      Model (update)
                                                                          encoders. This alignment enriches the model’s
                                                                          visual understanding and significantly boosts
                                                                          performance across in- and out-of-distribution
                                                                          tasks. Lavender requires just 0.13 million train-                             Figure 2. Lavender: Diffusion Instruction Tuning. Lavender
                                                                                                                                                        uses the text-vision attention maps of a Stable Diffusion Model,
                                                                          ing examples—2.5% of typical large-scale SFT
                                                                                                                                                        AttentionSDM , as a guiding objective for the attention of the
                                                                          datasets—and fine-tunes on standard hardware (8
                                                                                                                                                        target vision-language model (VLM), AttentionV LM . The Atten-
                                                                          GPUs) in a single day. It consistently improves                               tion Alignment module employs a 3-Layer ConvNet to transform
                                                                          state-of-the-art open-source multimodal LLMs                                  AttentionV LM to match AttentionSDM via an MSE loss, act-
                                                                          (e.g., Llama-3.2-11B, MiniCPM-Llama3-v2.5),                                   ing as a regularisation term during supervised fine-tuning.
                                                                          achieving up to 30% gains and a 68% boost on
                                                                          challenging out-of-distribution medical QA tasks.
                                                                          By efficiently transferring the visual expertise                              1. Introduction
                                                                          of image generators with minimal supervision,
                                                                          Lavender offers a scalable solution for more accu-                            Training frontier foundation models from scratch costs mil-
                                                                                                                                                        lions of dollars at minimum, requiring hundreds of GPUs
                                           1
                                             Centre for AI, AstraZeneca, Cambridge, UK 2 Google Deep-                                                   and millions to billions of data (DeepSeek-AI et al., 2024).
                                         Mind, UK. Correspondence to: <chen.jin@astrazeneca.com>.                                                       This challenge is even more pronounced in multimodal
                                         Proceedings of the 42 nd International Conference on Machine                                                   settings: vision-language models (VLMs) often face data
                                         Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025                                                    scarcity because collecting paired image-text datasets is ex-
                                         by the author(s).                                                                                              pensive (Zhu et al., 2024). A common workaround is to

                                                                                                                                                    1
```

## P35 — libero_2306.03310.pdf

PDF 第 1 页；共 44 页；SHA256 `ff7d943e2eb37760df684f3ae2931a5a35ef8ea465bd387e62afe49507b8e414`。

```text
                                             LIBERO: Benchmarking Knowledge Transfer for
                                                      Lifelong Robot Learning


                                                                  †
                                                                   Bo Liu∗, † Yifeng Zhu∗ , ‡ Chongkai Gao∗ , † Yihao Feng
                                                                         †
                                                                           Qiang Liu, † Yuke Zhu, †,§ Peter Stone
                                                            †
                                                              The University of Texas at Austin, § Sony AI, ‡ Tsinghua University
                                                               {bliu,yifengz,lqiang,yukez,pstone}@cs.utexas.edu
arXiv:2306.03310v2 [cs.AI] 14 Oct 2023




                                                                yihao.ac@gmail.com, gck20@mails.tsinghua.edu.cn



                                                                                         Abstract

                                                     Lifelong learning offers a promising paradigm of building a generalist agent that
                                                     learns and adapts over its lifespan. Unlike traditional lifelong learning problems in
                                                     image and text domains, which primarily involve the transfer of declarative knowl-
                                                     edge of entities and concepts, lifelong learning in decision-making (LLDM) also
                                                     necessitates the transfer of procedural knowledge, such as actions and behaviors.
                                                     To advance research in LLDM, we introduce LIBERO, a novel benchmark of
                                                     lifelong learning for robot manipulation. Specifically, LIBERO highlights five key
                                                     research topics in LLDM: 1) how to efficiently transfer declarative knowledge,
                                                     procedural knowledge, or the mixture of both; 2) how to design effective policy
                                                     architectures and 3) effective algorithms for LLDM; 4) the robustness of a lifelong
                                                     learner with respect to task ordering; and 5) the effect of model pretraining for
                                                     LLDM. We develop an extendible procedural generation pipeline that can in
                                                     principle generate infinitely many tasks. For benchmarking purpose, we create
                                                     four task suites (130 tasks in total) that we use to investigate the above-mentioned
                                                     research topics. To support sample-efficient learning, we provide high-quality
                                                     human-teleoperated demonstration data for all tasks. Our extensive experiments
                                                     present several insightful or even unexpected discoveries: sequential finetuning
                                                     outperforms existing lifelong learning methods in forward transfer, no single visual
                                                     encoder architecture excels at all types of knowledge transfer, and naive supervised
                                                     pretraining can hinder agents’ performance in the subsequent LLDM.2


                                         1       Introduction
                                         A longstanding goal in machine learning is to develop a generalist agent that can perform a wide
                                         range of tasks. While multitask learning [10] is one approach, it is computationally demanding and
                                         not adaptable to ongoing changes. Lifelong learning [65], however, offers a practical solution by
                                         amortizing the learning process over the agent’s lifespan. Its goal is to leverage prior knowledge to
                                         facilitate learning new tasks (forward transfer) and use the newly acquired knowledge to enhance
                                         performance on prior tasks (backward transfer).
                                         The main body of the lifelong learning literature has focused on how agents transfer declarative
                                         knowledge in visual or language tasks, which pertains to declarative knowledge about entities and
                                         concepts [7, 40]. Yet it is understudied how agents transfer knowledge in decision-making tasks,
                                         which involves a mixture of both declarative and procedural knowledge (knowledge about how to do
                                         something). Consider a scenario where a robot, initially trained to retrieve juice from a fridge, fails
                                             ∗
                                                 Equal contribution.
                                             2
                                                 Check the website at https://libero-project.github.io for the code and the datasets.


                                         37th Conference on Neural Information Processing Systems (NeurIPS 2023) Track on Datasets and Benchmarks.
```

## P36 — lingbot_va2_2607.08639.pdf

PDF 第 1 页；共 29 页；SHA256 `62c8dc106268d35de97f8d1fb0448681da238d11194da21e11b3c48e9b2cb6ac`。

```text
                                                        Native Video-Action Pretraining for
                                                           Generalizable Robot Control
                                               Qihang Zhang, Lin Li, Luyao Zhang, Shuai Yang, Yiming Luo, Shuaiting Li, Ruilin Wang, Junke Wang,
                                               Jiahao Shao, Gangwei Xu, Jiaming Zhou, Yishu Shen, Yudong Jin, Fangyi Xu, Shuailei Ma, Jiaqi Liao,
                                                  Guanxing Lu, Zifan Shi, Yongkun Wen, Yujie Zhao, Weixuan Tang, Xinyang Wang, Chaojian Li,
                                                          Jiapeng Zhu, Ka Leong Cheng, Nan Xue, Xing Zhu, Yujun Shen, Yinghao Xu†

                                                                                              †
                                                                                                  Project Lead




                                            The advent of video-action models offers a promising path for robot control. Nevertheless, we argue that
                                            repurposing video generative models designed for digital content creation is inherently inadequate for physical
                                            environments. To bridge this gap, we present LingBot-VA 2.0, a video-action foundation model built from
                                            the ground up for embodiment. Four core design principles showcase its evolution from LingBot-VA. (1)
arXiv:2607.08639v1 [cs.RO] 9 Jul 2026




                                            Departing from traditional reconstruction-focused VAEs, we introduce a semantic visual-action tokenizer,
                                            which aligns visual representations with both semantics and actions, improving instruction following and action
                                            precision in subsequent policy learning. (2) Given the strictly causal nature of temporal dynamics, we adopt a
                                            causal pretraining paradigm, training from scratch to circumvent the catastrophic forgetting that frequently
                                            occurs when adapting bidirectional architectures. (3) To meet the demands of high-frequency inference, our
                                            model employs a sparse MoE backbone, expanding model capacity without compromising efficiency. (4)
                                            Real-time closed-loop control is realized through an enhanced asynchronous inference scheme, which predicts
                                            future latents in parallel with action execution while re-grounding each rollout on the latest observation via
                                            learned forward dynamics. Real-world deployment validates LingBot-VA 2.0 as a robust foundation model,
                                            as evidenced by its few-shot generalization across complex manipulation tasks.

                                            Website: https://technology.robbyant.com/lingbot-va-v2




                                        1    Introduction
                                        Video-action models such as LingBot-VA [50] and DreamZero [115] have recently emerged as a powerful paradigm for
                                        generalist robot manipulation. Rather than mapping observations directly to actions, as reactive vision-language-action
                                        policies do [9, 11, 43], they jointly predict how a scene will evolve and how to act within it, grounding control in
                                        physical dynamics and improving sample efficiency and generalization [35, 54, 132]. Much of this capability, however,
                                        is inherited from their video-pretrained backbones, suggesting that generalist robot control depends as much on the
                                        pretrained foundation as on the policy learned upon it.
                                            Current video-action models are still largely built from components designed for generic video generation—a
                                        reconstruction-oriented VAE and a bidirectional video-diffusion backbone—with an action module added for robotics
                                        afterward. This starting point creates three concrete limitations. First, the representation is optimized for appearance
                                        rather than dynamics: pixel-reconstruction latents preserve visual detail but carry limited semantic and physical structure,
                                        and the separately attached action module leaves world states and actions in poorly aligned spaces. Second, inference is
                                        too slow for closed-loop control: high-dimensional video tokens and iterative denoising make first-generation video-
                                        action models costly to run at the frequencies real robots require. Third, the pretraining signal does not scale toward
                                        control: web-scale video is abundant, but generic video objectives do not teach how actions reshape the world, so the
                                        action signal remains tied to expensive robot data, limiting the control knowledge learned before downstream adaptation.
                                            These limitations are compounded by a structural mismatch: the backbone is pretrained with bidirectional attention,


                                                                                                      1
```

## P37 — lingbot_va_2601.21998.pdf

PDF 第 1 页；共 31 页；SHA256 `57bd993b298765c005fc143e2550243cecb1e76c0682653634a85d34045b0442`。

```text
                                                 Causal World Modeling for Robot Control
                                                         Lin Li∗        Qihang Zhang∗†      Yiming Luo∗      Shuai Yang       Ruilin Wang   Fei Han
                                                            Mingrui Yu        Zelin Gao     Nan Xue      Xing Zhu       Yujun Shen   Yinghao Xu‡

                                                                   ∗                           †                    ‡
                                                                       Equal Contribution          Project Lead         Corresponding Author




                                             This work highlights that video world modeling, alongside vision-language pre-training, establishes a fresh
arXiv:2601.21998v2 [cs.CV] 22 Mar 2026




                                             and independent foundation for robot learning. Intuitively, video world models provide the ability to “imagine”
                                             the near future by understanding the causality between actions and visual dynamics. Inspired by this, we
                                             introduce LingBot-VA, an autoregressive diffusion framework that learns frame prediction and policy execution
                                             simultaneously. Our model features three carefully crafted designs: (1) a shared latent space, integrating
                                             vision and action tokens, driven by a Mixture-of-Transformers (MoT) architecture, (2) a closed-loop rollout
                                             mechanism, allowing for ongoing acquisition of environmental feedback with ground-truth observations, (3)
                                             an asynchronous inference pipeline, parallelizing action prediction and motor execution to support efficient
                                             control. We evaluate our model on both simulation benchmarks and real-world scenarios, where it shows
                                             significant promise in long-horizon manipulation, data efficiency in post-training, and strong generalizability to
                                             novel configurations. The code and model are made publicly available to facilitate the community.

                                             Website: https://technology.robbyant.com/lingbot-va
                                             Github: https://github.com/robbyant/lingbot-va
                                             Checkpoints: https://huggingface.co/robbyant/lingbot-va




                                         1    Introduction
                                         Vision-Language-Action (VLA) models have emerged as a promising paradigm for general-purpose robotic manipula-
                                         tion [7, 11, 12, 34], demonstrating impressive capabilities in grounding linguistic instructions into visual perceptions
                                         across diverse objects and unstructured environments. However, beneath their apparent success lies a significant
                                         challenge: representation entanglement. Most existing VLAs adopt a feedforward paradigm that maps current
                                         observations to action sequences [17, 91], requiring a single neural network to simultaneously learn visual scene
                                         understanding, physical dynamics, and motor control from a unified supervision signal. This entanglement can create a
                                         bottleneck—the model must compress heterogeneous knowledge, ranging from high-dimensional visual semantics to
                                         low-dimensional motor commands, into a shared representation space. This often leads to limited sample efficiency and
                                         suboptimal generalization. Without explicit modeling of environmental evolution [25, 26, 82], reactive policies may rely
                                         on pattern matching rather than a principled understanding of physical dynamics.
                                             Recent attempts to bring world modeling into robotic policies span interactive neural simulators (e.g., UniSim [86]),
                                         chunk-based video-action diffusion models (e.g., UVA [40] and UWM [97]), and offline video generators for subgoal
                                         synthesis (e.g. Gen2Act [4], Act2Goal [95]). While conceptually appealing, these approaches face three primary
                                         limitations for effective closed-loop control. First, the reactivity gap: chunk/open-loop generation often rolls out long
                                         segments without incorporating real-time feedback, making it hard to adapt to disturbances. Second, limited long-term
                                         memory: chunk-wise generation can introduce inconsistencies over long horizons when history is not persistently cached.
                                         Third, causality: bidirectional attention within a segment allows future tokens to influence past predictions, which
                                         diverges from the causal nature of physical reality where the present depends only on the past. These observations
                                         motivate an autoregressive formulation for robust closed-loop reasoning.
                                             We propose LingBot-VA, an autoregressive diffusion world model that addresses these limitations through a unified



                                                                                                         1
```

## P38 — lingbot_video_2607.07675.pdf

PDF 第 1 页；共 51 页；SHA256 `9614b876cda35416a389690b164df91f9600d61121faab1801a2afb94524d647`。

```text
                                            Scaling Mixture-of-Experts Video Pretraining
                                                     for Embodied Intelligence
                                                 Shuailei Ma∗ , Jiaqi Liao∗ , Xinyang Wang∗ , Jingjing Wang∗ , Chaoran Feng, Zijing Hu, Chong Bao,
                                                  Zichen Xi, Yuqi Gan, Weisen Wang, Yanhong Zeng, Qin Zhao, Zifan Shi, Wei Wu, Hao Ouyang,
                                                 Qiuyu Wang, Shangzhan Zhang, Jiahao Shao, Yipengjing Sun, Liangxiao Hu, Lunke Pan, Nan Xue,
                                                              Kecheng Zheng, Yinghao Xu, Xing Zhu, Yujun Shen, Ka Leong Cheng†

                                                                                 ∗                           †
                                                                                     Equal Contribution          Project Lead




                                            Despite the recent promise in robot control, video generative models suffer from a domain mismatch due to their
                                            primary focus on content creation. For example, their design inherently prioritizes visual fidelity and creativity
                                            over computational efficiency and physical realism. In this work, we present LingBot-Video, a DiT-based
arXiv:2607.07675v1 [cs.CV] 8 Jul 2026




                                            video pretraining paradigm specifically tailored for embodied intelligence. From the architecture perspective,
                                            we adopt the Mixture-of-Experts (MoE), instead of dense, framework to achieve a better trade-off between
                                            modeling capacity and inference efficiency, and manage to scale it up from scratch. From the data perspective,
                                            we construct a data profiling engine that augments standard internet videos with extensive robot-oriented footage,
                                            encompassing manipulation, navigation, and egocentric perspectives, to equip the base model with an intrinsic
                                            understanding of actions and world dynamics. From the training perspective, we develop a multi-dimensional
                                            reward system to enforce the alignment regarding physical rationality and task completion, going beyond
                                            standard criteria such as aesthetics, prompt-following, and motion consistency. Comprehensive evaluations
                                            validate its performance and efficiency as a video foundation model. We contribute LingBot-Video as the
                                            inaugural large-scale, open-source MoE video foundation model to the community, in a pioneering effort to
                                            bridge digital creativity and physical actuation.

                                            Website: https://technology.robbyant.com/lingbot-video
                                            Github: https://github.com/robbyant/lingbot-video
                                            Checkpoints: https://huggingface.co/robbyant/lingbot-video




                                        1     Introduction
                                        Beyond their success in content creation, diffusion-based [7, 9, 36, 37, 81, 89, 105] and autoregressive [32, 101, 119]
                                        video models have demonstrated remarkable ability to synthesize temporally coherent and photorealistic sequences
                                        conditioned on text, images, and other control signals [10, 23, 31, 92, 96]. This capability has motivated a growing body
                                        of work that interprets video models as implicit simulators of the physical world, enabling their use in robotics [2, 52, 55],
                                        autonomous driving [77, 78], and interactive environments [10]. In this paradigm, video models serve not only as
                                        generative systems but also as predictive world models [4] that support planning, policy learning, and imagination-based
                                        control. However, translating these models from passive video generation to active embodied reasoning and intelligence
                                        remains an open challenge.
                                            Despite their promise, a fundamental gap persists between video generation models and embodied intelligence
                                        requirements. Most video foundation models are optimized for perceptual quality—such as realism, aesthetics, and text
                                        alignment—rather than physical correctness or controllability. While these objectives yield visually compelling results,
                                        they do not explicitly enforce consistency with physical interaction constraints, such as contact stability, rigid-body
                                        dynamics, or long-horizon state consistency under intervention. This highlights a key tension: while internet-scale video
                                        provides rich visual diversity, it does not guarantee fidelity to the constraints of embodied interaction.


                                                                                                      1
```

## P39 — lingbot_vla2_2607.06403.pdf

PDF 第 1 页；共 20 页；SHA256 `4d15ccc778af7ce315a1efe81a403a7611b5e659f0ddec5570f7b7973302dda1`。

```text
                                                          From Foundation to Application:
                                                         Improving VLA Models in Practice
                                                   Wei Wu∗ , Fangjing Wang∗ , Fan Lu, He Sun, Shi Liu, Yunnan Wang, Yibin Yan, Yong Wang,
                                              Shuailei Ma, Xinyang Wang, Yibin Liu, Shuai Yang, Tianxiang Zhou, Kejia Zhang, Lei Zhou, Cheng Su,
                                                Nan Xue, Bin Tan, Han Zhang, Youchao Zhang, Fei Liao, Xing Zhu, Yujun Shen, Kecheng Zheng†

                                                                               ∗                          †
                                                                                   Equal Contribution         Project Lead




                                            Despite recent progress of VLA foundation models, the disparity between laboratory conditions and real-
                                            world applications continues to impede their practical implementation. To bridge this gap, we present
                                            LingBot-VLA 2.0, which advances LingBot-VLA through improvements in three functional domains.
                                            (1) Generalization across tasks and embodiments. Compared to the previous version, we revamp the data
                                            processing pipeline and curate around 60,000 hours of data for pretraining, including 50,000 hours of robot
arXiv:2607.06403v1 [cs.RO] 7 Jul 2026




                                            trajectories spanning 20 robot configurations and 10,000 hours of egocentric human videos. (2) Expanded
                                            action space in addition to dual-arm hardware platforms. In particular, our system accommodates degrees of
                                            freedom for the heads, waists, mobile bases, and dexterous hands, thereby empowering the robots to tackle
                                            more complex tasks in practical scenarios. (3) Predictive dynamics modeling for improved temporal reasoning.
                                            Specifically, we formulate future prediction as a proxy task, facilitated by a video representation model for
                                            semantic priors and a depth estimation model for geometric cues. Evaluations on the GM-100 benchmark,
                                            conducted in a generalist setting, validate the beneficial impact of these proposed modifications. Furthermore,
                                            benefiting from the expanded pretraining data that covers whole-body degrees of freedom, LingBot-VLA-2.0
                                            demonstrates strong cross-embodiment long-horizon mobile manipulation capability across the two robotic
                                            platforms.

                                            Website: https://technology.robbyant.com/lingbot-vla-v2
                                            Github: https://github.com/robbyant/lingbot-vla-v2
                                            Checkpoints: https://huggingface.co/collections/robbyant/lingbot-vla-v2




                                        1    Introduction
                                        Vision-language-action (VLA) models [4–6, 16] have recently emerged as a promising paradigm for building generalist
                                        robot policies. A key advantage of this paradigm is that pretrained vision-language models provide rich multimodal
                                        alignment and semantic representations, enabling VLA models to better understand complex scenes and generalize
                                        across diverse tasks. Beyond such model-level priors, recent advances [5, 31] further show that scaling up robot data in
                                        both quantity and diversity can substantially improve the capability of VLA systems. Together, these developments have
                                        established VLA as a compelling foundation for robot learning.
                                            However, despite this rapid progress, a substantial gap remains between laboratory benchmarks and real-world
                                        deployment. In practice, robots are expected to operate under broader embodiment diversity, richer action spaces, and
                                        more dynamic environments than those considered in many existing VLA settings. First, generalization in practice is
                                        not only about transferring across tasks, but also about handling heterogeneous robot configurations and data sources.
                                        Another point is that many real-world platforms involve substantially more degrees of freedom than standard dual-arm
                                        manipulation setups, including head movement, waist, mobile-base control, and dexterous hands. Subsequently, real-
                                        world execution often requires anticipating future scene evolution and action consequences, rather than reacting only to
                                        current observations. These challenges collectively limit the practical utility of current VLA foundation models.


                                                                                                   1
```

## P40 — lingbot_vla_2601.18692.pdf

PDF 第 1 页；共 20 页；SHA256 `4c3a330ec97de773802a2d64799cee4226026b11863f7542b6e612a905cb505d`。

```text
                                                        A Pragmatic VLA Foundation Model
                                             Wei Wu∗ , Fan Lu∗ , Yunnan Wang∗ , Shuai Yang∗ , Shi Liu∗ , Fangjing Wang∗ , Qian Zhu, He Sun, Yong Wang,
                                                Shuailei Ma, Yiyu Ren, Kejia Zhang, Hui Yu, Jingmei Zhao, Shuai Zhou, Zhenqi Qiu, Houlong Xiong,
                                             Ziyu Wang, Zechen Wang, Ran Cheng, Yong-Lu Li, Yongtao Huang, Xing Zhu, Yujun Shen, Kecheng Zheng†

                                                                                 ∗                         †
                                                                                     Equal Contribution        Project Lead
arXiv:2601.18692v4 [cs.RO] 15 Jun 2026




                                             Offering great potential in robotic manipulation, a capable Vision-Language-Action (VLA) foundation model is
                                             expected to faithfully generalize across tasks and platforms while ensuring cost efficiency (e.g., data and GPU
                                             hours required for adaptation). To this end, we develop LingBot-VLA with around 20,000 hours of real-world
                                             data from 9 popular dual-arm robot configurations. Through a systematic assessment on 4 robotic platforms,
                                             each completing 100 tasks with 130 post-training episodes per task, our model achieves clear superiority over
                                             competitors, showcasing its strong performance and broad generalizability. We have also built an efficient
                                             codebase, which delivers a throughput of 261 samples per second with an 8-GPU training setup, representing a
                                             1.5 ∼ 2.8× (depending on the relied VLM base model) speedup over existing VLA-oriented codebases. The
                                             above features ensure that our model is well-suited for real-world deployment. To advance the field of robot
                                             learning, we provide open access to the code, base model, and benchmark data, with a focus on enabling more
                                             challenging tasks and promoting sound evaluation standards.

                                             Website: https://technology.robbyant.com/lingbot-vla
                                             Github: https://github.com/robbyant/lingbot-vla
                                             Checkpoints: https://huggingface.co/collections/robbyant/lingbot-vla




                                         1    Introduction
                                         Vision-Language-Action (VLA) foundation models [5, 6, 27] have emerged as a promising method for enabling robots
                                         to perform diverse manipulation tasks guided by natural language instructions. Through large-scale pre-training, these
                                         models acquire generalizable skills that can be rapidly adapted to diverse tasks and robotic platforms. Despite the
                                         significant progress, there remains a lack of comprehensive empirical studies on how real-robot performance scales with
                                         increasingly vast pre-training datasets. Moreover, the community lacks a highly optimized training codebase capable of
                                         efficiently conducting these scaling evaluations on massive volumes of data. Consequently, a fundamental question that
                                         demands investigation in the real-world setting is: How do VLA models truly scale with massive real-world robot data?
                                             Understanding the scaling behavior of VLA models is crucial for robotic learning, especially on vast and diverse
                                         real-world datasets. In this work, we provide a systematic empirical investigation into how success rates scale with
                                         respect to data volume and diversity during VLA pre-training By scaling pre-training data from 3,000 hours to 20,000
                                         hours, we demonstrate that downstream success rates improve consistently and substantially. Notably, this scaling
                                         behavior shows no signs of saturation even at the 20,000-hour mark, suggesting that VLA performance continues to
                                         benefit from increased data volume. These results provide the first empirical evidence of favorable scaling properties in
                                         real-world robot learning, offering critical insights for future VLA development and large-scale data curation.
                                             While scaling analysis reveals favorable performance trends, translating these insights into reliable, deployable
                                         systems necessitates rigorous evaluation on real robotic platforms at a large scale. Thanks to GM-100 [29], which
                                         provides 100 carefully designed tasks, we conduct a systematic assessment across 4 robotic platforms, involving 130
                                         episodes per task per embodiment. By emphasizing task diversity and multi-platform consistency, our evaluation
                                         framework provides a choice of new standards for sound VLA benchmarking.



                                                                                                     1
```

## P41 — lingbot_vla_v2_2607.06403.pdf

PDF 第 1 页；共 20 页；SHA256 `4d15ccc778af7ce315a1efe81a403a7611b5e659f0ddec5570f7b7973302dda1`。

```text
                                                          From Foundation to Application:
                                                         Improving VLA Models in Practice
                                                   Wei Wu∗ , Fangjing Wang∗ , Fan Lu, He Sun, Shi Liu, Yunnan Wang, Yibin Yan, Yong Wang,
                                              Shuailei Ma, Xinyang Wang, Yibin Liu, Shuai Yang, Tianxiang Zhou, Kejia Zhang, Lei Zhou, Cheng Su,
                                                Nan Xue, Bin Tan, Han Zhang, Youchao Zhang, Fei Liao, Xing Zhu, Yujun Shen, Kecheng Zheng†

                                                                               ∗                          †
                                                                                   Equal Contribution         Project Lead




                                            Despite recent progress of VLA foundation models, the disparity between laboratory conditions and real-
                                            world applications continues to impede their practical implementation. To bridge this gap, we present
                                            LingBot-VLA 2.0, which advances LingBot-VLA through improvements in three functional domains.
                                            (1) Generalization across tasks and embodiments. Compared to the previous version, we revamp the data
                                            processing pipeline and curate around 60,000 hours of data for pretraining, including 50,000 hours of robot
arXiv:2607.06403v1 [cs.RO] 7 Jul 2026




                                            trajectories spanning 20 robot configurations and 10,000 hours of egocentric human videos. (2) Expanded
                                            action space in addition to dual-arm hardware platforms. In particular, our system accommodates degrees of
                                            freedom for the heads, waists, mobile bases, and dexterous hands, thereby empowering the robots to tackle
                                            more complex tasks in practical scenarios. (3) Predictive dynamics modeling for improved temporal reasoning.
                                            Specifically, we formulate future prediction as a proxy task, facilitated by a video representation model for
                                            semantic priors and a depth estimation model for geometric cues. Evaluations on the GM-100 benchmark,
                                            conducted in a generalist setting, validate the beneficial impact of these proposed modifications. Furthermore,
                                            benefiting from the expanded pretraining data that covers whole-body degrees of freedom, LingBot-VLA-2.0
                                            demonstrates strong cross-embodiment long-horizon mobile manipulation capability across the two robotic
                                            platforms.

                                            Website: https://technology.robbyant.com/lingbot-vla-v2
                                            Github: https://github.com/robbyant/lingbot-vla-v2
                                            Checkpoints: https://huggingface.co/collections/robbyant/lingbot-vla-v2




                                        1    Introduction
                                        Vision-language-action (VLA) models [4–6, 16] have recently emerged as a promising paradigm for building generalist
                                        robot policies. A key advantage of this paradigm is that pretrained vision-language models provide rich multimodal
                                        alignment and semantic representations, enabling VLA models to better understand complex scenes and generalize
                                        across diverse tasks. Beyond such model-level priors, recent advances [5, 31] further show that scaling up robot data in
                                        both quantity and diversity can substantially improve the capability of VLA systems. Together, these developments have
                                        established VLA as a compelling foundation for robot learning.
                                            However, despite this rapid progress, a substantial gap remains between laboratory benchmarks and real-world
                                        deployment. In practice, robots are expected to operate under broader embodiment diversity, richer action spaces, and
                                        more dynamic environments than those considered in many existing VLA settings. First, generalization in practice is
                                        not only about transferring across tasks, but also about handling heterogeneous robot configurations and data sources.
                                        Another point is that many real-world platforms involve substantially more degrees of freedom than standard dual-arm
                                        manipulation setups, including head movement, waist, mobile-base control, and dexterous hands. Subsequently, real-
                                        world execution often requires anticipating future scene evolution and action consequences, rather than reacting only to
                                        current observations. These challenges collectively limit the practical utility of current VLA foundation models.


                                                                                                   1
```

## P42 — next_forcing.pdf

PDF 第 1 页；共 18 页；SHA256 `05ac346721d5487580649919f73e37df74ca87341496534b764738517e9f5903`。

```text
                                                    Next Forcing: Causal World Modeling with
                                                             Multi-Chunk Prediction


                                                        Gangwei Xu1,2          Qihang Zhang1,†          Jiaming Zhou1,4     Xing Zhu1
                                                                      Yujun Shen1       Xin Yang2,‡       Yinghao Xu3,1,‡
                                                                  1                2              3           4
                                                                      Robbyant         HUST           HKUST       HKUST (GZ)
                                                                         †                    ‡
arXiv:2606.11187v1 [cs.CV] 9 Jun 2026




                                                                             Project Lead         Corresponding Author



                                                                                            Abstract
                                                    Autoregressive video generation has emerged as a powerful paradigm for World
                                                    Action Models (WAMs). However, existing approaches suffer from slow training
                                                    convergence and limited converged accuracy, particularly at high frame rates, as
                                                    the training supervision is confined to the current chunk without explicit signals
                                                    about future dynamics; they also suffer from slow inference due to iterative video
                                                    denoising. In this paper, we present Next Forcing, a multi-chunk prediction (MCP)
                                                    framework for causal world modeling that enables faster training, higher accuracy,
                                                    and accelerated inference. Inspired by multi-token prediction in large language
                                                    models, Next Forcing introduces an MCP training objective that augments the main
                                                    model with lightweight auxiliary MCP modules to simultaneously denoise video
                                                    chunks at multiple future temporal horizons (next1 , next2 , next3 chunks). These
                                                    MCP modules form a causal chain across prediction depths, where intermediate
                                                    features fused from multiple layers of the main model are leveraged to predict
                                                    future dynamics, allowing near-future predictions to inform farther-future ones and
                                                    providing dense multi-scale temporal supervision back to the main model. During
                                                    training, the MCP modules significantly accelerate convergence and improve
                                                    converged accuracy, especially at high frame rates: at 50 fps, Next Forcing achieves
                                                    a 93.1% relative improvement over LingBot-VA at 5k training steps and 2.3×
                                                    faster convergence, and establishes new state-of-the-art results on the RoboTwin
                                                    benchmark (94.1/93.5% on Clean/Random). At inference, the MCP modules
                                                    can be retained to predict the next video chunk in parallel with the current one,
                                                    achieving 2× inference acceleration. Next Forcing also demonstrates significant
                                                    improvements on PhyWorld, a benchmark evaluating adherence to physical laws
                                                    in video generation, and over 50% FVD reduction on general video pretraining.
                                                    Project website: https://gangweix.github.io/next-forcing/.


                                        1    Introduction
                                        Videos capture how the physical world evolves and how agents act within it, recording the dynamics
                                        of physical interactions at scale [24, 25, 26, 50, 9, 1, 2]. Building on this, World Action Models
                                        (WAMs) [36, 62] have recently emerged as a new paradigm for embodied AI, learning manipulation
                                        policies by jointly modeling future video and actions. The dominant training objective for WAMs is
                                        teacher-forced next-chunk denoising, where the model denoises the noisy current chunk conditioned
                                        on clean past chunks from ground-truth data. Despite the recent achievements of this paradigm
                                        [36, 62], teacher-forced next-chunk denoising remains an inefficient and shortcut-prone training
                                        signal for video world models. More precisely, predicting the next chunk is a fundamentally local
                                        task, which admits an appearance shortcut [21]: since adjacent chunks are visually highly similar,
                                        much of the denoising loss can be driven down by learning a near-identity map from the clean past

                                        Preprint.
```

## P43 — openvla_2406.09246.pdf

Syntax Error: Couldn't find trailer dictionary
Syntax Error: Invalid XRef entry 254
Syntax Error: Invalid XRef entry 904
Syntax Error: Top-level pages object is wrong type (null)
Command Line Error: Wrong page range given: the first page (1) can not be after the last page (0).


## P44 — pi0_2410.24164.pdf

PDF 第 1 页；共 17 页；SHA256 `fbdfba56258bdbd220207ba63410c6c943e43a73e97a7a5a47ca0bc74204f821`。

```text
                                               π0: A Vision-Language-Action Flow Model for
                                                           General Robot Control
                                                                                    Physical Intelligence
                                             Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai,
                                           Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine,
                                          Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong,
                                                                       Anna Walling, Haohuan Wang, Ury Zhilinsky
                                                                        https://physicalintelligence.company/blog/pi0
arXiv:2410.24164v4 [cs.LG] 8 Jan 2026




                                                Z)0./GSO/,SGL                        G.G 0%G)     °2G)©¯SO/,SG )¡ G                
                                                                                          2.G .(
                                                   ./O/ ), G


                                                                                                                                                 O') O%G
                                                            O') O%G     20 )O/ %G

                                                                                                                                                    # ! 2/  .() /)
                                                                                          mlhg/%('(G0           /)S/,G%                        ,&&0'% )  
                                                          O()(./0G.G   /2G)2/20/.                                           0 /)
                                                                                                  2.G .G,)|{z                G2G.

                                                                                                                                                 GS2 ³)2. SG ),.³G.

                                                           %/,),pG     /. )%G(/
                                                                                                                  y&/%,)p. n


                                                                                                                                                 O 0p)&/%,)p. 

                                                           S G)0/&&GG    ÀGG2) O%G     Õ(pÓ'% ³)2/  .(), 
                                                                                                                                                     8::!7 2/  .() /)
                                                                                                                                                            'GG)  

                                                           GS2 ³),.³G.     G ) O%G
                                                                                                   p(p),G G. ³)  
                                                                                                                                                 2' ) GS)),.ÀG.


                                                           20 )pG%&     &% G)O/

                                                               ,)S³)S/.G±
                                                                                                   'GG)  
                                                                                                                                                 .G2%0G)22G.) /ÀG%




                                        Fig. 1: Our generalist robot policy uses a pre-trained vision-language model (VLM) backbone, as well as a diverse cross-
                                        embodiment dataset with a variety of dexterous manipulation tasks. The model is adapted to robot control by adding a separate
                                        action expert that produces continuous actions via flow matching, enabling precise and fluent manipulation skills. The model
                                        can then be used directly to perform tasks based on a prompt, or fine-tuned on high-quality data to enable complex multi-stage
                                        tasks, such as folding multiple articles of laundry or assembling a box.


                                           Abstract—Robot learning holds tremendous promise to unlock                      built on top of a pre-trained vision-language model (VLM)
                                        the full potential of flexible, general, and dexterous robot systems,              to inherit Internet-scale semantic knowledge. We then discuss
                                        as well as to address some of the deepest questions in artificial                  how this model can be trained on a large and diverse dataset
                                        intelligence. However, bringing robot learning to the level of                     from multiple dexterous robot platforms, including single-arm
                                        generality required for effective real-world systems faces major                   robots, dual-arm robots, and mobile manipulators. We evaluate
                                        obstacles in terms of data, generalization, and robustness. In                     our model in terms of its ability to perform tasks via direct
                                        this paper, we discuss how generalist robot policies (i.e., robot                  prompting, follow language instructions from people and from a
                                        foundation models) can address these challenges, and how we can                    high-level VLM policy, and its ability to acquire new skills via
                                        design effective generalist robot policies for complex and highly                  fine-tuning. Our results cover a wide variety of tasks, such as
                                        dexterous tasks. We propose a novel flow matching architecture                     laundry folding, table cleaning, and assembling boxes.

                                          Physical Intelligence, San Francisco, California, USA. Correspondance to:
                                        research@physicalintelligence.company
```

## P45 — prism.pdf

PDF 第 1 页；共 12 页；SHA256 `8a7d22932d61665701d1eadd5222aeca96031f0720e6dcfe5f02af48dd6bbf70`。

```text
                                                                   The Prism Hypothesis:
                                            Harmonizing Semantic and Pixel Representations via Unified Autoencoding

                                                            Weichen Fan1,2 Haiwen Diao1 Quan Wang2 Dahua Lin2 Ziwei Liu1,B
                                                               1
                                                                 S-Lab, Nanyang Technological University 2 SenseTime Research
                                                                             weichen002@e.ntu.edu.sg,                   haiwen.diao@ntu.edu.sg,
                                                                       {wangquan,dhlin}@sensetime.com,                            ziwei.liu@ntu.edu.sg
arXiv:2512.19693v5 [cs.CV] 1 Apr 2026




                                                             Github:                 https://github.com/WeichenFan/UAE.
                                                             Hugging Face:           https://huggingface.co/weepiess2383/UAE.



                                              Low-Level            (Image)
                                               Appearance                                                                                                                                             Low-Frequency
                                               Geometry …                                                                                                                                  V   iT …
                                                                                                                                                                          IP,        Intern
                                                                                                                                                                   , SigL                                  Visual
                                              High-Level                                                                                                       DINO
                                                                                                                                                       oders                                             Abstraction
                                               Category                                                   Frequency-Band                        tic Enc
                                               Attribute                                                                                   Seman
                                               Relation …                                                    Modulator
                                                                    (Text)                                                                  Pix                                                          UAE
                                                                                                                                               el E
                                              High-Level                                                                                           nco
                                                                                                                                                      der
                                               Category                                                                                                  s
                                                                                                                                                             SD                                          Visual
                                               Attribute                                                                                                        3-V
                                                                                                                                                                    A   E, F
                                               Relation …                                   Starti                                                                          lux                          Fidelity
                                                                                                                                                                               -VA
                                                                                                  ng / E                         ncoders                                             E…
                                                                                                        volvin            antic E
                                                                                                              g   from Sem                                                                            High-Frequency



                                        Figure 1. The Prism Hypothesis. Our conceptual “prism” decomposes various natural inputs into spectral components along frequency.
                                        Low frequency bands capture global semantics and abstract meaning, while high frequency bands encode local detail and fine visual texture.
                                        This motivates our Unified Autoencoding (UAE), which harmonizes semantic and pixel representations within a single latent space.

                                                                  Abstract                                             that UAE effectively unifies semantic abstraction and pixel-
                                                                                                                       level fidelity within a single latent space, achieving state-of-
                                                                                                                       the-art performance. Moreover, we show that UAE can be
                                        Deep representations across modalities are inherently in-
                                                                                                                       directly applied to pixel-space modeling, significantly im-
                                        tertwined. In this paper, we systematically analyze the
                                                                                                                       proving both FID and IS over the vanilla JIT baseline.
                                        spectral characteristics of various semantic and pixel en-
                                        coders. Interestingly, our study uncovers a highly inspiring                   1. Introduction
                                        and rarely explored correspondence between an encoder’s
                                        feature spectrum and its functional role: semantic encoders                    Trained on massive corpora, recent foundation models have
                                        primarily capture low-frequency components that encode                         profoundly reshaped perception and generation systems,
                                        abstract meaning, whereas pixel encoders additionally re-                      generalizing well across diverse downstream tasks [3, 24,
                                        tain high-frequency information that conveys fine-grained                      26, 30]. Yet, early advances in perception and generation
                                        detail. This heuristic finding offers a unifying perspective                   evolve along largely separate trajectories. Their objectives
                                        that ties encoder behavior to its underlying spectral struc-                   are typically distributed across distinct network structures,
                                        ture. We define it as the Prism Hypothesis, where each data                    e.g., employing pretrained semantic encoders [3, 26, 30],
                                        modality can be viewed as a projection of the natural world                    to capture high-level meaning, or pixel encoders [17, 32]
                                        onto a shared feature spectrum, just like the prism. Building                  to compress fine-grained visual detail. While each module
                                        on this insight, we propose Unified Autoencoding (UAE), a                      excels within its own domain, this fragmentation compels
                                        model that harmonizes semantic structure and pixel details                     subsequent unification efforts [10, 29, 38] to depend simul-
                                        via an innovative frequency-band modulator, enabling their                     taneously on semantic and pixel encoders, forcing networks
                                        seamless coexistence. Extensive experiments demonstrate                        to reconcile fundamentally heterogeneous representations.


                                                                                                                  1
```

## P46 — qwen25_vl_2502.13923.pdf

PDF 第 1 页；共 23 页；SHA256 `399a28f6ed79f4ece29cca93993fceb56fc304b1d779e49392aec05aff45aba5`。

```text
                                                                                                                             March 5, 2025


                                                               Qwen2.5-VL Technical Report
                                                                       Qwen Team, Alibaba Group

                                                                  https://chat.qwenlm.ai
                                                                  https://huggingface.co/Qwen
                                                                  https://modelscope.cn/organization/qwen
                                                                  https://github.com/QwenLM/Qwen2.5-VL

                                                                                Abstract

                                         We introduce Qwen2.5-VL, the latest flagship model of Qwen vision-language series,
                                         which demonstrates significant advancements in both foundational capabilities and
                                         innovative functionalities. Qwen2.5-VL achieves a major leap forward in understanding
                                         and interacting with the world through enhanced visual recognition, precise object local-
                                         ization, robust document parsing, and long-video comprehension. A standout feature of
arXiv:2502.13923v1 [cs.CV] 19 Feb 2025




                                         Qwen2.5-VL is its ability to localize objects using bounding boxes or points accurately. It
                                         provides robust structured data extraction from invoices, forms, and tables, as well as
                                         detailed analysis of charts, diagrams, and layouts. To handle complex inputs, Qwen2.5-
                                         VL introduces dynamic resolution processing and absolute time encoding, enabling it
                                         to process images of varying sizes and videos of extended durations (up to hours) with
                                         second-level event localization. This allows the model to natively perceive spatial scales
                                         and temporal dynamics without relying on traditional normalization techniques. By
                                         training a native dynamic-resolution Vision Transformer (ViT) from scratch and incorpo-
                                         rating Window Attention, we have significantly reduced computational overhead while
                                         maintaining native resolution. As a result, Qwen2.5-VL excels not only in static image
                                         and document understanding but also as an interactive visual agent capable of reasoning,
                                         tool usage, and task execution in real-world scenarios such as operating computers and
                                         mobile devices. The model achieves strong generalization across domains without requir-
                                         ing task-specific fine-tuning. Qwen2.5-VL is available in three sizes, addressing diverse
                                         use cases from edge AI to high-performance computing. The flagship Qwen2.5-VL-72B
                                         model matches state-of-the-art models like GPT-4o and Claude 3.5 Sonnet, particularly
                                         excelling in document and diagram understanding. The smaller Qwen2.5-VL-7B and
                                         Qwen2.5-VL-3B models outperform comparable competitors, offering strong capabilities
                                         even in resource-constrained environments. Additionally, Qwen2.5-VL maintains robust
                                         linguistic performance, preserving the core language competencies of the Qwen2.5 LLM.




                                                                                     1
```

## P47 — qwen2_vl_2409.12191.pdf

PDF 第 1 页；共 52 页；SHA256 `30bb0f6c9babf910295d6b7cd34b6f54824236f68782664c3c93fbf4803275d0`。

```text
                                         Qwen2-VL: Enhancing Vision-Language Model’s Perception
                                                     of the World at Any Resolution

                                                     Peng Wang* Shuai Bai* Sinan Tan* Shijie Wang* Zhihao Fan* Jinze Bai*†
                                                Keqin Chen Xuejing Liu Jialin Wang Wenbin Ge Yang Fan Kai Dang Mengfei Du
                                                 Xuancheng Ren Rui Men Dayiheng Liu Chang Zhou Jingren Zhou Junyang Lin†
arXiv:2409.12191v2 [cs.CV] 3 Oct 2024




                                                                           Qwen Team Alibaba Group


                                                                                                  Abstract

                                                           We present the Qwen2-VL Series, an advanced upgrade of the previous Qwen-VL models
                                                           that redefines the conventional predetermined-resolution approach in visual processing.
                                                           Qwen2-VL introduces the Naive Dynamic Resolution mechanism, which enables the
                                                           model to dynamically process images of varying resolutions into different numbers of
                                                           visual tokens. This approach allows the model to generate more efficient and accurate
                                                           visual representations, closely aligning with human perceptual processes. The model also
                                                           integrates Multimodal Rotary Position Embedding (M-RoPE), facilitating the effective
                                                           fusion of positional information across text, images, and videos. We employ a unified
                                                           paradigm for processing both images and videos, enhancing the model’s visual perception
                                                           capabilities. To explore the potential of large multimodal models, Qwen2-VL investigates
                                                           the scaling laws for large vision-language models (LVLMs). By scaling both the model
                                                           size-with versions at 2B, 8B, and 72B parameters-and the amount of training data, the
                                                           Qwen2-VL Series achieves highly competitive performance. Notably, the Qwen2-VL-72B
                                                           model achieves results comparable to leading models such as GPT-4o and Claude3.5-
                                                           Sonnet across various multimodal benchmarks, outperforming other generalist models.
                                                           Code is available at https://github.com/QwenLM/Qwen2-VL.



                                        1    Introduction

                                        In the realm of artificial intelligence, Large Vision-Language Models (LVLMs) represent a significant leap
                                        forward, building upon the strong textual processing capabilities of traditional large language models. These
                                        advanced models now encompass the ability to interpret and analyze a broader spectrum of data, including
                                        images, audio, and video. This expansion of capabilities has transformed LVLMs into indispensable tools for
                                        tackling a variety of real-world challenges. Recognized for their unique capacity to condense extensive and
                                        intricate knowledge into functional representations, LVLMs are paving the way for more comprehensive
                                        cognitive systems. By integrating diverse data forms, LVLMs aim to more closely mimic the nuanced ways in
                                        which humans perceive and interact with their environment. This allows these models to provide a more
                                        accurate representation of how we engage with and perceive our environment
                                        Recent advancements in large vision-language models (LVLMs) (Li et al., 2023c; Liu et al., 2023b; Dai et al.,
                                        2023; Zhu et al., 2023; Huang et al., 2023a; Bai et al., 2023b; Liu et al., 2023a; Wang et al., 2023b; OpenAI.,
                                        2023; Team et al., 2023) have led to significant improvements in a short span. These models (OpenAI, 2023;
                                        Touvron et al., 2023a,b; Chiang et al., 2023; Bai et al., 2023a) generally follow a common approach of visual
                                        encoder→cross-modal connector→LLM. This setup, combined with next-token prediction as the primary training
                                        method and the availability of high-quality datasets (Liu et al., 2023a; Zhang et al., 2023; Chen et al., 2023b;
                                            ∗ Equal   core contribution, † Corresponding author



                                                                                                      1
```

## P48 — qwen3_vl_2511.21631.pdf

PDF 第 1 页；共 42 页；SHA256 `ee075d08e67de1148d6437c6c1d481f7894183b8793905a2deb5f62664f49380`。

```text
                                                                                                                       December 1, 2025


                                                               Qwen3-VL Technical Report
                                                                              Qwen Team

                                                                 https://chat.qwen.ai
                                                                 https://huggingface.co/Qwen
                                                                 https://modelscope.cn/organization/qwen
                                                                 https://github.com/QwenLM/Qwen3-VL

                                                                               Abstract

                                         We introduce Qwen3-VL, the most capable vision–language model in the Qwen series to
                                         date, achieving superior performance across a broad range of multimodal benchmarks.
                                         It natively supports interleaved contexts of up to 256K tokens, seamlessly integrat-
                                         ing text, images, and video. The model family includes both dense (2B/4B/8B/32B)
arXiv:2511.21631v2 [cs.CV] 27 Nov 2025




                                         and mixture-of-experts (30B-A3B/235B-A22B) variants to accommodate diverse la-
                                         tency–quality trade-offs. Qwen3-VL delivers three core pillars: (i) markedly stronger
                                         pure-text understanding, surpassing comparable text-only backbones in several cases;
                                         (ii) robust long-context comprehension with a native 256K-token window for both text
                                         and interleaved multimodal inputs, enabling faithful retention, retrieval, and cross-
                                         referencing across long documents and videos; and (iii) advanced multimodal reasoning
                                         across single-image, multi-image, and video tasks, demonstrating leading performance
                                         on comprehensive evaluations such as MMMU and visual-math benchmarks (e.g., Math-
                                         Vista and MathVision). Architecturally, we introduce three key upgrades: (i) an enhanced
                                         interleaved-MRoPE for stronger spatial–temporal modeling across images and video; (ii)
                                         DeepStack integration, which effectively leverages multi-level ViT features to tighten
                                         vision–language alignment; and (iii) text-based time alignment for video, evolving from
                                         T-RoPE to explicit textual timestamp alignment for more precise temporal grounding. To
                                         balance text-only and multimodal learning objectives, we apply square-root reweight-
                                         ing, which boosts multimodal performance without compromising text capabilities.
                                         We extend pretraining to a context length of 256K tokens and bifurcate post-training
                                         into non-thinking and thinking variants to address distinct application requirements.
                                         Furthermore, we allocate additional compute resources to the post-training phase to
                                         further enhance model performance. Under comparable token budgets and latency
                                         constraints, Qwen3-VL achieves superior performance in both dense and Mixture-of-
                                         Experts (MoE) architectures. We envision Qwen3-VL serving as a foundational engine for
                                         image-grounded reasoning, agentic decision-making, and multimodal code intelligence
                                         in real-world workflows.




                                                                                    1
```

## P49 — qwen_robotmanip_2606.17846.pdf

PDF 第 1 页；共 44 页；SHA256 `193fff177785d4e9d9cc63dca02b61b4e250ec174535c45dafe7a625ed8ed025`。

```text
                                                                                                                                                                                                                           June 18, 2026


                                             Qwen-RobotManip Technical Report: Alignment Unlocks Scale for
                                                      Robotic Manipulation Foundation Models
                                                                                                                                    Qwen Team

                                                                                                     https://qwen.ai/blog?id=qwen-robotmanip
                                                                                                     https://github.com/QwenLM/Qwen-RobotManip

                                                                                                                                         Abstract

                                                         Foundation models in language and multimodality achieve strong generalization be-
                                                         cause heterogeneous data sources can be aligned under a unified formulation, and
                                                         abundant low-cost data from the internet allows diverse training signals to reinforce
                                                         one another at scale. In this report, we investigate whether this scaling recipe can be
                                                         applied to robotic manipulation to achieve genuine generalization. This is challeng-
arXiv:2606.17846v2 [cs.RO] 17 Jun 2026




                                                         ing because, unlike text, manipulation data is heterogeneous by nature, expensive to
                                                         collect, and narrow in diversity, making alignment and scale simultaneously difficult
                                                         to achieve. We present Q WEN -R OBOT M ANIP, a generalizable Vision-Language-Action
                                                         foundation model built upon Qwen-VL. Q WEN -R OBOT M ANIP introduces a unified
                                                         alignment framework across the representation, motion, and behavioral dimensions of
                                                         manipulation, making large-scale multi-source training coherent rather than conflicting.
                                                         This alignment capability in turn enables Q WEN -R OBOT M ANIP to absorb manipulation
                                                         data at a scale that prior training regimes could not sustain. To provide a scaling en-
                                                         gine for manipulation data, a human-to-robot synthesis pipeline converts egocentric
                                                         hand demonstrations into robot trajectories across 15 platforms, and a rigorous curation
                                                         pipeline harmonizes heterogeneous real-robot and synthetic datasets. To our surprise,
                                                         by leveraging only open-source robotic manipulation datasets and human demonstra-
                                                         tion videos without any proprietary data collection, Q WEN -R OBOT M ANIP constructs a
                                                         ∼38,100-hour pretraining corpus and already exhibits emergent generalization capabili-
                                                         ties, including zero-shot instruction following, robustness to perturbations, reactive error
                                                         recovery, and cross-embodiment knowledge transfer. In experiments, we further find
                                                         that most standard benchmarks systematically fail to capture the quality of pretraining.
                                                         Thus, we instead adopt OOD evaluation settings, including RoboCasa365, LIBERO-Plus,
                                                         EBench, RoboTwin-Clean2Rand, RoboTwin-IF (our new instruction-following bench-
                                                         mark), and RoboTwin-XE (our new cross-embodiment transfer benchmark), as our north
                                                         star for measuring genuine generalization. Q WEN -R OBOT M ANIP achieves substantially
                                                         better performance than prior state-of-the-art models, including π0.5 , across all OOD
                                                         settings, ranks 1st in RoboChallenge with a 20% relative improvement, and is validated
                                                         on real-robot platforms including AgileX ALOHA, Franka, UR, and ARX.



                                         1    Scaling Robotic Manipulation Data                            2   Unified Cross-Embodiment Alignment                                      3    Performance
                                                                                                                                                                                                  In-distribution
                                                                                                                                                                                                                               Real-World
                                                                                                                                                                                                   Performance
                                             Human-to-Robot Synthesis (15 platforms)                                                                                 Dexterous
                                                                                                                                                                                                                               Evaluation
                                                                                                                   Joint Pos.     EEF Pose           Gripper
                                                                                                                                                                       Hand

                                                                                             Representation Alignment
                                                                                             Shared canonical state vector
                                                                                                                                     Unified                     Motion Alignment
                                                   Diverse Robot Embodiments                                                        EEF Pose                   Camera-centric consistency                                               Cross-Embodiment
                                                                                                                                                                                                                                             Transfer

                                                                                      ...
                                                                                                                                                                                            Task & Scene
                                                                                                                                                                                            Generalization                         Instruction
                                                                                                                                                                                                                                    Following

                                                    Multi-source Data Curation

                                                                                                                                                                                             4     Scaling Law (Emergent Generalization)
                                                                                                           ...              ...                ...                         ...
                                                                                                                  Vision                  System Prompt           State Context
                                                        > 38,100 Hours
                                                Heterogeneous Manipulation Data                                  Behavior Alignment
                                                Robot Data   Human Videos   Synthetic Data                       System prompt and In-
                                                + Vision-Language Co-training Data                                 context adaptation




                                                                                                                                               1
```

## P50 — spikingbrain_2509.05276.pdf

PDF 第 1 页；共 40 页；SHA256 `ff1b04180cdfe2c189f25526ee5a8daec8de32d4e8b6ef015dd1d157ebb345be`。

```text
                                        SpikingBrain: Spiking Brain-inspired Large Models

                                        Yuqi Pan1,2,3 , Yupeng Feng1 , Jinghao Zhuang1 , Siyu Ding1 , Han Xu1,4 , Zehao Liu1,5 ,
                                        Bohan Sun1 , Yuhong Chou1,5 , Xuerui Qiu1,6 , Anlin Deng1 , Anjie Hu1,6,7 , Shurong Wang1,8 ,
                                        Peng Zhou9 , Man Yao1,2,3 , Jibin Wu5 , Jian Yang10 , Guoliang Sun10 , Bo Xu1,2∗ & Guoqi Li1,2,3∗

                                        1
                                          Institute of Automation, Chinese Academy of Sciences
                                        2
                                          Beijing Key Laboratory of Brain-Inspired General Intelligence Large Model
                                        3
                                          Key Laboratory of Brain Cognition and Brain-inspired Intelligence Technology
                                        4
                                          Beijing Academy of Artificial Intelligence 5 The Hong Kong Polytechnic University
                                        6
                                          Zhongguancun Academy 7 Beihang University 8 Zhejiang University
arXiv:2509.05276v4 [cs.LG] 8 May 2026




                                        9
                                          LuxiTech 10 MetaX Integrated Circuit Co., Ltd.



                                                                                                  Abstract

                                                    Mainstream Transformer-based large language models (LLMs) face significant efficiency
                                                    bottlenecks: training computation scales quadratically with sequence length, and inference
                                                    memory grows linearly. These constraints limit their ability to process long sequences
                                                    effectively. In addition, building large models on non-NVIDIA computing platforms poses
                                                    major challenges in achieving stable and efficient training and deployment. To address these
                                                    issues, we introduce SpikingBrain, a new family of brain-inspired models designed for efficient
                                                    long-context training and inference. SpikingBrain leverages the MetaX1 GPU cluster and
                                                    focuses on three core aspects: i) Model Architecture: linear and hybrid-linear attention
                                                    architectures with adaptive spiking neurons; ii) Algorithmic Optimizations: an efficient,
                                                    conversion-based training pipeline compatible with existing LLMs, along with a dedicated
                                                    spike coding framework; iii) System Engineering: customized training frameworks, op-
                                                    erator libraries, and parallelism strategies tailored to the MetaX hardware. Using these
                                                    techniques, we develop two models: SpikingBrain-7B, a linear LLM, and SpikingBrain-
                                                    76B, a hybrid-linear MoE LLM. These models demonstrate the feasibility of large-scale LLM
                                                    development on non-NVIDIA platforms, and our training framework supports weeks of stable
                                                    training on hundreds of MetaX GPUs with Model FLOPs Utilization (MFU) at expected
                                                    levels. SpikingBrain achieves performance comparable to open-source Transformer baselines
                                                    while using exceptionally low data resources (continual pre-training of ∼150B tokens). Our
                                                    models also significantly improve long-context efficiency and deliver inference with (partially)
                                                    constant memory and event-driven spiking behavior. For example, SpikingBrain-7B achieves
                                                    more than 100× speedup in Time to First Token (TTFT) for 4M-token sequences. Further-
                                                    more, the proposed spiking scheme achieves 69.15% sparsity, enabling low-power operation.
                                                    Overall, this work demonstrates the potential of brain-inspired mechanisms to drive the next
                                                    generation of efficient and scalable large model design. 2


                                        1     Introduction

                                        Recent advances in large language models (LLMs) built on the Transformer architecture (Vaswani et al.,
                                        2017) have been driven by the scaling law (Kaplan et al., 2020), which suggests that performance improves
                                        with larger model sizes and more data (OpenAI, 2025; Google DeepMind, 2025; Anthropic, 2025). However,
                                        this scale-driven approach comes with significant challenges: extremely high training costs, substantial energy
                                        consumption, and complex deployment pipelines. Therefore, achieving high performance and energy efficiency
                                            ∗ Corresponding   authors: xubo@ia.ac.cn and guoqi.li@ia.ac.cn
                                            1 https://www.metax-tech.com/en
                                            2 The   code for this project is publicly available at https://github.com/BICLab/SpikingBrain-7B.


                                                                                                        1
```

## P51 — visual grounding survey.pdf

PDF 第 1 页；共 30 页；SHA256 `4a10fda37e04e63e12ee7af5e980285976c3adeaa320ffee633bc9bea8a803dc`。

```text
                                        IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, NOVEMBER 2025                                                                         1




                                                            Toward Visual Grounding: A Survey
                                                                                 Linhui Xiao , Xiaoshan Yang , Xiangyuan Lan ,
                                                                  Yaowei Wang , Member, IEEE, and Changsheng Xu , Fellow, IEEE

                                             Abstract—Visual Grounding, also known as Referring Expression Comprehension and Phrase Grounding, aims to ground the specific
                                             region(s) within the image(s) based on the given expression text. This task simulates the common referential relationships between
                                             visual and linguistic modalities, enabling machines to develop human-like multimodal comprehension capabilities. Consequently, it has
                                             extensive applications in various domains. However, since 2021, visual grounding has witnessed significant advancements, with emerg-
                                             ing new concepts such as grounded pre-training, grounding multimodal LLMs, generalized visual grounding, and giga-pixel grounding,
                                             which have brought numerous new challenges. In this survey, we first examine the developmental history of visual grounding and provide
                                             an overview of essential background knowledge, including fundamental concepts and evaluation metrics. We systematically track and
                                             summarize the advancements, and then meticulously define and organize the various settings to standardize future research and ensure
                                             a fair comparison. In the dataset section, we compile a comprehensive list of current relevant datasets, conduct a fair comparative
                                             analysis, and provide ultimate performance prediction to inspire the development of new standard benchmarks. Additionally, we delve
                                             into numerous applications and highlight several advanced topics. Finally, we outline the challenges confronting visual grounding and
                                             propose valuable directions for future research, which may serve as inspiration for subsequent researchers. By extracting common
                                             technical details, this survey encompasses the representative work in each subtopic over the past decade. To the best of our knowledge,
arXiv:2412.20206v4 [cs.CV] 4 Aug 2026




                                             this paper represents the most comprehensive overview currently available in the field of visual grounding. This survey is designed to
                                             be suitable for both beginners and experienced researchers, serving as an invaluable resource for understanding key concepts and
                                             tracking the latest research developments. We keep tracing related work at https://github.com/linhuixiao/Awesome-Visual-Grounding.

                                             Index Terms—Visual Grounding, Referring Expression Comprehension, Phrase Grounding, Survey

                                                                                                                      ✦



                                        1    I NTRODUCTION

                                        I  N the field of artificial intelligence (AI) [1], [2], [3], [4],
                                           multimodal learning [5], [6] that combines visual perception
                                        and natural language understanding has emerged as a pivotal
                                                                                                                            "a man in a white hat
                                        approach for achieving human-like cognition in machines. At its                     and red jacket cross-
                                        core lies the integration of visual and linguistic cues, intending                     country skiing"   Visual Grounding Model
                                        to bridge the semantic gap between image scenes and language                                    Fig. 1: An illustration of visual grounding.
                                        descriptions. Visual Grounding (VG) [7], [8], [9] represents such                 tablish intrinsic connections between linguistic expressions and
                                        a fundamental pursuit, encompassing AI models’ ability to es-                     corresponding visual elements.
                                                                                                                              As depicted in Fig. 1, visual grounding, also known as Re-
                                        •   Linhui Xiao is with Pengcheng Laboratory (PCL), Shenzhen 518066,              ferring Expression Comprehension (REC) and Phrase Grounding
                                            China, also with Institute of Automation, Chinese Academy of Sciences         (PG), according to the classical definition [10], [11], [12], involves
                                            (CASIA), Beijing 100190, China, and also with School of Artificial
                                            Intelligence, University of Chinese Academy of Sciences (UCAS), Beijing       localizing a specific region within an image based on a given
                                            100049, China (e-mail: xiaolinhui16@mails.ucas.ac.cn).                        textual description, and such a description is called “referring ex-
                                        •   Xiaoshan Yang, and Changsheng Xu are with State Key Labora-                   pression” [7], [13], [14], [15], [16], [17], [18], [19]. The objective
                                            tory of Multimodal Artificial Intelligence Systems (MAIS), Institute of
                                            Automation, Chinese Academy of Sciences (CASIA), Beijing 100190,
                                                                                                                          of this task is to emulate the prevalent referential relationships in
                                            China, also with Pengcheng Laboratory (PCL), Shenzhen 518066, China,          social conversations, equipping machines with human-like multi-
                                            and also with School of Artificial Intelligence, University of Chi-           modal comprehension capabilities. Consequently, it has extensive
                                            nese Academy of Sciences (UCAS), Beijing 100049, China (e-mail: xi-           applications in visual language navigation [20], human-machine
                                            aoshan.yang@nlpr.ia.ac.cn, csxu@nlpr.ia.ac.cn).
                                        •   Xiangyuan Lan is with Pengcheng Laboratory (PCL), Shenzhen 518066,            dialogue [21], [22], visual question answering [23], [24], and other
                                            China (e-mail: lanxy@pcl.ac.cn).                                              related domains [25].
                                        •   Yaowei Wang is with Harbin Institute of Technology (Shenzhen), Shenzhen           The continuous advancements in deep learning, including
                                            518055, China, and also with Pengcheng Laboratory (PCL), Shenzhen
                                            518066, China (e-mail: wangyw@pcl.ac.cn).
                                                                                                                          visual grounding, are driven by three fundamental elements:
                                        •   Corresponding author: Changsheng Xu.                                          data, algorithms, and computing power [26]. From a data per-
                                        •   This work was supported in part by the Major Key Project of PCL under         spective, the grounding task involves three essential types of
                                            Grant PCL2025A14, in part by the National Natural Science Foundation          data: images, referring expressions, and referred
                                            of China under Grants U23A20387, 62322212, 62036012, 62072455,
                                            62536003, 62402252, in part by National Science and Technology Major          bounding boxes. However, obtaining such paired triplet data
                                            Project under Grant 2021ZD0112200, and also in part by CAS Project for        is not straightforward, despite images being more readily avail-
                                            Young Scientists in Basic Research (YSBR-116).                                able among these three types. Challenges arise when acquiring
                                        •   Digital Object Identifier https:// doi.org/ 10.1109/ TPAMI.2025.3630635
                                                                                                                          expression text and corresponding bounding boxes. Firstly, visual

                                                                  1520-9210 © 2025 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.
```

## P52 — 脑启发的新型混合模型架构（内容远比标题丰富）-石润林 (1).pdf

PDF 第 1 页；共 34 页；SHA256 `7ef92d49529c077d7e7b041c9938c87c6aa9688ca2b79c1069260a982d84d0e5`。

```text
              脑启发的新型混合模型架构

石润林
shirunlin@tju.edu.cn
shirunlin2026@ia.cn.cn
shishuimu25@163.com

Disclaimer / 声明
本研究为本人本科毕设内容，由本人独立完成，使用中科院自动化所脑图谱与类脑智能实验室计算资源，展示内容仅代
表个人观点，不代表任何机构、实验室或导师的立场。
```
