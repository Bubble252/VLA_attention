"""Assemble the audit ledger from retrieved records and explicit reviewer annotations."""
import json
import re
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
NOTES = {
 'Robot Trains Robot': ('机器人保护与真实 humanoid 学习', '机械臂教师/潜变量适应', '主动保护支持真实训练', 'humanoid RL', 0),
 'Learning to Discern': ('异质示范质量评估', '轨迹嵌入与偏好质量评估', '分离示范质量与风格', '离线模仿学习', 1),
 'Untangling Dense': ('绳结解开', '关键点与几何规划', '任务几何结构支持控制', '机器人绳索', 1),
 'On the Similarities': ('对比嵌入几何', '相似度理论与辅助方差损失', '批大小影响正负对几何', '一般表征学习', 1),
 'Weakly-Supervised': ('医学分割', '几何与对比损失', '低对比边界的表征区分', '医学图像', 0),
 'MameLoshnLM': ('意第绪语建模', '语料与继续预训练', '低资源语言数据质量', 'NLP', 0),
 'Mitigating State': ('VLA 控制状态混淆', '当前未来视觉→动作 chunk 的逆动力学辅助任务', '动作监督塑造视觉区别', 'VLA/CALVIN/SimplerEnv', 3),
 'Comparison of Vision': ('goal state action token 方法比较', '比较条目，未核实为原创论文', '目标状态与低层动作区别', 'VLA', 1),
 'ALFRED-L': ('交互环境中的语言与动作', '摘要仅元数据，方法待核实', '语言作用需隔离测量', '交互具身任务', 1),
 'SpatialVLA': ('空间动作表示', '空间感知表示与自适应 action grids', '空间编码支持迁移', 'VLA', 1),
 'Learning Representations Via': ('行为相关 RL 表征', 'robust behavioral metric；摘要缺失，未原文核实', '与动作距离学习存在潜在邻近', 'RL', 2),
 'Deployment-time': ('部署可靠性', '摘要缺失，方法未核实', '暂不能判断具体重合', 'robot policy', 1),
 'ConSERT': ('句子表示', '自监督对比框架', '文本增强用于语义表征', 'NLP', 0),
 'DiffCSE': ('句子表示', 'difference-based contrastive learning', '文本表征变化', 'NLP', 0),
 'Self-Guided': ('句子表示', '自引导对比学习', '文本表示迁移', 'NLP', 0),
 'Grounding Hierarchical': ('层级 VLA 对齐', '摘要缺失，标题提示 explicit language-action alignment', '不能仅凭标题确定机制', 'VLA', 1),
 'Vision-Language-Action and': ('VLA 综述', '综述，非原方法', '分类与挑战', 'VLA', 1),
 'Transforming the vision': ('非技术章节噪声', '无相关方法证据', '不适用', '非目标领域', 0),
 'Peer Review': ('聚类论文审稿记录', '非研究论文原文', '不适用', '多视图聚类', 0),
 'A Qualitative': ('语言相似性表达', '属性空间框架', '定性语义相似关系', '语言知识表示', 0),
 'CLOP-DiT': ('单细胞生成', '对比跨模态与扩散', '元数据生成条件', '生物信息', 0),
 "Enhancing SCGAN": ('一般解耦表示', 'contrastive SSIM；摘要缺失', '未核实', '一般视觉', 0),
}
PRIOR = [
 ('RS-CL', 'Contrastive Representation Regularization for Vision-Language-Action Models', '2025-10 / PDF v4 2026-05', 'https://arxiv.org/abs/2510.01711', '引用追溯：inverse-dynamics 文献 [45]', 'VLA 表征缺乏控制状态敏感性', 'summary token + proprio 距离 soft weights + InfoNCE + view cutoff，与原动作损失联合', '物理状态距离可塑造控制相关表征', 'GR00T/RoboCasa/LIBERO/真机', ['match','partial','match','match'], 'ICML 2026（PDF 首页）', '§2.1–2.2 Eq.(2)–(5)；Table 2 LIBERO；Table 3 比较 TCN', '使用 proprio 距离，非我们的语言区域 action-chunk triplet；不能说 exact duplicate。'),
 ('PSE', 'Contrastive Behavioral Similarity Embeddings for Generalization in Reinforcement Learning', '2021-01 / PDF v2 2021-03', 'https://arxiv.org/abs/2101.05265', 'model-recall，PDF 已核验', '跨环境行为泛化', 'policy similarity metric 的当前动作分布差异+未来状态递归；contrastive metric embeddings', '行为相似决定表示相似', 'RL/LQR/jumping/Distracting Control', ['partial','match','match','partial'], 'ICLR 2021（PDF 首页）', '§3 Eq.(2)；§4 Eq.(4)；Fig.3 联合 RL/IL 与辅助损失', '需要专家/近似最优策略，保留未来行为；有限 H=8 action L1 不继承其理论。'),
 ('Inverse dynamics VLA', 'Mitigating State Aliasing in Vision-Language-Action Models via Inverse Dynamics Learning', '2026-05', 'https://arxiv.org/abs/2605.29577', '本轮 OpenAlex 检索，PDF 已核验', 'VLA 相似视觉对应不同动作的状态混淆', '当前/未来编码器特征预测中间 action chunk，附加 pseudo time reversal', '直接动作监督视觉编码器可减轻混淆', 'CALVIN ABC-D/SimplerEnv；LIBERO probing', ['match','partial','match','match'], 'arXiv preprint', '§3.1 inverse action loss；§3.2 PTR 的非精确反演声明；§4 和 frozen encoder probes', '逆动力学非 triplet；未来观测只用于训练辅助头。'),
 ('FOCAL-VLA', 'FOCAL-VLA: Subtask-Guided Geometry Distillation and Implicit World Modeling for Vision-Language-Action Models', '2026-09', 'https://arxiv.org/abs/2609.21228', '上一轮检索，本轮 PDF 核验', '当前子任务所需空间与未来动态理解', '子任务实体 mask pooling VGGT/Track4World 教师特征，geometry/motion queries 条件化动作', '全场景蒸馏包含无关信息，任务区域应随子任务改变', 'π0.5/LIBERO/真机', ['partial','partial','match','match'], 'arXiv preprint', '§III-B Eq.(1)–(2)；§III-C Eq.(3)–(4)；§III-D 原生 context 路径保留', '依赖离线子任务语义和实体 masks；绝对教师特征蒸馏，不是动作距离监督。'),
 ('Copper', 'Copper-Policy: Focus on the Representation for Robust Robot Manipulation', '2026-09', 'https://arxiv.org/abs/2609.32779', '用户本地 PDF', '学习有利于控制的紧凑世界表征', '意图条件 online/EMA compactor + future flow prediction + 当前帧细节动作通路', '任务进展与执行细节互补，target space 随 policy 学习', 'LIBERO/Plus/RoboTwin/真机', ['partial','differ','match','match'], 'arXiv preprint（用户本地副本）', '§3.2 Eq.(2)–(3)；§3.4–3.5 asymmetric interface；§4.5 Table 7', '未来条件为 task intention，非任意候选动作后果；完整架构不是训练期 triplet。'),
 ('TCN', 'Time-Contrastive Networks: Self-Supervised Learning from Video', '2017-04 / PDF v3 2018-03', 'https://arxiv.org/abs/1704.06888', 'model-recall，PDF 已核验，显式扩展检索年份', '视角不同但语义相同/外观相近但功能不同的操作状态', '同步多视角 positives + temporal negatives + triplet margin', '不变性与功能敏感性需同时学习', '机器人模仿/倒水/人体姿态', ['partial','partial','match','match'], 'arXiv preprint（本次 PDF 未显示正式 venue）', 'Abstract；§III-A triplet inequality；§III-B/C reward 与 imitation 使用', '同步视角/时间负对，非 action-chunk 距离；有单视角变体。'),
 ('DBC', 'Learning Invariant Representations for Reinforcement Learning without Reconstruction', '2020-06 / PDF v2 2021-04', 'https://arxiv.org/abs/2006.10742', 'model-recall，PDF 已核验', '像素控制抗无关背景', 'latent 距离拟合 reward difference + transition Wasserstein distance', '保留行为信息无需像素重建', '视觉 MuJoCo/CARLA RL', ['partial','partial','match','partial'], 'ICLR 2021（PDF 首页）', '§3 bisimulation Definition 2；§4 encoder loss；§3 partial observability 假设', '用 rewards/transitions，不等价于即时 action distance，理论不能借用。'),
]

def main():
    timestamp = datetime.now(timezone.utc).isoformat()
    raw = json.loads((ROOT/'search_raw.json').read_text())
    pool = {}
    for source, records in raw['results'].items():
        for p in records:
            key = ' '.join(p['title'].lower().split())
            if key not in pool:
                pool[key] = dict(p, sources=[source])
            elif source not in pool[key]['sources']:
                pool[key]['sources'].append(source)
    ledger = []
    for p in pool.values():
        note = next((v for k,v in NOTES.items() if p['title'].startswith(k)), None)
        if note is None:
            raise ValueError('Unreviewed record: '+p['title'])
        problem, mechanism, insight, domain, score = note
        ledger.append(dict(title=p['title'], date=p.get('publication_date') or str(p.get('year') or '未核实'),
                           source=p.get('url'), provenance=p['sources'], abstract=p.get('abstract',''),
                           problem=problem, mechanism=mechanism, insight=insight, domain=domain,
                           abstract_overlap_score=score, fulltext_verified=p['title'].startswith('Mitigating State')))
    for _,title,date,url,prov,problem,mechanism,insight,domain,axes,venue,evidence,scope in PRIOR:
        if ' '.join(title.lower().split()) in pool:
            continue
        ledger.append(dict(title=title,date=date,source=url,provenance=[prov],
                           problem=problem,mechanism=mechanism,insight=insight,domain=domain,
                           abstract_overlap_score=sum(x=='match' for x in axes),fulltext_verified=True))
    (ROOT/'audit_records.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2))
    errors = ('arXiv 第一查询 SSLError/SSLEOFError；DBLP 三查询 JSON 解析失败；'
              'OpenReview: openreview not installed；Semantic Scholar 三查询 HTTP 429。'
              '其余 API 有返回。失败和低召回不构成不存在近邻的证据。')
    counts = ', '.join(f'{k}={len(v)}' for k,v in raw['results'].items())
    step2 = f'# Step 2 — 检索与去重\n\n{timestamp}\n\n查询：\n'+ '\n'.join('- '+q for q in raw['queries'])
    step2 += f'\n\n年份 2020–2026，各源每查询上限 3。结果：{counts}。24 原始记录，{len(pool)} 标题去重记录；补充 6 篇（3 篇 model-recall、2 篇上一轮近邻/用户文献、1 篇引用追溯），合计 {len(ledger)}。\n\n{errors}\n\nTCN 2017 是历史机制近邻，显式超出 API 年份保留。原始摘要及来源保存在 search_raw.json / audit_records.json。'
    (ROOT/'step2.md').write_text(step2)
    structured=[]
    for i,p in enumerate(ledger,1):
        structured.append(f"### {i}. {p['title']}\n\n- Title: {p['title']}\n- Date: {p['date']}\n- Problem framing: {p['problem']}\n- Core mechanism: {p['mechanism']}\n- Key insight: {p['insight']}\n- Application domain: {p['domain']}\n- Overlap score: {p['abstract_overlap_score']}/4，摘要筛选值，非最终判定\n- Source: {p['source']}；{', '.join(p['provenance'])}\n")
    (ROOT/'step3.md').write_text('# Step 3 — 完整摘要筛选台账\n\n'+timestamp+'\n\n摘要缺失记录保留不确定性；不以标题冒充原文核验。\n\n'+'\n'.join(structured))
    step4 = '# Step 4 — 原文候选选择\n\n'+timestamp+'\n\n选择 7 篇：\n'
    for name,title,_,_,_,_,_,insight,_,_,_,_,_ in PRIOR:
        step4 += f'- {name}: {title}。选择原因：{insight}。\n'
    step4 += '\n检索到 robust behavioral metric 2022（摘要缺失）尚未全文核验，作为残余风险保留；以已获原文且机制更直接的 PSE/RS-CL 优先。其余噪声记录见 step3。'
    (ROOT/'step4.md').write_text(step4)
    verified=[]
    compare=[]
    for name,title,date,url,prov,problem,mechanism,insight,domain,axes,venue,evidence,scope in PRIOR:
        n=sum(x=='match' for x in axes); level=5-n
        verified.append(dict(name=name,title=title,date=date,url=url,venue=venue,problem=problem,
                             mechanism=mechanism,insight=insight,domain=domain,axes=axes,
                             evidence=evidence,scope=scope,matching_axes=n,level=level))
        compare.append(f'- **{name}**\n  - Title: {title}\n  - Date: {date}\n  - Source: {url}\n  - Problem framing: {problem}\n  - Core mechanism: {mechanism}\n  - Key insight: {insight}\n  - Application domain: {domain}\n  - Verified venue: {venue}\n  - Closest-passage evidence: {evidence}\n  - Assumptions & scope: {scope}\n  - Refined overlap (problem/mechanism/insight/domain): {" / ".join(axes)}\n  - Verdict: {n}/4 match，Level {level}。partial 不自动算 match。\n')
    (ROOT/'verified_candidates.json').write_text(json.dumps(verified,ensure_ascii=False,indent=2))
    (ROOT/'step5.md').write_text('# Step 5 — 原文核验\n\n'+timestamp+'\n\n六篇 PDF 已下载并验证为 PDF，以 pdftotext -layout 提取；Copper 复用用户本地副本。以下是有章节锚点的方法/设置定向阅读，未声称逐字通读全部附录。\n\n'+'\n'.join(compare))
    proposal='- **Proposed work**\n  - Title: 语言目标条件下的动作可分辨视觉表征（原候选）\n  - Date: 2026-10-16，未发表\n  - Source: provisional_recommendation.json\n  - Problem framing: VLA 状态混淆与任务相关视觉证据\n  - Core mechanism: 冻结 source/target map pooling + action-chunk distance 正负对 + triplet loss，原生策略通路保留\n  - Key insight: 控制差异应体现为任务相关视觉表示差异\n  - Application domain: OpenVLA/OFT、LIBERO；尚未实验\n'
    (ROOT/'step6.md').write_text('# Step 6 — 四轴比较\n\n'+timestamp+'\n\n'+proposal+'\n'+'\n'.join(compare))
    delta='相较 RS-CL 用 proprioception 距离组织全局 VLM 摘要表示，本候选在可比本体状态下，用真实示范 action chunk 差异组织 source/target 语言区域表示，拟改善语言目标变化时的动作选择；收益尚待实验，区域汇聚与动作相似性分别已有先例，不能单凭组合声称强创新。'
    concerns='''## 原候选的三个关键问题

1. **创新性**：RS-CL 已做 VLA 关系监督，PSE 已做行为距离决定 embedding，FOCAL 已做任务区域汇聚。仅把 proprio 距离换 action 距离、InfoNCE 换 triplet、全局 pooling 换词图，不足以支撑强主贡献。
2. **方法前提**：相同 H=8 动作不等于相同任务状态，尤其共享接近动作、不同后续目标；相同状态也可有多种成功动作。旧候选没有完整解决 false positives / false negatives。静态视觉不能辨别不可观测接触/历史，不能用距离损失补出输入中不存在的信息。
3. **辅助头逃避**：pooling 向量分得开不保证原 action head 使用该差异；map-only 坐标可能单独预测任务阶段。需 native-branch 梯度、冻结 probe、动作遮挡/干预验证，不能只看 embedding loss。

## 已执行的逻辑反例

`mechanism_counterexamples.py` 已执行并输出 `mechanism_counterexamples.json`。它构造了两个反例：有限 H=8 动作 chunk 完全相同，但更远期动作不同；可观测输入完全相同，但成功示范可采用不同动作。前者否定“短动作一致必然等于长期行为等价”，后者否定“动作不同必然要求确定性视觉表征可分”。这些是方法前提的逻辑反例，不是 LIBERO 实验，也不证明经过严格筛选的辅助损失一定无效。

## 原候选处置

保留为低成本基线或机制探索，不以它冻结论文主创新；没有发现一个已核验论文逐项覆盖全部细节，但这不等于已通过新颖性门。

## 单独的新提案（未完成独立检索，不冒充本次审计通过）

将问题收窄为“指令重定向时，动作变化是否来自正确视觉对象”：在同一物理场景/本体状态上比较两个合法目标指令，再分别干预各自目标的视觉证据，测量语言效应与视觉效应的交互；用真实可执行演示或可靠规划器取得目标变化后的动作监督，禁止给错指令沿用原动作标签。目标是避免仅对词向量/固定位置作反应，而不是用 map containment 限制所有执行证据。这更接近原 VLM 归因工作的延续，但与 counterfactual grounding、causal representation、LIT/Anchor-Align 等仍需独立审计。

该提案的最小判别实验必须同时匹配数据、重定向示范、增强预算和训练步数：普通 BC + 同一数据、语义监督、关系度量监督、新交互约束。若收益来自新增示范/增强，不能归因给新损失。遮挡是分布扰动，不自动提供因果真值。
'''
    step7='# Step 7 — 判定与剩余差异\n\n'+timestamp+'\n\n**Level 2 — High Overlap**（已核验集合最小 level）。RS-CL 与 inverse-dynamics VLA 均在问题、洞见、应用三轴重合；具体损失与空间读取实现部分不同。PSE 提供最直接的机制祖先。\n\n'+delta+'\n\n'+concerns
    (ROOT/'step7.md').write_text(step7)
    report='# VLA 候选创新性审计（2026-10-16）\n\n## Verdict\n\n**Level 2 — High Overlap**。原候选不宜作为已验证创新的论文主方法；尚未查到完整逐项重复。此结论为当前可核验文献集合的保守判断，受检索失败与候选上限影响。\n\n## Delta\n\n'+delta+'\n\n## Decomposed claim\n\n'+(ROOT/'step1.md').read_text().split('## 被审计的原始候选')[1].split('## 不能偷换')[0]+'\n\n## Structured papers\n\n'+step2+'\n\n'+'\n'.join(structured)+'\n\n## Comparison result\n\n'+proposal+'\n'+'\n'.join(compare)+'\n'+concerns
    (ROOT/'report.md').write_text(report)
    (ROOT/'allinone.md').write_text('# 检索完整记录\n\n'+step2+'\n\n'+'\n'.join(structured)+'\n\n## 检索解释\n\n相关簇：行为度量（PSE/DBC/RS-CL）、状态混淆（逆动力学）、任务空间蒸馏（FOCAL）、未来表征（Copper）。本次只是 bounded novelty search；不能从小规模且多源失败的检索推断领域趋势或可靠作者引文排行。引用数不作为新颖性证据；噪声记录完整保留。')
    print(f'{len(ledger)} records; {len(verified)} full-text candidates; min level={min(p["level"] for p in verified)}')

if __name__=='__main__':
    main()
