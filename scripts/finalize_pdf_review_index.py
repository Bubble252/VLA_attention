"""Build the reading index from the inspected, manually authored 52 entries."""
import collections
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'references/abstract_review'
rows = json.loads((D / 'inventory.json').read_text())
entries = {}
for file in sorted(D.glob('0[123]_*.md')):
    parts = re.split(r'^## (P\d+) — (.*)$', file.read_text(), flags=re.M)
    for i in range(1, len(parts), 3):
        ident, title, body = parts[i:i+3]
        assert ident not in entries, ident
        entries[ident] = (file.name, title, body)
assert set(entries) == {r['id'] for r in rows}, 'Incomplete coverage'
repaired = ROOT / 'references/recovered/openvla_2406.09246v3.pdf'
assert repaired.exists()
recovery = dict(id='P43', original='references/papers/openvla_2406.09246.pdf',
    reason='pdftotext: missing trailer / invalid XRef / invalid pages object',
    recovered=str(repaired.relative_to(ROOT)), source='https://arxiv.org/pdf/2406.09246',
    observed_version='2406.09246v3', pages=37, read_pages=[1,2],
    sha256=hashlib.sha256(repaired.read_bytes()).hexdigest(), date='2026-09-16')
(D / 'source_recovery.json').write_text(json.dumps(recovery, ensure_ascii=False, indent=2))
groups = collections.defaultdict(list)
for r in rows:
    if 'sha256' in r:
        groups[r['sha256']].append(r['id'])
lines = ['# 52 份 PDF 的阅读状态与来源索引', '',
    '日期：2026-09-16。以下“已读”仅表示摘要/概要阅读，全文实验复现不在本轮范围。', '',
    '52 个文件对应 52 个条目；P39/P41 字节完全重复。P23/P24/P50 为同研究家族，独立论文数不以文件数充当。', '',
    '| ID | 原始 PDF | 页数 | 阅读定位/状态 | 人工归纳 |',
    '|---|---|---:|---|---|']
for r in rows:
    f, title, body = entries[r['id']]
    assert '来源' in body and ('借鉴' in body or '重合' in body), r['id']
    ident = r['id']
    status = '摘要/首页概要已读；PDF p.' + str(r.get('evidence_page',1))
    if ident=='P17': status='p.1 已读；实际为 OREOS LiDAR 定位，文件名误导'
    if ident=='P26': status='p.2 无标题摘要已读；DOI 占位符不可引用'
    if ident=='P41': status='p.1 已读；与 P39 字节重复，不独立计数'
    if ident=='P43': status='原件损坏；补读 arXiv v3 p.1–2，37 页'
    if ident=='P52': status='幻灯片无摘要；概览及机制页 p.1–16 已读，非全文精读'
    link = quote('../papers/'+r['file'],safe='/')
    lines.append(f"| {ident} | [{r['file']}]({link}) | {r.get('pages','异常')} | {status} | [{title}]({f}) |")
lines += ['', '## 审计结论', '',
    '- 归纳覆盖 52/52 文件，具体区别、重合与借鉴均有条目。',
    '- 原始文件保留；P43 补读副本及 SHA256 见 source_recovery.json。',
    '- 摘要不能确认的 benchmark 版本、训练配置、数学等价性均留待正文/代码精读，不虚报已复现。',
    '- 页码为 PDF 页面序号，不一定等于纸面页码。',
    '- 未来目录新增 PDF 后，须重新盘点并人工增加条目；不能仅运行抽取脚本就称已读。', '']
(D / '05_inventory_and_reading_status.md').write_text('\n'.join(lines))
print('PASS: 52 files / 52 manual entries; duplicates:',[v for v in groups.values() if len(v)>1])
