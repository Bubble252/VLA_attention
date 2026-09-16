"""Extract local PDF evidence; no network requests or model-generated claims."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'references' / 'abstract_review'
OUT.mkdir(exist_ok=True)
CACHE = ROOT / 'outputs' / 'pdf_text'
CACHE.mkdir(parents=True, exist_ok=True)
records = []
for n, pdf in enumerate(sorted((ROOT / 'references/papers').glob('*.pdf')), 1):
    ident = f'P{n:02d}'
    proc = subprocess.run(['pdftotext', '-layout', str(pdf), '-'], capture_output=True)
    if proc.returncode:
        records.append(dict(id=ident, file=pdf.name,
            sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
            error=proc.stderr.decode(errors='replace')))
        continue
    raw = proc.stdout.decode('utf-8', errors='replace')
    (CACHE / f'{ident}.txt').write_text(raw)
    pages = raw.split('\f')
    if not pages[-1].strip():
        pages.pop()
    # Layout text is evidence only; column order and abstract boundaries are
    # inspected manually. Never label this automated window as a full abstract.
    hits = [(p + 1, t) for p, t in enumerate(pages[:8])
            if re.search(r'\bA\s*BSTRACT\b|摘\s*要', t, re.I)]
    p, t = hits[0] if hits else (1, pages[0] if pages else '')
    # Manually checked no-heading opening summaries / false body hits.
    manual_pages = {'ImageWAM.pdf': 1, 'Robust-WAM.pdf': 1,
                    'WAM_survey.pdf': 1, 'WordCon.pdf': 2,
                    'prism.pdf': 1}
    if pdf.name in manual_pages:
        p = manual_pages[pdf.name]
        t = pages[p-1]
    records.append(dict(id=ident, file=pdf.name, pages=len(pages),
        sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
        evidence_page=p, abstract_marker_found=bool(hits),
        cache=f'outputs/pdf_text/{ident}.txt',
        evidence=t))
(OUT / 'inventory.json').write_text(json.dumps(records, ensure_ascii=False, indent=2))
lines = ['# 本地 PDF 来源与摘要页证据', '',
         '本文件由脚本提取，保留页面阅读顺序以便复核；双栏文本可能交错。不是自动翻译或已完成精读的声明。', '']
for r in records:
    lines += [f"## {r['id']} — {r['file']}", '']
    if 'error' in r:
        lines += [r['error'], '']
    else:
        lines += [f"PDF 第 {r['evidence_page']} 页；共 {r['pages']} 页；SHA256 `{r['sha256']}`。",
                  '', '```text', r['evidence'].rstrip(), '```', '']
(OUT / 'source_evidence.md').write_text('\n'.join(lines))
for r in records:
    print(r['id'], r['file'], 'pages=' + str(r.get('pages')), 'abstract=' + str(r.get('abstract_marker_found')))
