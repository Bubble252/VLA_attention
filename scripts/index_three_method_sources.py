"""Build page-located evidence anchors for the three-method analytical reading."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
definitions = {
    'DB': ('P06', "Don't Blind Your VLA.pdf", {
        'METHOD': ([5], 'Sec.6.1–6.2, Eq.7–10', 'Let'),
        'ABLATION': ([8,9], 'Sec.8.2–8.5, Tables 5–8', '8.2'),
        'CONFIG': ([11,12,13], 'Appendix A.1/A.5/A.6, Tables 9–13', 'Table 9'),
        'EVAL': ([3,4,6], 'VL-Think and OOD evaluation protocol', '4.1'),
        'RESULT': ([6,7], 'Tables 1–3 and Sec.7.6', 'Table 1'),
    }),
    'AA': ('P10', 'Generalizable VLA Finetuning.pdf', {
        'METHOD': ([4,5], 'Sec.3.1–3.3, Eq.1–5', '3.1'),
        'SETUP': ([5,6,10,22], 'Sec.4.1/4.4; App.C.3', 'Baselines'),
        'CONFIG': ([19], 'App.B.1–B.2', 'B.1'),
        'COST': ([20], 'App.B.3, Table 6', 'B.3'),
        'DIAGNOSTIC': ([27,28,29], 'App.E.1–E.4/F', 'E.1'),
        'ABLATION': ([8], 'Sec.4.3, Table 3', 'Both anchoring'),
        'CONTROLS': ([8,21], 'Table 4 and App.B.5', 'Alignment is not'),
        'RESULT': ([6,8,9,10,11], 'Table 1/3/4; Fig.7–9', 'Table 1'),
    }),
    'PA': ('P18', 'PosA-VLA.pdf', {
        'MAPS': ([3,4], 'Fig.2/3; Sec.3.2, Eq.2–4', 'Figure 2'),
        'LOSS': ([5,6], 'Sec.3.3–3.4, Eq.5–17', 'Spatial Attention'),
        'ABLATION': ([8], 'Table 4 and Sec.4.4', 'Table 4'),
        'CONFIG': ([6,7,10], 'Sec.4.2 and Appendix Sec.6', '4.2.'),
        'SIM': ([11], 'Appendix Sec.9, Table 7', 'Table 7'),
        'LIMIT': ([12], 'Appendix Sec.13', '13. Limitations'),
    }),
}
blocks=[]
for prefix,(ident,filename,anchors) in definitions.items():
    pdf=ROOT/'references/papers'/filename
    pages=(ROOT/'outputs/pdf_text'/f'{ident}.txt').read_text().split('\f')
    for suffix,(selected,section,term) in anchors.items():
        evidence=[{'pdf_page':p,'original':pages[p-1]} for p in selected]
        blocks.append({'id':prefix+'-'+suffix,'pdf':str(pdf.relative_to(ROOT)),
            'sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'section':section,
            'pages':selected,'evidence':evidence,
            'reading_note':'Focused original-text analysis; page-layout extraction may interleave columns. Not a full bilingual translation.'})
out=ROOT/'references/abstract_review/three_methods_source_map.json'
out.write_text(json.dumps({'scope':'method/baseline/benchmark comparative reading', 'blocks':blocks},ensure_ascii=False,indent=2))
print('Evidence anchors:',len(blocks))
