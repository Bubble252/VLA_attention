"""Summarize V3 fine-lambda and teacher-temperature conditioning sweeps."""
from __future__ import annotations
import argparse, json
from pathlib import Path

SPECS=(
    ('l0p06_t1p00',.06,1.), ('l0p08_t1p00',.08,1.), ('l0p12_t1p00',.12,1.), ('l0p15_t1p00',.15,1.),
    ('l0p10_t0p50',.10,.5), ('l0p10_t0p75',.10,.75), ('l0p10_t1p25',.10,1.25), ('l0p10_t1p50',.10,1.5),
)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    rows=[]
    for tag,lmb,temp in SPECS:
        d=json.loads((a.root/f'V3_{tag}'/'report.json').read_text())['summary']
        rows.append({'tag':tag,'lambda_sem':lmb,'teacher_temperature':temp,'pointing':d['pointing'],'mass_in_box':d['mass_in_box'],'top20_box_iou':d['top20_box_iou']})
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps({'rows':rows},indent=2)+'\n')
    lines=['# V3 Conditioning Sweep', '', '| tag | lambda | teacher temperature | pointing | mass-in-box | IoU |','|---|---:|---:|---:|---:|---:|']
    for r in rows: lines.append(f"| {r['tag']} | {r['lambda_sem']:.2f} | {r['teacher_temperature']:.2f} | {r['pointing']:.4f} | {r['mass_in_box']:.4f} | {r['top20_box_iou']:.4f} |")
    best=max(rows,key=lambda r:(r['mass_in_box']+r['pointing'],r['top20_box_iou']))
    lines+=['',f"Best provisional joint mass+pointing candidate: `{best['tag']}`. It must still beat the coarse λ=0.1,T=1 anchor on more than one metric and be confirmed with a second seed before selection."]
    (a.output.parent/'analysis.md').write_text('\n'.join(lines)+'\n'); print(a.output)
if __name__=='__main__': main()
