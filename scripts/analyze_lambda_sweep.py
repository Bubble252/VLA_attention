"""Summarize fixed-seed V3/V4 semantic-loss lambda sweep reports."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--steps',type=int,default=100); a=p.parse_args()
    rows=[]
    for lam in ('0p00','0p01','0p03','0p10','0p30'):
        for variant in ('V3','V4'):
            f=a.root/f'{variant}_l{lam}'/'report.json'
            d=json.loads(f.read_text())['summary']; rows.append({'variant':variant,'lambda_sem':float(lam.replace('p','.')),'pointing':d['pointing'],'mass_in_box':d['mass_in_box'],'top20_box_iou':d['top20_box_iou'],'n':d['n']})
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps({'steps':a.steps,'rows':rows},indent=2)+'\n')
    lines=['# Semantic-loss lambda sweep', '', f'Fixed {a.steps}-step training on the same F0v2 manifest and seed; evaluation uses the same 64 image-disjoint held-out records.', '', '| variant | lambda_sem | pointing | mass-in-box | top20 box IoU |','|---|---:|---:|---:|---:|']
    for r in rows: lines.append(f"| {r['variant']} | {r['lambda_sem']:.2f} | {r['pointing']:.4f} | {r['mass_in_box']:.4f} | {r['top20_box_iou']:.4f} |")
    for variant in ('V3','V4'):
        candidates=[r for r in rows if r['variant']==variant]; best=max(candidates,key=lambda r:(r['mass_in_box'],r['pointing']))
        lines += ['', f"Best directional {variant} setting by mass then pointing: lambda={best['lambda_sem']:.2f}. This is a selection result, not a final significance claim."]
    lines += ['', '## Interpretation', '', 'Use the sweep to decide whether a stable intermediate lambda improves both spatial metrics relative to lambda=0. If no lambda is consistently better, change map normalization/temperature before increasing data. Do not select a lambda using only one metric or one variant.']
    (a.output.parent/'analysis.md').write_text('\n'.join(lines)+'\n'); print(a.output)
if __name__=='__main__': main()
