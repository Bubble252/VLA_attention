"""Aggregate held-out geometry-sweep evaluations over seeds."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(); p.add_argument('--input-dir',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    rows=[]
    for path in sorted(a.input_dir.glob('geom16_*_seed*_heldout/report.json')):
        report=json.loads(path.read_text()); summary=report['summary']; name=path.parent.name
        match=re.fullmatch(r'geom16_(.+)_seed(\d+)_heldout', name)
        if not match: raise ValueError(f'unexpected report directory: {name}')
        mode, seed=match.group(1), int(match.group(2))
        rows.append({'mode':mode,'seed':seed,**{k:v for k,v in summary.items() if isinstance(v,(float,int))}})
    if not rows: raise SystemExit(f'no heldout reports under {a.input_dir}')
    metrics=[k for k,v in rows[0].items() if k not in ('mode','seed')]
    summary={}
    for mode in sorted({x['mode'] for x in rows}):
        group=[x for x in rows if x['mode']==mode]
        summary[mode]={'n_seeds':len(group),'seeds':[x['seed'] for x in group], 'metrics':{m:{'mean':float(np.mean([x[m] for x in group])), 'std':float(np.std([x[m] for x in group],ddof=1)) if len(group)>1 else 0.0, 'values':[x[m] for x in group]} for m in metrics}}
    payload={'reports':rows,'summary':summary}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(payload,indent=2)+'\n')
    lines=['# Geometry loss sweep: held-out summary','','All four map objectives use the frozen calibration grid/threshold and three training seeds. Values are means across the same image-disjoint held-out set; seed mean±sample-SD are reported below.','','| loss | seeds | calibrated IoU | soft-IoU | top20 IoU | mass-in-box | pointing |','|---|---|---:|---:|---:|---:|---:|']
    for mode,d in summary.items():
        m=d['metrics']; seedstr=','.join(map(str,d['seeds']))
        def fmt(key):
            x=m.get(key,{}); return f"{x.get('mean',float('nan')):.4f}±{x.get('std',float('nan')):.4f}"
        lines.append(f"| {mode} | {seedstr} | {fmt('calibrated_box_iou')} | {fmt('common_soft_iou')} | {fmt('common_box_iou_top20')} | {fmt('common_mass_in_box')} | {fmt('pointing')} |")
    a.output.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(a.output)

if __name__=='__main__': main()
