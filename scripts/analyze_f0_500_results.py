"""Summarize the completed 500-step F0 confirmation run."""
import argparse, json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    names={x: f'{x}_500step_heldout_report.json' for x in ('V1','V2','V3','V4')}
    data={x:json.loads((a.root/f).read_text())['summary'] for x,f in names.items()}
    out={'heldout_500step':data,'deltas':{
      'V3_minus_V1':{k:data['V3'][k]-data['V1'][k] for k in ('pointing','mass_in_box','top20_box_iou')},
      'V4_minus_V2':{k:data['V4'][k]-data['V2'][k] for k in ('pointing','mass_in_box','top20_box_iou')}}}
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    table=''.join(f"| {x} | {d['pointing']:.4f} | {d['mass_in_box']:.4f} | {d['top20_box_iou']:.4f} |\n" for x,d in data.items())
    text='''# F0 500-step confirmation

The 500-step run uses the same 256 training rows, seed 17, loss weights, and 64 image-disjoint held-out records. All four training loss traces are finite and each checkpoint was restored for evaluation.

| model | pointing | mass-in-box | top20 box IoU |
|---|---:|---:|---:|
'''+table+'''\n## Decision\n\nV3 is below V1 on pointing and IoU but slightly above on mass-in-box. V4 is below V2 on pointing and mass-in-box and nearly unchanged on IoU. The 500-step confirmation therefore does not support the required `V3 > V1` and `V4 > V2` claim. The semantic loss is not yet a validated improvement under this configuration. The correct SD teacher map still beats same-image wrong-word, wrong-image, and random controls on pointing and mass, so the teacher signal itself is meaningful; the failure is in transferring that signal through the current training objective. Do not start F1-10k unchanged.

Next research action: audit semantic-loss gradient scale and schedule, verify map normalization/temperature, then run a small lambda sweep with the same held-out protocol. Keep all current results as negative/partial evidence; do not delete them.
'''
    (a.output.parent/'analysis_500step.md').write_text(text)
    print(a.output)
if __name__=='__main__': main()
