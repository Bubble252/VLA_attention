"""Paired bootstrap comparison of two held-out attribution reports."""
from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser(); p.add_argument('--candidate',type=Path,required=True); p.add_argument('--baseline',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--seed',type=int,default=23); a=p.parse_args()
 import numpy as np
 c=json.loads(a.candidate.read_text())['rows']; b=json.loads(a.baseline.read_text())['rows']; bm={x['sample_id']:x for x in b}
 if set(x['sample_id'] for x in c)!=set(bm): raise ValueError('sample IDs differ')
 rng=np.random.default_rng(a.seed); out={}
 for key in ('pointing','mass_in_box','top20_box_iou'):
  d=np.array([r['metrics'][key]-bm[r['sample_id']]['metrics'][key] for r in c],float); boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)
  out[key]={'mean_delta':float(d.mean()),'positive_fraction':float((d>0).mean()),'ci95':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
