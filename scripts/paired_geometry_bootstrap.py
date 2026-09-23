"""Paired sample bootstrap for geometry-loss sweeps versus matched V1.

Bootstrap resamples held-out image-phrase rows within each paired training
seed, then averages deltas over the fixed seed set. This is uncertainty over
held-out examples conditional on the selected training seeds; it is not a
population-level training-seed significance test.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

METRICS=("calibrated_box_iou","common_soft_iou","common_box_iou_top10","common_box_iou_top20",
         "common_box_iou_top30","common_mass_in_box","pointing","mass_in_box")

def load_rows(path:Path):
    report=json.loads(path.read_text())
    rows={row["sample_id"]:row["metrics"] for row in report["rows"]}
    if not rows: raise ValueError(f"empty report: {path}")
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--baseline-template",required=True,help="JSON path template with {seed}")
    p.add_argument("--candidate-template",required=True,help="JSON path template with {loss} and {seed}")
    p.add_argument("--losses",default="kl,kl_rank,kl_moment,js")
    p.add_argument("--seeds",default="17,29,41")
    p.add_argument("--bootstrap",type=int,default=10000)
    p.add_argument("--seed",type=int,default=20260923)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); seeds=[int(x) for x in a.seeds.split(",")]; losses=a.losses.split(","); rng=np.random.default_rng(a.seed)
    result={"bootstrap_samples":a.bootstrap,"training_seeds":seeds,"resampling":"paired sample IDs within seed; aggregate fixed seeds","metrics":list(METRICS),"comparisons":{}}
    for loss in losses:
        by_metric={m:[] for m in METRICS}; per_seed={}
        for seed in seeds:
            b=load_rows(Path(a.baseline_template.format(seed=seed)))
            c=load_rows(Path(a.candidate_template.format(loss=loss,seed=seed)))
            ids=sorted(set(b)&set(c))
            if len(ids)!=len(b) or len(ids)!=len(c):
                raise ValueError(f"sample mismatch for {loss} seed {seed}: baseline={len(b)}, candidate={len(c)}, intersection={len(ids)}")
            per_seed[str(seed)]={}
            for metric in METRICS:
                delta=np.asarray([float(c[i][metric])-float(b[i][metric]) for i in ids],dtype=np.float64)
                by_metric[metric].append(delta)
                per_seed[str(seed)][metric]=float(delta.mean())
        summary={}
        for metric, arrays in by_metric.items():
            observed=float(np.mean([x.mean() for x in arrays]))
            reps=np.empty(a.bootstrap,dtype=np.float64)
            for j in range(a.bootstrap):
                seed_means=[]
                for arr in arrays:
                    idx=rng.integers(0,len(arr),size=len(arr))
                    seed_means.append(float(arr[idx].mean()))
                reps[j]=float(np.mean(seed_means))
            summary[metric]={"mean_delta":observed,"ci95":[float(np.quantile(reps,.025)),float(np.quantile(reps,.975))],"positive_fraction":float(np.mean(reps>0))}
        result["comparisons"][loss]={"per_seed_delta":per_seed,"aggregate":summary}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(a.output)

if __name__=="__main__": main()
