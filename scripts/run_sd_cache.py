"""Robust sequential SD null-text cache driver for aligned JSONL manifests."""
from __future__ import annotations

import argparse, json, subprocess, sys
from pathlib import Path


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--runner',type=Path,required=True); p.add_argument('--python',dest='python_bin',required=True); p.add_argument('--model',type=Path,required=True); p.add_argument('--dataset-root',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--cache',type=Path,required=True); p.add_argument('--steps',type=int,default=20); p.add_argument('--inner-steps',type=int,default=10); p.add_argument('--seed',type=int,default=17); args=p.parse_args()
    args.cache.mkdir(parents=True,exist_ok=True); failures=[]
    for line in args.manifest.read_text().splitlines():
        row=json.loads(line); key=row['sample_id'].replace(':','_'); output=args.cache/(key+'.json')
        if output.exists(): continue
        command=[args.python_bin,str(args.runner),'--model',str(args.model),'--image',str(args.dataset_root/row['image_path']),'--caption',row['caption'],'--phrase',row['phrase'],'--output',str(output),'--steps',str(args.steps),'--inner-steps',str(args.inner_steps),'--resolution','16','--seed',str(args.seed)]
        result=subprocess.run(command,env={**__import__('os').environ,'CUDA_VISIBLE_DEVICES':'1'})
        if result.returncode: failures.append({'sample_id':row['sample_id'],'returncode':result.returncode})
    (args.cache/'failures.json').write_text(json.dumps(failures,indent=2)+'\n')
    print(json.dumps({'maps':len(list(args.cache.glob('*.json'))),'failures':len(failures)}))

if __name__=='__main__': main()
