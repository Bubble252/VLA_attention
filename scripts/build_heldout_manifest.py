"""Build a deterministic test-only held-out manifest disjoint by image ID."""
import argparse, json, random
from pathlib import Path

def main():
 p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--test',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--n',type=int,default=64); p.add_argument('--seed',type=int,default=23); a=p.parse_args()
 train={json.loads(x)['image_id'] for x in a.train.read_text().splitlines() if x}; rows=[json.loads(x) for x in a.test.read_text().splitlines() if json.loads(x)['image_id'] not in train]; chosen=random.Random(a.seed).sample(rows,a.n); a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in chosen)); print(len(chosen))
if __name__=='__main__': main()
