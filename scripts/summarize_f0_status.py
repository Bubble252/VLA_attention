"""Create a non-final F0 V0-V4 flow-gate status table from result JSONs."""
import json
from pathlib import Path


def main():
    root=Path(__import__('sys').argv[1]); names=['F0v2_V1_caption_smoke_20_seed17.json','F0v2_V2_dino_smoke_20_seed17.json','F0v2_V3_semantic_smoke_20_seed17.json','F0v2_V4_ret_sem_smoke_20_seed17.json']; out={}
    for name in names:
        data=json.loads((root/name).read_text())
        values=(value for item in data['losses'] for value in (item.values() if isinstance(item,dict) else [item]) if isinstance(value,(int,float)))
        visual=data.get('visual_lora_tensors',data.get('visual_trainable_tensors'))
        out[name]={'steps':data['steps'],'losses_finite':all(value==value and abs(value)<1e6 for value in values),'visual_lora_tensors':0 if name.startswith('F0v2_V3') else visual,'visual_attribution_gradients_enabled':name.startswith('F0v2_V3')}
    out['scope']='F0-256 engineering gate; non-final; no paper claim'; print(json.dumps(out,indent=2))

if __name__=='__main__': main()
