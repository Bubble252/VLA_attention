"""Validate every record/tensor against the exact frozen teacher inputs."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(manifest,cache,kind):
    import numpy as np
    rows=[json.loads(x) for x in manifest.read_text().splitlines()]
    ids=[r['sample_id'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate sample IDs')
    expected=set();failures=[];records=[]
    for row in rows:
        key=row['sample_id'].replace(':','_');expected.add(key+'.npy')
        try:
            path=cache/(key+'.npy');meta_path=cache/(key+'.json')
            m=json.loads(meta_path.read_text());values=np.load(path,allow_pickle=False)
            if m['sample_id']!=row['sample_id']:raise ValueError('ID mismatch')
            if not np.isfinite(values).all():raise ValueError('Nonfinite tensor')
            if not m.get('teacher_frozen'):raise ValueError('Teacher not frozen')
            if kind=='semantic':
                if m['method']!='ddim_nulltext_fp32_reconstruction_v3' or m['capture_stage']!='final_reconstruction_only':raise ValueError('Wrong extraction recipe')
                if m['attention_tensors']!=100 or m['inversion_steps']!=20 or m['inner_steps']!=10:raise ValueError('Wrong step/count')
                if m['caption']!=row['caption'] or m['phrase']!=row['phrase']:raise ValueError('Language mismatch')
                if m.get('phrase_occurrence','first')!=row.get('phrase_occurrence','first'):raise ValueError('Occurrence mismatch')
                if list(values.shape)!=[16,16] or (values<0).any() or not np.isclose(values.sum(),1.,atol=1e-5):raise ValueError('Invalid spatial distribution')
                if not np.isfinite(m['mean_reconstruction_mse']):raise ValueError('Nonfinite reconstruction')
                if m['map_sha256']!=sha(path):raise ValueError('Map checksum mismatch')
            else:
                if m['signal_type']!='visual_features' or list(values.shape)!=[256,1024]:raise ValueError('Wrong feature schema')
                if m['output_sha256']!=sha(path):raise ValueError('Feature checksum mismatch')
            if row.get('image_sha256') and m['image_sha256']!=row['image_sha256']:raise ValueError('Image mismatch')
            records.append({'sample_id':row['sample_id'],'tensor_sha256':sha(path),'metadata_sha256':sha(meta_path)})
        except Exception as exc:failures.append({'sample_id':row['sample_id'],'error':str(exc)})
    extras=sorted({f.name for f in cache.glob('*.npy')}-expected)
    return {'kind':kind,'manifest_sha256':sha(manifest),'expected':len(rows),'valid':len(records),
            'failures':failures,'extra_files':extras,'passed':not failures and not extras,
            'content_digest':hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--kind',choices=['semantic','retention'],required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=validate(a.manifest,a.cache,a.kind)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result));return 0 if result['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
