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
    if not rows:raise ValueError('Empty manifest')
    expected=set();failures=[];records=[];recipes=set()
    manifest_digest=sha(manifest)
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
                if m.get('manifest_sha256')!=manifest_digest:raise ValueError('Manifest provenance mismatch')
                if m.get('guidance_scale')!=7.5 or m.get('attention_resolution')!=16:raise ValueError('CFG/grid differs from frozen recipe')
                if m.get('numerics')!='fp32,deterministic_algorithms,math_sdpa,tf32_off,cublas4096:8':raise ValueError('Numerical backend not frozen')
                instruction=row.get('instruction',row['caption'])
                if m.get('instruction_sha256')!=hashlib.sha256(instruction.encode()).hexdigest():raise ValueError('Instruction hash mismatch')
                if not isinstance(m.get('seed'),int):raise ValueError('Missing seed')
                if len(m.get('extractor_sha256',''))!=64 or len(m.get('image_sha256',''))!=64:raise ValueError('Missing extractor/image hashes')
                recipes.add((m['seed'],m['extractor_sha256'],m['numerics']))
                for field in ('episode_id','timestep','camera','split','role','instruction'):
                    if field in row and m.get('source_provenance',{}).get(field)!=row[field]:raise ValueError(f'Wrong {field} provenance')
            else:
                if m['signal_type']!='visual_features' or list(values.shape)!=[256,1024]:raise ValueError('Wrong feature schema')
                if m['output_sha256']!=sha(path):raise ValueError('Feature checksum mismatch')
                if m.get('source_manifest_sha256')!=manifest_digest:raise ValueError('Feature manifest provenance mismatch')
                if m.get('revision')!='9d0413465e8a91e67bbf2c1ad342815478d1b906':raise ValueError('Teacher revision differs')
                recipes.add((m['revision'],m.get('extractor_sha256'),m.get('preprocessing')))
            if row.get('image_sha256') and m['image_sha256']!=row['image_sha256']:raise ValueError('Image mismatch')
            records.append({'sample_id':row['sample_id'],'tensor_sha256':sha(path),'metadata_sha256':sha(meta_path)})
        except Exception as exc:failures.append({'sample_id':row['sample_id'],'error':str(exc)})
    extras=sorted({f.name for f in cache.glob('*.npy')}-expected)
    if len(recipes)>1:failures.append({'error':'Multiple producer recipes in one cache'})
    return {'kind':kind,'manifest_sha256':sha(manifest),'expected':len(rows),'valid':len(records),
            'failures':failures,'extra_files':extras,'passed':not failures and not extras,
            'content_digest':hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--kind',choices=['semantic','retention'],required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=validate(a.manifest,a.cache,a.kind)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    temporary = a.output.with_suffix(a.output.suffix + '.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(a.output)
    print(json.dumps(result));return 0 if result['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
