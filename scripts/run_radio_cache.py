"""Frozen C-RADIOv3-L spatial features for image-addressed LIBERO inputs."""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    import torch
    import numpy as np
    from PIL import Image
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--source-lock',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    entry=next(e for e in json.loads(a.source_lock.read_text())['entries'] if e['repo']=='nvidia/C-RADIOv3-L')
    for item in entry['files']:
        path=a.model/item['path']
        if item.get('sha256'):valid=sha(path)==item['sha256']
        else:valid=hashlib.sha1(f'blob {path.stat().st_size}\0'.encode()+path.read_bytes()).hexdigest()==item['git_blob_id']
        if not valid:raise ValueError(f'Upstream hash mismatch: {path}')
    # Direct import of the hash-verified local package avoids old HF dynamic
    # loader missing transitive relative imports; no source code is modified.
    spec=importlib.util.spec_from_file_location('radio_local',a.model/'hf_model.py',submodule_search_locations=[str(a.model)])
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    model=module.RADIOModel.from_pretrained(a.model,local_files_only=True).eval().cuda().requires_grad_(False)
    torch.backends.cuda.matmul.allow_tf32=False
    rows=[json.loads(x) for x in a.manifest.read_text().splitlines()]
    a.output.mkdir(parents=True)
    manifest_sha=sha(a.manifest);code_sha=sha(__file__);receipts=[]
    for index,row in enumerate(rows):
        path=a.dataset_root/row['image_path']
        if sha(path)!=row['image_sha256']:raise ValueError('Image changed after export')
        with Image.open(path) as image:array=np.asarray(image.convert('RGB')).copy()
        pixels=torch.from_numpy(array).permute(2,0,1).unsqueeze(0).cuda().float()/255.
        pixels=torch.nn.functional.interpolate(pixels,size=(256,256),mode='bilinear',align_corners=False)
        with torch.no_grad():
            summary,features=model(pixels)
            if index==0:
                _,again=model(pixels)
                torch.testing.assert_close(features,again,rtol=1e-5,atol=1e-6)
        patch=int(model.config.patch_size);grid=[256//patch,256//patch]
        if features.shape[1]!=grid[0]*grid[1] or not torch.isfinite(features).all():raise ValueError('Invalid patch features')
        name=row['sample_id']+'.npy'; out=a.output/name
        array=features[0].float().cpu().numpy()
        with out.with_suffix('.tmp').open('wb') as f:np.save(f,array)
        out.with_suffix('.tmp').replace(out)
        receipt={**row,'signal_type':'visual_features','teacher':'nvidia/C-RADIOv3-L',
            'revision':entry['revision'],'license':entry['license'],'teacher_frozen':True,
            'preprocessing':'exported224 RGB/255 -> bilinear256 align_corners=False; model internal input conditioner; no external mean/std',
            'feature_grid':grid,'feature_dim':int(features.shape[-1]),'feature_shape':list(array.shape),
            'model_patch_size':patch,'source_manifest_sha256':manifest_sha,'extractor_sha256':code_sha,
            'output_file':name,'output_sha256':sha(out),'dtype':str(array.dtype)}
        (a.output/(row['sample_id']+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
        receipts.append(receipt)
        progress={'total':len(rows),'completed':index+1,'failed':0,'updated_at_unix':time.time()}
        temp=a.output/'progress.tmp';temp.write_text(json.dumps(progress));temp.replace(a.output/'progress.json')
        if (index+1)%100==0:print(json.dumps(progress),flush=True)
    manifest=a.output/'cache_manifest.jsonl'
    manifest.write_text(''.join(json.dumps(x)+'\n' for x in receipts))
    (a.output/'cache_manifest.sha256').write_text(sha(manifest)+'\n')
    (a.output/'failures.json').write_text('[]\n')
    print('RADIO_CACHE_OK',len(rows),sha(manifest),flush=True)


if __name__=='__main__':main()
