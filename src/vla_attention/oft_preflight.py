"""Non-GPU provenance checks shared by real OFT interface and B0 runs."""
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def verify_snapshot(model,source_lock):
    source=next(e for e in json.loads(Path(source_lock).read_text())['entries'] if e['role']=='training_initialization')
    for item in source['files']:
        path=Path(model)/item['path']
        if not path.is_file():raise FileNotFoundError(path)
        if item.get('size') is not None and path.stat().st_size!=item['size']:
            raise ValueError(f'Snapshot size mismatch: {path}')
        if item.get('sha256'):
            valid=sha(path)==item['sha256']
        else:
            valid=hashlib.sha1(f'blob {path.stat().st_size}\0'.encode()+path.read_bytes()).hexdigest()==item.get('git_blob_id')
        if not valid:raise ValueError(f'Snapshot hash mismatch: {path}')
    return source


def require_p1(path,model_revision,train_sha,stats_sha):
    report=json.loads(Path(path).read_text())
    required=['action_shape','gradient_finite','gradient_nonzero','repeatability',
              'normalization_roundtrip','module_restore','visual_grid']
    if report.get('status')!='passed' or any(not report.get(k) for k in required):
        raise ValueError('P1 incomplete')
    if report['action_shape']!=[1,8,7]:raise ValueError('Wrong P1 action schema')
    expected={'model_revision':model_revision,'train_manifest_sha256':train_sha,'statistics_sha256':stats_sha}
    for key,value in expected.items():
        if report.get(key)!=value:raise ValueError(f'P1 provenance mismatch: {key}')
    for key in ['repeatability','normalization_roundtrip','module_restore']:
        if report[key].get('passed') is not True:raise ValueError(f'Failed P1 gate: {key}')
    return report
