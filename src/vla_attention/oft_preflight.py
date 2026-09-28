"""Non-GPU provenance checks shared by real OFT interface and B0 runs."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def require_gpu_window(path):
    """Require an audited, freshly checked and idle *selected* physical GPU.

    nvidia-smi uses physical IDs independently of CUDA_VISIBLE_DEVICES. Bind
    CUDA to the verified UUID so CUDA ordinal ordering cannot pick another GPU.
    Call before importing torch/TensorFlow or creating any GPU context.
    """
    gate = json.loads(Path(path).read_text())
    if gate.get('cache_audit_passed') is not True or gate.get('exclusive_b0_window') is not True:
        raise ValueError('Resource/cache gate not passed')
    if not 0 <= time.time() - gate.get('checked_at_unix', 0) < 300:
        raise ValueError('Resource window must be checked within 5 minutes')
    selected = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if not selected or ',' in selected:
        raise ValueError('Explicit single CUDA_VISIBLE_DEVICES GPU required')
    inventory = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    devices = [tuple(field.strip() for field in line.split(','))
               for line in inventory.splitlines() if line.strip()]
    matches = [uuid for index, uuid in devices if selected in (index, uuid)]
    if len(matches) != 1:
        raise ValueError('Selected physical GPU not uniquely found')
    uuid = matches[0]
    if gate.get('gpu_uuid') != uuid:
        raise ValueError('Resource window GPU UUID mismatch')
    processes = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    for line in processes.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(',')]
        if len(fields) != 2 or not fields[0].startswith('GPU-') or not fields[1].isdigit():
            raise RuntimeError('Cannot verify GPU compute-process inventory')
        if fields[0] == uuid:
            raise RuntimeError('Selected GPU compute processes still present')
    os.environ['CUDA_VISIBLE_DEVICES'] = uuid
    return gate


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
