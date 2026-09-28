"""Bind B0 state, LoRA adapter and continuation evidence to one checkpoint."""
import json
from pathlib import Path

from vla_attention.oft_preflight import sha


def checkpoint_files(root):
    root = Path(root)
    required = ['state.pt', 'state.sha256', 'config.json', 'offline_eval.json',
                'continuation_reference.json', 'adapter/adapter_config.json']
    weights = [p for p in (root / 'adapter').glob('adapter_model.*')
               if p.suffix in ('.safetensors', '.bin')]
    if len(weights) != 1:
        raise ValueError('Exactly one saved LoRA weights file required')
    required.append(str(weights[0].relative_to(root)))
    for name in required:
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    # Include auxiliary adapter files too, without hashing mutable rollout outputs.
    return sorted(set(required) | {str(p.relative_to(root))
                                  for p in (root / 'adapter').rglob('*') if p.is_file()})


def seal_checkpoint(root):
    root = Path(root)
    inventory = {'schema': 'oft_b0_checkpoint_v1',
                 'files': {name: sha(root / name) for name in checkpoint_files(root)}}
    target = root / 'checkpoint_manifest.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(inventory, indent=2) + '\n')
    temporary.replace(target)
    return sha(target)


def verify_checkpoint(root):
    root = Path(root)
    target = root / 'checkpoint_manifest.json'
    inventory = json.loads(target.read_text())
    if inventory.get('schema') != 'oft_b0_checkpoint_v1':
        raise ValueError('Unknown B0 checkpoint schema')
    if set(inventory['files']) != set(checkpoint_files(root)):
        raise ValueError('Checkpoint file coverage changed')
    for name, expected in inventory['files'].items():
        if sha(root / name) != expected:
            raise ValueError(f'Checkpoint hash mismatch: {name}')
    return sha(target)


def require_restored_checkpoint(root, restore_dir):
    digest = verify_checkpoint(root)
    restore_dir = Path(restore_dir)
    report = json.loads((restore_dir / 'restore_report.json').read_text())
    continuation = json.loads((restore_dir / 'continuation_report.json').read_text())
    if (report.get('restored') is not True or report.get('max_prediction_error') != 0
            or report.get('checkpoint_manifest_sha256') != digest):
        raise ValueError('Exact restore evidence missing or from another checkpoint')
    if (continuation.get('passed') is not True
            or continuation.get('checkpoint_manifest_sha256') != digest
            or continuation.get('probe') != json.loads((Path(root) / 'continuation_reference.json').read_text())):
        raise ValueError('Optimizer continuation evidence missing or mismatched')
    return digest
