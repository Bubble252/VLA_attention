import json

import pytest

from vla_attention.oft_checkpoint import seal_checkpoint, verify_checkpoint, require_restored_checkpoint


def make_checkpoint(root):
    (root / 'adapter').mkdir(parents=True)
    for name in ['state.pt', 'state.sha256', 'config.json', 'offline_eval.json',
                 'adapter/adapter_model.safetensors', 'adapter/adapter_config.json']:
        (root / name).write_text('{}')
    (root / 'continuation_reference.json').write_text('{"parameter_sha256":"test"}')
    return seal_checkpoint(root)


@pytest.mark.parametrize('filename', ['adapter/adapter_model.safetensors', 'adapter/adapter_config.json',
                                     'state.pt', 'continuation_reference.json'])
def test_corrupt_component_rejected(tmp_path, filename):
    make_checkpoint(tmp_path)
    (tmp_path / filename).write_text('changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        verify_checkpoint(tmp_path)


def test_missing_weights_cannot_be_sealed(tmp_path):
    make_checkpoint(tmp_path)
    (tmp_path / 'adapter/adapter_model.safetensors').unlink()
    with pytest.raises(ValueError, match='weights'):
        seal_checkpoint(tmp_path)


def test_rollout_requires_matching_exact_restore_and_continuation(tmp_path):
    root = tmp_path / 'checkpoint'
    digest = make_checkpoint(root)
    assert verify_checkpoint(root) == digest
    restored = tmp_path / 'restored'; restored.mkdir()
    report = {'restored': True, 'max_prediction_error': 0, 'checkpoint_manifest_sha256': digest}
    continuation = {'passed': True, 'checkpoint_manifest_sha256': digest,
                    'probe': {'parameter_sha256': 'test'}}
    (restored / 'restore_report.json').write_text(json.dumps(report))
    with pytest.raises(FileNotFoundError):
        require_restored_checkpoint(root, restored)
    (restored / 'continuation_report.json').write_text(json.dumps(continuation))
    assert require_restored_checkpoint(root, restored) == digest
    report['max_prediction_error'] = 1e-8
    (restored / 'restore_report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='Exact restore'):
        require_restored_checkpoint(root, restored)
    report['max_prediction_error'] = 0
    (restored / 'restore_report.json').write_text(json.dumps(report))
    continuation['checkpoint_manifest_sha256'] = 'another-checkpoint'
    (restored / 'continuation_report.json').write_text(json.dumps(continuation))
    with pytest.raises(ValueError, match='continuation'):
        require_restored_checkpoint(root, restored)
