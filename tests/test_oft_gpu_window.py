import json
import time

import pytest

from vla_attention.oft_preflight import require_gpu_window


def setup_window(tmp_path, monkeypatch, processes):
    path = tmp_path / 'window.json'
    path.write_text(json.dumps({'cache_audit_passed': True, 'exclusive_b0_window': True,
                               'checked_at_unix': time.time(), 'gpu_uuid': 'GPU-second'}))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    def query(args, **kwargs):
        return '0, GPU-first\n1, GPU-second\n' if '--query-gpu=index,uuid' in args else processes
    monkeypatch.setattr('vla_attention.oft_preflight.subprocess.check_output', query)
    return path


def test_other_gpu_jobs_do_not_block_selected_idle_gpu(tmp_path, monkeypatch):
    path = setup_window(tmp_path, monkeypatch, 'GPU-first, 123\n')
    require_gpu_window(path)
    import os
    assert os.environ['CUDA_VISIBLE_DEVICES'] == 'GPU-second'
    require_gpu_window(path)  # UUID-bound repeat check also resolves.


def test_selected_gpu_busy_is_rejected(tmp_path, monkeypatch):
    path = setup_window(tmp_path, monkeypatch, 'GPU-first, 123\nGPU-second, 456\n')
    with pytest.raises(RuntimeError, match='still present'):
        require_gpu_window(path)


@pytest.mark.parametrize('device', ['', '0,1', '7', 'GPU-missing'])
def test_ambiguous_or_missing_selection_is_rejected(tmp_path, monkeypatch, device):
    path = setup_window(tmp_path, monkeypatch, '')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', device)
    with pytest.raises(ValueError):
        require_gpu_window(path)


def test_gate_cannot_be_reused_for_another_gpu(tmp_path, monkeypatch):
    path = setup_window(tmp_path, monkeypatch, '')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    with pytest.raises(ValueError, match='UUID mismatch'):
        require_gpu_window(path)


def test_unknown_process_output_fails_closed(tmp_path, monkeypatch):
    path = setup_window(tmp_path, monkeypatch, 'N/A\n')
    with pytest.raises(RuntimeError, match='inventory'):
        require_gpu_window(path)


def test_stale_window_rejected(tmp_path, monkeypatch):
    path = setup_window(tmp_path, monkeypatch, '')
    data = json.loads(path.read_text()); data['checked_at_unix'] -= 301
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='5 minutes'):
        require_gpu_window(path)
