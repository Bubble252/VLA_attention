import json
from scripts.audit_teacher_cache_stages import audit


def test_mixed_stage_cache_is_rejected(tmp_path):
    p = tmp_path / 'sample.json'
    p.write_text(json.dumps({'attention_tensors': 1400, 'inversion_steps': 20}))
    result = audit(tmp_path)
    assert not result['stage_count_gate']
    assert result['invalid'][0]['expected'] == 100


def test_reference_count_and_empty_cache(tmp_path):
    assert not audit(tmp_path)['stage_count_gate']
    (tmp_path / 'sample.json').write_text(json.dumps({'attention_tensors': 100, 'inversion_steps': 20}))
    assert audit(tmp_path)['stage_count_gate']
