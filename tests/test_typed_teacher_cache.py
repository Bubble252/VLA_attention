import json
import hashlib
import pytest
np=pytest.importorskip('numpy')
from scripts.validate_typed_teacher_cache import validate


def test_mixed_stage_and_corrupted_output_are_rejected(tmp_path):
    cache=tmp_path/'cache';cache.mkdir()
    row={'sample_id':'a','caption':'bowl plate','phrase':'plate','phrase_occurrence':'last'}
    manifest=tmp_path/'manifest.jsonl';manifest.write_text(json.dumps(row)+'\n')
    np.save(cache/'a.npy',np.ones((16,16))/256)
    meta={**row,'method':'ddim_nulltext_fp32_reconstruction_v3','capture_stage':'final_reconstruction_only',
          'teacher_frozen':True,'attention_tensors':100,'inversion_steps':20,'inner_steps':10,
          'mean_reconstruction_mse':.01,'map_sha256':hashlib.sha256((cache/'a.npy').read_bytes()).hexdigest()}
    (cache/'a.json').write_text(json.dumps(meta))
    assert validate(manifest,cache,'semantic')['passed']
    meta['attention_tensors']=1400;(cache/'a.json').write_text(json.dumps(meta))
    assert not validate(manifest,cache,'semantic')['passed']
    meta['attention_tensors']=100;(cache/'a.json').write_text(json.dumps(meta))
    np.save(cache/'a.npy',np.zeros((16,16)))
    assert not validate(manifest,cache,'semantic')['passed']
