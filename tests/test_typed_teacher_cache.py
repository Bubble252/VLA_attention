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
    meta.update(manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),guidance_scale=7.5,
                attention_resolution=16,numerics='fp32,deterministic_algorithms,math_sdpa,tf32_off,cublas4096:8',
                instruction_sha256=hashlib.sha256(row['caption'].encode()).hexdigest(),seed=17,
                extractor_sha256='a'*64,image_sha256='b'*64)
    (cache/'a.json').write_text(json.dumps(meta))
    assert validate(manifest,cache,'semantic')['passed']
    meta['attention_tensors']=1400;(cache/'a.json').write_text(json.dumps(meta))
    assert not validate(manifest,cache,'semantic')['passed']
    meta['attention_tensors']=100;(cache/'a.json').write_text(json.dumps(meta))
    np.save(cache/'a.npy',np.zeros((16,16)))
    assert not validate(manifest,cache,'semantic')['passed']


def test_empty_cache_is_not_a_success(tmp_path):
    m=tmp_path/'m.jsonl';m.write_text('')
    with pytest.raises(ValueError,match='Empty'):validate(m,tmp_path,'semantic')
