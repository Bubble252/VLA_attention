import hashlib
import json
import pytest
from vla_attention.oft_preflight import verify_snapshot,require_p1


def test_processor_metadata_is_hashed_not_just_weights(tmp_path):
    f=tmp_path/'config.json';f.write_bytes(b'{}')
    lock=tmp_path/'lock.json'
    lock.write_text(json.dumps({'entries':[{'role':'training_initialization','revision':'r','files':[
        {'path':'config.json','size':2,'git_blob_id':hashlib.sha1(b'blob 2\0{}').hexdigest()}]}]}))
    assert verify_snapshot(tmp_path,lock)['revision']=='r'
    f.write_bytes(b'[]')
    with pytest.raises(ValueError,match='hash mismatch'):verify_snapshot(tmp_path,lock)


def test_incomplete_or_mismatched_p1_cannot_unlock_b0(tmp_path):
    path=tmp_path/'p1.json'
    report={'status':'passed','action_shape':[1,8,7],'gradient_finite':True,'gradient_nonzero':True,
            'repeatability':{'passed':True},'module_restore':{'passed':True},
            'normalization_roundtrip':{'passed':True},'visual_grid':{'per_camera':[16,16]},
            'model_revision':'r','train_manifest_sha256':'t','statistics_sha256':'s'}
    path.write_text(json.dumps(report));require_p1(path,'r','t','s')
    with pytest.raises(ValueError,match='provenance'):require_p1(path,'new','t','s')
    report['gradient_nonzero']=False;path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='incomplete'):require_p1(path,'r','t','s')
