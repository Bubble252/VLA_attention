import hashlib
import json
import pytest
from scripts import download_verified_ranges as d


def test_interrupted_download_resumes_without_discarding_bytes(tmp_path,monkeypatch):
    data=b'abcdefghijk';target=tmp_path/'weights';expected=hashlib.sha256(data).hexdigest()
    calls=[]
    def first(url,start,end,total):
        calls.append(start)
        if start>=4:raise OSError('network interruption')
        return data[start:end+1]
    monkeypatch.setattr(d,'fetch_chunk',first)
    with pytest.raises(OSError):d.download('url',target,len(data),expected,chunk_bytes=4,retries=1)
    assert (tmp_path/'weights.ranges.partial').read_bytes()==b'abcd'
    def resumed(url,start,end,total):
        calls.append(start);return data[start:end+1]
    monkeypatch.setattr(d,'fetch_chunk',resumed)
    d.download('url',target,len(data),expected,chunk_bytes=4,retries=1)
    assert target.read_bytes()==data
    assert calls==[0,4,4,8]


def test_source_change_and_bad_hash_are_rejected(tmp_path,monkeypatch):
    target=tmp_path/'weights'
    monkeypatch.setattr(d,'fetch_chunk',lambda *args:b'bad')
    with pytest.raises(ValueError,match='SHA256'):d.download('url',target,3,'0'*64,retries=1)
    assert not target.exists()
    with pytest.raises(ValueError,match='source changed'):d.download('new-url',target,3,'0'*64,retries=1)
