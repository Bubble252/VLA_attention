"""Bounded HTTP Range download with durable offsets and final SHA verification.

Never retries in place over the destination. Each chunk is validated against
Content-Range before append; interrupted responses cannot erase prior bytes.
"""
import argparse
import fcntl
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()


def fetch_chunk(url,start,end,total,timeout=45):
    request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
    with urllib.request.urlopen(request,timeout=timeout) as response:
        expected=f'bytes {start}-{end}/{total}'
        if response.status!=206 or response.headers.get('Content-Range')!=expected:
            raise ValueError('Server did not return requested byte range')
        chunk=response.read(end-start+2)
        if len(chunk)!=end-start+1:
            raise ValueError('Incomplete or oversized byte range')
        return chunk


def download(url,target,total,expected_sha,*,chunk_bytes=8*1024*1024,retries=8):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    partial=target.with_name(target.name+'.ranges.partial')
    provenance=target.with_name(target.name+'.ranges.json')
    lock=target.with_name(target.name+'.ranges.lock')
    identity={'url':url,'size':total,'sha256':expected_sha}
    with lock.open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if target.exists():
            if target.stat().st_size==total and sha(target)==expected_sha:return
            raise ValueError('Existing target differs; preserve it for investigation')
        if provenance.exists():
            if json.loads(provenance.read_text())!=identity:raise ValueError('Resume source changed')
        else:
            if partial.exists():raise ValueError('Partial lacks source provenance')
            provenance.write_text(json.dumps(identity,indent=2)+'\n')
        start=partial.stat().st_size if partial.exists() else 0
        if start>total:raise ValueError('Partial longer than expected')
        with partial.open('ab') as out:
            while start<total:
                end=min(total-1,start+chunk_bytes-1)
                for attempt in range(retries):
                    try:
                        chunk=fetch_chunk(url,start,end,total)
                        break
                    except Exception as exc:
                        print(json.dumps({'event':'range_retry','offset':start,'attempt':attempt+1,
                                          'error':type(exc).__name__}),flush=True)
                        if attempt==retries-1:raise
                        time.sleep(min(20,2*(attempt+1)))
                out.write(chunk);out.flush();os.fsync(out.fileno())
                start+=len(chunk)
                print(json.dumps({'event':'download_progress','file':target.name,'bytes':start,'total':total}),flush=True)
        if sha(partial)!=expected_sha:raise ValueError('Final upstream SHA256 mismatch; partial retained')
        os.replace(partial,target)
        print(json.dumps({'event':'verified','file':str(target),'sha256':expected_sha}),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-lock',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--repo',required=True)
    p.add_argument('--filename',required=True)
    p.add_argument('--endpoint',default='https://huggingface.co')
    a=p.parse_args()
    e=next(x for x in json.loads(a.source_lock.read_text())['entries'] if x['repo']==a.repo)
    f=next(x for x in e['files'] if x['path']==a.filename)
    if not f.get('sha256') or not f.get('size'):raise ValueError('Range mode requires upstream size/SHA256')
    prefix='datasets/' if e['repo_type']=='dataset' else ''
    url=f"{a.endpoint.rstrip('/')}/{prefix}{e['repo']}/resolve/{e['revision']}/{f['path']}"
    target=a.root/('data' if prefix else 'models')/a.repo.replace('/','--')/a.filename
    download(url,target,f['size'],f['sha256'])


if __name__=='__main__':main()
