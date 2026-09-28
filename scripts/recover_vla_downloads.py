"""Retry missing pinned artifacts through a mirror; verify before promotion."""
import argparse
import fcntl
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from download_verified_ranges import download,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source-lock',type=Path,required=True)
    p.add_argument('--endpoint',default='https://hf-mirror.com');a=p.parse_args()
    with (a.root/'recovery.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        failures=[]
        for e in json.loads(a.source_lock.read_text())['entries']:
            if e['repo_type']!='model':continue
            dest=a.root/'models'/e['repo'].replace('/','--')
            staging=a.root/'mirror_staging'/e['repo'].replace('/','--')
            for f in e['files']:
                target=dest/f['path']
                if target.exists():continue # full source-lock re-audit still required
                url=f"{a.endpoint}/{e['repo']}/resolve/{e['revision']}/{f['path']}"
                stage=staging/f['path'];stage.parent.mkdir(parents=True,exist_ok=True)
                try:
                    if f.get('sha256'):
                        download(url,stage,f['size'],f['sha256'])
                    else:
                        with urllib.request.urlopen(url,timeout=30) as response:data=response.read()
                        blob=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
                        if blob!=f['git_blob_id']:raise ValueError('Upstream Git blob mismatch')
                        stage.write_bytes(data)
                    target.parent.mkdir(parents=True,exist_ok=True)
                    os.link(stage,target) # fails rather than overwrite another completed producer
                    print('PROMOTED_VERIFIED',str(target),flush=True)
                except Exception as exc:
                    failures.append({'repo':e['repo'],'path':f['path'],'error':repr(exc)})
                    print('RECOVERY_FAILED',e['repo'],f['path'],type(exc).__name__,flush=True)
        (a.root/'artifacts/recovery_failures.json').write_text(json.dumps(failures,indent=2)+'\n')
        if failures:raise SystemExit(2)
        print('RECOVERY_COMPLETE_REAUDIT_REQUIRED',flush=True)


if __name__=='__main__':main()
