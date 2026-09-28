"""Serial, resumable download of revision-locked VLA artifacts (CPU only)."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def digest(path, git_blob=False):
    h = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        h.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lock', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    lock = json.loads(args.lock.read_text())
    log = args.root / 'artifacts/download_receipts.jsonl'
    log.parent.mkdir(parents=True, exist_ok=True)
    # Metadata/data first lets CPU audits proceed while large weights download.
    entries = sorted(lock['entries'], key=lambda x: x['repo_type'] != 'dataset')
    failures = []
    for entry in entries:
        dest = args.root / ('data' if entry['repo_type'] == 'dataset' else 'models') / entry['repo'].replace('/', '--')
        dest.mkdir(parents=True, exist_ok=True)
        files = sorted(entry['files'], key=lambda f: f.get('size') or 0)
        for spec in files:
            target = dest / spec['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            expected = spec.get('sha256') or spec.get('git_blob_id')
            is_git = not bool(spec.get('sha256'))
            if target.exists():
                if expected and digest(target, is_git) == expected:
                    print('VERIFIED_EXISTING', target, flush=True)
                    continue
                raise ValueError(f'Existing file failed hash; preserve for investigation: {target}')
            partial = target.with_name(target.name + '.partial')
            prefix = 'datasets/' if entry['repo_type'] == 'dataset' else ''
            url = f"https://huggingface.co/{prefix}{entry['repo']}/resolve/{entry['revision']}/{spec['path']}"
            print('DOWNLOAD', entry['repo'], spec['path'], flush=True)
            started = time.time()
            try:
                subprocess.run(['curl', '--http1.1', '--fail', '--location', '--silent', '--show-error',
                                '--retry', '8', '--retry-all-errors', '--retry-delay', '5', '--connect-timeout', '25',
                                '--speed-time', '120', '--speed-limit', '1024', '--limit-rate', '15M',
                                '--continue-at', '-', '--output', str(partial), url], check=True)
                if spec.get('size') is not None and partial.stat().st_size != spec['size']:
                    raise ValueError(f'Size mismatch: {partial}')
                actual = digest(partial, is_git)
                if not expected or actual != expected:
                    raise ValueError(f'Upstream hash mismatch: {partial}')
                os.replace(partial, target)
            except Exception as exc:
                failures.append(str(target))
                print('DOWNLOAD_FAILED_CONTINUE', spec['path'], repr(exc), flush=True)
                with log.open('a') as handle:
                    handle.write(json.dumps({'repo': entry['repo'], 'revision': entry['revision'],
                                             'path': str(target), 'upstream_hash_verified': False,
                                             'error': repr(exc)}) + '\n')
                continue
            receipt = {'repo': entry['repo'], 'revision': entry['revision'], 'role': entry['role'],
                       'path': str(target), 'sha256': digest(target), 'upstream_hash_verified': True,
                       'seconds': time.time() - started}
            with log.open('a') as handle:
                handle.write(json.dumps(receipt) + '\n')
            print('VERIFIED', spec['path'], flush=True)
    if failures:
        raise SystemExit(f'VLA_SOURCES_INCOMPLETE: {len(failures)} files failed; retry preserves verified files')
    print('VLA_SOURCES_DOWNLOAD_OK', flush=True)


if __name__ == '__main__':
    main()
