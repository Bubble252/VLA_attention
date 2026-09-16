"""Push selected review/plan Markdown via DocSync and verify all text blocks.

Uses existing app credentials; never prints secrets. Does not recurse folders,
delete unrelated cloud documents, or trust DocSync's success counters.
"""
import collections
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--nearest-only', action='store_true', help='Sync only new three-method analysis and its two updated entry documents')
parser.add_argument('--blindvla-only', action='store_true', help='Sync BlindVLA code audit and its updated study/plan documents')
args = parser.parse_args()
TOOL = ROOT.parent / 'doc-sync-main'
sys.path.insert(0, str(TOOL / 'src'))
os.chdir(TOOL)
from doc_sync.converter import MarkdownToFeishu
from doc_sync.feishu_client import FeishuClient
from doc_sync.sync.manager import SyncManager

OUT = ROOT / 'outputs/pdf_review_sync'
OUT.mkdir(parents=True, exist_ok=True)
cfg_path = TOOL / 'sync_config.json'
cfg = json.loads(cfg_path.read_text())
session = requests.Session()
BASE = 'https://open.feishu.cn/open-apis/'

def api(method, path, **kwargs):
    r = session.request(method, BASE + path, timeout=40, **kwargs)
    data = r.json()
    if r.status_code >= 400 or data.get('code', 0) != 0:
        # Do not dump response or request headers (could contain credentials).
        raise RuntimeError(f'Feishu {path}: HTTP {r.status_code}, code={data.get("code")}')
    return data

auth = api('POST', 'auth/v3/tenant_access_token/internal', json={
    'app_id': cfg['feishu_app_id'], 'app_secret': cfg['feishu_app_secret']})
session.headers['Authorization'] = 'Bearer ' + auth['tenant_access_token']
client = FeishuClient(cfg['feishu_app_id'], cfg['feishu_app_secret'])

def paginated(path, params, key):
    items = []
    while True:
        d = api('GET', path, params=params).get('data', {})
        items.extend(d.get(key, []))
        if not d.get('has_more'):
            return items
        token = d.get('page_token')
        if not token:
            raise RuntimeError('Missing page token')
        params = dict(params, page_token=token)

def text_blocks(blocks):
    result = collections.Counter()
    for b in blocks:
        if b.get('block_type') != 1:
            for v in b.values():
                if isinstance(v, dict) and 'elements' in v:
                    content = ''.join(e.get('text_run', {}).get('content', '')
                                      for e in v['elements'])
                    content = re.sub(r'\s+', '', content).replace('\u200b', '')
                    if content:
                        result[content] += 1
        children = b.get('children', [])
        result.update(text_blocks([c for c in children if isinstance(c, dict)]))
    return result

targets = []
review = ROOT / 'references/abstract_review'
for name in ['README.md', '01_vla_alignment_and_shortcuts.md',
             '02_world_models_and_dynamic_teachers.md',
             '03_vlm_teachers_and_background.md',
             '04_selection_and_experiment_decisions.md',
             '05_inventory_and_reading_status.md']:
    targets.append((review / name, 'DAB9w31AUiuZJAkN5B7cukoGn2g', 'PDF摘要归纳-' + Path(name).stem))
for name in ['01_project_background.md', '02_technical_stack.md', '03_execution_plan.md']:
    targets.append((ROOT / 'doc' / name, 'KodSw9aQXiCNygkmN9bcOWU3nrd', Path(name).stem))
for name in ['00_paper_blueprint.md', '04_system_and_method.md',
             '05_experiments_and_expected_conclusions.md',
             '08_model_baseline_benchmark_options.md', 'README.md']:
    targets.append((ROOT / 'paper_draft' / name, 'RGgnwMcVUi4xd5klGX9cAThinBc', Path(name).stem))

if args.nearest_only:
    targets = [
        (review / '07_three_nearest_methods_deep_read.md', 'DAB9w31AUiuZJAkN5B7cukoGn2g', 'PDF摘要归纳-07_three_nearest_methods_deep_read'),
        (review / 'README.md', 'DAB9w31AUiuZJAkN5B7cukoGn2g', 'PDF摘要归纳-README'),
        (ROOT / 'paper_draft/05_experiments_and_expected_conclusions.md', 'RGgnwMcVUi4xd5klGX9cAThinBc', '05_experiments_and_expected_conclusions'),
    ]

manifest_path = OUT / 'verified.json'
if args.blindvla_only:
    targets = [
        (review / '08_blindvla_code_audit.md', 'DAB9w31AUiuZJAkN5B7cukoGn2g', 'PDF摘要归纳-08_blindvla_code_audit'),
        (review / '07_three_nearest_methods_deep_read.md', 'DAB9w31AUiuZJAkN5B7cukoGn2g', 'PDF摘要归纳-07_three_nearest_methods_deep_read'),
        (review / 'README.md', 'DAB9w31AUiuZJAkN5B7cukoGn2g', 'PDF摘要归纳-README'),
        (ROOT / 'doc/03_execution_plan.md', 'KodSw9aQXiCNygkmN9bcOWU3nrd', '03_execution_plan'),
        (ROOT / 'paper_draft/05_experiments_and_expected_conclusions.md', 'RGgnwMcVUi4xd5klGX9cAThinBc', '05_experiments_and_expected_conclusions'),
    ]
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
for local, folder, title in targets:
    source = local.read_text()
    expected = text_blocks(MarkdownToFeishu().parse(source))
    assert expected, local
    listing = paginated('drive/v1/files', {'folder_token': folder, 'page_size': 200}, 'files')
    matching = [f for f in listing if f['name'] == title and f['type'] == 'docx']
    if len(matching) > 1:
        raise RuntimeError(f'Ambiguous duplicate document name: {title}')
    if matching:
        token = matching[0]['token']
    else:
        created = api('POST', 'docx/v1/documents', json={'folder_token': folder, 'title': title})
        token = created['data']['document']['document_id']
    block_path = f'docx/v1/documents/{token}/blocks'
    before = paginated(block_path, {'page_size': 500}, 'items')
    if expected != text_blocks(before):
        (OUT / (token + '.before.json')).write_text(json.dumps(before, ensure_ascii=False))
        SyncManager(str(local), token, force=True, vault_root=str(ROOT), client=client).run()
    for attempt in range(3):
        after = paginated(block_path, {'page_size': 500}, 'items')
        actual = text_blocks(after)
        if expected == actual:
            break
        time.sleep(1)
    else:
        (OUT / (token + '.mismatch.json')).write_text(json.dumps({
            'missing': dict(expected-actual), 'extra': dict(actual-expected)}, ensure_ascii=False, indent=2))
        raise RuntimeError(f'Cloud text mismatch: {title}; see outputs/pdf_review_sync')
    rel = str(local.relative_to(ROOT))
    manifest[rel] = {'title': title, 'folder': folder, 'document': token,
        'url': 'https://feishu.cn/docx/' + token, 'sha256': hashlib.sha256(source.encode()).hexdigest(),
        'text_blocks': sum(expected.values()), 'verified_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'verification': 'Full normalized text-block multiset equality; includes nested/table cell text.'}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print('VERIFIED', rel, manifest[rel]['url'], flush=True)

# Register each review as a file-to-document task: no folder-token ambiguity
# and no need to expose the machine-readable/full-source files in Feishu.
latest = json.loads(cfg_path.read_text())
tasks = latest.setdefault('tasks', [])
for local, folder, title in [t for t in targets if t[0].parent == review]:
    tasks[:] = [t for t in tasks if t.get('local') != str(local)]
    tasks.append({'note': title, 'local': str(local),
                  'cloud': manifest[str(local.relative_to(ROOT))]['document'],
                  'vault_root': str(ROOT), 'enabled': True, 'force': True})
backup = TOOL / 'sync_config.json.bak.pdf_review'
if not backup.exists():
    backup.write_text(cfg_path.read_text())
    backup.chmod(0o600)
cfg_path.write_text(json.dumps(latest, ensure_ascii=False, indent=2))
cfg_path.chmod(0o600)
print('ALL VERIFIED:', len(targets), 'documents; review tasks registered', flush=True)
