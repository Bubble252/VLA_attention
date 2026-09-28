"""Enumerate source task identities and downloaded RLDS metadata without GPU."""
import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    repo = a.workspace / 'repos/libero'
    src = repo / 'libero/libero/benchmark/libero_suite_task_map.py'
    tasks = next(ast.literal_eval(n.value) for n in ast.parse(src.read_text()).body
                 if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'libero_task_map' for t in n.targets))
    inventory = []
    for suite, names in tasks.items():
        for index, name in enumerate(names):
            bddl = repo / 'libero/libero/bddl_files' / suite / (name + '.bddl')
            init = repo / 'libero/libero/init_files' / suite / (name + '.pruned_init')
            inventory.append({'suite': suite, 'task_order_index': 0, 'task_id': index,
                              'name': name, 'bddl_sha256': sha(bddl) if bddl.exists() else None,
                              'init_states_file': str(init), 'init_file_present': init.exists()})
    data = a.workspace / 'data/openvla--modified_libero_rlds/libero_spatial_no_noops/1.0.0'
    info = json.loads((data / 'dataset_info.json').read_text())
    payload = {'source_commit': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'],text=True).strip(),
               'tasks': inventory, 'dataset_info_sha256': sha(data / 'dataset_info.json'),
               'declared_splits': info['splits'],
               'declared_episodes': sum(int(n) for s in info['splits'] for n in s['shardLengths']),
               'actual_episodes_read': False,
               'next_gate': 'Read records, retain episode identity, train-only normalization, no invented train/val split.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(payload, f, indent=2)
    print(json.dumps({'tasks':len(inventory), 'declared_episodes':payload['declared_episodes'],
                      'missing_bddl':sum(x['bddl_sha256'] is None for x in inventory)}))


if __name__ == '__main__':
    main()
