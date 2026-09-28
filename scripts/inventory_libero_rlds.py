"""CPU-only real-episode inventory, deterministic splits and train-only stats.

Reads official TFRecords directly, without TensorFlow/GPU initialization.
Rollout init-state identities are deliberately not fabricated from demo IDs.
"""
import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path


def partition(rows, seed):
    ordered = sorted(rows, key=lambda r: hashlib.sha256(f'{seed}:{r["episode_id"]}'.encode()).hexdigest())
    if len(ordered) < 10:
        raise ValueError('too few real episodes for train/validation/offline split')
    nval = max(1, len(ordered) // 5)
    return {'train': ordered[2*nval:], 'validation': ordered[:nval], 'offline_eval': ordered[nval:2*nval]}


def main():
    import numpy as np
    from PIL import Image
    from tfrecord.reader import tfrecord_iterator, extract_feature_dict
    from tfrecord import example_pb2
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--task', default='pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate')
    p.add_argument('--seed', type=int, default=17)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    typename = {'byte':'bytes_list', 'float':'float_list', 'int':'int64_list'}
    rows, seen, arrays = [], set(), {}
    info = json.loads((a.data / 'dataset_info.json').read_text())
    expected = sum(int(v) for s in info['splits'] for v in s['shardLengths'])
    for file in sorted(a.data.glob('*.tfrecord-*')):
        for index, raw in enumerate(tfrecord_iterator(str(file), None)):
            example = example_pb2.Example()
            example.ParseFromString(raw)
            x = extract_feature_dict(example.features, None, typename)
            actions = x['steps/action'].reshape(-1, 7)
            states = x['steps/observation/state'].reshape(-1, 8)
            steps = len(actions)
            if not np.isfinite(actions).all() or not np.isfinite(states).all():
                raise ValueError(f'Nonfinite episode {file}:{index}')
            if len(states) != steps or steps < 8:
                raise ValueError(f'Invalid episode length {file}:{index}')
            instructions = set(v.decode() for v in x['steps/language_instruction'])
            if len(instructions) != 1:
                raise ValueError('Changing instruction inside an episode')
            language = next(iter(instructions))
            task = language.replace(' ', '_')
            payload_hash = hashlib.sha256(raw).hexdigest()
            if payload_hash in seen:
                raise ValueError(f'Duplicate serialized episode {file}:{index}')
            seen.add(payload_hash)
            shapes = {}
            # Decode first and last frames in both views; hash ALL images for
            # detecting identical trajectories serialized under different IDs.
            semantic_hash = hashlib.sha256(actions.tobytes() + states.tobytes() + language.encode())
            for name in ['image', 'wrist_image']:
                images = x['steps/observation/' + name]
                if len(images) != steps:
                    raise ValueError('Image/action length mismatch')
                for encoded in images:
                    semantic_hash.update(hashlib.sha256(encoded).digest())
                for t in [0, steps-1]:
                    with Image.open(io.BytesIO(images[t])) as im:
                        im.load()
                        if im.size != (256,256) or im.mode != 'RGB':
                            raise ValueError('Unexpected camera image geometry')
                shapes[name] = [256,256,3]
            if not x['steps/is_first'][0] or not x['steps/is_last'][-1]:
                raise ValueError('Episode boundary flags missing')
            row = {'episode_id': payload_hash, 'content_hash': semantic_hash.hexdigest(),
                   'source_shard': str(file), 'record_index': index, 'data_revision': a.revision,
                   'original_file_path': x['episode_metadata/file_path'].decode(),
                   'task_name': task, 'language': language, 'steps': steps, 'valid_chunks_h8': steps-7,
                   'action_shape':[steps,7], 'state_shape':[steps,8], 'camera_shapes':shapes,
                   'terminal':bool(x['steps/is_terminal'][-1]),
                   'raw_gripper_values':np.unique(actions[:,-1]).tolist()}
            rows.append(row)
            if task == a.task:
                arrays[payload_hash] = (actions,states)
        print('SHARD_READ',file.name,len(rows),flush=True)
    if len(rows) != expected or len({r['content_hash'] for r in rows}) != len(rows):
        raise ValueError('Episode coverage mismatch or duplicate trajectory content')
    selected = [r for r in rows if r['task_name'] == a.task]
    splits = partition(selected, a.seed)
    def write_rows(name, values):
        path=a.output/name
        path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in values))
        return hashlib.sha256(path.read_bytes()).hexdigest()
    hashes={'inventory.jsonl':write_rows('inventory.jsonl',rows)}
    for name, values in splits.items():
        hashes[name+'.jsonl']=write_rows(name+'.jsonl',values)
    train_actions=np.concatenate([arrays[r['episode_id']][0] for r in splits['train']])
    train_states=np.concatenate([arrays[r['episode_id']][1] for r in splits['train']])
    train_actions[:,-1]=1-np.clip(train_actions[:,-1],0,1)
    stats={}
    for name, values in [('action',train_actions),('proprio',train_states)]:
        stats[name]={k:v.tolist() for k,v in {'q01':np.quantile(values,.01,axis=0),
                     'q99':np.quantile(values,.99,axis=0),'mean':values.mean(0),'std':values.std(0)}.items()}
    stats['action']['mask']=[True]*6+[False]
    stats['provenance']={'train_manifest_sha256':hashes['train.jsonl'],'raw_gripper_transform':'1-clip(raw,0,1)',
                         'scope':'selected task train episodes only; not official all-suite statistics'}
    (a.output/'train_statistics.json').write_text(json.dumps(stats,indent=2)+'\n')
    summary={'actual_episodes':len(rows),'tasks':dict(Counter(r['task_name'] for r in rows)),
             'selected_task':a.task,'selected_episodes':len(selected),'splits':{k:len(v) for k,v in splits.items()},
             'hashes':hashes,'seed':a.seed,'rollout_manifest':'pending real init-state linkage; no fabricated demo IDs',
             'image_validation':'first/last decoded; all frames included in content hash',
             'schema_audit_only':True}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
