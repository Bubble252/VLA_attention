"""Prepare one real, episode-isolated OFT batch using official transforms.

No model weights are loaded. Keeps full H=8 labels and both camera views.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU preflight requires CUDA_VISIBLE_DEVICES empty')
    import numpy as np
    import torch
    import tensorflow as tf
    from PIL import Image
    from transformers import AutoTokenizer
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor
    from prismatic.vla.datasets.datasets import RLDSBatchTransform
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--statistics',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    row=json.loads(a.manifest.read_text().splitlines()[0])
    records=tf.data.TFRecordDataset([row['source_shard']],num_parallel_reads=1)
    raw=next(iter(records.skip(row['record_index']).take(1))).numpy()
    if hashlib.sha256(raw).hexdigest()!=row['episode_id']:
        raise ValueError('Episode payload changed since inventory')
    features=tf.train.Example.FromString(raw).features.feature
    x={}
    for key,value in features.items():
        kind=value.WhichOneof('kind')
        values=getattr(value,kind).value
        x[key]=np.asarray(values,dtype=object if kind=='bytes_list' else None)
    actions=x['steps/action'].reshape(-1,7).copy()
    states=x['steps/observation/state'].reshape(-1,8).copy()
    stats=json.loads(a.statistics.read_text())
    # Official LIBERO standardization; stats are measured only on training demos.
    actions[:,-1]=1-np.clip(actions[:,-1],0,1)
    def normalize(values, name):
        low=np.array(stats[name]['q01']); high=np.array(stats[name]['q99'])
        mask=np.array(stats[name].get('mask',[True]*values.shape[-1]))
        return np.where(mask,np.clip(2*(values-low)/(high-low+1e-8)-1,-1,1),values).astype(np.float32)
    actions=normalize(actions,'action')
    proprio=normalize(states,'proprio')
    def image(name):
        # RLDS stores JPEGs; no additional 180-degree rotation here.
        im=tf.io.decode_jpeg(x['steps/observation/'+name][0],channels=3)
        im=tf.image.resize(im,[224,224],method='lanczos3',antialias=True)
        return tf.cast(tf.clip_by_value(tf.round(im),0,255),tf.uint8).numpy()[None]
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True,use_fast=True)
    processor=PrismaticImageProcessor.from_pretrained(a.model,local_files_only=True)
    transform=RLDSBatchTransform(ActionTokenizer(tokenizer),tokenizer,processor.apply_transform,
                                  PurePromptBuilder,use_wrist_image=True,use_proprio=True)
    record={'dataset_name':b'libero_spatial_no_noops','action':actions[:8],
            'task':{'language_instruction':x['steps/language_instruction'][0]},
            'observation':{'image_primary':image('image'),'image_wrist':image('wrist_image'),'proprio':proprio[:1]}}
    instance=transform(record)
    collator=PaddedCollatorForActionPrediction(tokenizer.model_max_length,tokenizer.pad_token_id)
    batch=collator([instance])
    # Official collator squeezes B=1; retain a batch axis for the proprio MLP.
    batch['proprio']=batch['proprio'].reshape(1,8)
    mask=get_current_action_mask(batch['labels'][:,1:]) | get_next_actions_mask(batch['labels'][:,1:])
    assert mask.sum().item()==56, 'action tokenizer did not preserve all 56 action slots'
    assert tuple(batch['actions'].shape)==(1,8,7)
    assert tuple(batch['pixel_values'].shape)==(1,12,224,224)
    assert not torch.cuda.is_initialized()
    a.output.mkdir(parents=True)
    torch.save(batch,a.output/'batch.pt')
    payload={'episode_id':row['episode_id'],'task':row['task_name'],'source_shard':row['source_shard'],
             'record_index':row['record_index'],'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
             'statistics_sha256':hashlib.sha256(a.statistics.read_bytes()).hexdigest(),
             'shapes':{k:list(v.shape) for k,v in batch.items() if hasattr(v,'shape')},
             'action_slots':int(mask.sum()),'cuda_initialized':False,
             'preprocessing':'RLDS JPEG decode, Lanczos3 resize224, processor normalization, no augmentation for CPU gate',
             'next_gate':'not a forward/backward/restore/rollout test'}
    (a.output/'report.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload))


if __name__=='__main__':
    main()
