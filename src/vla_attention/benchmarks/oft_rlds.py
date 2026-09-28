"""Episode-isolated official RLDS data for short native OFT smoke runs.

Retains every valid H=8 transition and delegates action tokenization/collation
to pinned OFT. TensorFlow is constrained to CPU before any dataset operation.
"""
import hashlib
import io
import json
from pathlib import Path


def normalize(values, stats):
    import numpy as np
    low=np.asarray(stats['q01']); high=np.asarray(stats['q99'])
    mask=np.asarray(stats.get('mask',[True]*values.shape[-1]))
    result=np.where(mask,np.clip(2*(values-low)/(high-low+1e-8)-1,-1,1),values)
    # This is part of the official bounds-q99 contract, even with a false mask.
    constant=np.asarray(stats['min'])==np.asarray(stats['max'])
    return np.where(constant,0.,result).astype(np.float32)


class EpisodeDataset:
    def __init__(self, manifest, statistics, model_path):
        import tensorflow as tf
        from transformers import AutoTokenizer
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor
        from prismatic.vla.datasets.datasets import RLDSBatchTransform
        from prismatic.vla.action_tokenizer import ActionTokenizer
        from prismatic.models.backbones.llm.prompting import PurePromptBuilder
        from prismatic.util.data_utils import PaddedCollatorForActionPrediction
        tf.config.set_visible_devices([], 'GPU')
        self.tf=tf
        self.manifest=Path(manifest)
        self.rows=[json.loads(line) for line in self.manifest.read_text().splitlines() if line.strip()]
        self.stats=json.loads(Path(statistics).read_text())
        for name in ['action','proprio']:
            if not all(k in self.stats[name] for k in ['q01','q99','min','max']):
                raise ValueError('Stats missing official bounds/constant-dimension fields')
        self.tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,use_fast=True)
        self.processor=PrismaticImageProcessor.from_pretrained(model_path,local_files_only=True)
        self.transform=RLDSBatchTransform(ActionTokenizer(self.tokenizer),self.tokenizer,
                      self.processor.apply_transform,PurePromptBuilder,use_wrist_image=True,use_proprio=True)
        self.collator=PaddedCollatorForActionPrediction(self.tokenizer.model_max_length,self.tokenizer.pad_token_id)
        self.index=[(i,t) for i,row in enumerate(self.rows) for t in range(row['steps']-7)]
        if not self.index: raise ValueError('No complete action chunks')
        self._loaded_index=None
        self._loaded=None

    def __len__(self): return len(self.index)

    def _load_episode(self,i):
        import numpy as np
        if self._loaded_index==i: return self._loaded
        row=self.rows[i]
        records=self.tf.data.TFRecordDataset([row['source_shard']],num_parallel_reads=1)
        options=self.tf.data.Options()
        options.threading.private_threadpool_size=1
        records=records.with_options(options)
        raw=next(iter(records.skip(row['record_index']).take(1))).numpy()
        if hashlib.sha256(raw).hexdigest()!=row['episode_id']:
            raise ValueError('Episode hash mismatch')
        ex=self.tf.train.Example.FromString(raw)
        x={}
        for key,feature in ex.features.feature.items():
            kind=feature.WhichOneof('kind')
            x[key]=np.asarray(getattr(feature,kind).value,dtype=object if kind=='bytes_list' else None)
        action=x['steps/action'].reshape(-1,7).copy()
        action[:,-1]=1-np.clip(action[:,-1],0,1)
        x['normalized_actions']=normalize(action,self.stats['action'])
        x['normalized_states']=normalize(x['steps/observation/state'].reshape(-1,8),self.stats['proprio'])
        self._loaded=x; self._loaded_index=i
        return x

    def __getitem__(self,index):
        i,t=self.index[index]; x=self._load_episode(i); tf=self.tf
        def image(name):
            im=tf.io.decode_jpeg(x['steps/observation/'+name][t],channels=3)
            im=tf.image.resize(im,[224,224],method='lanczos3',antialias=True)
            return tf.cast(tf.clip_by_value(tf.round(im),0,255),tf.uint8).numpy()[None]
        sample={'dataset_name':b'libero_spatial_no_noops','action':x['normalized_actions'][t:t+8],
                'task':{'language_instruction':x['steps/language_instruction'][t]},
                'observation':{'image_primary':image('image'),'image_wrist':image('wrist_image'),
                               'proprio':x['normalized_states'][t:t+1]}}
        result=self.transform(sample)
        result['episode_id']=self.rows[i]['episode_id']; result['timestep']=t
        return result

    def collate(self,samples):
        result=self.collator(samples)
        result['proprio']=result['proprio'].reshape(len(samples),8)
        result['episode_ids']=[s['episode_id'] for s in samples]
        result['timesteps']=[s['timestep'] for s in samples]
        return result
