"""Export real train observations for three typed teacher caches, no test data."""
import argparse
import hashlib
import io
import json
from pathlib import Path


def main():
    import tensorflow as tf
    import numpy as np
    from PIL import Image
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    p=argparse.ArgumentParser()
    p.add_argument('--train-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(x) for x in a.train_manifest.read_text().splitlines()]
    manifest_sha=hashlib.sha256(a.train_manifest.read_bytes()).hexdigest()
    frames=[];semantic=[]
    for row in rows:
        if row['task_name']!='pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate':
            raise ValueError('Task phrase mapping has not been specified for this task')
        options=tf.data.Options();options.threading.private_threadpool_size=1
        ds=tf.data.TFRecordDataset([row['source_shard']],num_parallel_reads=1).with_options(options)
        raw=next(iter(ds.skip(row['record_index']).take(1))).numpy()
        if hashlib.sha256(raw).hexdigest()!=row['episode_id']:raise ValueError('Episode hash drift')
        features=tf.train.Example.FromString(raw).features.feature
        for t in range(row['steps']-7):
            for camera in ['image','wrist_image']:
                encoded=features['steps/observation/'+camera].bytes_list.value[t]
                pixels=tf.io.decode_jpeg(encoded,channels=3)
                pixels=tf.image.resize(pixels,[224,224],method='lanczos3',antialias=True)
                pixels=tf.cast(tf.clip_by_value(tf.round(pixels),0,255),tf.uint8).numpy()
                ident=f"{row['episode_id']}_{t:04d}_{camera}"
                rel=Path('images')/row['episode_id']/f'{t:04d}_{camera}.png'
                path=a.output/rel;path.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(pixels).save(path)
                entry={'sample_id':ident,'episode_id':row['episode_id'],'timestep':t,'camera':camera,
                       'split':'train','task_name':row['task_name'],'instruction':row['language'],
                       'image_path':str(rel),'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                       'source_jpeg_sha256':hashlib.sha256(encoded).hexdigest(),'source_manifest_sha256':manifest_sha,
                       'preprocessing':'RLDS JPEG->TF Lanczos3 224x224 round uint8; no rotation/crop/augmentation'}
                frames.append(entry)
                for role,phrase,occ in [('source','the black bowl','first'),('target','the plate','last')]:
                    semantic.append({**entry,'sample_id':ident+'_'+role,'role':role,'phrase':phrase,
                                     'phrase_occurrence':occ,'caption':row['language']})
        print('EPISODE_EXPORTED',row['episode_id'],flush=True)
    for name,data in [('frames',frames),('semantic',semantic)]:
        path=a.output/(name+'.jsonl')
        path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in data))
        (a.output/(name+'.sha256')).write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'\n')
    # Same frozen observations reserved for SAEB; does not choose a VLM checkpoint.
    (a.output/'saeb_pending.json').write_text(json.dumps({'input_manifest':'frames.jsonl',
         'status':'blocked_until_frozen_vlm_checkpoint_and_output_scalar_validated',
         'no_latents_generated':True},indent=2)+'\n')
    (a.output/'summary.json').write_text(json.dumps({'episodes':len(rows),'frames':len(frames),'semantic_maps':len(semantic),
         'train_manifest_sha256':manifest_sha,'all_valid_h8_frames':True,'camera_count':2,
         'phrase_mapping':'task-specific source/target; last target mention, no box supervision'},indent=2)+'\n')
    print('EXPORT_OK',len(frames),len(semantic))


if __name__=='__main__':main()
