"""Native OpenVLA-OFT B0 train/restore/offline gate; NOT a full benchmark.

Requires an explicit resource-window artifact; never auto-launches alongside
cache workers. No teacher, SAEB or auxiliary loss is used.
"""
import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--source-lock',type=Path,required=True)
    p.add_argument('--train-manifest',type=Path,required=True)
    p.add_argument('--eval-manifest',type=Path,required=True)
    p.add_argument('--statistics',type=Path,required=True)
    p.add_argument('--gpu-window',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=30)
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--restore',type=Path)
    a=p.parse_args()
    if not 20<=a.steps<=50:raise ValueError('B0 smoke limited to 20–50 steps')
    if a.output.exists():raise FileExistsError(a.output)
    gate=json.loads(a.gpu_window.read_text())
    if gate.get('cache_audit_passed') is not True or gate.get('exclusive_b0_window') is not True:
        raise ValueError('Resource/cache gate not passed')
    if not 0<=time.time()-gate.get('checked_at_unix',0)<300:
        raise ValueError('Resource window must be checked within 5 minutes')
    # The artifact alone is insufficient: reject any existing GPU compute job.
    running=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if running:raise RuntimeError('GPU compute processes still present; do not start smoke')
    lock=json.loads(a.source_lock.read_text())
    source=next(e for e in lock['entries'] if e['role']=='training_initialization')
    for item in source['files']:
        f=a.model/item['path']
        if not f.is_file():raise FileNotFoundError(f)
        if item.get('sha256') and sha(f)!=item['sha256']:raise ValueError(f'Weight SHA mismatch: {f}')
    train_ids={json.loads(x)['episode_id'] for x in a.train_manifest.read_text().splitlines()}
    eval_ids={json.loads(x)['episode_id'] for x in a.eval_manifest.read_text().splitlines()}
    if train_ids & eval_ids:raise ValueError('Train/eval episodes overlap')
    import numpy as np
    import torch
    from peft import LoraConfig,get_peft_model,PeftModel
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.projectors import ProprioProjector
    from vla_attention.benchmarks.oft_rlds import EpisodeDataset
    from vla_attention.adapters.oft_forward import forward_l1
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    # Native OFT classes use pinned external code, not mutable HF custom-code copying.
    config=OpenVLAConfig.from_pretrained(a.model,local_files_only=True)
    base=OpenVLAForActionPrediction.from_pretrained(a.model,config=config,local_files_only=True,
              torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation='eager').cuda()
    base.vision_backbone.set_num_images_in_input(2)
    patch_count=base.vision_backbone.get_num_patches()*2
    if a.restore:
        policy=PeftModel.from_pretrained(base,a.restore/'adapter',is_trainable=True)
    else:
        policy=get_peft_model(base,LoraConfig(r=32,lora_alpha=16,lora_dropout=.0,
                                            target_modules='all-linear',init_lora_weights='gaussian'))
    head=L1RegressionActionHead(input_dim=base.llm_dim,hidden_dim=base.llm_dim).cuda().bfloat16()
    proprio=ProprioProjector(llm_dim=base.llm_dim,proprio_dim=8).cuda().bfloat16()
    params=[x for m in [policy,head,proprio] for x in m.parameters() if x.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=5e-4)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100000],gamma=.1)
    train=EpisodeDataset(a.train_manifest,a.statistics,a.model)
    evaluation=EpisodeDataset(a.eval_manifest,a.statistics,a.model)
    metadata={'model_revision':source['revision'],'train_manifest_sha256':sha(a.train_manifest),
              'eval_manifest_sha256':sha(a.eval_manifest),'statistics_sha256':sha(a.statistics),
              'seed':a.seed,'steps':a.steps,'action_loss':'L1','chunk':8,'attention':'eager',
              'augmentation':False,'purpose':'B0 engineering smoke, not official full recipe'}
    if json.loads(a.statistics.read_text())['provenance']['train_manifest_sha256'] != metadata['train_manifest_sha256']:
        raise ValueError('Normalization stats were not generated from this training split')
    a.output.mkdir(parents=True)
    (a.output/'config.json').write_text(json.dumps(metadata,indent=2)+'\n')
    def forward(batch):return forward_l1(policy,head,proprio,batch,device='cuda',visual_token_count=patch_count)
    def predictions():
        for m in [policy,head,proprio]:m.eval()
        result=[]
        # One fixed sample per held-out episode.
        indices=[next(k for k,(i,t) in enumerate(evaluation.index) if i==j) for j in range(len(evaluation.rows))]
        with torch.no_grad():
            for index in indices:
                batch=evaluation.collate([evaluation[index]])
                out=forward(batch)
                result.append({'episode_id':batch['episode_ids'][0],'l1':float(out['loss']),
                               'prediction':out['prediction'].float().cpu().tolist()})
        return result
    if a.restore:
        if sha(a.restore/'state.pt') != (a.restore/'state.sha256').read_text().strip():
            raise ValueError('Saved training state SHA mismatch')
        state=torch.load(a.restore/'state.pt',map_location='cpu')
        if state['metadata']!=metadata:raise ValueError('Restore provenance mismatch')
        head.load_state_dict(state['head']);proprio.load_state_dict(state['proprio'])
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
        actual=predictions();expected=state['eval_predictions']
        if len(actual)!=len(expected):raise ValueError('Restore eval coverage mismatch')
        max_error=0.
        for now,before in zip(actual,expected):
            if now['episode_id']!=before['episode_id']:raise ValueError('Restore episode mismatch')
            max_error=max(max_error,float(np.max(np.abs(np.array(now['prediction'])-np.array(before['prediction'])))))
        if max_error>1e-3:raise ValueError(f'Restore prediction drift: {max_error}')
        (a.output/'restore_report.json').write_text(json.dumps({'max_prediction_error':max_error,
                'optimizer_state_entries':len(optimizer.state),'scheduler_state':scheduler.state_dict(),
                'saved_step':state['step'],'restored':True,'eval_predictions':actual},indent=2)+'\n')
        return
    order=list(range(len(train)));random.Random(a.seed).shuffle(order)
    for m in [policy,head,proprio]:m.train()
    for step in range(a.steps):
        started=time.time();batch=train.collate([train[order[step%len(order)]]])
        optimizer.zero_grad(set_to_none=True);out=forward(batch);out['loss'].backward()
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        optimizer.step();scheduler.step()
        record={'step':step+1,'action_l1':float(out['loss'].detach()),'grad_norm':float(norm),
                'lr':scheduler.get_last_lr()[0],'seconds':time.time()-started,
                'peak_memory_bytes':torch.cuda.max_memory_allocated(),
                'episode_id':batch['episode_ids'][0],'timestep':batch['timesteps'][0]}
        with (a.output/'train_metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    measured=predictions()
    policy.save_pretrained(a.output/'adapter')
    state={'metadata':metadata,'step':a.steps,'head':head.state_dict(),'proprio':proprio.state_dict(),
           'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
           'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
           'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
           'order':order,'eval_predictions':measured}
    torch.save(state,a.output/'state.pt.tmp');(a.output/'state.pt.tmp').replace(a.output/'state.pt')
    (a.output/'offline_eval.json').write_text(json.dumps(measured,indent=2)+'\n')
    (a.output/'state.sha256').write_text(sha(a.output/'state.pt')+'\n')
    print('B0_TRAIN_FINISHED_RESTORE_AND_ROLLOUT_PENDING',flush=True)


if __name__=='__main__':main()
