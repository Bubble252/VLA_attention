"""Native OpenVLA-OFT B0 train/restore/offline gate; NOT a full benchmark.

Requires an explicit resource-window artifact; never auto-launches alongside
cache workers. No teacher, SAEB or auxiliary loss is used.
"""
import argparse
import os
import hashlib
import json
import random
import time
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
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
    p.add_argument('--p1-report',type=Path)
    a=p.parse_args()
    if not 20<=a.steps<=50:raise ValueError('B0 smoke limited to 20–50 steps')
    if a.output.exists():raise FileExistsError(a.output)
    from vla_attention.oft_preflight import verify_snapshot,require_p1,require_gpu_window
    gate=require_gpu_window(a.gpu_window)
    source=verify_snapshot(a.model,a.source_lock)
    # Hashing large weights can outlast the resource window. Refresh the
    # gate externally and rerun if stale; never load based on an old check.
    require_gpu_window(a.gpu_window)
    if a.p1_report is None:raise ValueError('A real passing P1 report is required')
    require_p1(a.p1_report,source['revision'],sha(a.train_manifest),sha(a.statistics))
    train_ids={json.loads(x)['episode_id'] for x in a.train_manifest.read_text().splitlines()}
    eval_ids={json.loads(x)['episode_id'] for x in a.eval_manifest.read_text().splitlines()}
    if train_ids & eval_ids:raise ValueError('Train/eval episodes overlap')
    from vla_attention.oft_checkpoint import verify_checkpoint,seal_checkpoint
    restored_digest=verify_checkpoint(a.restore) if a.restore else None
    import numpy as np
    import torch
    from peft import LoraConfig,get_peft_model,PeftModel
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.projectors import ProprioProjector
    from vla_attention.benchmarks.oft_rlds import EpisodeDataset
    from vla_attention.adapters.oft_forward import forward_l1
    from vla_attention.resume_probe import optimizer_probe
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    # Native OFT classes use pinned external code, not mutable HF custom-code copying.
    config=OpenVLAConfig.from_pretrained(a.model,local_files_only=True)
    base=OpenVLAForActionPrediction.from_pretrained(a.model,config=config,local_files_only=True,
              torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation='sdpa').cuda()
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
              'seed':a.seed,'steps':a.steps,'action_loss':'L1','chunk':8,'attention':'official_fork_sdpa_bidirectional',
              'augmentation':False,'p1_report_sha256':sha(a.p1_report),
              'purpose':'B0 engineering smoke, not official full recipe'}
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
            current=np.array(now['prediction']);saved=np.array(before['prediction'])
            if current.shape!=saved.shape or not np.isfinite(current).all() or not np.isfinite(saved).all():
                raise ValueError('Restore prediction shape/nonfinite mismatch')
            max_error=max(max_error,float(np.max(np.abs(current-saved))))
        if max_error!=0.:raise ValueError(f'Restore prediction drift: {max_error}')
        (a.output/'restore_report.json').write_text(json.dumps({'max_prediction_error':max_error,
                'optimizer_state_entries':len(optimizer.state),'scheduler_state':scheduler.state_dict(),
                'saved_step':state['step'],'restored':True,'prediction_equality':'exact',
                'checkpoint_manifest_sha256':restored_digest,
                'eval_predictions':actual},indent=2)+'\n')
        # Reset RNG after evaluation; match the uninterrupted continuation from
        # the saved state, including optimizer/scheduler and data cursor.
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
        item=state['order'][state['step']%len(state['order'])]
        batch=train.collate([train[item]])
        probe=optimizer_probe([('policy',policy),('head',head),('proprio',proprio)],optimizer,scheduler,
                              lambda:forward(batch)['loss'])
        expected_probe=json.loads((a.restore/'continuation_reference.json').read_text())
        # Compare semantically with a small tolerance: BF16 matmul/reduction order
        # can differ between two fresh CUDA processes even when the checkpoint,
        # optimizer moments, RNG state and data cursor are identical. Exact
        # prediction restore remains a separate zero-drift gate above.
        numeric=('loss','grad_norm')
        for key in numeric:
            if abs(float(probe[key])-float(expected_probe[key])) > 1e-4:
                raise ValueError(f'Optimizer continuation differs after restore: {key}')
        if probe['lr'] != expected_probe['lr'] or probe['scheduler_epoch'] != expected_probe['scheduler_epoch']:
            raise ValueError('Optimizer scheduler continuation differs after restore')
        (a.output/'continuation_report.json').write_text(json.dumps({'passed':True,'probe':probe,
                'expected_probe':expected_probe,'comparison':'loss/grad_norm abs_tol=1e-4; lr and scheduler exact',
                'checkpoint_manifest_sha256':restored_digest},indent=2)+'\n')
        return
    order=list(range(len(train)));random.Random(a.seed).shuffle(order)
    for m in [policy,head,proprio]:m.train()
    for step in range(a.steps):
        started=time.time();batch=train.collate([train[order[step%len(order)]]])
        optimizer.zero_grad(set_to_none=True);out=forward(batch);out['loss'].backward()
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        optimizer.step();scheduler.step()
        record={'step':step+1,'action_l1':float(out['loss'].detach()),'grad_norm':float(norm),
                'finite_loss':bool(torch.isfinite(out['loss'])),'finite_gradients':bool(torch.isfinite(norm)),
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
    # Diagnostic step is performed only after saving the exact smoke state.
    # It is never saved as the trained checkpoint or included in training curves.
    item=order[a.steps%len(order)];batch=train.collate([train[item]])
    probe=optimizer_probe([('policy',policy),('head',head),('proprio',proprio)],optimizer,scheduler,
                          lambda:forward(batch)['loss'])
    (a.output/'continuation_reference.json').write_text(json.dumps(probe,indent=2)+'\n')
    seal_checkpoint(a.output)
    print('B0_TRAIN_FINISHED_RESTORE_AND_ROLLOUT_PENDING',flush=True)


if __name__=='__main__':main()
