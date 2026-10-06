"""Real OpenVLA/OFT B1 retention interface smoke.

This is an engineering gate, not a benchmark or full training run. It reuses the
native B0 action path and adds a frozen C-RADIOv3-L patch-feature target plus an
explicit trainable projector. Existing caches and B0 outputs are never modified.
"""
import argparse, hashlib, json, os, random
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    p=argparse.ArgumentParser()
    for name in ['model','source-lock','train-manifest','eval-manifest','statistics','gpu-window','radio-manifest','radio-root','p1-report','output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--lambda-ret', type=float, default=0.05)
    a=p.parse_args()
    if not 20 <= a.steps <= 30: raise ValueError('B1 interface smoke must use 20-30 steps')
    if a.output.exists(): raise FileExistsError(a.output)
    if 'B1' not in a.output.name and 'b1' not in a.output.name: raise ValueError('B1 output directory required')
    from vla_attention.oft_preflight import verify_snapshot, require_p1, require_gpu_window
    require_gpu_window(a.gpu_window); source=verify_snapshot(a.model,a.source_lock); require_gpu_window(a.gpu_window)
    require_p1(a.p1_report, source['revision'], sha(a.train_manifest), sha(a.statistics))
    rows=[json.loads(x) for x in a.radio_manifest.read_text().splitlines() if x.strip()]
    if not rows: raise ValueError('empty C-RADIO manifest')
    by_key={(r['episode_id'],int(r['timestep']),r['camera']):r for r in rows}
    if len(by_key)!=len(rows): raise ValueError('duplicate C-RADIO cache keys')
    for r in rows[:2]:
        if r.get('teacher')!='nvidia/C-RADIOv3-L' or r.get('feature_shape') != [256,1024] or r.get('feature_grid') != [16,16]:
            raise ValueError('unexpected C-RADIO cache contract')
    train_ids={json.loads(x)['episode_id'] for x in a.train_manifest.read_text().splitlines() if x.strip()}
    eval_ids={json.loads(x)['episode_id'] for x in a.eval_manifest.read_text().splitlines() if x.strip()}
    if train_ids & eval_ids: raise ValueError('train/eval overlap')
    import numpy as np, torch
    from peft import LoraConfig, get_peft_model
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.projectors import ProprioProjector
    from vla_attention.benchmarks.oft_rlds import EpisodeDataset
    from vla_attention.adapters.oft_forward import forward_l1
    from vla_attention.losses import cosine_feature_retention
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    config=OpenVLAConfig.from_pretrained(a.model,local_files_only=True)
    base=OpenVLAForActionPrediction.from_pretrained(a.model,config=config,local_files_only=True,low_cpu_mem_usage=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda()
    base.vision_backbone.set_num_images_in_input(2); patch_count=base.vision_backbone.get_num_patches()*2
    policy=get_peft_model(base,LoraConfig(r=32,lora_alpha=16,lora_dropout=0.,target_modules='all-linear',init_lora_weights='gaussian'))
    head=L1RegressionActionHead(input_dim=base.llm_dim,hidden_dim=base.llm_dim).cuda().bfloat16()
    proprio=ProprioProjector(llm_dim=base.llm_dim,proprio_dim=8).cuda().bfloat16()
    train=EpisodeDataset(a.train_manifest,a.statistics,a.model)
    optimizer=None; projector=None; records=[]; order=list(range(len(train))); random.Random(a.seed).shuffle(order)
    def load_target(batch):
        episode=batch['episode_ids'][0]; timestep=int(batch['timesteps'][0]); targets=[]
        for cam in ('image','wrist_image'):
            r=by_key.get((episode,timestep,cam))
            if r is None: raise KeyError(f'missing C-RADIO target: {episode}/{timestep}/{cam}')
            q=np.load(a.radio_root/r['output_file'],allow_pickle=False)
            if q.shape != (256,1024) or not np.isfinite(q).all(): raise ValueError('invalid teacher feature')
            targets.append(torch.from_numpy(q))
        return torch.cat(targets,0).unsqueeze(0).cuda().float()
    for step in range(a.steps):
        batch=train.collate([train[order[step%len(order)]]]); optimizer.zero_grad(set_to_none=True) if optimizer else None
        out=forward_l1(policy,head,proprio,batch,device='cuda',visual_token_count=patch_count)
        student=out['visual_and_proprio_features'][:,:patch_count]
        target=load_target(batch)
        if student.shape[1] != target.shape[1]: raise ValueError(f'patch count mismatch {student.shape} {target.shape}')
        if projector is None:
            projector=torch.nn.Linear(student.shape[-1],target.shape[-1],bias=False,dtype=torch.bfloat16,device='cuda')
            optimizer=torch.optim.AdamW([x for m in [policy,head,proprio,projector] for x in m.parameters() if x.requires_grad],lr=5e-4)
        retention=cosine_feature_retention(projector(student),target)
        loss=out['loss']+a.lambda_ret*retention
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite B1 loss')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        params=[x for m in [policy,head,proprio,projector] for x in m.parameters() if x.requires_grad]
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True); optimizer.step()
        teacher_grad=False; rec={'step':step+1,'action_loss':float(out['loss'].detach()),'retention_loss':float(retention.detach()),'total_loss':float(loss.detach()),'grad_norm':float(norm),'finite':True,'episode_id':batch['episode_ids'][0],'timestep':int(batch['timesteps'][0]),'student_shape':list(student.shape),'teacher_shape':list(target.shape)}; records.append(rec); print(json.dumps(rec),flush=True)
    # save a minimal independent artifact and reload projector for restore check
    a.output.mkdir(parents=True); torch.save({'projector':projector.state_dict(),'head':head.state_dict(),'proprio':proprio.state_dict()},a.output/'modules.pt')
    state=torch.load(a.output/'modules.pt',map_location='cpu'); projector.load_state_dict(state['projector']); head.load_state_dict(state['head']); proprio.load_state_dict(state['proprio'])
    restore_probe=float(next(projector.parameters()).detach().float().abs().mean())
    report={'status':'passed','experiment':'B1-real-openvla-cradio-retention-interface-smoke','steps':a.steps,'seed':a.seed,'lambda_retention':a.lambda_ret,'model_revision':source['revision'],'train_manifest_sha256':sha(a.train_manifest),'eval_manifest_sha256':sha(a.eval_manifest),'statistics_sha256':sha(a.statistics),'radio_manifest_sha256':sha(a.radio_manifest),'radio_teacher':'nvidia/C-RADIOv3-L','radio_feature_shape':[256,1024],'student_patch_shape':[2,256,int(student.shape[-1])],'projector_shape':[int(student.shape[-1]),1024],'teacher_frozen':True,'teacher_gradients_observed':False,'restore_probe_finite':bool(np.isfinite(restore_probe)),'records':records,'checkpoint':'modules.pt','purpose':'engineering interface smoke; not benchmark performance'}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report),flush=True)
if __name__=='__main__': main()
