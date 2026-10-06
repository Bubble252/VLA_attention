"""OpenVLA teacher-map controls smoke (wrong-word, wrong-image, area-matched random).

Uses frozen SD semantic maps as spatial targets and the native action-loss
gradient×activation map as the student signal. Engineering gate only.
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
    for name in ['model','source-lock','train-manifest','eval-manifest','statistics','gpu-window','semantic-root','semantic-manifest','p1-report','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--steps',type=int,default=20); p.add_argument('--seed',type=int,default=17)
    p.add_argument('--lambda-sem',type=float,default=0.10); p.add_argument('--temperature',type=float,default=1.0); p.add_argument('--control',choices=['correct','wrong_word','wrong_image','random','role_swapped_same_image','area_matched_random'],default='correct')
    a=p.parse_args()
    if not 20<=a.steps<=30: raise ValueError('B2 smoke must use 20-30 steps')
    if a.output.exists(): raise FileExistsError(a.output)
    if 'control' not in a.output.name.lower(): raise ValueError('controls output directory required')
    from vla_attention.oft_preflight import verify_snapshot,require_p1,require_gpu_window
    require_gpu_window(a.gpu_window); source=verify_snapshot(a.model,a.source_lock); require_gpu_window(a.gpu_window)
    require_p1(a.p1_report,source['revision'],sha(a.train_manifest),sha(a.statistics))
    sem_rows=[json.loads(x) for x in a.semantic_manifest.read_text().splitlines() if x.strip()]
    by_key={(r['episode_id'],int(r['timestep']),r['camera'],r['role']):r for r in sem_rows}
    if len(by_key)!=len(sem_rows): raise ValueError('duplicate semantic cache keys')
    for r in sem_rows[:2]:
        side=a.semantic_root / (r['sample_id'] + '.json')
        meta=json.loads(side.read_text())
        if meta.get('attention_resolution')!=16 or meta.get('seed')!=17 or meta.get('teacher_frozen') is not True: raise ValueError('unexpected semantic cache contract')
    train_ids={json.loads(x)['episode_id'] for x in a.train_manifest.read_text().splitlines() if x.strip()}; eval_ids={json.loads(x)['episode_id'] for x in a.eval_manifest.read_text().splitlines() if x.strip()}
    if train_ids & eval_ids: raise ValueError('train/eval overlap')
    import numpy as np, torch
    from peft import LoraConfig,get_peft_model
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.projectors import ProprioProjector
    from vla_attention.benchmarks.oft_rlds import EpisodeDataset
    from vla_attention.adapters.oft_forward import forward_l1
    from vla_attention.losses import semantic_map_kl
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    config=OpenVLAConfig.from_pretrained(a.model,local_files_only=True)
    base=OpenVLAForActionPrediction.from_pretrained(a.model,config=config,local_files_only=True,low_cpu_mem_usage=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda(); base.vision_backbone.set_num_images_in_input(2); patch_count=base.vision_backbone.get_num_patches()*2
    policy=get_peft_model(base,LoraConfig(r=32,lora_alpha=16,lora_dropout=0.,target_modules='all-linear',init_lora_weights='gaussian'))
    head=L1RegressionActionHead(input_dim=base.llm_dim,hidden_dim=base.llm_dim).cuda().bfloat16(); proprio=ProprioProjector(llm_dim=base.llm_dim,proprio_dim=8).cuda().bfloat16()
    train=EpisodeDataset(a.train_manifest,a.statistics,a.model); order=list(range(len(train))); random.Random(a.seed).shuffle(order); optimizer=None; records=[]
    def load_maps(batch):
        ep=batch['episode_ids'][0]; ts=int(batch['timesteps'][0]); maps=[]
        for cam in ('image','wrist_image'):
            vals=[]
            for role in ('source','target'):
                r=by_key.get((ep,ts,cam,role))
                if r is None: raise KeyError(f'missing semantic map {ep}/{ts}/{cam}/{role}')
                q=np.load(a.semantic_root/(r['sample_id']+'.npy'),allow_pickle=False).astype('float32')
                if q.shape!=(16,16) or not np.isfinite(q).all() or (q<0).any(): raise ValueError('invalid semantic map')
                vals.append(q.reshape(-1))
            maps.append(np.mean(vals,axis=0))
        out=np.concatenate(maps)
        if a.control=='random':
            rng=np.random.default_rng(17 + int(batch['timesteps'][0])); out=rng.random(out.shape).astype('float32')
        elif a.control=='area_matched_random':
            rng=np.random.default_rng(17000 + int(batch['timesteps'][0])); matched=[]
            for target in maps:
                k=max(1, int(np.count_nonzero(target >= np.quantile(target, 0.8))))
                vals=np.zeros(256, dtype='float32'); idx=rng.choice(256, size=k, replace=False)
                vals[idx]=rng.random(k).astype('float32'); matched.append(vals)
            out=np.concatenate(matched)
        elif a.control=='role_swapped_same_image':
            swapped=[]
            for cam in ('image','wrist_image'):
                r=by_key.get((ep,ts,cam,'target'))
                if r is None: raise KeyError(f'missing same-image target map {ep}/{ts}/{cam}')
                q=np.load(a.semantic_root/(r['sample_id']+'.npy'),allow_pickle=False).astype('float32')
                if q.shape!=(16,16) or not np.isfinite(q).all() or (q<0).any(): raise ValueError('invalid same-image swapped map')
                swapped.append(q.reshape(-1))
            out=np.concatenate(swapped)
        elif a.control in ('wrong_word','wrong_image'):
            candidates=[x for x in sem_rows if x['camera']=='image' and x['role']=='source' and x['episode_id']!=ep]
            if not candidates: raise RuntimeError('no negative control candidate')
            peer_ep=candidates[0]['episode_id']; other=[]
            for cam in ('image','wrist_image'):
                r=next((x for x in sem_rows if x['episode_id']==peer_ep and int(x['timestep'])==int(candidates[0]['timestep']) and x['camera']==cam and x['role']=='source'),None)
                if r is None: raise RuntimeError(f'negative control missing peer camera: {peer_ep}/{cam}')
                q=np.load(a.semantic_root/(r['sample_id']+'.npy'),allow_pickle=False).astype('float32')
                if q.shape!=(16,16) or not np.isfinite(q).all(): raise ValueError('invalid negative map')
                other.append(q.reshape(-1))
            out=np.concatenate(other)
        return torch.from_numpy(out).cuda().reshape(1,-1)
    for step in range(a.steps):
        batch=train.collate([train[order[step%len(order)]]]); captured=[]
        def capture(_m,_i,o):
            if isinstance(o, tuple): o=o[0]
            if torch.is_tensor(o):
                o.retain_grad(); captured[:] = [o]
            return o
        hook=policy.base_model.model.vision_backbone.register_forward_hook(capture)
        out=forward_l1(policy,head,proprio,batch,device='cuda',visual_token_count=patch_count)
        out['loss'].backward(retain_graph=True)
        hook.remove()
        if not captured or captured[0].grad is None: raise RuntimeError('visual backbone gradient unavailable')
        features=captured[0][:,:patch_count]; grad=captured[0].grad[:,:patch_count]
        student=(grad.float()*features.float()).sum(-1).abs().reshape(1,2,256)
        teacher=load_maps(batch).reshape(1,512); loss_sem=semantic_map_kl(student.reshape(1,512),teacher,teacher_temperature=a.temperature); loss=out['loss']+a.lambda_sem*loss_sem
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite B2 loss')
        params=[x for m in [policy,head,proprio] for x in m.parameters() if x.requires_grad]; optimizer=torch.optim.AdamW(params,lr=5e-4) if optimizer is None else optimizer; optimizer.zero_grad(set_to_none=True); loss.backward(); norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True); optimizer.step()
        records.append({'step':step+1,'action_loss':float(out['loss'].detach()),'semantic_loss':float(loss_sem.detach()),'total_loss':float(loss.detach()),'grad_norm':float(norm),'student_map_shape':list(student.shape),'teacher_map_shape':list(teacher.shape),'finite':True,'episode_id':batch['episode_ids'][0],'timestep':int(batch['timesteps'][0])}); print(json.dumps(records[-1]),flush=True)
    a.output.mkdir(parents=True); report={'status':'passed','experiment':'VLA-teacher-map-controls-smoke','steps':a.steps,'seed':a.seed,'control':a.control,'lambda_semantic':a.lambda_sem,'temperature':a.temperature,'model_revision':source['revision'],'train_manifest_sha256':sha(a.train_manifest),'eval_manifest_sha256':sha(a.eval_manifest),'statistics_sha256':sha(a.statistics),'semantic_manifest_sha256':sha(a.semantic_manifest),'teacher':'SD1.5 null-text reconstruction v3','teacher_frozen':True,'student_signal':'action-loss gradient times activation','student_map_shape':[2,256],'teacher_map_shape':[2,256],'records':records,'purpose':'engineering interface smoke; not benchmark performance'}; (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report),flush=True)
if __name__=='__main__': main()
