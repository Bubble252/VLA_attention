"""Real OpenVLA-OFT native interface gate, no training updates.

Run in an exclusive GPU window after cache smokes. No eager fallback.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    p=argparse.ArgumentParser()
    for name in ['model','source-lock','train-manifest','statistics','gpu-window','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    gate=json.loads(a.gpu_window.read_text())
    if gate.get('cache_audit_passed') is not True or gate.get('exclusive_b0_window') is not True:
        raise ValueError('Cache/resource gate not passed')
    if not 0<=time.time()-gate.get('checked_at_unix',0)<300:raise ValueError('Stale resource check')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU compute jobs present')
    from vla_attention.oft_preflight import verify_snapshot,sha
    source=verify_snapshot(a.model,a.source_lock)
    import numpy as np
    import torch
    from peft import get_peft_model,LoraConfig
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.projectors import ProprioProjector
    from vla_attention.benchmarks.oft_rlds import EpisodeDataset
    from vla_attention.adapters.oft_forward import forward_l1
    torch.manual_seed(17);torch.cuda.manual_seed_all(17)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config=OpenVLAConfig.from_pretrained(a.model,local_files_only=True)
    base=OpenVLAForActionPrediction.from_pretrained(a.model,config=config,local_files_only=True,
        low_cpu_mem_usage=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda()
    base.vision_backbone.set_num_images_in_input(2)
    patch=base.vision_backbone.featurizer.patch_embed
    other=base.vision_backbone.fused_featurizer.patch_embed
    grid=tuple(map(int,patch.grid_size))
    if tuple(other.grid_size)!=grid:raise ValueError('Fused towers have different grids')
    n=grid[0]*grid[1]
    if n!=base.vision_backbone.get_num_patches():raise ValueError('Patch grid mismatch')
    policy=get_peft_model(base,LoraConfig(r=32,lora_alpha=16,lora_dropout=0.,target_modules='all-linear',init_lora_weights='gaussian')).eval()
    head=L1RegressionActionHead(base.llm_dim,base.llm_dim).cuda().bfloat16().eval()
    proprio=ProprioProjector(base.llm_dim,8).cuda().bfloat16().eval()
    data=EpisodeDataset(a.train_manifest,a.statistics,a.model);batch=data.collate([data[0]])
    def forward():return forward_l1(policy,head,proprio,batch,device='cuda',visual_token_count=2*n)
    out=forward();features=out['visual_and_proprio_features']
    if features.shape[1]!=2*n+1:raise ValueError('Unexpected camera/proprio block')
    gradient=torch.autograd.grad(out['loss'],features)[0][:,:2*n]
    signed=(gradient.float()*features[:,:2*n].float()).sum(-1)
    maps=signed.abs().reshape(2,*grid).detach().cpu().numpy()
    grad_ok=bool(np.isfinite(maps).all()); nonzero=bool(np.any(maps>0))
    expected=out['prediction'].detach().float().cpu()
    with torch.no_grad():repeat=forward()['prediction'].float().cpu()
    repeat_error=float((expected-repeat).abs().max())
    stats=data.stats['action']; y=batch['actions'].cpu().numpy();lo=np.array(stats['q01']);hi=np.array(stats['q99'])
    mask=np.array(stats['mask']);constant=np.array(stats['min'])==np.array(stats['max'])
    native=np.where(mask, .5*(y+1)*(hi-lo+1e-8)+lo,y)
    from vla_attention.benchmarks.oft_rlds import normalize
    renorm=normalize(native,stats)
    norm_error=float(np.max(np.abs(renorm-y)))
    # Check gripper round trip on command values, not a claim of physical EEF units.
    gripper_env=1-2*y[:,:,-1];gripper_back=(1-gripper_env)/2
    gripper_error=float(np.max(np.abs(gripper_back-y[:,:,-1])))
    a.output.mkdir(parents=True)
    torch.save({'head':head.state_dict(),'proprio':proprio.state_dict()},a.output/'modules.pt')
    with torch.no_grad():
        next(head.parameters()).add_(1)
        next(proprio.parameters()).add_(1)
    state=torch.load(a.output/'modules.pt',map_location='cpu')
    head.load_state_dict(state['head']);proprio.load_state_dict(state['proprio'])
    with torch.no_grad():restored=forward()['prediction'].float().cpu()
    restore_error=float((restored-expected).abs().max())
    np.save(a.output/'action_attribution.npy',maps)
    provenance=[]
    for view,camera in enumerate(['image','wrist_image']):
        for i in range(n):
            row,col=divmod(i,grid[1])
            provenance.append({'feature_index':view*n+i,'language_sequence_index':1+view*n+i,
                               'camera':camera,'row':row,'column':col,
                               'normalized_box':[col/grid[1],row/grid[0],(col+1)/grid[1],(row+1)/grid[0]]})
    (a.output/'token_provenance.json').write_text(json.dumps(provenance)+'\n')
    passed=grad_ok and nonzero and repeat_error==0 and restore_error==0 and norm_error<1e-6 and gripper_error<1e-6
    report={'status':'passed' if passed else 'failed','model_revision':source['revision'],
      'train_manifest_sha256':sha(a.train_manifest),'statistics_sha256':sha(a.statistics),
      'action_shape':list(expected.shape),'chunk_horizon':8,'scalar':'native_mean_L1',
      'action_loss':float(out['loss'].detach()),'gradient_finite':grad_ok,'gradient_nonzero':nonzero,
      'repeatability':{'passed':repeat_error==0,'max_abs_error':repeat_error},
      'module_restore':{'passed':restore_error==0,'max_abs_error':restore_error,'scope':'head/proprio only; full new-process optimizer restore belongs to B0'},
      'normalization_roundtrip':{'passed':norm_error<1e-6 and gripper_error<1e-6,'max_abs_error':norm_error,'gripper_error':gripper_error,
                                 'scope':'clipped normalized data; physical controller scaling requires environment audit'},
      'visual_grid':{'per_camera':list(grid),'camera_order':['image','wrist_image'],'proprio_feature_index':2*n,'source':'runtime patch_embed.grid_size'},
      'episode_id':batch['episode_ids'][0],'timestep':batch['timesteps'][0],
      'files':{'token_provenance_sha256':sha(a.output/'token_provenance.json'),'attribution_sha256':sha(a.output/'action_attribution.npy')},
      'checkpoint_role':'base OpenVLA plus seed17 untrained OFT head/projector; engineering only'}
    (a.output/'p1_interface.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if not passed:raise RuntimeError('Real OFT P1 failed')


if __name__=='__main__':main()
