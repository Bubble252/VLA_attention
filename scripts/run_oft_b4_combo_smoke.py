"""OpenVLA B4 combined retention + semantic + containment smoke."""
import argparse,hashlib,json,os,random
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
 return h.hexdigest()

def main():
 os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
 p=argparse.ArgumentParser()
 for n in ['model','source-lock','train-manifest','eval-manifest','statistics','gpu-window','semantic-root','semantic-manifest','radio-root','radio-manifest','p1-report','output']: p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--steps',type=int,default=20); p.add_argument('--seed',type=int,default=17); p.add_argument('--lambda-sem',type=float,default=.1); p.add_argument('--lambda-contain',type=float,default=.02)
 a=p.parse_args();
 if not 20<=a.steps<=30: raise ValueError('B3 smoke must use 20-30 steps')
 if a.output.exists(): raise FileExistsError(a.output)
 if 'B4' not in a.output.name and 'b4' not in a.output.name: raise ValueError('B4 output directory required')
 from vla_attention.oft_preflight import verify_snapshot,require_p1,require_gpu_window
 require_gpu_window(a.gpu_window); source=verify_snapshot(a.model,a.source_lock); require_gpu_window(a.gpu_window); require_p1(a.p1_report,source['revision'],sha(a.train_manifest),sha(a.statistics))
 rows=[json.loads(x) for x in a.semantic_manifest.read_text().splitlines() if x.strip()]; by={(r['episode_id'],int(r['timestep']),r['camera'],r['role']):r for r in rows}
 if len(by)!=len(rows): raise ValueError('duplicate semantic keys')
 radio_rows=[json.loads(x) for x in a.radio_manifest.read_text().splitlines() if x.strip()]; radio_by={(r['episode_id'],int(r['timestep']),r['camera']):r for r in radio_rows}
 if len(radio_by)!=len(radio_rows): raise ValueError('duplicate radio keys')
 import numpy as np,torch
 from peft import LoraConfig,get_peft_model
 from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
 from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
 from prismatic.models.action_heads import L1RegressionActionHead
 from prismatic.models.projectors import ProprioProjector
 from vla_attention.benchmarks.oft_rlds import EpisodeDataset
 from vla_attention.adapters.oft_forward import forward_l1
 from vla_attention.losses import semantic_map_kl,action_evidence_containment,cosine_feature_retention
 torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
 config=OpenVLAConfig.from_pretrained(a.model,local_files_only=True); base=OpenVLAForActionPrediction.from_pretrained(a.model,config=config,local_files_only=True,low_cpu_mem_usage=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda(); base.vision_backbone.set_num_images_in_input(2); n=base.vision_backbone.get_num_patches(); patches=2*n
 policy=get_peft_model(base,LoraConfig(r=32,lora_alpha=16,lora_dropout=0.,target_modules='all-linear',init_lora_weights='gaussian')); head=L1RegressionActionHead(input_dim=base.llm_dim,hidden_dim=base.llm_dim).cuda().bfloat16(); proprio=ProprioProjector(llm_dim=base.llm_dim,proprio_dim=8).cuda().bfloat16(); train=EpisodeDataset(a.train_manifest,a.statistics,a.model); order=list(range(len(train))); random.Random(a.seed).shuffle(order); opt=None; projector=None; recs=[]
 def radio(batch):
  ep=batch['episode_ids'][0]; ts=int(batch['timesteps'][0]); vals=[]
  for cam in ('image','wrist_image'):
   r=radio_by.get((ep,ts,cam));
   if r is None: raise KeyError(f'missing radio target {ep}/{ts}/{cam}')
   q=np.load(a.radio_root/r['output_file'],allow_pickle=False)
   if q.shape!=(256,1024) or not np.isfinite(q).all(): raise ValueError('invalid radio feature')
   vals.append(torch.from_numpy(q))
  return torch.cat(vals).unsqueeze(0).cuda().float()
 def maps(batch):
  ep=batch['episode_ids'][0]; ts=int(batch['timesteps'][0]); out=[]
  for cam in ('image','wrist_image'):
   vals=[]
   for role in ('source','target'):
    r=by.get((ep,ts,cam,role));
    if r is None: raise KeyError(f'missing semantic map {ep}/{ts}/{cam}/{role}')
    q=np.load(a.semantic_root/(r['sample_id']+'.npy'),allow_pickle=False)
    if q.shape!=(16,16) or not np.isfinite(q).all() or (q<0).any(): raise ValueError('invalid semantic map')
    vals.append(torch.from_numpy(q).float().reshape(-1))
   vals=torch.stack(vals); out.append(vals.mean(0))
  return torch.cat(out).cuda()
 for step in range(a.steps):
  batch=train.collate([train[order[step%len(order)]]]); captured=[]
  def hookfn(_m,_i,o):
   if isinstance(o,tuple): o=o[0]
   if torch.is_tensor(o): o.retain_grad(); captured[:]=[o]
   return o
  h=policy.base_model.model.vision_backbone.register_forward_hook(hookfn); out=forward_l1(policy,head,proprio,batch,device='cuda',visual_token_count=patches); out['loss'].backward(retain_graph=True); h.remove()
  if not captured or captured[0].grad is None: raise RuntimeError('A_act gradient unavailable')
  feat=captured[0][:,:patches]; grad=captured[0].grad[:,:patches]; act=(grad.float()*feat.float()).sum(-1).abs().reshape(1,patches); teacher_feat=radio(batch);
  if projector is None: projector=torch.nn.Linear(feat.shape[-1],teacher_feat.shape[-1],bias=False,dtype=torch.bfloat16,device='cuda')
  retention=cosine_feature_retention(projector(feat),teacher_feat)
  support=maps(batch).reshape(1,patches); support=(support/support.max().clamp_min(1e-8)).clamp(0,1)
  sem=semantic_map_kl(act,support); contain=action_evidence_containment(act/(act.sum(-1,keepdim=True)+1e-8),support); loss=out['loss']+0.05*retention+a.lambda_sem*sem+a.lambda_contain*contain
  if not torch.isfinite(loss): raise FloatingPointError('nonfinite B3 loss')
  params=[x for m in [policy,head,proprio,projector] for x in m.parameters() if x.requires_grad]; opt=torch.optim.AdamW(params,lr=5e-4) if opt is None else opt; opt.zero_grad(set_to_none=True); loss.backward(); norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True); opt.step()
  rec={'step':step+1,'action_loss':float(out['loss'].detach()),'semantic_loss':float(sem.detach()),'containment_loss':float(contain.detach()),'retention_loss':float(retention.detach()),'total_loss':float(loss.detach()),'grad_norm':float(norm),'a_act_shape':[1,patches],'support_shape':[1,patches],'finite':True,'episode_id':batch['episode_ids'][0],'timestep':int(batch['timesteps'][0])}; recs.append(rec); print(json.dumps(rec),flush=True)
 a.output.mkdir(parents=True); report={'status':'passed','experiment':'B4-real-openvla-combined-retention-semantic-containment-smoke','steps':a.steps,'seed':a.seed,'lambda_semantic':a.lambda_sem,'lambda_containment':a.lambda_contain,'model_revision':source['revision'],'train_manifest_sha256':sha(a.train_manifest),'eval_manifest_sha256':sha(a.eval_manifest),'statistics_sha256':sha(a.statistics),'semantic_manifest_sha256':sha(a.semantic_manifest),'student_signal':'action-loss gradient times activation','retention_teacher':'nvidia/C-RADIOv3-L','retention_projector_shape':[int(feat.shape[-1]),1024],'support':'source/target semantic map union proxy','teacher_frozen':True,'records':recs,'purpose':'engineering interface smoke; not benchmark performance'}; (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report),flush=True)
if __name__=='__main__': main()
