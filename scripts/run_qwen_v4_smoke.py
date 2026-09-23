"""V4 smoke: V2 DINO retention plus V3 semantic attribution."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from scripts.run_qwen_caption_smoke import last_subsequence

def phrase_positions(full_ids, caption, phrase, tokenizer):
    caption_ids = tokenizer(caption, add_special_tokens=False)['input_ids']
    caption_start = last_subsequence(full_ids, caption_ids)
    char_start = caption.lower().find(phrase.lower())
    if char_start < 0:
        raise ValueError('phrase absent from caption')
    prefix_ids = tokenizer(caption[:char_start], add_special_tokens=False)['input_ids']
    phrase_ids = tokenizer(caption[char_start:char_start + len(phrase)], add_special_tokens=False)['input_ids']
    return caption_start + len(prefix_ids), phrase_ids

def main():
 import numpy as np, torch
 from PIL import Image
 from peft import LoraConfig,TaskType,get_peft_model
 from qwen_vl_utils import process_vision_info
 from transformers import AutoImageProcessor,AutoModel,AutoProcessor,Qwen2_5_VLForConditionalGeneration
 from vla_attention.data import jsonl
 from vla_attention.losses import semantic_map_loss,resample_spatial_distribution
 from vla_attention.teachers.retention import cosine_retention_loss,resample_teacher_features
 p=argparse.ArgumentParser(); p.add_argument('--model',type=Path,required=True); p.add_argument('--dino',type=Path,required=True); p.add_argument('--dataset-root',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--cache',type=Path,required=True); p.add_argument('--lora-config',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--checkpoint',type=Path,default=None); p.add_argument('--max-steps',type=int,default=20); p.add_argument('--lambda-ret',type=float,default=.1); p.add_argument('--lambda-sem',type=float,default=.1); p.add_argument('--teacher-temperature',type=float,default=1.0); p.add_argument('--semantic-warmup-steps',type=int,default=0); p.add_argument('--semantic-loss-mode',choices=['kl','js','kl_rank','kl_moment'],default='kl'); p.add_argument('--common-resolution',type=int,default=32); p.add_argument('--rank-quantile',type=float,default=.25); p.add_argument('--rank-margin',type=float,default=.5); p.add_argument('--rank-weight',type=float,default=1.0); p.add_argument('--moment-weight',type=float,default=1.0); p.add_argument('--seed',type=int,default=17); a=p.parse_args()
 cfg=json.loads(a.lora_config.read_text()); torch.manual_seed(a.seed); proc=AutoProcessor.from_pretrained(a.model,local_files_only=True,use_fast=False); dp=AutoImageProcessor.from_pretrained(a.dino,local_files_only=True,use_fast=False); dino=AutoModel.from_pretrained(a.dino,torch_dtype=torch.bfloat16,local_files_only=True).eval().cuda(); [x.requires_grad_(False) for x in dino.parameters()]
 base=Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model,torch_dtype=torch.bfloat16,local_files_only=True,attn_implementation='eager').cuda(); pc=cfg['peft']; model=get_peft_model(base,LoraConfig(task_type=TaskType.CAUSAL_LM,r=pc['r'],lora_alpha=pc['lora_alpha'],lora_dropout=pc['lora_dropout'],bias=pc['bias'],target_modules=pc['target_modules_regex'])); [x.requires_grad_(True) for x in model.base_model.model.model.visual.parameters()]
 captured=[]
 def hook(_m,_i,o): captured[:]=[o]; return o
 h=model.base_model.model.model.visual.merger.register_forward_hook(hook); projector=None; opt=None; rows=list(jsonl(a.manifest)); logs=[]
 try:
  for step in range(a.max_steps):
   row=rows[step%len(rows)]; key=row['sample_id'].replace(':','_'); teacher_np=np.load(a.cache/(key+'.npy')); msgs=[{'role':'user','content':[{'type':'image','image':str(a.dataset_root/row['image_path'])},{'type':'text','text':row['prompt']}]},{'role':'assistant','content':[{'type':'text','text':row['caption']}]}]; text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=False); ims,vid=process_vision_info(msgs); batch=proc(text=[text],images=ims,videos=vid,padding=True,return_tensors='pt'); batch={k:v.cuda() if hasattr(v,'cuda') else v for k,v in batch.items()}; ids=proc.tokenizer(row['caption'],add_special_tokens=False)['input_ids']; st=last_subsequence(batch['input_ids'][0].tolist(),ids); labels=torch.full_like(batch['input_ids'],-100); labels[0,st:st+len(ids)]=batch['input_ids'][0,st:st+len(ids)]
   q=model.base_model.model.model; emb=q.get_input_embeddings()(batch['input_ids']); ie=q.get_image_features(batch['pixel_values'],batch['image_grid_thw']); ie=torch.cat(ie,dim=0).to(emb.device,emb.dtype); mask,_=q.get_placeholder_mask(batch['input_ids'],inputs_embeds=emb,image_features=ie); emb=emb.masked_scatter(mask,ie).detach().requires_grad_(True); out=model(input_ids=batch['input_ids'],inputs_embeds=emb,attention_mask=batch.get('attention_mask'),image_grid_thw=batch.get('image_grid_thw'),labels=labels); feat=emb[mask].reshape(1,-1,emb.shape[-1]); ps,ph=phrase_positions(batch['input_ids'][0].tolist(),row['caption'],row['phrase'],proc.tokenizer); score=out.logits[0,ps-1:ps-1+len(ph)].float().log_softmax(-1).gather(1,torch.tensor(ph,device='cuda').unsqueeze(1)).sum(); grad=torch.autograd.grad(score,emb,create_graph=True,retain_graph=True)[0][mask].reshape(1,-1,emb.shape[-1]); student=(grad*feat).sum(-1).abs(); gh,gw=(batch['image_grid_thw'][0][1:].tolist()); merge=base.config.vision_config.spatial_merge_size; teacher=torch.from_numpy(teacher_np).to('cuda',dtype=student.dtype).reshape(1,-1,1); teacher=resample_teacher_features(teacher,source_height=16,source_width=16,target_height=gh//merge,target_width=gw//merge).squeeze(-1)
   pixels={k:v.cuda() for k,v in dp(images=Image.open(a.dataset_root/row['image_path']).convert('RGB'),return_tensors='pt').items()}; tf=dino(**pixels).last_hidden_state[:,1:]; teacher_feat=resample_teacher_features(tf,source_height=16,source_width=16,target_height=gh//merge,target_width=gw//merge)
   if projector is None: projector=torch.nn.Linear(teacher_feat.shape[-1],feat.shape[-1],bias=False,dtype=torch.bfloat16,device='cuda'); opt=torch.optim.AdamW([x for n,x in model.named_parameters() if x.requires_grad and '.visual.' not in n]+list(projector.parameters()),lr=cfg['optimizer']['learning_rate'])
   common=a.common_resolution; student=resample_spatial_distribution(student,source_height=gh//merge,source_width=gw//merge,target_height=common,target_width=common); teacher=resample_spatial_distribution(teacher,source_height=gh//merge,source_width=gw//merge,target_height=common,target_width=common); ret=cosine_retention_loss(feat,projector(teacher_feat)); sem=semantic_map_loss(student,teacher,mode=a.semantic_loss_mode,teacher_temperature=a.teacher_temperature,rank_quantile=a.rank_quantile,rank_margin=a.rank_margin,rank_weight=a.rank_weight,moment_weight=a.moment_weight); warmup=min(1.0,(step+1)/a.semantic_warmup_steps) if a.semantic_warmup_steps else 1.0; effective_lambda=a.lambda_sem*warmup; loss=out.loss+a.lambda_ret*ret+effective_lambda*sem; opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(projector.parameters()),cfg['optimizer']['gradient_clip_norm']); opt.step(); logs.append({'total':float(loss.detach()),'caption':float(out.loss.detach()),'retention':float(ret.detach()),'semantic':float(sem.detach()),'effective_lambda_sem':effective_lambda})
 finally: h.remove()
 if a.checkpoint is not None:
  a.checkpoint.mkdir(parents=True,exist_ok=True); model.save_pretrained(a.checkpoint); torch.save(projector.state_dict(),a.checkpoint/'dino_projector.pt')
 vis=[n for n,x in model.named_parameters() if x.requires_grad and '.visual.' in n and 'lora_' in n]; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps({'experiment':'V4-caption-plus-retention-plus-semantic-smoke','steps':a.max_steps,'lambda_ret':a.lambda_ret,'lambda_sem':a.lambda_sem,'teacher_temperature':a.teacher_temperature,'semantic_warmup_steps':a.semantic_warmup_steps,'semantic_loss_mode':a.semantic_loss_mode,'common_resolution':[a.common_resolution,a.common_resolution],'rank_quantile':a.rank_quantile,'rank_margin':a.rank_margin,'rank_weight':a.rank_weight,'moment_weight':a.moment_weight,'losses':logs,'visual_lora_tensors':len(vis),'dino_frozen':True},indent=2)+'\n'); print(a.output)
if __name__=='__main__': main()
