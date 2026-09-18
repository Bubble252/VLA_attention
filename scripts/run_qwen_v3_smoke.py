"""V3 smoke: caption LoRA plus phrase-score gradient attribution alignment."""
from __future__ import annotations

import argparse, json
from pathlib import Path
from scripts.run_qwen_caption_smoke import last_subsequence


def phrase_positions_in_caption(full_ids, caption, phrase, tokenizer):
    caption_ids = tokenizer(caption, add_special_tokens=False)['input_ids']
    caption_start = last_subsequence(full_ids, caption_ids)
    char_start = caption.lower().find(phrase.lower())
    if char_start < 0:
        raise ValueError('phrase text is absent from caption')
    prefix_ids = tokenizer(caption[:char_start], add_special_tokens=False)['input_ids']
    phrase_ids = tokenizer(caption[char_start:char_start + len(phrase)], add_special_tokens=False)['input_ids']
    return caption_start + len(prefix_ids), phrase_ids


def main() -> int:
    import numpy as np
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from vla_attention.data import jsonl
    from vla_attention.losses import semantic_map_kl
    from vla_attention.teachers.retention import resample_teacher_features
    p=argparse.ArgumentParser(); p.add_argument('--model',type=Path,required=True); p.add_argument('--dataset-root',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--cache',type=Path,required=True); p.add_argument('--lora-config',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--max-steps',type=int,default=20); p.add_argument('--lambda-sem',type=float,default=0.1); p.add_argument('--seed',type=int,default=17); a=p.parse_args()
    cfg=json.loads(a.lora_config.read_text()); torch.manual_seed(a.seed); processor=AutoProcessor.from_pretrained(a.model,local_files_only=True,use_fast=False)
    base=Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model,torch_dtype=torch.bfloat16,local_files_only=True,attn_implementation='eager').cuda(); pc=cfg['peft']; model=get_peft_model(base,LoraConfig(task_type=TaskType.CAUSAL_LM,r=pc['r'],lora_alpha=pc['lora_alpha'],lora_dropout=pc['lora_dropout'],bias=pc['bias'],target_modules=pc['target_modules_regex']))
    captured=[]
    for parameter in model.base_model.model.model.visual.parameters():
        parameter.requires_grad_(True)  # gradients for attribution only; excluded from optimizer
    def hook(_m,_i,o):
        captured[:] = [o]
        return o
    h=model.base_model.model.model.visual.merger.register_forward_hook(hook); opt=torch.optim.AdamW((x for x in model.parameters() if x.requires_grad),lr=cfg['optimizer']['learning_rate'],betas=tuple(cfg['optimizer']['betas']),eps=cfg['optimizer']['eps']); rows=list(jsonl(a.manifest)); logs=[]
    try:
      for step in range(a.max_steps):
        row=rows[step%len(rows)]; key=row['sample_id'].replace(':','_'); teacher_np=np.load(a.cache/(key+'.npy'))
        messages=[{'role':'user','content':[{'type':'image','image':str(a.dataset_root/row['image_path'])},{'type':'text','text':row['prompt']}]},{'role':'assistant','content':[{'type':'text','text':row['caption']}]}]
        text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=False); images,videos=process_vision_info(messages); batch=processor(text=[text],images=images,videos=videos,padding=True,return_tensors='pt'); batch={k:v.cuda() if hasattr(v,'cuda') else v for k,v in batch.items()}
        caption_ids=processor.tokenizer(row['caption'],add_special_tokens=False)['input_ids']; start=last_subsequence(batch['input_ids'][0].tolist(),caption_ids); labels=torch.full_like(batch['input_ids'],-100); labels[0,start:start+len(caption_ids)]=batch['input_ids'][0,start:start+len(caption_ids)]
        captured.clear()
        qmodel=model.base_model.model.model
        inputs_embeds=qmodel.get_input_embeddings()(batch['input_ids'])
        image_embeds=qmodel.get_image_features(batch['pixel_values'],batch['image_grid_thw'])
        image_embeds=torch.cat(image_embeds,dim=0).to(inputs_embeds.device,inputs_embeds.dtype)
        image_mask,_=qmodel.get_placeholder_mask(batch['input_ids'],inputs_embeds=inputs_embeds,image_features=image_embeds)
        inputs_embeds=inputs_embeds.masked_scatter(image_mask,image_embeds).detach().requires_grad_(True)
        out=model(input_ids=batch['input_ids'],inputs_embeds=inputs_embeds,attention_mask=batch.get('attention_mask'),image_grid_thw=batch.get('image_grid_thw'),labels=labels)
        features=inputs_embeds[image_mask].reshape(1,-1,inputs_embeds.shape[-1])
        phrase_start,phrase_ids=phrase_positions_in_caption(batch['input_ids'][0].tolist(),row['caption'],row['phrase'],processor.tokenizer); logp=out.logits[0,phrase_start-1:phrase_start-1+len(phrase_ids)].float().log_softmax(-1); phrase_score=logp.gather(1,torch.tensor(phrase_ids,device='cuda').unsqueeze(1)).sum()
        gradient_full=torch.autograd.grad(phrase_score,inputs_embeds,create_graph=True,retain_graph=True)[0]
        gradient=gradient_full[image_mask].reshape(1,-1,inputs_embeds.shape[-1])
        student=(gradient*features).sum(-1).abs()
        grid_t,gh,gw=batch['image_grid_thw'][0].tolist(); merge=base.config.vision_config.spatial_merge_size; teacher=torch.from_numpy(teacher_np).to('cuda',dtype=student.dtype).reshape(1,-1,1); teacher=resample_teacher_features(teacher,source_height=16,source_width=16,target_height=gh//merge,target_width=gw//merge).squeeze(-1)
        sem=semantic_map_kl(student,teacher); loss=out.loss+a.lambda_sem*sem
        if not torch.isfinite(loss): raise RuntimeError('non-finite V3 loss')
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['optimizer']['gradient_clip_norm']); opt.step(); logs.append({'total':float(loss.detach()),'caption':float(out.loss.detach()),'semantic':float(sem.detach())})
    finally: h.remove()
    vis=[n for n,x in model.named_parameters() if x.requires_grad and '.visual.' in n and 'lora_' in n]; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps({'experiment':'V3-caption-plus-semantic-smoke','steps':a.max_steps,'lambda_sem':a.lambda_sem,'losses':logs,'visual_lora_tensors':len(vis),'visual_attribution_gradients_enabled':True,'teacher_cache':str(a.cache)},indent=2)+'\n'); print(a.output)
if __name__=='__main__': main()
