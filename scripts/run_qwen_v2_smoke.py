"""V2 smoke: caption LoRA plus frozen DINO retention."""
from __future__ import annotations

import argparse, json
from pathlib import Path

from scripts.run_qwen_caption_smoke import last_subsequence


def main() -> int:
    import torch
    from PIL import Image
    from peft import LoraConfig, TaskType, get_peft_model
    from qwen_vl_utils import process_vision_info
    from transformers import AutoImageProcessor, AutoModel, AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from vla_attention.data import jsonl
    from vla_attention.teachers.retention import cosine_retention_loss, resample_teacher_features

    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True); parser.add_argument('--dino',type=Path,required=True)
    parser.add_argument('--dataset-root',type=Path,required=True); parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--lora-config',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-steps',type=int,default=20); parser.add_argument('--lambda-ret',type=float,default=0.1); parser.add_argument('--seed',type=int,default=17)
    args=parser.parse_args(); cfg=json.loads(args.lora_config.read_text()); torch.manual_seed(args.seed)
    processor=AutoProcessor.from_pretrained(args.model,local_files_only=True,use_fast=False)
    dino_processor=AutoImageProcessor.from_pretrained(args.dino,local_files_only=True,use_fast=False)
    dino=AutoModel.from_pretrained(args.dino,torch_dtype=torch.bfloat16,local_files_only=True).eval().cuda()
    for p in dino.parameters(): p.requires_grad_(False)
    base=Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model,torch_dtype=torch.bfloat16,local_files_only=True).cuda()
    peft=cfg['peft']; model=get_peft_model(base,LoraConfig(task_type=TaskType.CAUSAL_LM,r=peft['r'],lora_alpha=peft['lora_alpha'],lora_dropout=peft['lora_dropout'],bias=peft['bias'],target_modules=peft['target_modules_regex']))
    captured=[]
    def hook(_m,_i,o): captured[:] = [o]
    handle=model.base_model.model.model.visual.merger.register_forward_hook(hook)
    projector=None; records=list(jsonl(args.manifest)); losses=[]
    try:
      for step in range(args.max_steps):
        sample=records[step%len(records)]; image_path=args.dataset_root/sample['image_path']
        messages=[{'role':'user','content':[{'type':'image','image':str(image_path)},{'type':'text','text':sample['prompt']}]},{'role':'assistant','content':[{'type':'text','text':sample['caption']}]}]
        text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=False); images,videos=process_vision_info(messages)
        batch=processor(text=[text],images=images,videos=videos,padding=True,return_tensors='pt'); batch={k:v.cuda() if hasattr(v,'cuda') else v for k,v in batch.items()}
        caption_ids=processor.tokenizer(sample['caption'],add_special_tokens=False)['input_ids']; start=last_subsequence(batch['input_ids'][0].tolist(),caption_ids)
        labels=torch.full_like(batch['input_ids'],-100); labels[0,start:start+len(caption_ids)]=batch['input_ids'][0,start:start+len(caption_ids)]
        captured.clear(); out=model(**batch,labels=labels); qwen=captured[0].reshape(1,-1,captured[0].shape[-1])
        with torch.no_grad():
          pixels={k:v.cuda() for k,v in dino_processor(images=Image.open(image_path).convert('RGB'),return_tensors='pt').items()}; dino_features=dino(**pixels).last_hidden_state[:,1:]
        grid_t,grid_h,grid_w=batch['image_grid_thw'][0].tolist(); merge=base.config.vision_config.spatial_merge_size
        teacher=resample_teacher_features(dino_features,source_height=16,source_width=16,target_height=grid_h//merge,target_width=grid_w//merge)
        if projector is None:
          projector=torch.nn.Linear(teacher.shape[-1],qwen.shape[-1],bias=False,dtype=torch.bfloat16,device='cuda'); optimizer=torch.optim.AdamW(list(p for p in model.parameters() if p.requires_grad)+list(projector.parameters()),lr=cfg['optimizer']['learning_rate'],betas=tuple(cfg['optimizer']['betas']),eps=cfg['optimizer']['eps'])
        retention=cosine_retention_loss(qwen,projector(teacher)); loss=out.loss+args.lambda_ret*retention
        if not torch.isfinite(loss): raise RuntimeError('non-finite V2 loss')
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(projector.parameters()),cfg['optimizer']['gradient_clip_norm']); optimizer.step()
        losses.append({'total':float(loss.detach()),'caption':float(out.loss.detach()),'retention':float(retention.detach())})
    finally: handle.remove()
    visual=[n for n,p in model.named_parameters() if p.requires_grad and '.visual.' in n]
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps({'experiment':'V2-caption-plus-dino-smoke','steps':args.max_steps,'lambda_ret':args.lambda_ret,'losses':losses,'visual_trainable_tensors':len(visual),'dino_frozen':True},indent=2)+'\n'); print(args.output)

if __name__=='__main__': main()
