"""V1 caption-SFT smoke with strict assistant-token loss masking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def last_subsequence(sequence: list[int], query: list[int]) -> int:
    for index in range(len(sequence) - len(query), -1, -1):
        if sequence[index : index + len(query)] == query:
            return index
    raise ValueError("caption token span is absent from teacher-forced input")


def main() -> int:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from vla_attention.data import jsonl

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lora-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    config = json.loads(args.lora_config.read_text())
    torch.manual_seed(args.seed)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, use_fast=config["processor_use_fast"])
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.bfloat16, local_files_only=True).cuda()
    peft = config["peft"]
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=peft["r"], lora_alpha=peft["lora_alpha"], lora_dropout=peft["lora_dropout"], bias=peft["bias"], target_modules=peft["target_modules_regex"]))
    visual_trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad and ".visual." in name]
    if visual_trainable:
        raise RuntimeError(f"visual encoder unexpectedly trainable: {visual_trainable[:3]}")
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=config["optimizer"]["learning_rate"], betas=tuple(config["optimizer"]["betas"]), eps=config["optimizer"]["eps"], weight_decay=config["optimizer"]["weight_decay"])
    records = list(jsonl(args.manifest))
    losses = []
    model.train()
    for step in range(args.max_steps):
        sample = records[step % len(records)]
        messages = [{"role": "user", "content": [{"type": "image", "image": str(args.dataset_root / sample["image_path"])}, {"type": "text", "text": sample["prompt"]}]}, {"role": "assistant", "content": [{"type": "text", "text": sample["caption"]}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        images, videos = process_vision_info(messages)
        batch = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt")
        batch = {key: value.cuda() if hasattr(value, "cuda") else value for key, value in batch.items()}
        caption_ids = processor.tokenizer(sample["caption"], add_special_tokens=False)["input_ids"]
        start = last_subsequence(batch["input_ids"][0].tolist(), caption_ids)
        labels = torch.full_like(batch["input_ids"], -100)
        labels[0, start : start + len(caption_ids)] = batch["input_ids"][0, start : start + len(caption_ids)]
        loss = model(**batch, labels=labels).loss
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite caption CE")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["optimizer"]["gradient_clip_norm"])
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"experiment": "V1-caption-smoke", "steps": args.max_steps, "losses": losses, "visual_trainable_tensors": len(visual_trainable), "processor_use_fast": config["processor_use_fast"]}, indent=2) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
