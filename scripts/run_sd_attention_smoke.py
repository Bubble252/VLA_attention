"""Image-latent cross-attention hook smoke. Not a formal inversion calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    import numpy as np
    import torch
    from diffusers import StableDiffusionPipeline
    from PIL import Image
    from vla_attention.teachers.sd_attention import install_cross_attention_capture

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16, local_files_only=True, safety_checker=None, requires_safety_checker=False).to("cuda")
    store = install_cross_attention_capture(pipe.unet, conditional_cfg_half=True)
    image = Image.open(args.image).convert("RGB").resize((512, 512))
    pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0).float().div(127.5).sub(1).to("cuda", dtype=torch.float16)
    with torch.no_grad():
        latent = pipe.vae.encode(pixels).latent_dist.mean * pipe.vae.config.scaling_factor
    prompt = pipe.tokenizer(args.prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    cond = pipe.text_encoder(prompt.input_ids.cuda())[0]
    uncond = pipe.text_encoder(pipe.tokenizer("", padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt").input_ids.cuda())[0]
    pipe.scheduler.set_timesteps(20, device="cuda")
    timestep = pipe.scheduler.timesteps[len(pipe.scheduler.timesteps) // 2]
    noisy = pipe.scheduler.add_noise(latent, torch.randn_like(latent), timestep.unsqueeze(0))
    with torch.no_grad():
        pipe.unet(torch.cat([noisy, noisy]), timestep, encoder_hidden_states=torch.cat([uncond, cond]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"kind": "sd_image_latent_attention_smoke", "formal_calibration_eligible": False, "reason": "no DDIM/null-text inversion", "prompt": args.prompt, "seed": args.seed, "resolutions": {str(side): len(items) for side, items in store.maps.items()}}
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
