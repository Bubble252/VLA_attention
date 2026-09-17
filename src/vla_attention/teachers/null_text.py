"""Null-text optimization over a DDIM image inversion trajectory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class NullTextResult:
    inverse_latents: list[Any]
    unconditional_embeddings: list[Any]
    reconstruction_losses: list[float]


def invert_ddim(unet: Any, inverse_scheduler: Any, latent: Any, conditional: Any, *, steps: int, device: str) -> list[Any]:
    """Return [x_0, ..., x_T] under conditional DDIM inversion."""
    inverse_scheduler.set_timesteps(steps, device=device)
    latents = [latent.detach()]
    current = latent.detach()
    for timestep in inverse_scheduler.timesteps:
        with __import__("torch").no_grad():
            noise = unet(current, timestep, encoder_hidden_states=conditional).sample
            current = inverse_scheduler.step(noise, timestep, current).prev_sample.detach()
        latents.append(current)
    return latents


def optimize_null_text(unet: Any, scheduler: Any, inverse_latents: list[Any], unconditional: Any, conditional: Any, *, steps: int, guidance_scale: float, inner_steps: int, device: str) -> NullTextResult:
    """Port Lavender's per-timestep unconditional-embedding optimization."""
    import torch
    import torch.nn.functional as functional

    scheduler.set_timesteps(steps, device=device)
    if len(inverse_latents) != len(scheduler.timesteps) + 1:
        raise ValueError("inversion trajectory length must match forward scheduler steps")
    current = inverse_latents[-1].detach()
    optimized, losses = [], []
    for index, timestep in enumerate(scheduler.timesteps):
        target = inverse_latents[-2 - index].detach()
        candidate = unconditional.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([candidate], lr=1e-2 * (1.0 - index / 100.0))
        with torch.no_grad():
            conditional_noise = unet(current, timestep, encoder_hidden_states=conditional).sample
        for _ in range(inner_steps):
            unconditional_noise = unet(current, timestep, encoder_hidden_states=candidate).sample
            noise = unconditional_noise + guidance_scale * (conditional_noise - unconditional_noise)
            reconstructed = scheduler.step(noise, timestep, current).prev_sample
            loss = functional.mse_loss(reconstructed, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        optimized.append(candidate.detach())
        losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            context = torch.cat([candidate.detach(), conditional])
            pair = torch.cat([current, current])
            prediction = unet(pair, timestep, encoder_hidden_states=context).sample
            uncond_noise, cond_noise = prediction.chunk(2)
            guided = uncond_noise + guidance_scale * (cond_noise - uncond_noise)
            current = scheduler.step(guided, timestep, current).prev_sample.detach()
    return NullTextResult(inverse_latents, optimized, losses)
