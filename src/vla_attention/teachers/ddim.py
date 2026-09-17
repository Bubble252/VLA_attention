"""Deterministic DDIM inversion primitive for image-conditioned teacher maps."""

from __future__ import annotations

from typing import Any


def invert_latent(unet: Any, scheduler: Any, latent: Any, text_embeddings: Any, *, steps: int, device: str) -> Any:
    """Run conditional DDIM inversion without null-text optimization.

    The caller records `method=ddim_conditional_no_nulltext`; this is image
    conditioned and deterministic, but not an exact reproduction of Lavender's
    null-text inversion pipeline.
    """
    scheduler.set_timesteps(steps, device=device)
    current = latent
    for timestep in scheduler.timesteps:
        noise_prediction = unet(current, timestep, encoder_hidden_states=text_embeddings).sample
        current = scheduler.step(noise_prediction, timestep, current).prev_sample
    return current
