"""Modern Diffusers cross-attention capture for the SD1.5 calibration adapter.

This retains the attention-probability part of Lavender's teacher mechanism but
does not claim to reproduce its null-text image inversion on its own.  Callers
must label maps with their inversion method.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import isqrt
from typing import Any


@dataclass
class CrossAttentionStore:
    """CPU summaries of conditional cross attention by spatial query resolution."""

    maps: dict[int, list[Any]] = field(default_factory=lambda: defaultdict(list))

    def record(self, attention_probs: Any) -> None:
        # Shape: batch*heads, query_tokens, text_tokens. Conditional CFG half is last.
        heads_times_batch, query_tokens, _ = attention_probs.shape
        side = isqrt(query_tokens)
        if side * side != query_tokens:
            return
        conditional = attention_probs[heads_times_batch // 2 :] if heads_times_batch % 2 == 0 else attention_probs
        self.maps[side].append(conditional.detach().float().cpu())


class CapturingAttnProcessor:
    """Diffusers 0.38-compatible processor matching AttnProcessor arithmetic."""

    def __init__(self, store: CrossAttentionStore) -> None:
        self.store = store

    def __call__(self, attn: Any, hidden_states: Any, encoder_hidden_states: Any = None, attention_mask: Any = None, temb: Any = None, *args: Any, **kwargs: Any) -> Any:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        is_cross = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key, value = attn.to_k(encoder_hidden_states), attn.to_v(encoder_hidden_states)
        query, key, value = attn.head_to_batch_dim(query), attn.head_to_batch_dim(key), attn.head_to_batch_dim(value)
        probabilities = attn.get_attention_scores(query, key, attention_mask)
        if is_cross:
            self.store.record(probabilities)
        hidden_states = attn.batch_to_head_dim(probabilities @ value)
        hidden_states = attn.to_out[1](attn.to_out[0](hidden_states))
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def install_cross_attention_capture(unet: Any) -> CrossAttentionStore:
    """Replace all UNet attention processors; caller owns the loaded pipeline."""
    store = CrossAttentionStore()
    unet.set_attn_processor({name: CapturingAttnProcessor(store) for name in unet.attn_processors})
    return store
