"""P1-C student for single-step lateral correction and release prediction.

The student consumes frozen route evidence (or a zeroed evidence control) plus
the base action and proprioception.  It predicts a bounded xy residual for the
first action only and a release logit.  z and rotation have no output path, so
they are exactly zero rather than merely small.
"""
from __future__ import annotations

from typing import Any

import torch


class P1Student(torch.nn.Module):
    """Parameter-matched control and evidence arms for P1-C."""

    def __init__(
        self,
        *,
        feature_dim: int = 4096,
        action_dim: int = 7,
        horizon: int = 8,
        proprio_dim: int = 8,
        geometry_dim: int = 6,
        hidden_dim: int = 256,
        feature_bottleneck_dim: int = 128,
        dropout: float = 0.1,
        max_translation_delta: float = 0.10,
        gate_mode: str = "tanh",
        release_bias: float = -4.0,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or action_dim <= 0 or horizon <= 0 or proprio_dim <= 0:
            raise ValueError("feature/action/horizon/proprio dimensions must be positive")
        if geometry_dim < 0 or feature_bottleneck_dim <= 0 or hidden_dim <= 0:
            raise ValueError("geometry/bottleneck/hidden dimensions are invalid")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if max_translation_delta <= 0:
            raise ValueError("max_translation_delta must be positive")
        if gate_mode not in ("tanh", "direct_zero_head"):
            raise ValueError("gate_mode must be tanh or direct_zero_head")

        self.feature_dim = int(feature_dim)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.proprio_dim = int(proprio_dim)
        self.geometry_dim = int(geometry_dim)
        self.feature_bottleneck_dim = int(feature_bottleneck_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_translation_delta = float(max_translation_delta)
        self.gate_mode = str(gate_mode)

        self.route_projection = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, self.feature_bottleneck_dim),
            torch.nn.LayerNorm(self.feature_bottleneck_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(float(dropout)),
        )
        input_dim = (
            4 * self.feature_bottleneck_dim
            + self.geometry_dim
            + self.proprio_dim
            + self.horizon * self.action_dim
        )
        self.input_norm = torch.nn.LayerNorm(input_dim)
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(input_dim, self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.GELU(),
        )
        self.xy_head = torch.nn.Linear(self.hidden_dim, 2)
        self.correction_head = torch.nn.Linear(self.hidden_dim, 1)
        self.release_head = torch.nn.Linear(self.hidden_dim, 1)
        # Start as an exact action no-op.  The release head starts with a
        # conservative negative bias and learns from the 88 positive triggers.
        self.gate_param = torch.nn.Parameter(torch.zeros(1))
        if self.gate_mode == "direct_zero_head":
            torch.nn.init.zeros_(self.xy_head.weight)
            torch.nn.init.zeros_(self.xy_head.bias)
        torch.nn.init.zeros_(self.correction_head.weight)
        torch.nn.init.zeros_(self.correction_head.bias)
        torch.nn.init.zeros_(self.release_head.weight)
        torch.nn.init.constant_(self.release_head.bias, float(release_bias))
        self.register_buffer(
            "xy_scale",
            torch.tensor([self.max_translation_delta] * 2, dtype=torch.float32),
            persistent=True,
        )

    def forward(
        self,
        z_source,
        z_target,
        z_global,
        proprio,
        base_action,
        geometry=None,
        *,
        use_route: bool,
        use_geometry: bool,
    ) -> dict[str, Any]:
        values = [z_source, z_target, z_global]
        if any(value.ndim != 2 for value in values):
            raise ValueError("evidence inputs must be [batch, dim]")
        if any(value.shape[-1] != self.feature_dim for value in values):
            raise ValueError("evidence dimension does not match student")
        if len({value.shape[0] for value in values}) != 1:
            raise ValueError("evidence batch sizes differ")
        batch = z_source.shape[0]
        if proprio.shape != (batch, self.proprio_dim):
            raise ValueError("proprio must be [batch, proprio_dim]")
        if base_action.shape != (batch, self.horizon, self.action_dim):
            raise ValueError("base_action must be [batch, horizon, action_dim]")
        if self.geometry_dim > 0:
            if geometry is None:
                geometry = torch.zeros(
                    (batch, self.geometry_dim), dtype=z_source.dtype, device=z_source.device
                )
            elif geometry.shape != (batch, self.geometry_dim):
                raise ValueError(f"geometry must be [batch, {self.geometry_dim}]")
        elif geometry is not None:
            raise ValueError("geometry must be absent when geometry_dim=0")

        if not use_route:
            z_source = torch.zeros_like(z_source)
            z_target = torch.zeros_like(z_target)
            z_global = torch.zeros_like(z_global)
        if not use_geometry:
            geometry = torch.zeros_like(geometry)

        route_features = torch.cat(
            [
                self.route_projection(z_source),
                self.route_projection(z_target),
                self.route_projection(z_global),
                self.route_projection(z_target - z_source),
            ],
            dim=-1,
        )
        features = torch.cat(
            [route_features, geometry, proprio, base_action.reshape(batch, -1)], dim=-1
        )
        hidden = self.trunk(self.input_norm(features))
        if self.gate_mode == "direct_zero_head":
            gate = torch.ones((), dtype=hidden.dtype, device=hidden.device)
            correction_logit = self.correction_head(hidden).reshape(batch)
            correction_active = torch.sigmoid(correction_logit).reshape(batch, 1)
            delta_xy = correction_active * self.xy_scale * torch.tanh(self.xy_head(hidden))
        else:
            gate = torch.tanh(self.gate_param).reshape(())
            correction_logit = self.correction_head(hidden).reshape(batch)
            correction_active = torch.sigmoid(correction_logit).reshape(batch, 1)
            delta_xy = gate * correction_active * self.xy_scale * torch.tanh(self.xy_head(hidden))
        release_logit = self.release_head(hidden).reshape(batch)
        return {
            "delta_xy": delta_xy,
            "correction_logit": correction_logit,
            "release_logit": release_logit,
            "gate": gate,
        }


def arm_inputs(arm: str) -> tuple[bool, bool]:
    """Return ``(use_route, use_geometry)`` for a P1-C arm."""

    if arm == "R1-CAP":
        return False, False
    if arm == "R2-ROUTE":
        return True, False
    if arm == "R3-GEO":
        return False, True
    raise ValueError(f"unknown P1-C arm: {arm}")
