"""
Flow Matching for Generative Modeling (Lipman et al., 2023)

Learns a velocity field v_theta(x_t, t) that transports noise to data
along straight (optimal transport) paths.

Convention:
    t in [0, 1] where t=0 is data and t=1 is pure noise
    x_t = (1 - t) * x_0 + t * epsilon,  epsilon ~ N(0, I)
    v_target = epsilon - x_0   (the straight-line velocity)

Training:
    Loss = E_{t, x_0, eps} || v_theta(x_t, t) - v_target ||^2

Sampling (Euler method, from noise to data):
    Start at x_1 ~ N(0, I), integrate dx/dt = -v_theta(x_t, t) from t=1 to t=0
"""

from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMethod


class FlowMatching(BaseMethod):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_timesteps: int = 1000,
    ):
        """
        Args:
            model: Neural network that predicts velocity v_theta(x_t, t).
            device: Device to run computations on.
            num_timesteps: Used to scale continuous t in [0,1] to integer
                           range for the UNet's sinusoidal embeddings.
        """
        super().__init__(model, device)
        self.num_timesteps = int(num_timesteps)

    # =========================================================================
    # Training loss
    # =========================================================================

    def compute_loss(self, x_0: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the flow matching training loss (conditional flow matching objective).

        Loss = E_{t, x_0, eps} || v_theta(x_t, t_scaled) - (eps - x_0) ||^2

        Args:
            x_0: Clean data samples (batch_size, C, H, W)

        Returns:
            loss: Scalar loss tensor
            metrics: Dict of logging metrics
        """
        batch_size = x_0.shape[0]

        # Sample random t ~ U(0, 1)
        t = torch.rand(batch_size, device=self.device)

        # Sample noise
        epsilon = torch.randn_like(x_0)

        # Reshape t for broadcasting: (B,) -> (B, 1, 1, 1)
        t_broadcast = t.view(batch_size, 1, 1, 1)

        # Interpolate: x_t = (1 - t) * x_0 + t * epsilon
        x_t = (1.0 - t_broadcast) * x_0 + t_broadcast * epsilon

        # Target velocity: v = epsilon - x_0
        v_target = epsilon - x_0

        # Scale t to integer range for UNet's timestep embedding
        t_scaled = (t * (self.num_timesteps - 1)).long()

        # Predict velocity
        v_pred = self.model(x_t, t_scaled)

        # MSE loss
        loss = F.mse_loss(v_pred, v_target)

        metrics = {
            'loss': loss.detach(),
            'mse': loss.detach(),
        }

        return loss, metrics

    # =========================================================================
    # Sampling (Euler integration)
    # =========================================================================

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate samples via Euler integration from t=1 (noise) to t=0 (data).

        x_{t - dt} = x_t + dt * v_theta(x_t, t)
        where v_theta predicts (eps - x_0) and we integrate *backward* in t,
        so the update is: x_{t - dt} = x_t - dt * v_theta(x_t, t)

        Args:
            batch_size: Number of samples to generate
            image_shape: (C, H, W)
            num_steps: Number of Euler steps (default 100)

        Returns:
            samples: (batch_size, C, H, W)
        """
        self.eval_mode()

        if num_steps is None:
            num_steps = 100

        # Start from pure noise at t = 1
        x = torch.randn(batch_size, *image_shape, device=self.device)

        dt = 1.0 / num_steps

        # Euler integration from t=1 to t=0
        for step in range(num_steps):
            t = 1.0 - step * dt  # current time, going from 1 -> 0

            # Scale to integer range for UNet
            t_int = int(t * (self.num_timesteps - 1))
            t_batch = torch.full(
                (batch_size,), t_int, device=self.device, dtype=torch.long
            )

            # Predict velocity
            v_pred = self.model(x, t_batch)

            # Euler step: move toward data (negative direction in t)
            x = x - dt * v_pred

        return x

    # =========================================================================
    # Device / state
    # =========================================================================

    def to(self, device: torch.device) -> "FlowMatching":
        super().to(device)
        self.device = device
        return self

    def state_dict(self) -> Dict:
        state = super().state_dict()
        state["num_timesteps"] = self.num_timesteps
        return state

    @classmethod
    def from_config(cls, model: nn.Module, config: dict, device: torch.device) -> "FlowMatching":
        fm_config = config.get("flow_matching", {})
        return cls(
            model=model,
            device=device,
            num_timesteps=fm_config.get("num_timesteps", 1000),
        ).to(device)
