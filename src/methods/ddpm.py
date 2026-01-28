"""
Denoising Diffusion Probabilistic Models (DDPM)
"""

import math
from typing import Dict, Tuple, Optional, Literal, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMethod


class DDPM(BaseMethod):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_timesteps: int,
        beta_start: float,
        beta_end: float,
    ):
        super().__init__(model, device)

        self.num_timesteps = int(num_timesteps)
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Create beta schedule (linear schedule)
        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        self.register_buffer('betas', betas)
        
        # Compute alphas
        alphas = 1.0 - betas
        self.register_buffer('alphas', alphas)
        
        # Compute cumulative product of alphas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        
        # Alphas cumprod at previous timestep (for reverse process)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Pre-compute values for forward process q(x_t | x_0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        # Pre-compute values for reverse process posterior q(x_{t-1} | x_t, x_0)
        # Posterior variance: β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        
        # Clip the log to avoid numerical issues
        self.register_buffer('posterior_log_variance_clipped', 
                           torch.log(torch.clamp(posterior_variance, min=1e-20)))
        
        # Pre-compute coefficients for reverse process mean
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1.0))

    # =========================================================================
    # Helper functions
    # Pro tips: If you have a lot of pseudo parameters that you will specify for each
    # model run but will be fixed once you specified them (say in your config),
    # then you can use super().register_buffer(...) for these parameters

    # Pro tips 2: If you need a specific broadcasting for your tensors,
    # it's a good idea to write a general helper function for that
    # =========================================================================
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        """
        Extract coefficients at specified timesteps and reshape for broadcasting.
        
        Args:
            a: Tensor to extract from (1D tensor of length num_timesteps)
            t: Timestep indices (batch_size,)
            x_shape: Shape of the input tensor for proper broadcasting
            
        Returns:
            Extracted values reshaped to (batch_size, 1, 1, 1) for broadcasting
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    # =========================================================================
    # Forward process
    # =========================================================================

    def forward_process(
        self, 
        x_0: torch.Tensor, 
        t: torch.Tensor, 
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implement the forward diffusion process q(x_t | x_0).
        
        Using the reparameterization trick:
        x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
        
        Args:
            x_0: Clean data samples of shape (batch_size, channels, height, width)
            t: Timestep indices of shape (batch_size,)
            noise: Optional noise tensor. If None, samples from N(0, I)
        
        Returns:
            x_t: Noisy samples at timestep t
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Extract coefficients for timestep t and reshape for broadcasting
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        # Apply noise: x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_t, noise

    # =========================================================================
    # Training loss
    # =========================================================================

    def compute_loss(self, x_0: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the DDPM training loss (simplified objective).
        
        The loss is: L_simple = E_{t, x_0, ε} [ || ε - ε_θ(x_t, t) ||^2 ]
        where x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε

        Args:
            x_0: Clean data samples of shape (batch_size, channels, height, width)
            **kwargs: Additional method-specific arguments
        
        Returns:
            loss: Scalar loss tensor for backpropagation
            metrics: Dictionary of metrics for logging (e.g., {'mse': 0.1})
        """
        batch_size = x_0.shape[0]
        
        # Sample random timesteps uniformly
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device, dtype=torch.long)
        
        # Sample noise
        noise = torch.randn_like(x_0)
        
        # Get noisy images at timestep t
        x_t, _ = self.forward_process(x_0, t, noise)
        
        # Predict the noise
        noise_pred = self.model(x_t, t)
        
        # Compute MSE loss between predicted and actual noise
        loss = F.mse_loss(noise_pred, noise)
        
        # Return loss and metrics
        metrics = {
            'loss': loss.detach(),
            'mse': loss.detach(),
        }
        
        return loss, metrics

    # =========================================================================
    # Reverse process (sampling)
    # =========================================================================
    
    @torch.no_grad()
    def reverse_process(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Implement one step of the DDPM reverse (denoising) process p(x_{t-1} | x_t).
        
        The reverse process mean is:
        μ_θ(x_t, t) = (1 / sqrt(α_t)) * (x_t - (β_t / sqrt(1 - ᾱ_t)) * ε_θ(x_t, t))
        
        Sample: x_{t-1} = μ_θ(x_t, t) + σ_t * z, where z ~ N(0, I) if t > 1, else z = 0

        Args:
            x_t: Noisy samples at time t (batch_size, channels, height, width)
            t: Timestep tensor of shape (batch_size,)
        
        Returns:
            x_prev: Noisy samples at time t-1 (batch_size, channels, height, width)
        """
        # Predict noise
        noise_pred = self.model(x_t, t)
        
        # Extract coefficients
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x_t.shape)
        betas_t = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        
        # Compute the mean of the reverse process (DDPM Eq. 11)
        # μ_θ = (1 / sqrt(α_t)) * (x_t - (β_t / sqrt(1 - ᾱ_t)) * ε_θ)
        model_mean = sqrt_recip_alphas_t * (x_t - (betas_t / sqrt_one_minus_alphas_cumprod_t) * noise_pred)
        
        # Add noise if not at t=0
        if t[0] > 0:
            # Extract posterior variance
            posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            x_prev = model_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            # At t=0, don't add noise
            x_prev = model_mean
        
        return x_prev

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Implement DDPM sampling: Start from pure noise x_T ~ N(0, I) and 
        iteratively denoise using the reverse process.

        Args:
            batch_size: Number of samples to generate
            image_shape: Shape of each image (channels, height, width)
            num_steps: Number of sampling steps (defaults to num_timesteps)
            **kwargs: Additional method-specific arguments
        
        Returns:
            samples: Generated samples of shape (batch_size, *image_shape)
        """
        self.eval_mode()
        
        if num_steps is None:
            num_steps = self.num_timesteps
        
        # Start from pure noise x_T ~ N(0, I)
        x = torch.randn(batch_size, *image_shape, device=self.device)
        
        # Compute timesteps to use (evenly spaced across full range)
        if num_steps < self.num_timesteps:
            # Evenly spaced timesteps from T-1 down to 0
            timesteps = torch.linspace(
                self.num_timesteps - 1, 0, num_steps, 
                dtype=torch.long, device=self.device
            )
        else:
            # Use all timesteps
            timesteps = torch.arange(
                self.num_timesteps - 1, -1, -1, 
                dtype=torch.long, device=self.device
            )
        
        # Iterate through selected timesteps
        for t in timesteps:
            # Create batch of timesteps
            t_batch = torch.full((batch_size,), t.item(), device=self.device, dtype=torch.long)
            
            # One step of reverse process
            x = self.reverse_process(x, t_batch)
        
        return x

    # =========================================================================
    # Device / state
    # =========================================================================

    def to(self, device: torch.device) -> "DDPM":
        super().to(device)
        self.device = device
        return self

    def state_dict(self) -> Dict:
        state = super().state_dict()
        state["num_timesteps"] = self.num_timesteps
        state["beta_start"] = self.beta_start
        state["beta_end"] = self.beta_end
        return state

    @classmethod
    def from_config(cls, model: nn.Module, config: dict, device: torch.device) -> "DDPM":
        ddpm_config = config.get("ddpm", config)
        return cls(
            model=model,
            device=device,
            num_timesteps=ddpm_config["num_timesteps"],
            beta_start=ddpm_config["beta_start"],
            beta_end=ddpm_config["beta_end"],
        ).to(device)
