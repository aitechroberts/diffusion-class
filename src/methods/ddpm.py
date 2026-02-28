"""
Denoising Diffusion Probabilistic Models (DDPM)
"""

import math
from typing import Callable, Dict, List, Tuple, Optional, Literal

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
    # DDIM sampling (deterministic, fewer steps, same trained model)
    # =========================================================================

    @torch.no_grad()
    def ddim_sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: int = 100,
        **kwargs,
    ) -> torch.Tensor:
        """
        DDIM sampling (Song et al., 2020).

        Uses a deterministic update rule that allows far fewer steps than DDPM
        while reusing the exact same trained noise-prediction model.

        Algorithm:
            1. Create timestep subsequence [tau_S, ..., tau_1] evenly spaced
            2. For each step:
                eps   = eps_theta(x_t, t)
                x0_hat = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)
                x_{t_prev} = sqrt(alpha_bar_{t_prev}) * x0_hat
                             + sqrt(1 - alpha_bar_{t_prev}) * eps

        Args:
            batch_size: Number of samples to generate
            image_shape: (C, H, W)
            num_steps: Number of DDIM steps (e.g. 100 instead of 1000)

        Returns:
            samples: (batch_size, C, H, W)
        """
        self.eval_mode()

        # Build evenly-spaced timestep subsequence from T-1 down to 0
        # We need num_steps+1 points so we have (t, t_prev) pairs
        step_indices = torch.linspace(
            self.num_timesteps - 1, 0, num_steps + 1,
        ).long().to(self.device)

        # Start from pure noise
        x = torch.randn(batch_size, *image_shape, device=self.device)

        for i in range(num_steps):
            t = step_indices[i]
            t_prev = step_indices[i + 1]

            t_batch = torch.full((batch_size,), t.item(), device=self.device, dtype=torch.long)

            # Predict noise
            eps = self.model(x, t_batch)

            # alpha_bar values
            alpha_bar_t = self.alphas_cumprod[t]
            alpha_bar_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=self.device)

            # Predict x_0
            x0_pred = (x - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)

            # DDIM deterministic step
            x = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1.0 - alpha_bar_prev) * eps

        return x

    # =========================================================================
    # RX-DDIM sampling (Richardson extrapolation on DDIM, same trained model)
    # Reference: Choi et al., "Enhanced Diffusion Sampling via Extrapolation
    #            with Multiple ODE Solutions", ICLR 2025
    # =========================================================================

    @staticmethod
    def tau_to_timestep(tau: float, alphas_cumprod: torch.Tensor) -> int:
        """Map a noise-level fraction *tau* in [0, 1] to the nearest
        diffusion timestep index whose forward-process noise std matches.
        """
        noise_levels = torch.sqrt(1.0 - alphas_cumprod)
        return (noise_levels - tau).abs().argmin().item()

    @torch.no_grad()
    def _rx_ddim_core(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: int,
        k: int,
        extrapolate_fn: Callable[
            [torch.Tensor, torch.Tensor, List[float]], torch.Tensor
        ],
        x_init: Optional[torch.Tensor] = None,
        step_indices_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Shared RX-DDIM loop with a pluggable extrapolation strategy.

        Args:
            batch_size: Number of samples to generate.
            image_shape: (C, H, W).
            num_steps: Total DDIM steps.
            k: Extrapolation interval (extrapolate every k steps).
            extrapolate_fn: ``fn(x_k_step, x_1_step, lam_js) -> x``
                where *lam_js* is the list of per-sub-step lambda values
                in gamma-space.
            x_init: Optional starting tensor.  When *None* (default),
                sampling starts from pure Gaussian noise.
            step_indices_override: Optional custom timestep schedule.
                When *None*, an evenly-spaced schedule from T-1 to 0 is
                built automatically.

        Returns:
            samples: (batch_size, C, H, W)
        """
        self.eval_mode()

        if step_indices_override is not None:
            step_indices = step_indices_override
        else:
            step_indices = torch.linspace(
                self.num_timesteps - 1, 0, num_steps + 1,
            ).long().to(self.device)

        gammas = torch.sqrt(
            (1.0 - self.alphas_cumprod) / self.alphas_cumprod
        )

        if x_init is not None:
            x = x_init
        else:
            x = torch.randn(batch_size, *image_shape, device=self.device)

        i = 0
        while i < num_steps:
            block_size = min(k, num_steps - i)

            x_k_step = x.clone()
            first_eps = None

            for j in range(block_size):
                t = step_indices[i + j]
                t_prev = step_indices[i + j + 1]
                t_batch = torch.full(
                    (batch_size,), t.item(),
                    device=self.device, dtype=torch.long,
                )

                eps = self.model(x_k_step, t_batch)
                if j == 0:
                    first_eps = eps

                ab_t = self.alphas_cumprod[t]
                ab_prev = self.alphas_cumprod[t_prev]

                x0_pred = (
                    (x_k_step - torch.sqrt(1.0 - ab_t) * eps)
                    / torch.sqrt(ab_t)
                )
                x_k_step = (
                    torch.sqrt(ab_prev) * x0_pred
                    + torch.sqrt(1.0 - ab_prev) * eps
                )

            if block_size == k and k > 1:
                t_start = step_indices[i]
                t_end = step_indices[i + k]

                ab_start = self.alphas_cumprod[t_start]
                ab_end = self.alphas_cumprod[t_end]

                x0_pred_1 = (
                    (x - torch.sqrt(1.0 - ab_start) * first_eps)
                    / torch.sqrt(ab_start)
                )
                x_1_step = (
                    torch.sqrt(ab_end) * x0_pred_1
                    + torch.sqrt(1.0 - ab_end) * first_eps
                )

                gamma_start = gammas[t_start]
                gamma_end = gammas[t_end]
                h = gamma_start - gamma_end

                lam_js: List[float] = []
                for j in range(block_size):
                    g_curr = gammas[step_indices[i + j]]
                    g_next = gammas[step_indices[i + j + 1]]
                    lam_js.append(((g_curr - g_next) / h).item())

                x = extrapolate_fn(x_k_step, x_1_step, lam_js)
            else:
                x = x_k_step

            i += block_size

        return x

    # ----- Standard RX-DDIM (original) ------------------------------------

    @torch.no_grad()
    def rx_ddim_sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: int = 100,
        k: int = 2,
        **kwargs,
    ) -> torch.Tensor:
        """RX-DDIM sampling — Richardson extrapolation applied to DDIM.

        Every *k* DDIM steps, two ODE solutions over the same time interval
        are combined to cancel the leading truncation-error term:

            x_tilde = (x_k_step - S * x_1_step) / (1 - S)      [Eq. 18]

        where S = sum_j lambda_j^p and p = 2 for first-order DDIM.

        Args:
            batch_size: Number of samples to generate.
            image_shape: (C, H, W).
            num_steps: Total DDIM steps (e.g. 50 or 100).
            k: Extrapolation interval (k = 2 recommended).

        Returns:
            samples: (batch_size, C, H, W)
        """
        p = 2

        def _standard_extrapolate(
            x_k_step: torch.Tensor,
            x_1_step: torch.Tensor,
            lam_js: List[float],
        ) -> torch.Tensor:
            S = sum(lj ** p for lj in lam_js)
            return (x_k_step - S * x_1_step) / (1.0 - S)

        return self._rx_ddim_core(
            batch_size, image_shape, num_steps, k, _standard_extrapolate,
        )

    # ----- Condition-Number-Gated RX-DDIM (CNG) ---------------------------

    @torch.no_grad()
    def cng_rx_ddim_sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: int = 100,
        k: int = 2,
        tau: float = 0.3,
        s_param: float = 0.1,
        **kwargs,
    ) -> torch.Tensor:
        """Condition-Number-Gated RX-DDIM sampling.

        Blends the Richardson-extrapolated result with the plain DDIM
        result using a sigmoid gate on the conditioning factor
        kappa = |1 - S|.  When kappa is large (many steps, well-
        conditioned), the gate passes through full RX-DPM.  When kappa
        approaches zero (few steps, ill-conditioned), the gate smoothly
        falls back to plain DDIM, preventing the explosive error
        amplification that makes standard RX-DPM produce noise at
        very low step counts.

        Args:
            batch_size: Number of samples to generate.
            image_shape: (C, H, W).
            num_steps: Total DDIM steps.
            k: Extrapolation interval (k = 2 recommended).
            tau: Sigmoid centre — kappa values below this shift toward
                 DDIM; above toward full RX.
            s_param: Sigmoid sharpness (smaller = sharper transition).

        Returns:
            samples: (batch_size, C, H, W)
        """
        p = 2

        def _cng_extrapolate(
            x_k_step: torch.Tensor,
            x_1_step: torch.Tensor,
            lam_js: List[float],
        ) -> torch.Tensor:
            S = sum(lj ** p for lj in lam_js)
            kappa = abs(1.0 - S)
            alpha = torch.sigmoid(
                torch.tensor((kappa - tau) / s_param, device=x_k_step.device)
            )
            x_rx = (x_k_step - S * x_1_step) / (1.0 - S)
            return alpha * x_rx + (1.0 - alpha) * x_k_step

        return self._rx_ddim_core(
            batch_size, image_shape, num_steps, k, _cng_extrapolate,
        )

    # ----- Dual-Order RX-DDIM (DO) ----------------------------------------

    @torch.no_grad()
    def do_rx_ddim_sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: int = 100,
        k: int = 2,
        p1: int = 2,
        p2: int = 3,
        do_threshold: float = 0.1,
        **kwargs,
    ) -> torch.Tensor:
        """Dual-Order RX-DDIM sampling — embedded-RK-style error estimation.

        Computes Richardson extrapolation at two candidate error orders
        *p1* and *p2* simultaneously.  The relative disagreement between
        the two estimates signals whether the polynomial error assumption
        holds at the current step size.  A sigmoid gate blends the p1
        estimate with plain DDIM based on this disagreement, providing
        automatic adaptation without hand-tuned step-count thresholds.

        Analogous to embedded Runge-Kutta methods (RK45 / Dormand-Prince)
        which use two different order solutions to estimate local error
        and decide whether to accept or reject a step.

        Args:
            batch_size: Number of samples to generate.
            image_shape: (C, H, W).
            num_steps: Total DDIM steps.
            k: Extrapolation interval (k = 2 recommended).
            p1: Primary assumed error order (default 2 for Euler/DDIM).
            p2: Secondary order used for comparison (default 3).
            do_threshold: Relative disagreement level at the sigmoid
                midpoint — larger values tolerate more disagreement
                before gating toward DDIM.

        Returns:
            samples: (batch_size, C, H, W)
        """

        def _do_extrapolate(
            x_k_step: torch.Tensor,
            x_1_step: torch.Tensor,
            lam_js: List[float],
        ) -> torch.Tensor:
            S1 = sum(lj ** p1 for lj in lam_js)
            S2 = sum(lj ** p2 for lj in lam_js)
            x_rx1 = (x_k_step - S1 * x_1_step) / (1.0 - S1)
            x_rx2 = (x_k_step - S2 * x_1_step) / (1.0 - S2)
            disagreement = (
                torch.norm(x_rx1 - x_rx2)
                / (torch.norm(x_k_step) + 1e-8)
            )
            alpha = torch.sigmoid(
                torch.tensor(
                    (1.0 / (disagreement.item() + 1e-8)
                     - 1.0 / do_threshold) * do_threshold,
                    device=x_k_step.device,
                )
            )
            return alpha * x_rx1 + (1.0 - alpha) * x_k_step

        return self._rx_ddim_core(
            batch_size, image_shape, num_steps, k, _do_extrapolate,
        )

    # ----- Distillation Warm-Started RX-DDIM (DWS) -------------------------

    @torch.no_grad()
    def dws_rx_ddim_sample(
        self,
        distilled_model: nn.Module,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        tau: float = 0.3,
        n_rx_steps: int = 2,
        k: int = 2,
        extrapolation: str = "standard",
        tau_cng: float = 0.3,
        s_param: float = 0.1,
        p1: int = 2,
        p2: int = 3,
        do_threshold: float = 0.1,
        **kwargs,
    ) -> torch.Tensor:
        """Distillation Warm-Started RX-DDIM sampling.

        Uses a distilled model for 1 NFE to produce a coarse x0 estimate,
        re-noises it to an intermediate timestep controlled by *tau*, then
        runs RX-DPM over the resulting short, well-conditioned trajectory.

        Total NFE = 1 (distillation) + n_rx_steps.

        Args:
            distilled_model: A 1-step distilled model (same interface as
                the base UNet — takes (x_t, t) and predicts eps).
            batch_size: Number of samples to generate.
            image_shape: (C, H, W).
            tau: Noise-level fraction for re-noising (0 = no noise,
                 1 = pure noise).  Typical sweet spot: 0.2-0.4.
            n_rx_steps: Number of RX-DPM refinement steps (2 or 3).
            k: Extrapolation interval for RX-DPM (k = 2 recommended).
            extrapolation: Which extrapolation strategy to use for the
                RX refinement pass: ``"standard"``, ``"cng"``, or
                ``"do"``.
            tau_cng: CNG sigmoid centre (only used when extrapolation="cng").
            s_param: CNG sigmoid sharpness (only used when extrapolation="cng").
            p1: DO primary error order (only used when extrapolation="do").
            p2: DO secondary error order (only used when extrapolation="do").
            do_threshold: DO disagreement threshold (only used when
                extrapolation="do").

        Returns:
            samples: (batch_size, C, H, W)
        """
        self.eval_mode()
        distilled_model.eval()

        # --- Step 1: 1-NFE distilled estimate --------------------------------
        # The distilled model uses v-prediction:  v = alpha*eps - sigma*x0
        # so  x0 = alpha*z - sigma*v  (where alpha=sqrt(abar), sigma=sqrt(1-abar))
        z = torch.randn(batch_size, *image_shape, device=self.device)
        t_T = self.num_timesteps - 1
        t_batch = torch.full(
            (batch_size,), t_T, device=self.device, dtype=torch.long,
        )
        v_pred = distilled_model(z, t_batch)
        abar_T = self.alphas_cumprod[t_T]
        alpha_T = torch.sqrt(abar_T)
        sigma_T = torch.sqrt(1.0 - abar_T)
        x0_hat = alpha_T * z - sigma_T * v_pred

        # --- Step 2: Re-noise to intermediate timestep -----------------------
        t_start = self.tau_to_timestep(tau, self.alphas_cumprod)
        t_start = max(t_start, 1)  # avoid t=0 which leaves no room to denoise
        abar_start = self.alphas_cumprod[t_start]
        epsilon = torch.randn_like(x0_hat)
        x_tau = (
            torch.sqrt(abar_start) * x0_hat
            + torch.sqrt(1.0 - abar_start) * epsilon
        )

        # --- Step 3: Build short schedule and select extrapolation fn --------
        step_indices = torch.linspace(
            t_start, 0, n_rx_steps + 1,
        ).long().to(self.device)

        p_base = 2

        if extrapolation == "cng":
            def _extrapolate(x_k, x_1, lam_js):
                S = sum(lj ** p_base for lj in lam_js)
                kappa = abs(1.0 - S)
                alpha = torch.sigmoid(
                    torch.tensor((kappa - tau_cng) / s_param, device=x_k.device)
                )
                x_rx = (x_k - S * x_1) / (1.0 - S)
                return alpha * x_rx + (1.0 - alpha) * x_k
        elif extrapolation == "do":
            def _extrapolate(x_k, x_1, lam_js):
                S1 = sum(lj ** p1 for lj in lam_js)
                S2 = sum(lj ** p2 for lj in lam_js)
                x_rx1 = (x_k - S1 * x_1) / (1.0 - S1)
                x_rx2 = (x_k - S2 * x_1) / (1.0 - S2)
                disag = torch.norm(x_rx1 - x_rx2) / (torch.norm(x_k) + 1e-8)
                a = torch.sigmoid(torch.tensor(
                    (1.0 / (disag.item() + 1e-8) - 1.0 / do_threshold)
                    * do_threshold,
                    device=x_k.device,
                ))
                return a * x_rx1 + (1.0 - a) * x_k
        else:
            def _extrapolate(x_k, x_1, lam_js):
                S = sum(lj ** p_base for lj in lam_js)
                return (x_k - S * x_1) / (1.0 - S)

        return self._rx_ddim_core(
            batch_size, image_shape, n_rx_steps, k, _extrapolate,
            x_init=x_tau,
            step_indices_override=step_indices,
        )

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
