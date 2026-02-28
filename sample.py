"""
Sampling Script for Diffusion / Flow Matching Models

Generate samples from a trained model. By default, saves individual images to avoid
memory issues with large sample counts. Use --grid to generate a single grid image.

Supported methods:
    ddpm           - DDPM sampling (standard reverse process)
    ddim           - DDIM sampling (deterministic, fewer steps, reuses DDPM checkpoint)
    rx_ddim        - RX-DDIM sampling (DDIM + Richardson extrapolation, reuses DDPM checkpoint)
    cng_rx_ddim    - Condition-Number-Gated RX-DDIM (adaptive gate on extrapolation)
    do_rx_ddim     - Dual-Order RX-DDIM (embedded-RK-style error estimation)
    dws_rx_ddim    - Distillation Warm-Started RX-DDIM (distill + renoise + RX refine)
    pd             - Progressive Distillation v-prediction (1-step or multi-step DDIM)
    flow_matching  - Flow Matching sampling (Euler integration)

Usage:
    # DDPM sampling
    python sample.py --checkpoint ckpt.pt --method ddpm --num_samples 64

    # DDIM sampling (uses the same DDPM checkpoint, fewer steps)
    python sample.py --checkpoint ckpt.pt --method ddim --num_steps 100 --num_samples 64

    # RX-DDIM sampling (improved DDIM with extrapolation, same checkpoint)
    python sample.py --checkpoint ckpt.pt --method rx_ddim --num_steps 100 --k 2 --num_samples 64

    # CNG-RX-DDIM (condition-number-gated, good for very few steps)
    python sample.py --checkpoint ckpt.pt --method cng_rx_ddim --num_steps 5 --k 2 --tau 0.3 --s_param 0.1

    # DO-RX-DDIM (dual-order adaptive, self-calibrating)
    python sample.py --checkpoint ckpt.pt --method do_rx_ddim --num_steps 5 --k 2 --do_threshold 0.1

    # DWS-RX-DDIM (distillation warm-start + RX refinement, 3 total NFE)
    python sample.py --checkpoint ckpt.pt --method dws_rx_ddim --distilled_checkpoint pd_final.pt --dws_tau 0.3 --n_rx_steps 2

    # Flow Matching sampling
    python sample.py --checkpoint ckpt.pt --method flow_matching --num_steps 100

    # PD v-prediction (Progressive Distillation 1-step)
    python sample.py --checkpoint pd_stage5_final.pt --method pd --num_steps 1 --num_samples 8 --grid

    # Generate a grid image
    python sample.py --checkpoint ckpt.pt --method ddpm --num_samples 16 --grid
"""

import os
import sys
import argparse
from datetime import datetime

import yaml
import torch
from tqdm import tqdm

from src.models import create_model_from_config
from src.data import save_image
from src.methods import DDPM, FlowMatching
from src.utils import EMA


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load checkpoint and return model, config, and EMA."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    model = create_model_from_config(config).to(device)
    model.load_state_dict(checkpoint['model'])
    
    # Create EMA and load
    ema = EMA(model, decay=config['training']['ema_decay'])
    ema.load_state_dict(checkpoint['ema'])
    
    return model, config, ema


@torch.no_grad()
def ddim_sample_v_pred(model, alphas_cumprod, image_shape, num_steps, batch_size, device):
    """DDIM sampling with a v-prediction model (Progressive Distillation)."""
    num_timesteps = len(alphas_cumprod)
    schedule = torch.linspace(num_timesteps - 1, 0, num_steps + 1).long().to(device)
    z = torch.randn(batch_size, *image_shape, device=device)

    for i in range(num_steps):
        t = schedule[i].item()
        t_next = schedule[i + 1].item()
        abar_t = alphas_cumprod[t]
        abar_next = alphas_cumprod[t_next]
        alpha_t = torch.sqrt(abar_t)
        sigma_t = torch.sqrt(1.0 - abar_t)

        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        v = model(z, t_batch)

        x0 = alpha_t * z - sigma_t * v
        eps = sigma_t * z + alpha_t * v

        z = torch.sqrt(abar_next) * x0 + torch.sqrt(1.0 - abar_next) * eps

    return z


def save_samples(
    samples: torch.Tensor,
    save_path: str,
    nrow: int = 8,
) -> None:
    """
    Save generated samples as images.

    Args:
        samples: Generated samples tensor with shape (num_samples, C, H, W).
        save_path: File path to save the image grid.
        nrow: Number of images per row in the grid.
    """
    from src.data import unnormalize
    
    # Unnormalize from [-1, 1] to [0, 1]
    samples = unnormalize(samples)
    
    # Save as grid
    save_image(samples, save_path, nrow=nrow)


def main():
    parser = argparse.ArgumentParser(description='Generate samples from trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--method', type=str, required=True,
                       choices=['ddpm', 'ddim', 'rx_ddim', 'cng_rx_ddim', 'do_rx_ddim', 'dws_rx_ddim', 'pd', 'flow_matching'],
                       help='Sampling method')
    parser.add_argument('--num_samples', type=int, default=64,
                       help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='samples',
                       help='Directory to save individual images (default: samples)')
    parser.add_argument('--grid', action='store_true',
                       help='Save as grid image instead of individual images')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for grid (only used with --grid, default: samples_<timestamp>.png)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for generation')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    # Sampling arguments
    parser.add_argument('--num_steps', type=int, default=None,
                       help='Number of sampling steps (default: from config)')
    parser.add_argument('--k', type=int, default=2,
                       help='Extrapolation interval for RX-DDIM variants (default: 2)')
    parser.add_argument('--tau', type=float, default=0.3,
                       help='CNG-RX-DDIM: sigmoid centre for condition-number gate (default: 0.3)')
    parser.add_argument('--s_param', type=float, default=0.1,
                       help='CNG-RX-DDIM: sigmoid sharpness (default: 0.1)')
    parser.add_argument('--p1', type=int, default=2,
                       help='DO-RX-DDIM: primary assumed error order (default: 2)')
    parser.add_argument('--p2', type=int, default=3,
                       help='DO-RX-DDIM: secondary error order for comparison (default: 3)')
    parser.add_argument('--do_threshold', type=float, default=0.1,
                       help='DO-RX-DDIM: disagreement threshold for gating (default: 0.1)')
    parser.add_argument('--distilled_checkpoint', type=str, default=None,
                       help='DWS-RX-DDIM: path to distilled (PD) model checkpoint')
    parser.add_argument('--dws_tau', type=float, default=0.3,
                       help='DWS-RX-DDIM: re-noising level (default: 0.3)')
    parser.add_argument('--n_rx_steps', type=int, default=2,
                       help='DWS-RX-DDIM: number of RX refinement steps (default: 2)')
    parser.add_argument('--extrapolation', type=str, default='standard',
                       choices=['standard', 'cng', 'do'],
                       help='DWS-RX-DDIM: extrapolation strategy for refinement (default: standard)')
    
    # Other options
    parser.add_argument('--no_ema', action='store_true',
                       help='Use training weights instead of EMA weights')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    model, config, ema = load_checkpoint(args.checkpoint, device)
    
    # Create method
    distilled_model = None
    pd_alphas_cumprod = None
    if args.method in ('ddpm', 'ddim', 'rx_ddim', 'cng_rx_ddim', 'do_rx_ddim', 'dws_rx_ddim'):
        method = DDPM.from_config(model, config, device)
        if args.method == 'dws_rx_ddim':
            if args.distilled_checkpoint is None:
                raise ValueError("--distilled_checkpoint is required for dws_rx_ddim")
            print(f"Loading distilled model from {args.distilled_checkpoint}...")
            d_ckpt = torch.load(args.distilled_checkpoint, map_location=device)
            distilled_model = create_model_from_config(config).to(device)
            distilled_model.load_state_dict(d_ckpt['model'])
            if 'ema' in d_ckpt and not args.no_ema:
                d_ema = EMA(distilled_model, decay=config['training']['ema_decay'])
                d_ema.load_state_dict(d_ckpt['ema'])
                d_ema.apply_shadow()
            distilled_model.eval()
    elif args.method == 'pd':
        ddpm_cfg = config['ddpm']
        betas = torch.linspace(
            ddpm_cfg['beta_start'], ddpm_cfg['beta_end'],
            ddpm_cfg['num_timesteps'], dtype=torch.float32,
        )
        pd_alphas_cumprod = torch.cumprod(1.0 - betas, dim=0).to(device)
        method = None
    elif args.method == 'flow_matching':
        method = FlowMatching.from_config(model, config, device)
    else:
        raise ValueError(f"Unknown method: {args.method}. Supported: ddpm, ddim, pd, flow_matching.")
    
    # Apply EMA weights
    if not args.no_ema:
        print("Using EMA weights")
        ema.apply_shadow()
    else:
        print("Using training weights (no EMA)")

    if method is not None:
        method.eval_mode()
    
    # Image shape
    data_config = config['data']
    image_shape = (data_config['channels'], data_config['image_size'], data_config['image_size'])
    
    # Generate samples
    print(f"Generating {args.num_samples} samples...")

    all_samples = []
    remaining = args.num_samples
    sample_idx = 0

    # Create output directory if saving individual images
    if not args.grid:
        os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        pbar = tqdm(total=args.num_samples, desc="Generating samples")
        while remaining > 0:
            batch_size = min(args.batch_size, remaining)

            num_steps = args.num_steps
            if num_steps is None:
                num_steps = config.get('sampling', {}).get('num_steps', 1)

            if args.method == 'pd':
                samples = ddim_sample_v_pred(
                    model, pd_alphas_cumprod, image_shape,
                    num_steps=num_steps,
                    batch_size=batch_size,
                    device=device,
                )
            elif args.method == 'ddim':
                samples = method.ddim_sample(
                    batch_size=batch_size,
                    image_shape=image_shape,
                    num_steps=num_steps,
                )
            elif args.method == 'rx_ddim':
                samples = method.rx_ddim_sample(
                    batch_size=batch_size,
                    image_shape=image_shape,
                    num_steps=num_steps,
                    k=args.k,
                )
            elif args.method == 'cng_rx_ddim':
                samples = method.cng_rx_ddim_sample(
                    batch_size=batch_size,
                    image_shape=image_shape,
                    num_steps=num_steps,
                    k=args.k,
                    tau=args.tau,
                    s_param=args.s_param,
                )
            elif args.method == 'do_rx_ddim':
                samples = method.do_rx_ddim_sample(
                    batch_size=batch_size,
                    image_shape=image_shape,
                    num_steps=num_steps,
                    k=args.k,
                    p1=args.p1,
                    p2=args.p2,
                    do_threshold=args.do_threshold,
                )
            elif args.method == 'dws_rx_ddim':
                samples = method.dws_rx_ddim_sample(
                    distilled_model=distilled_model,
                    batch_size=batch_size,
                    image_shape=image_shape,
                    tau=args.dws_tau,
                    n_rx_steps=args.n_rx_steps,
                    k=args.k,
                    extrapolation=args.extrapolation,
                    tau_cng=args.tau,
                    s_param=args.s_param,
                    p1=args.p1,
                    p2=args.p2,
                    do_threshold=args.do_threshold,
                )
            else:
                samples = method.sample(
                    batch_size=batch_size,
                    image_shape=image_shape,
                    num_steps=num_steps,
                )

            # Save individual images immediately or collect for grid
            if args.grid:
                all_samples.append(samples)
            else:
                for i in range(samples.shape[0]):
                    img_path = os.path.join(args.output_dir, f"{sample_idx:06d}.png")
                    # save_samples(samples, img_path, 1)
                    save_samples(samples[i:i+1], img_path, 1) # save each sample individually
                    sample_idx += 1

            remaining -= batch_size
            pbar.update(batch_size)

        pbar.close()

    # Save samples
    if args.grid:
        # Concatenate all samples for grid
        all_samples = torch.cat(all_samples, dim=0)[:args.num_samples]

        if args.output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = f"samples_{timestamp}.png"

        save_samples(all_samples, args.output, nrow=8)
        print(f"Saved grid to {args.output}")
    else:
        print(f"Saved {args.num_samples} individual images to {args.output_dir}")

    # Restore EMA if applied
    if not args.no_ema:
        ema.restore()


if __name__ == '__main__':
    main()
