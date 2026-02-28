"""
Progressive Distillation KID Evaluation Script

Evaluates a v-prediction Progressive Distillation checkpoint by generating
samples at specified step counts and computing KID via torch-fidelity.

Two modes:
  --mode kid   Generate samples for a single step count, compute KID.
               Designed to be called in parallel (one Modal container per config).

  --mode sweep Run the full KID sweep across all step counts in sequence
               on a single GPU. Writes kid_pd_vs_nfe.csv.

Default step counts: 1, 2, 3, 4, 5, 10

Usage:
    # Single step KID (called by Modal worker)
    python evaluate_pd.py --mode kid \\
        --checkpoint /data/logs/pd_distill/.../pd_stage5_final.pt \\
        --num_steps 1 \\
        --num_samples 5000 --batch_size 32 \\
        --output_dir /data/benchmark/pd_1 \\
        --dataset_path /data/celeba_images

    # Full sweep on one GPU
    python evaluate_pd.py --mode sweep \\
        --checkpoint /data/logs/pd_distill/.../pd_stage5_final.pt \\
        --step_counts 1,2,3,4,5,10 \\
        --num_samples 5000 --batch_size 32 \\
        --dataset_path /data/celeba_images \\
        --output_csv benchmark/kid_pd_vs_nfe.csv
"""

import os
import sys
import csv
import argparse
import subprocess

import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pd_model(checkpoint_path, device):
    """Load a PD checkpoint, apply EMA, return (model, alphas_cumprod, config)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from src.models import create_model_from_config
    from src.utils import EMA

    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]

    model = create_model_from_config(config).to(device)
    model.load_state_dict(ckpt["model"])

    if "ema" in ckpt:
        ema = EMA(model, decay=config["training"]["ema_decay"])
        ema.load_state_dict(ckpt["ema"])
        ema.apply_shadow()

    model.eval()

    ddpm_cfg = config["ddpm"]
    betas = torch.linspace(
        ddpm_cfg["beta_start"], ddpm_cfg["beta_end"],
        ddpm_cfg["num_timesteps"], dtype=torch.float32,
    )
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0).to(device)

    data_cfg = config["data"]
    image_shape = (
        data_cfg["channels"], data_cfg["image_size"], data_cfg["image_size"]
    )
    return model, alphas_cumprod, image_shape, config


@torch.no_grad()
def ddim_sample_v_pred(model, alphas_cumprod, image_shape, num_steps, batch_size, device):
    """DDIM sampling with a v-prediction student model.

    Converts v -> x0 -> eps at each step so the DDIM update is correct.
    """
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


def save_individual_images(samples, output_dir, start_idx=0):
    """Save batch of samples as individual PNG files."""
    from src.data import unnormalize, save_image

    samples = unnormalize(samples)
    for i in range(samples.shape[0]):
        path = os.path.join(output_dir, f"{start_idx + i:06d}.png")
        save_image(samples[i: i + 1], path, nrow=1)


def run_fidelity_kid(output_dir, dataset_path, batch_size):
    """Run torch-fidelity KID and return (kid_mean, kid_std)."""
    fidelity_cmd = [
        "fidelity",
        "--gpu", "0",
        "--batch-size", str(batch_size),
        "--kid",
        "--input1", output_dir,
        "--input2", dataset_path,
    ]
    result = subprocess.run(fidelity_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"fidelity stderr: {result.stderr}", file=sys.stderr)
        result.check_returncode()

    kid_mean, kid_std = None, None
    for line in result.stdout.strip().splitlines():
        if "kernel_inception_distance_mean" in line:
            kid_mean = float(line.split(":")[-1].strip())
        elif "kernel_inception_distance_std" in line:
            kid_std = float(line.split(":")[-1].strip())

    if kid_mean is None:
        print(f"Could not parse KID from output:\n{result.stdout}", file=sys.stderr)
        sys.exit(1)

    return kid_mean, kid_std


def run_fidelity_fid(output_dir, dataset_path, batch_size):
    """Run torch-fidelity FID and return fid_value."""
    fidelity_cmd = [
        "fidelity",
        "--gpu", "0",
        "--batch-size", str(batch_size),
        "--fid",
        "--input1", output_dir,
        "--input2", dataset_path,
    ]
    result = subprocess.run(fidelity_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"fidelity stderr: {result.stderr}", file=sys.stderr)
        result.check_returncode()

    fid_value = None
    for line in result.stdout.strip().splitlines():
        if "frechet_inception_distance" in line:
            fid_value = float(line.split(":")[-1].strip())

    if fid_value is None:
        print(f"Could not parse FID from output:\n{result.stdout}", file=sys.stderr)
        sys.exit(1)

    return fid_value


# =========================================================================
# Mode: kid (single step count — used by Modal workers)
# =========================================================================

def run_kid_mode(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphas_cumprod, image_shape, config = load_pd_model(args.checkpoint, device)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating {args.num_samples} samples "
          f"(v-pred PD, {args.num_steps} steps)...")
    remaining = args.num_samples
    sample_idx = 0

    with torch.no_grad():
        pbar = tqdm(total=args.num_samples, desc="Sampling")
        while remaining > 0:
            bs = min(args.batch_size, remaining)
            samples = ddim_sample_v_pred(
                model, alphas_cumprod, image_shape,
                num_steps=args.num_steps,
                batch_size=bs,
                device=device,
            )
            save_individual_images(samples, args.output_dir, start_idx=sample_idx)
            sample_idx += bs
            remaining -= bs
            pbar.update(bs)
        pbar.close()

    print("Running torch-fidelity KID...")
    kid_mean, kid_std = run_fidelity_kid(
        args.output_dir, args.dataset_path, args.batch_size)

    result_line = f"pd_vpred,{args.num_steps},{args.num_steps},{kid_mean},{kid_std}"
    print(f"RESULT:{result_line}")
    return result_line


# =========================================================================
# Mode: sweep (all step counts on one GPU)
# =========================================================================

def run_sweep_mode(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphas_cumprod, image_shape, config = load_pd_model(args.checkpoint, device)

    step_counts = [int(s) for s in args.step_counts.split(",")]
    rows = []

    for num_steps in step_counts:
        output_dir = os.path.join(
            os.path.dirname(args.output_csv),
            f"pd_generated_{num_steps}steps",
        )
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Steps={num_steps}: generating {args.num_samples} samples...")
        print(f"{'=' * 60}")

        remaining = args.num_samples
        sample_idx = 0

        with torch.no_grad():
            pbar = tqdm(total=args.num_samples, desc=f"Sampling @ {num_steps} steps")
            while remaining > 0:
                bs = min(args.batch_size, remaining)
                samples = ddim_sample_v_pred(
                    model, alphas_cumprod, image_shape,
                    num_steps=num_steps,
                    batch_size=bs,
                    device=device,
                )
                save_individual_images(samples, output_dir, start_idx=sample_idx)
                sample_idx += bs
                remaining -= bs
                pbar.update(bs)
            pbar.close()

        print(f"Running torch-fidelity KID for {num_steps} steps...")
        kid_mean, kid_std = run_fidelity_kid(
            output_dir, args.dataset_path, args.batch_size)

        print(f"  steps={num_steps}  KID={kid_mean:.6f} ± {kid_std:.6f}")
        rows.append(("pd_vpred", num_steps, num_steps, kid_mean, kid_std))

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "num_steps", "nfe", "kid_mean", "kid_std"])
        for row in rows:
            writer.writerow(row)

    print(f"\nKID results written to {args.output_csv}")
    for row in rows:
        print(f"  steps={row[1]:>2d}  KID={row[3]:.6f} ± {row[4]:.6f}")


# =========================================================================
# Mode: fid (single step count — FID evaluation)
# =========================================================================

def run_fid_mode(args):
    import shutil

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphas_cumprod, image_shape, config = load_pd_model(args.checkpoint, device)

    if os.path.isdir(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    print(f"Generating {args.num_samples} samples "
          f"(v-pred PD, {args.num_steps} steps) for FID...")
    remaining = args.num_samples
    sample_idx = 0

    with torch.no_grad():
        pbar = tqdm(total=args.num_samples, desc="Sampling")
        while remaining > 0:
            bs = min(args.batch_size, remaining)
            samples = ddim_sample_v_pred(
                model, alphas_cumprod, image_shape,
                num_steps=args.num_steps,
                batch_size=bs,
                device=device,
            )
            save_individual_images(samples, args.output_dir, start_idx=sample_idx)
            sample_idx += bs
            remaining -= bs
            pbar.update(bs)
        pbar.close()

    print("Running torch-fidelity FID...")
    fid_value = run_fidelity_fid(
        args.output_dir, args.dataset_path, args.batch_size)

    print(f"FID = {fid_value:.4f}  (steps={args.num_steps})")
    print(f"RESULT:pd_vpred,{args.num_steps},{fid_value}")


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Progressive Distillation evaluation (KID / FID)")
    parser.add_argument("--mode", required=True, choices=["kid", "sweep", "fid"],
                        help="kid: single KID; sweep: KID across steps; fid: single FID")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to PD final checkpoint (pd_stage5_final.pt)")
    parser.add_argument("--num_steps", type=int, default=1,
                        help="Step count for --mode kid / fid")
    parser.add_argument("--step_counts", type=str, default="1,2,3,4,5,10",
                        help="Comma-separated step counts for --mode sweep")
    parser.add_argument("--num_samples", type=int, default=5000,
                        help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for generation and fidelity")
    parser.add_argument("--output_dir", type=str, default="benchmark/pd_generated",
                        help="Output dir for generated images (--mode kid / fid)")
    parser.add_argument("--output_csv", type=str, default="benchmark/kid_pd_vs_nfe.csv",
                        help="Output CSV path (--mode sweep)")
    parser.add_argument("--dataset_path", type=str, default="/data/celeba_images",
                        help="Path to real CelebA images for reference")

    args = parser.parse_args()

    if args.mode == "kid":
        run_kid_mode(args)
    elif args.mode == "sweep":
        run_sweep_mode(args)
    elif args.mode == "fid":
        run_fid_mode(args)


if __name__ == "__main__":
    main()
