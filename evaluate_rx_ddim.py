"""
RX-DDIM Benchmark Script

Two evaluation modes:
  --mode kid   Generate samples for a single (method, step_count) config,
               compute KID via torch-fidelity, print parseable result line.
               Designed to be called in parallel (one Modal container per config).

  --mode l2    Run full L2 error sweep across all step counts on a single GPU.
               Uses 8 samples with shared initial noise vs 1000-step ground truth.
               Writes l2_vs_nfe.csv.

Usage:
    # KID for one config (called by Modal worker)
    python evaluate_rx_ddim.py --mode kid \\
        --checkpoint ckpt.pt --method rx_ddim --num_steps 10 \\
        --num_samples 5000 --batch_size 32 \\
        --output_dir /data/benchmark/rx_ddim_10 \\
        --dataset_path /data/celeba_images

    # L2 sweep (called once)
    python evaluate_rx_ddim.py --mode l2 \\
        --checkpoint ckpt.pt --rx_step_counts 1,5,10,25,50 \\
        --num_samples_l2 8 --seed 42 \\
        --output_csv benchmark/l2_vs_nfe.csv
"""

import os
import sys
import csv
import argparse
import subprocess

import torch
from tqdm import tqdm


def load_model(checkpoint_path, device):
    """Load checkpoint, apply EMA, return method + image_shape + config."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from src.models import create_model_from_config
    from src.methods import DDPM
    from src.utils import EMA

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model = create_model_from_config(config).to(device)
    model.load_state_dict(checkpoint["model"])

    ema = EMA(model, decay=config["training"]["ema_decay"])
    ema.load_state_dict(checkpoint["ema"])
    ema.apply_shadow()

    method = DDPM.from_config(model, config, device)
    method.eval_mode()

    data_cfg = config["data"]
    image_shape = (data_cfg["channels"], data_cfg["image_size"], data_cfg["image_size"])
    return method, image_shape, config


def load_distilled_model(checkpoint_path, config, device):
    """Load a distilled model checkpoint, apply EMA, return nn.Module."""
    from src.models import create_model_from_config
    from src.utils import EMA

    ckpt = torch.load(checkpoint_path, map_location=device)
    distilled = create_model_from_config(config).to(device)
    distilled.load_state_dict(ckpt["model"])

    if "ema" in ckpt:
        ema = EMA(distilled, decay=config["training"]["ema_decay"])
        ema.load_state_dict(ckpt["ema"])
        ema.apply_shadow()

    distilled.eval()
    return distilled


def save_individual_images(samples, output_dir, start_idx=0):
    """Save batch of samples as individual PNG files."""
    from src.data import unnormalize, save_image

    samples = unnormalize(samples)
    for i in range(samples.shape[0]):
        path = os.path.join(output_dir, f"{start_idx + i:06d}.png")
        save_image(samples[i : i + 1], path, nrow=1)


# =========================================================================
# Mode: kid
# =========================================================================

def run_kid_mode(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method, image_shape, config = load_model(args.checkpoint, device)

    distilled_model = None
    if args.method == "dws_rx_ddim":
        if args.distilled_checkpoint is None:
            raise ValueError("--distilled_checkpoint is required for dws_rx_ddim")
        distilled_model = load_distilled_model(
            args.distilled_checkpoint, config, device)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating {args.num_samples} samples: method={args.method}, steps={args.num_steps}")
    remaining = args.num_samples
    sample_idx = 0

    with torch.no_grad():
        pbar = tqdm(total=args.num_samples, desc="Generating")
        while remaining > 0:
            bs = min(args.batch_size, remaining)
            if args.method == "rx_ddim":
                samples = method.rx_ddim_sample(
                    batch_size=bs, image_shape=image_shape,
                    num_steps=args.num_steps, k=args.k,
                )
            elif args.method == "cng_rx_ddim":
                samples = method.cng_rx_ddim_sample(
                    batch_size=bs, image_shape=image_shape,
                    num_steps=args.num_steps, k=args.k,
                    tau=args.tau, s_param=args.s_param,
                )
            elif args.method == "do_rx_ddim":
                samples = method.do_rx_ddim_sample(
                    batch_size=bs, image_shape=image_shape,
                    num_steps=args.num_steps, k=args.k,
                    p1=args.p1, p2=args.p2,
                    do_threshold=args.do_threshold,
                )
            elif args.method == "dws_rx_ddim":
                samples = method.dws_rx_ddim_sample(
                    distilled_model=distilled_model,
                    batch_size=bs, image_shape=image_shape,
                    tau=args.dws_tau, n_rx_steps=args.n_rx_steps,
                    k=args.k, extrapolation=args.extrapolation,
                    tau_cng=args.tau, s_param=args.s_param,
                    p1=args.p1, p2=args.p2,
                    do_threshold=args.do_threshold,
                )
            else:
                samples = method.ddim_sample(
                    batch_size=bs, image_shape=image_shape,
                    num_steps=args.num_steps,
                )
            save_individual_images(samples, args.output_dir, start_idx=sample_idx)
            sample_idx += bs
            remaining -= bs
            pbar.update(bs)
        pbar.close()

    print(f"Running torch-fidelity KID...")
    fidelity_cmd = [
        "fidelity",
        "--gpu", "0",
        "--batch-size", str(args.batch_size),
        "--kid",
        "--input1", args.output_dir,
        "--input2", args.dataset_path,
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

    nfe = args.num_steps
    result_line = f"{args.method},{args.num_steps},{nfe},{kid_mean},{kid_std}"
    print(f"RESULT:{result_line}")
    return result_line


# =========================================================================
# Mode: l2
# =========================================================================

def run_l2_mode(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method, image_shape, config = load_model(args.checkpoint, device)

    distilled_model = None
    if args.distilled_checkpoint is not None:
        distilled_model = load_distilled_model(
            args.distilled_checkpoint, config, device)

    rx_steps = [int(s) for s in args.rx_step_counts.split(",")]
    n_samples = args.num_samples_l2

    print(f"L2 benchmark: {n_samples} samples, rx_steps={rx_steps}")
    print(f"Ground truth: DDIM @ 1000 steps")

    with torch.no_grad():
        # Generate ground truth with fixed seed
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        gt = method.ddim_sample(
            batch_size=n_samples, image_shape=image_shape, num_steps=1000,
        )
        gt_flat = gt.reshape(n_samples, -1)
        gt_norm = torch.norm(gt_flat, dim=1)

        def _seed():
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(args.seed)

        def _record(name, n_steps, out):
            flat = out.reshape(n_samples, -1)
            l2 = torch.norm(flat - gt_flat, dim=1) / gt_norm
            mse = torch.mean((flat - gt_flat) ** 2, dim=1)
            rows.append((
                name, n_steps, n_steps,
                l2.mean().item(), l2.std().item(),
                mse.mean().item(), mse.std().item(),
            ))
            print(f"  {name:<14s} steps={n_steps:>3d}  L2={l2.mean():.6f}")

        rows = []
        for rx_n in rx_steps:
            ddim_n = 2 * rx_n

            _seed()
            _record("rx_ddim", rx_n, method.rx_ddim_sample(
                batch_size=n_samples, image_shape=image_shape,
                num_steps=rx_n, k=args.k,
            ))

            _seed()
            _record("cng_rx_ddim", rx_n, method.cng_rx_ddim_sample(
                batch_size=n_samples, image_shape=image_shape,
                num_steps=rx_n, k=args.k,
                tau=args.tau, s_param=args.s_param,
            ))

            _seed()
            _record("do_rx_ddim", rx_n, method.do_rx_ddim_sample(
                batch_size=n_samples, image_shape=image_shape,
                num_steps=rx_n, k=args.k,
                p1=args.p1, p2=args.p2,
                do_threshold=args.do_threshold,
            ))

            _seed()
            _record("ddim", ddim_n, method.ddim_sample(
                batch_size=n_samples, image_shape=image_shape,
                num_steps=ddim_n,
            ))

            if distilled_model is not None:
                _seed()
                dws_nfe = 1 + rx_n  # 1 distilled + rx_n refinement
                _record("dws_rx_ddim", dws_nfe, method.dws_rx_ddim_sample(
                    distilled_model=distilled_model,
                    batch_size=n_samples, image_shape=image_shape,
                    tau=args.dws_tau, n_rx_steps=rx_n,
                    k=args.k, extrapolation=args.extrapolation,
                    tau_cng=args.tau, s_param=args.s_param,
                    p1=args.p1, p2=args.p2,
                    do_threshold=args.do_threshold,
                ))

    # Sort by nfe for cleaner output
    rows.sort(key=lambda r: (r[2], r[0]))

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "num_steps", "nfe", "l2_mean", "l2_std", "mse_mean", "mse_std"])
        for row in rows:
            writer.writerow(row)

    print(f"\nL2 results written to {args.output_csv}")


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="RX-DDIM benchmark evaluation")
    parser.add_argument("--mode", required=True, choices=["kid", "l2"],
                        help="Evaluation mode: kid (single config) or l2 (full sweep)")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--k", type=int, default=2, help="RX-DDIM extrapolation interval")

    # KID mode args
    parser.add_argument("--method", type=str, default="ddim",
                        choices=["ddim", "rx_ddim", "cng_rx_ddim", "do_rx_ddim", "dws_rx_ddim"])
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="benchmark/generated")
    parser.add_argument("--dataset_path", type=str, default="/data/celeba_images")

    # Adaptive RX-DDIM hyperparameters
    parser.add_argument("--tau", type=float, default=0.3,
                        help="CNG-RX-DDIM: sigmoid centre for condition-number gate")
    parser.add_argument("--s_param", type=float, default=0.1,
                        help="CNG-RX-DDIM: sigmoid sharpness")
    parser.add_argument("--p1", type=int, default=2,
                        help="DO-RX-DDIM: primary assumed error order")
    parser.add_argument("--p2", type=int, default=3,
                        help="DO-RX-DDIM: secondary error order for comparison")
    parser.add_argument("--do_threshold", type=float, default=0.1,
                        help="DO-RX-DDIM: disagreement threshold for gating")

    # DWS-RX-DDIM args
    parser.add_argument("--distilled_checkpoint", type=str, default=None,
                        help="DWS-RX-DDIM: path to distilled model checkpoint")
    parser.add_argument("--dws_tau", type=float, default=0.3,
                        help="DWS-RX-DDIM: re-noising level")
    parser.add_argument("--n_rx_steps", type=int, default=2,
                        help="DWS-RX-DDIM: number of RX refinement steps")
    parser.add_argument("--extrapolation", type=str, default="standard",
                        choices=["standard", "cng", "do"],
                        help="DWS-RX-DDIM: extrapolation strategy")

    # L2 mode args
    parser.add_argument("--rx_step_counts", type=str, default="1,5,10,25,50")
    parser.add_argument("--num_samples_l2", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_csv", type=str, default="benchmark/l2_vs_nfe.csv")

    args = parser.parse_args()

    if args.mode == "kid":
        run_kid_mode(args)
    elif args.mode == "l2":
        run_l2_mode(args)


if __name__ == "__main__":
    main()
