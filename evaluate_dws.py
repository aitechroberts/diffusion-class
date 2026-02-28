"""
DWS-RX-DDIM KID Evaluation — NFE Sweep

Evaluates DWS-RX-DDIM at varying n_rx_steps (NFE = 1 + n_rx_steps) with
a fixed tau.  Uses the original DDPM as the base model and the PD-distilled
v-prediction checkpoint as the 1-step warm-start.

Generates 5000 samples per config, computes KID via torch-fidelity, writes
kid_dws_vs_nfe.csv.

Usage:
    python evaluate_dws.py \\
        --base_checkpoint /data/.../ddpm_final.pt \\
        --distilled_checkpoint /data/.../pd_stage5_final.pt \\
        --rx_step_counts 1,2,3,4,5,10 \\
        --tau 0.3 \\
        --num_samples 5000 --batch_size 32 \\
        --dataset_path /data/celeba_images \\
        --output_csv /data/benchmark/kid_dws_vs_nfe.csv
"""

import os
import sys
import csv
import shutil
import argparse
import subprocess

import torch
from tqdm import tqdm


def load_base_model(checkpoint_path, device):
    """Load the original DDPM checkpoint, apply EMA, return (method, image_shape, config)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from src.models import create_model_from_config
    from src.methods import DDPM
    from src.utils import EMA

    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]

    model = create_model_from_config(config).to(device)
    model.load_state_dict(ckpt["model"])

    if "ema" in ckpt:
        ema = EMA(model, decay=config["training"]["ema_decay"])
        ema.load_state_dict(ckpt["ema"])
        ema.apply_shadow()

    method = DDPM.from_config(model, config, device)
    method.eval_mode()

    data_cfg = config["data"]
    image_shape = (data_cfg["channels"], data_cfg["image_size"], data_cfg["image_size"])
    return method, image_shape, config


def load_distilled_model(checkpoint_path, config, device):
    """Load the PD v-prediction distilled checkpoint, apply EMA."""
    from src.models import create_model_from_config
    from src.utils import EMA

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = create_model_from_config(config).to(device)
    model.load_state_dict(ckpt["model"])

    if "ema" in ckpt:
        ema = EMA(model, decay=config["training"]["ema_decay"])
        ema.load_state_dict(ckpt["ema"])
        ema.apply_shadow()

    model.eval()
    return model


def save_individual_images(samples, output_dir, start_idx=0):
    from src.data import unnormalize, save_image
    samples = unnormalize(samples)
    for i in range(samples.shape[0]):
        path = os.path.join(output_dir, f"{start_idx + i:06d}.png")
        save_image(samples[i:i+1], path, nrow=1)


def run_fidelity_kid(output_dir, dataset_path, batch_size):
    fidelity_cmd = [
        "fidelity", "--gpu", "0",
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
        print(f"Could not parse KID:\n{result.stdout}", file=sys.stderr)
        sys.exit(1)
    return kid_mean, kid_std


def main():
    parser = argparse.ArgumentParser(description="DWS-RX-DDIM KID sweep over n_rx_steps")
    parser.add_argument("--base_checkpoint", required=True,
                        help="Path to original DDPM checkpoint")
    parser.add_argument("--distilled_checkpoint", required=True,
                        help="Path to PD distilled checkpoint (v-prediction)")
    parser.add_argument("--rx_step_counts", type=str, default="1,2,3,4,5,10",
                        help="Comma-separated n_rx_steps values to sweep")
    parser.add_argument("--tau", type=float, default=0.3,
                        help="Re-noising level for DWS")
    parser.add_argument("--k", type=int, default=2,
                        help="RX extrapolation interval")
    parser.add_argument("--extrapolation", type=str, default="standard",
                        choices=["standard", "cng", "do"])
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataset_path", type=str, default="/data/celeba_images")
    parser.add_argument("--output_csv", type=str, default="benchmark/kid_dws_vs_nfe.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method, image_shape, config = load_base_model(args.base_checkpoint, device)
    distilled_model = load_distilled_model(args.distilled_checkpoint, config, device)

    step_counts = [int(s) for s in args.rx_step_counts.split(",")]
    rows = []

    for n_rx in step_counts:
        nfe = 1 + n_rx
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(args.output_csv)),
            f"dws_generated_rx{n_rx}",
        )
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        print(f"\n{'='*60}")
        print(f"n_rx_steps={n_rx}  tau={args.tau}  NFE={nfe}")
        print(f"Generating {args.num_samples} DWS-RX-DDIM samples...")
        print(f"{'='*60}")

        remaining = args.num_samples
        sample_idx = 0

        with torch.no_grad():
            pbar = tqdm(total=args.num_samples, desc=f"DWS rx={n_rx}")
            while remaining > 0:
                bs = min(args.batch_size, remaining)
                samples = method.dws_rx_ddim_sample(
                    distilled_model=distilled_model,
                    batch_size=bs,
                    image_shape=image_shape,
                    tau=args.tau,
                    n_rx_steps=n_rx,
                    k=args.k,
                    extrapolation=args.extrapolation,
                )
                save_individual_images(samples, output_dir, start_idx=sample_idx)
                sample_idx += bs
                remaining -= bs
                pbar.update(bs)
            pbar.close()

        print(f"Running torch-fidelity KID (n_rx={n_rx})...")
        kid_mean, kid_std = run_fidelity_kid(
            output_dir, args.dataset_path, args.batch_size)

        print(f"  n_rx={n_rx}  NFE={nfe}  KID={kid_mean:.6f} +/- {kid_std:.6f}")
        rows.append(("dws_rx_ddim", n_rx, nfe, args.tau, kid_mean, kid_std))

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "n_rx_steps", "nfe", "tau", "kid_mean", "kid_std"])
        for row in rows:
            writer.writerow(row)

    print(f"\nResults written to {args.output_csv}")
    for row in rows:
        print(f"  n_rx={row[1]}  NFE={row[2]}  tau={row[3]}  KID={row[4]:.6f}")


if __name__ == "__main__":
    main()
