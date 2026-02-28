"""
DWS-RX-DDIM KID Evaluation — Tau Sweep

Evaluates DWS-RX-DDIM at varying tau values with fixed n_rx_steps=1
(total NFE = 2 for every config).  The tau parameter controls how much
the distilled warm-start gets re-noised before the single RX refinement step.

Generates 5000 samples per tau, computes KID via torch-fidelity, writes
kid_dws_tau_sweep.csv.

Usage:
    python evaluate_tau_sweep.py \\
        --base_checkpoint /data/.../ddpm_final.pt \\
        --distilled_checkpoint /data/.../pd_stage5_final.pt \\
        --tau_values 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \\
        --n_rx_steps 1 \\
        --num_samples 5000 --batch_size 32 \\
        --dataset_path /data/celeba_images \\
        --output_csv /data/benchmark/kid_dws_tau_sweep.csv
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
    parser = argparse.ArgumentParser(
        description="DWS-RX-DDIM KID sweep over tau (fixed n_rx_steps)")
    parser.add_argument("--base_checkpoint", required=True,
                        help="Path to original DDPM checkpoint")
    parser.add_argument("--distilled_checkpoint", required=True,
                        help="Path to PD distilled checkpoint (v-prediction)")
    parser.add_argument("--tau_values", type=str,
                        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
                        help="Comma-separated tau values to sweep")
    parser.add_argument("--n_rx_steps", type=int, default=1,
                        help="Fixed number of RX refinement steps (default 1)")
    parser.add_argument("--k", type=int, default=2,
                        help="RX extrapolation interval")
    parser.add_argument("--extrapolation", type=str, default="standard",
                        choices=["standard", "cng", "do"])
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataset_path", type=str, default="/data/celeba_images")
    parser.add_argument("--output_csv", type=str,
                        default="benchmark/kid_dws_tau_sweep.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method, image_shape, config = load_base_model(args.base_checkpoint, device)
    distilled_model = load_distilled_model(args.distilled_checkpoint, config, device)

    tau_values = [float(t) for t in args.tau_values.split(",")]
    nfe = 1 + args.n_rx_steps
    rows = []

    for tau in tau_values:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(args.output_csv)),
            f"dws_tau_{tau:.2f}",
        )
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        print(f"\n{'='*60}")
        print(f"tau={tau:.2f}  n_rx_steps={args.n_rx_steps}  NFE={nfe}")
        print(f"Generating {args.num_samples} DWS-RX-DDIM samples...")
        print(f"{'='*60}")

        remaining = args.num_samples
        sample_idx = 0

        with torch.no_grad():
            pbar = tqdm(total=args.num_samples, desc=f"DWS tau={tau:.2f}")
            while remaining > 0:
                bs = min(args.batch_size, remaining)
                samples = method.dws_rx_ddim_sample(
                    distilled_model=distilled_model,
                    batch_size=bs,
                    image_shape=image_shape,
                    tau=tau,
                    n_rx_steps=args.n_rx_steps,
                    k=args.k,
                    extrapolation=args.extrapolation,
                )
                save_individual_images(samples, output_dir, start_idx=sample_idx)
                sample_idx += bs
                remaining -= bs
                pbar.update(bs)
            pbar.close()

        print(f"Running torch-fidelity KID (tau={tau:.2f})...")
        kid_mean, kid_std = run_fidelity_kid(
            output_dir, args.dataset_path, args.batch_size)

        print(f"  tau={tau:.2f}  NFE={nfe}  KID={kid_mean:.6f} +/- {kid_std:.6f}")
        rows.append(("dws_rx_ddim", tau, args.n_rx_steps, nfe, kid_mean, kid_std))

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "tau", "n_rx_steps", "nfe", "kid_mean", "kid_std"])
        for row in rows:
            writer.writerow(row)

    print(f"\nResults written to {args.output_csv}")
    for row in rows:
        print(f"  tau={row[1]:.2f}  NFE={row[3]}  KID={row[4]:.6f}")


if __name__ == "__main__":
    main()
