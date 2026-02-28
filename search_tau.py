"""
Tau Hyperparameter Search for DWS-RX-DPM

Grid search over (tau, n_rx_steps) measuring L2 error against a high-quality
DDIM 1000-step ground truth.  Outputs a CSV and prints the optimal config.

Usage:
    python search_tau.py \
        --checkpoint ckpt.pt \
        --distilled_checkpoint pd_final.pt \
        --num_samples 8 --seed 42 \
        --output_csv benchmark/tau_search.csv
"""

import os
import sys
import csv
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import create_model_from_config
from src.methods import DDPM
from src.utils import EMA


def load_model(checkpoint_path, device):
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
    image_shape = (
        data_cfg["channels"],
        data_cfg["image_size"],
        data_cfg["image_size"],
    )
    return method, image_shape, config


def load_distilled(checkpoint_path, config, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    distilled = create_model_from_config(config).to(device)
    distilled.load_state_dict(ckpt["model"])

    if "ema" in ckpt:
        ema = EMA(distilled, decay=config["training"]["ema_decay"])
        ema.load_state_dict(ckpt["ema"])
        ema.apply_shadow()

    distilled.eval()
    return distilled


def main():
    parser = argparse.ArgumentParser(
        description="Grid search for optimal (tau, n_rx_steps)")
    parser.add_argument("--checkpoint", required=True,
                        help="Base DDPM checkpoint")
    parser.add_argument("--distilled_checkpoint", required=True,
                        help="Distilled (PD) model checkpoint")
    parser.add_argument("--tau_values", type=str,
                        default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50",
                        help="Comma-separated tau values to try")
    parser.add_argument("--n_rx_values", type=str, default="2,3",
                        help="Comma-separated n_rx_steps values to try")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--extrapolation", type=str, default="standard",
                        choices=["standard", "cng", "do"])
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_csv", type=str,
                        default="benchmark/tau_search.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    method, image_shape, config = load_model(args.checkpoint, device)
    distilled = load_distilled(args.distilled_checkpoint, config, device)

    tau_vals = [float(t) for t in args.tau_values.split(",")]
    n_rx_vals = [int(n) for n in args.n_rx_values.split(",")]
    n_samples = args.num_samples

    print(f"Grid: {len(tau_vals)} tau x {len(n_rx_vals)} n_rx = "
          f"{len(tau_vals) * len(n_rx_vals)} configs")

    with torch.no_grad():
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        gt = method.ddim_sample(
            batch_size=n_samples, image_shape=image_shape, num_steps=1000)
        gt_flat = gt.reshape(n_samples, -1)
        gt_norm = torch.norm(gt_flat, dim=1)

    rows = []
    best = (None, None, float("inf"))

    for tau in tau_vals:
        for n_rx in n_rx_vals:
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(args.seed)

            with torch.no_grad():
                out = method.dws_rx_ddim_sample(
                    distilled_model=distilled,
                    batch_size=n_samples,
                    image_shape=image_shape,
                    tau=tau,
                    n_rx_steps=n_rx,
                    k=args.k,
                    extrapolation=args.extrapolation,
                )

            flat = out.reshape(n_samples, -1)
            l2 = torch.norm(flat - gt_flat, dim=1) / gt_norm
            l2_mean = l2.mean().item()
            l2_std = l2.std().item()

            rows.append((tau, n_rx, l2_mean, l2_std))
            nfe = 1 + n_rx
            print(f"  tau={tau:.2f}  n_rx={n_rx}  NFE={nfe}  "
                  f"L2={l2_mean:.6f} +/- {l2_std:.6f}")

            if l2_mean < best[2]:
                best = (tau, n_rx, l2_mean)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tau", "n_rx_steps", "l2_mean", "l2_std"])
        for row in rows:
            writer.writerow(row)

    print(f"\nResults written to {args.output_csv}")
    print(f"Optimal: tau={best[0]:.2f}, n_rx_steps={best[1]}, L2={best[2]:.6f}")


if __name__ == "__main__":
    main()
