#!/usr/bin/env python3
"""
Benchmark wall-clock inference speed for DDIM and RX-DDIM.

Measures samples/second and seconds/sample for each (method, num_steps)
configuration. Outputs timing_benchmark.csv.

Usage:
    python benchmark_timing.py --checkpoint ckpt.pt
    python benchmark_timing.py --checkpoint ckpt.pt --batch_size 64 --num_warmup 2 --num_trials 5
"""

import os
import sys
import csv
import time
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
    image_shape = (data_cfg["channels"], data_cfg["image_size"], data_cfg["image_size"])
    return method, image_shape


def time_sampling(method, image_shape, sample_fn, batch_size, num_warmup, num_trials, device):
    """Time a sampling function, return (mean_seconds_per_batch, std)."""
    # Warmup (fills CUDA caches, JIT, etc.)
    with torch.no_grad():
        for _ in range(num_warmup):
            sample_fn(method, batch_size, image_shape)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Timed trials
    times = []
    with torch.no_grad():
        for _ in range(num_trials):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            sample_fn(method, batch_size, image_shape)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

    times_t = torch.tensor(times)
    return times_t.mean().item(), times_t.std().item()


def main():
    parser = argparse.ArgumentParser(description="Benchmark DDIM/RX-DDIM wall-clock speed")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_warmup", type=int, default=2, help="Warmup iterations (not timed)")
    parser.add_argument("--num_trials", type=int, default=5, help="Timed iterations to average")
    parser.add_argument("--k", type=int, default=2, help="RX-DDIM extrapolation interval")
    parser.add_argument("--step_counts", type=str, default="1,3,5,10,25,50,100",
                        help="Comma-separated list of step counts to benchmark")
    parser.add_argument("--output_csv", type=str, default="timing_benchmark.csv")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    method, image_shape = load_model(args.checkpoint, device)

    step_counts = [int(s) for s in args.step_counts.split(",")]
    configs = []
    for n in step_counts:
        configs.append(("ddim", n))
        configs.append(("rx_ddim", n))
    configs.sort(key=lambda x: (x[1], x[0]))

    rows = []
    print(f"\nBenchmarking (batch_size={args.batch_size}, "
          f"warmup={args.num_warmup}, trials={args.num_trials})")
    print("-" * 72)
    print(f"{'method':<10} {'steps':>5} {'nfe':>5} {'sec/batch':>12} "
          f"{'samples/sec':>12} {'sec/sample':>12}")
    print("-" * 72)

    for method_name, num_steps in configs:
        nfe = num_steps

        if method_name == "rx_ddim":
            def sample_fn(m, bs, shape, ns=num_steps, k=args.k):
                return m.rx_ddim_sample(batch_size=bs, image_shape=shape, num_steps=ns, k=k)
        else:
            def sample_fn(m, bs, shape, ns=num_steps):
                return m.ddim_sample(batch_size=bs, image_shape=shape, num_steps=ns)

        mean_sec, std_sec = time_sampling(
            method, image_shape, sample_fn,
            args.batch_size, args.num_warmup, args.num_trials, device,
        )

        samples_per_sec = args.batch_size / mean_sec
        sec_per_sample = mean_sec / args.batch_size

        print(f"{method_name:<10} {num_steps:>5} {nfe:>5} "
              f"{mean_sec:>11.4f}s {samples_per_sec:>11.2f} {sec_per_sample:>11.5f}")

        rows.append({
            "method": method_name,
            "num_steps": num_steps,
            "nfe": nfe,
            "batch_time_mean": mean_sec,
            "batch_time_std": std_sec,
            "samples_per_sec": samples_per_sec,
            "sec_per_sample": sec_per_sample,
        })

    print("-" * 72)

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {args.output_csv}")


if __name__ == "__main__":
    main()
