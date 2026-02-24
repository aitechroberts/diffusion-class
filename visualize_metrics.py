#!/usr/bin/env python3
"""Visualize L2 and KID metrics vs NFE (and vs wall-clock speed if available)."""

import csv
import os
import matplotlib.pyplot as plt


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def group_by_method(rows, x_key="nfe", mean_key=None, std_key=None):
    groups = {}
    for r in rows:
        m = r["method"]
        if m not in groups:
            groups[m] = []
        groups[m].append(
            (float(r[x_key]), float(r[mean_key]), float(r[std_key]))
        )
    for m in groups:
        groups[m].sort(key=lambda x: x[0])
    return groups


def build_timing_lookup(timing_rows):
    """Map (method, num_steps) -> samples_per_sec."""
    lookup = {}
    for r in timing_rows:
        key = (r["method"], int(r["num_steps"]))
        lookup[key] = float(r["samples_per_sec"])
    return lookup


def merge_with_timing(metric_rows, timing_lookup, mean_key, std_key):
    """Attach samples_per_sec to each metric row, group by method."""
    groups = {}
    for r in metric_rows:
        key = (r["method"], int(r["num_steps"]))
        if key not in timing_lookup:
            continue
        m = r["method"]
        if m not in groups:
            groups[m] = []
        groups[m].append((
            timing_lookup[key],
            float(r[mean_key]),
            float(r[std_key]),
        ))
    for m in groups:
        groups[m].sort(key=lambda x: x[0])
    return groups


# ── Load data ──────────────────────────────────────────────────────────
l2_rows = load_csv("l2_vs_nfe.csv")
kid_rows = load_csv("kid_vs_nfe.csv")

has_timing = os.path.exists("timing_benchmark.csv")
if has_timing:
    timing_rows = load_csv("timing_benchmark.csv")
    timing_lookup = build_timing_lookup(timing_rows)

# ── 1. L2 vs NFE ──────────────────────────────────────────────────────
l2_groups = group_by_method(l2_rows, mean_key="l2_mean", std_key="l2_std")
fig, ax = plt.subplots(figsize=(8, 5))
for method, points in l2_groups.items():
    nfe, mean, std = zip(*points)
    ax.errorbar(nfe, mean, yerr=std, label=method, marker="o", capsize=3)
ax.set_xlabel("NFE (Number of Function Evaluations)")
ax.set_ylabel("L2 Mean")
ax.set_title("L2 vs NFE: rx_ddim vs ddim")
ax.legend()
ax.set_xscale("log")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("l2_vs_nfe.png", dpi=150)
plt.close()
print("Saved l2_vs_nfe.png")

# ── 2. KID vs NFE ─────────────────────────────────────────────────────
kid_groups = group_by_method(kid_rows, mean_key="kid_mean", std_key="kid_std")
fig, ax = plt.subplots(figsize=(8, 5))
for method, points in kid_groups.items():
    nfe, mean, std = zip(*points)
    ax.errorbar(nfe, mean, yerr=std, label=method, marker="o", capsize=3)
ax.set_xlabel("NFE (Number of Function Evaluations)")
ax.set_ylabel("KID Mean")
ax.set_title("KID vs NFE: rx_ddim vs ddim")
ax.legend()
ax.set_xscale("log")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("kid_vs_nfe.png", dpi=150)
plt.close()
print("Saved kid_vs_nfe.png")

# ── 3 & 4. Quality vs Samples/sec (only if timing data exists) ───────
if has_timing:
    # L2 vs samples/sec
    l2_speed = merge_with_timing(l2_rows, timing_lookup, "l2_mean", "l2_std")
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, points in l2_speed.items():
        sps, mean, std = zip(*points)
        ax.errorbar(sps, mean, yerr=std, label=method, marker="o", capsize=3)
    ax.set_xlabel("Samples / Second")
    ax.set_ylabel("L2 Mean (lower is better)")
    ax.set_title("L2 vs Inference Speed: rx_ddim vs ddim")
    ax.legend()
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("l2_vs_speed.png", dpi=150)
    plt.close()
    print("Saved l2_vs_speed.png")

    # KID vs samples/sec
    kid_speed = merge_with_timing(kid_rows, timing_lookup, "kid_mean", "kid_std")
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, points in kid_speed.items():
        sps, mean, std = zip(*points)
        ax.errorbar(sps, mean, yerr=std, label=method, marker="o", capsize=3)
    ax.set_xlabel("Samples / Second")
    ax.set_ylabel("KID Mean (lower is better)")
    ax.set_title("KID vs Inference Speed: rx_ddim vs ddim")
    ax.legend()
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("kid_vs_speed.png", dpi=150)
    plt.close()
    print("Saved kid_vs_speed.png")

    # Timing bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f"{r['method']}\n{r['num_steps']} steps" for r in timing_rows]
    sps_vals = [float(r["samples_per_sec"]) for r in timing_rows]
    colors = ["#1f77b4" if r["method"] == "rx_ddim" else "#ff7f0e" for r in timing_rows]
    bars = ax.bar(labels, sps_vals, color=colors)
    ax.set_ylabel("Samples / Second")
    ax.set_title("Inference Speed by Method and Step Count")
    ax.grid(True, alpha=0.3, axis="y")
    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color="#1f77b4", label="rx_ddim"),
        mpatches.Patch(color="#ff7f0e", label="ddim"),
    ])
    plt.tight_layout()
    plt.savefig("timing_bar.png", dpi=150)
    plt.close()
    print("Saved timing_bar.png")
else:
    print("\nNo timing_benchmark.csv found — skipping speed plots.")
    print("Run: python benchmark_timing.py --checkpoint <ckpt.pt>")
    print("Then re-run this script to get quality-vs-speed plots.")
