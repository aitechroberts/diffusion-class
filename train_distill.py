"""
Progressive Distillation Training Script (v3 — v-prediction, DDIM teacher)

Faithful implementation of Salimans & Ho (ICLR 2022), Algorithm 2:
  - Plain DDIM teacher (2 steps per block, epsilon-prediction)
  - Student uses v-prediction parameterization for gradient stability
  - Distillation target x-tilde via DDIM inversion (Appendix G, Eq 43)
  - Loss: unweighted MSE in v-space (= SNR+1 weighted x-space MSE)
  - Clean 2x halving per stage

Stage plan (5 stages, starting from 32-step DDIM):
    32 -> 16 -> 8 -> 4 -> 2 -> 1

Checkpoint saving:
    Every 200 iterations for ALL stages (for LCSC).

NaN safeguards:
    Skip gradient updates on NaN/Inf loss, abort after 50 consecutive.

Usage:
    python train_distill.py --config configs/pd_distill.yaml

    python train_distill.py --config configs/pd_distill.yaml \\
        --teacher_checkpoint /data/checkpoints/ddpm_final.pt
"""

import math
import os
import sys
import argparse
import copy
import time
from datetime import datetime
from typing import List, Optional

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm import tqdm

import wandb
from PIL import Image as PILImage

from src.models import create_model_from_config
from src.data import create_dataloader_from_config, unnormalize, save_image
from src.methods import DDPM
from src.utils import EMA


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, optimizer, ema, scaler, step, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "config": config,
    }
    if ema is not None:
        checkpoint["ema"] = ema.state_dict()
    torch.save(checkpoint, path)
    print(f"  Saved checkpoint to {path}")


def load_teacher_model(checkpoint_path, config, device):
    """Load a teacher model from a standard DDPM checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)

    teacher = create_model_from_config(config).to(device)

    if "model" in ckpt:
        teacher.load_state_dict(ckpt["model"])
    else:
        teacher.load_state_dict(ckpt)

    if "ema" in ckpt:
        ema = EMA(teacher, decay=config["training"]["ema_decay"])
        ema.load_state_dict(ckpt["ema"])
        ema.apply_shadow()

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    return teacher


# ---------------------------------------------------------------------------
# Teacher: 2-step DDIM (epsilon-prediction, per Algorithm 2)
# ---------------------------------------------------------------------------

@torch.no_grad()
def teacher_two_step_ddim(
    teacher: nn.Module,
    z_t: torch.Tensor,
    t_start: int,
    t_mid: int,
    t_end: int,
    alphas_cumprod: torch.Tensor,
    teacher_is_v_pred: bool = False,
) -> torch.Tensor:
    """Run 2 DDIM steps with the teacher: t_start -> t_mid -> t_end.

    Returns z_t'' (the final noisy latent after 2 steps).

    When teacher_is_v_pred=False the teacher output is decoded as epsilon.
    When teacher_is_v_pred=True the teacher output is decoded as v, where
    v = alpha*eps - sigma*x0, so x0 = alpha*z - sigma*v and
    eps = sigma*z + alpha*v.
    """
    bs = z_t.shape[0]
    device = z_t.device

    # Step 1: t_start -> t_mid
    abar_s = alphas_cumprod[t_start]
    abar_m = alphas_cumprod[t_mid]
    alpha_s = torch.sqrt(abar_s)
    sig_s = torch.sqrt(1.0 - abar_s)

    t_batch = torch.full((bs,), t_start, device=device, dtype=torch.long)
    out1 = teacher(z_t, t_batch)

    if teacher_is_v_pred:
        x0_1 = alpha_s * z_t - sig_s * out1
        eps1 = sig_s * z_t + alpha_s * out1
    else:
        eps1 = out1
        x0_1 = (z_t - sig_s * eps1) / alpha_s

    alpha_m = torch.sqrt(abar_m)
    sig_m = torch.sqrt(1.0 - abar_m)
    z_mid = alpha_m * x0_1 + sig_m * eps1

    # Step 2: t_mid -> t_end
    abar_e = alphas_cumprod[t_end]
    sig_e = torch.sqrt(1.0 - abar_e)

    t_batch_m = torch.full((bs,), t_mid, device=device, dtype=torch.long)
    out2 = teacher(z_mid, t_batch_m)

    if teacher_is_v_pred:
        x0_2 = alpha_m * z_mid - sig_m * out2
        eps2 = sig_m * z_mid + alpha_m * out2
    else:
        eps2 = out2
        x0_2 = (z_mid - sig_m * eps2) / alpha_m

    z_end = torch.sqrt(abar_e) * x0_2 + sig_e * eps2

    return z_end


# ---------------------------------------------------------------------------
# DDIM inversion: compute distillation target x-tilde (Appendix G, Eq 43)
# ---------------------------------------------------------------------------

def compute_x_tilde(z_t, z_end, abar_start, abar_end):
    """DDIM inversion target: the x the student must predict so that its
    single DDIM step from t_start to t_end matches the teacher's z_end.

    x_tilde = (z_end - (sig_end/sig_start)*z_t)
              / (sqrt(abar_end) - (sig_end/sig_start)*sqrt(abar_start))
    """
    sig_s = torch.sqrt(1.0 - abar_start)
    sig_e = torch.sqrt(1.0 - abar_end)
    sig_ratio = sig_e / sig_s
    alpha_s = torch.sqrt(abar_start)
    alpha_e = torch.sqrt(abar_end)
    return (z_end - sig_ratio * z_t) / (alpha_e - sig_ratio * alpha_s)


# ---------------------------------------------------------------------------
# v-prediction helpers
# ---------------------------------------------------------------------------

def x_to_v(z_t, x, abar):
    """Convert x-prediction target to v-space: v = (alpha*z_t - x) / sigma."""
    return (torch.sqrt(abar) * z_t - x) / torch.sqrt(1.0 - abar)


@torch.no_grad()
def ddim_sample_v_pred(model, alphas_cumprod, image_shape, num_steps, device):
    """DDIM sampling where the model predicts v instead of epsilon."""
    num_timesteps = len(alphas_cumprod)
    schedule = torch.linspace(num_timesteps - 1, 0, num_steps + 1).long().to(device)
    bs = image_shape[0]
    z = torch.randn(bs, *image_shape[1:], device=device)

    for i in range(num_steps):
        t = schedule[i].item()
        t_next = schedule[i + 1].item()

        abar_t = alphas_cumprod[t]
        abar_next = alphas_cumprod[t_next]

        t_batch = torch.full((bs,), t, device=device, dtype=torch.long)
        v = model(z, t_batch)

        x0 = torch.sqrt(abar_t) * z - torch.sqrt(1.0 - abar_t) * v
        eps = torch.sqrt(1.0 - abar_t) * z + torch.sqrt(abar_t) * v

        z = torch.sqrt(abar_next) * x0 + torch.sqrt(1.0 - abar_next) * eps

    return z


# ---------------------------------------------------------------------------
# Core PD training for one stage (Algorithm 2, Salimans & Ho 2022)
# ---------------------------------------------------------------------------

def train_one_stage(
    stage_idx: int,
    teacher: nn.Module,
    student: nn.Module,
    teacher_steps: int,
    student_steps: int,
    dataloader,
    config: dict,
    device: torch.device,
    log_dir: str,
    start_step: int = 0,
    optimizer=None,
    ema=None,
    scaler=None,
    wandb_run=None,
    global_step_offset: int = 0,
    teacher_is_v_pred: bool = False,
):
    """Train one PD stage following Algorithm 2.

    The teacher's N-step schedule is divided into student_steps blocks of
    exactly 2 teacher steps each (clean 2x halving).  For each block the
    teacher runs 2 DDIM steps; we compute the DDIM-inversion target x-tilde
    (Eq 43), convert to v-tilde, and train the student's v-prediction via
    unweighted MSE in v-space (= SNR+1 weighted x-space MSE).
    """
    training_cfg = config["training"]
    ddpm_cfg = config["ddpm"]
    distill_cfg = config["distillation"]
    num_iters = training_cfg["num_iterations"]
    log_every = training_cfg.get("log_every", 100)
    gradient_clip = training_cfg["gradient_clip_norm"]
    mixed_precision = config["infrastructure"]["mixed_precision"]

    num_timesteps = ddpm_cfg["num_timesteps"]
    beta_start = ddpm_cfg["beta_start"]
    beta_end = ddpm_cfg["beta_end"]

    betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)

    save_every = distill_cfg.get("save_every", 200)
    max_nan = distill_cfg.get("max_nan_consecutive", 50)
    sample_every = training_cfg.get("sample_every", 5000)
    num_samples = training_cfg.get("num_samples", 16)

    data_cfg = config["data"]
    image_shape = (data_cfg["channels"], data_cfg["image_size"], data_cfg["image_size"])

    samples_dir = os.path.join(log_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    N = teacher_steps
    teacher_schedule = torch.linspace(
        num_timesteps - 1, 0, N + 1).long().to(device)

    n_blocks = student_steps  # = N // 2

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=training_cfg["learning_rate"],
            betas=tuple(training_cfg["betas"]),
            weight_decay=training_cfg["weight_decay"],
        )
    if ema is None:
        ema = EMA(student, decay=training_cfg["ema_decay"])
    if scaler is None:
        device_type = "cuda" if device.type == "cuda" else "cpu"
        scaler = GradScaler(device_type, enabled=mixed_precision)

    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    student.train()
    data_iter = iter(dataloader)
    epoch = 0
    loss_accum = 0.0
    loss_count = 0
    nan_count = 0
    t0 = time.time()
    device_type = "cuda" if device.type == "cuda" else "cpu"

    pbar = tqdm(
        range(start_step, num_iters), initial=start_step, total=num_iters,
        desc=f"Stage {stage_idx + 1} ({N}->{student_steps} v-pred PD)")

    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            data_iter = iter(dataloader)
            batch = next(data_iter)

        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        x_0 = batch.to(device)

        block_idx = torch.randint(0, n_blocks, (1,)).item()
        idx_s = 2 * block_idx
        t_start = teacher_schedule[idx_s].item()
        t_mid = teacher_schedule[idx_s + 1].item()
        t_end = teacher_schedule[idx_s + 2].item()

        abar_s = alphas_cumprod[t_start]
        noise = torch.randn_like(x_0)
        z_t = torch.sqrt(abar_s) * x_0 + torch.sqrt(1.0 - abar_s) * noise

        # Teacher: 2 DDIM steps -> z_end
        with torch.no_grad():
            z_end = teacher_two_step_ddim(
                teacher, z_t, t_start, t_mid, t_end, alphas_cumprod,
                teacher_is_v_pred=teacher_is_v_pred)

        # DDIM inversion target x-tilde (Eq 43)
        with torch.no_grad():
            abar_e = alphas_cumprod[t_end]
            x_tilde = compute_x_tilde(z_t, z_end, abar_s, abar_e)
            v_tilde = x_to_v(z_t, x_tilde, abar_s)

        # Student predicts v
        optimizer.zero_grad()
        with autocast(device_type, enabled=mixed_precision):
            bs = z_t.shape[0]
            t_batch = torch.full(
                (bs,), t_start, device=device, dtype=torch.long)
            v_hat = student(z_t, t_batch)
            loss = F.mse_loss(v_hat, v_tilde)

        loss_val = loss.item()
        if not math.isfinite(loss_val):
            nan_count += 1
            if nan_count % 10 == 1:
                print(f"  WARNING: NaN/Inf loss at step {step + 1} "
                      f"(consecutive={nan_count})")
            optimizer.zero_grad(set_to_none=True)
            if wandb_run is not None:
                wandb.log({
                    "train/nan_event": 1,
                    "train/nan_consecutive": nan_count,
                }, step=global_step_offset + step + 1)
            if nan_count >= max_nan:
                print(f"  ABORTING stage: {max_nan} consecutive NaN steps")
                break
            continue
        nan_count = 0

        scaler.scale(loss).backward()
        if gradient_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()

        ema.update()

        loss_accum += loss_val
        loss_count += 1

        if (step + 1) % log_every == 0 and loss_count > 0:
            avg_loss = loss_accum / loss_count
            elapsed = time.time() - t0
            steps_per_sec = loss_count / elapsed
            pbar.set_postfix(loss=f"{avg_loss:.6f}", sps=f"{steps_per_sec:.1f}")

            if wandb_run is not None:
                try:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/steps_per_sec": steps_per_sec,
                        "train/stage": stage_idx + 1,
                        "train/stage_step": step + 1,
                    }, step=global_step_offset + step + 1)
                except Exception as e:
                    print(f"  WARNING: wandb log failed: {e}")

            loss_accum = 0.0
            loss_count = 0
            t0 = time.time()

        if (step + 1) % save_every == 0:
            path = os.path.join(
                ckpt_dir, f"pd_stage{stage_idx + 1}_step{step + 1:07d}.pt")
            ema.apply_shadow()
            save_checkpoint(path, student, optimizer, ema, scaler, step + 1, config)
            ema.restore()

        if (step + 1) % sample_every == 0:
            print(f"\n  Generating {num_samples} v-pred DDIM samples "
                  f"(student @ {student_steps} steps)...")
            ema.apply_shadow()
            student.eval()
            with torch.no_grad():
                samples = ddim_sample_v_pred(
                    student, alphas_cumprod,
                    (num_samples, *image_shape),
                    num_steps=student_steps,
                    device=device,
                )
            samples_unnorm = unnormalize(samples)
            nrow = int(math.sqrt(num_samples))
            sample_path = os.path.join(
                samples_dir,
                f"stage{stage_idx + 1}_step{step + 1:07d}.png")
            save_image(samples_unnorm, sample_path, nrow=nrow)
            print(f"  Saved samples to {sample_path}")

            if wandb_run is not None:
                try:
                    img = PILImage.open(sample_path)
                    wandb.log({
                        "samples": wandb.Image(
                            img,
                            caption=f"Stage {stage_idx+1} "
                                    f"({N}->{student_steps}) step {step+1}"),
                    }, step=global_step_offset + step + 1)
                except Exception as e:
                    print(f"  WARNING: wandb image log failed: {e}")

            ema.restore()
            student.train()

    ema.apply_shadow()
    final_path = os.path.join(ckpt_dir, f"pd_stage{stage_idx + 1}_final.pt")
    save_checkpoint(final_path, student, optimizer, ema, scaler,
                    step + 1 if 'step' in dir() else num_iters, config)
    ema.restore()

    return final_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Progressive Distillation (v-prediction, Algorithm 2)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to pd_distill.yaml config")
    parser.add_argument("--teacher_checkpoint", type=str, default=None,
                        help="Override teacher checkpoint path from config")
    parser.add_argument("--resume_stage", type=int, default=None,
                        help="Resume from this stage index (0-based)")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Checkpoint to resume from within the stage")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        and config["infrastructure"]["device"] != "cpu"
        else "cpu"
    )
    print(f"Using device: {device}")

    seed = config["infrastructure"]["seed"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    distill_cfg = config["distillation"]
    stages = distill_cfg["stages"]
    teacher_ckpt_path = args.teacher_checkpoint or distill_cfg["teacher_checkpoint"]
    log_dir = config["logging"]["dir"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(log_dir, f"pd_distill_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    wandb_run = None
    wandb_cfg = config.get("logging", {}).get("wandb", {})
    if wandb_cfg.get("enabled", False):
        try:
            wandb_run = wandb.init(
                project=wandb_cfg.get("project", "cmu-10799-diffusion"),
                entity=wandb_cfg.get("entity", None),
                name=f"pd_distill_{timestamp}",
                config=config,
                dir=log_dir,
                tags=["pd_distill", "v_pred"],
            )
            print(f"Weights & Biases: {wandb_run.url}")
        except Exception as e:
            print(f"WARNING: Failed to initialize wandb: {e}")

    dataloader = create_dataloader_from_config(config, split="train")
    print(f"Dataset size: {len(dataloader.dataset)}")

    num_iters = config["training"]["num_iterations"]
    print(f"\nProgressive Distillation (v-prediction, Algorithm 2): "
          f"{len(stages)} stages")
    print(f"  Teacher checkpoint: {teacher_ckpt_path}")
    for i, s in enumerate(stages):
        print(f"  Stage {i + 1}: {s['teacher_steps']} -> "
              f"{s['student_steps']} steps")
    print()

    teacher = load_teacher_model(teacher_ckpt_path, config, device)

    start_stage = args.resume_stage or 0

    # Stage 0 teacher is the original DDPM (eps-prediction).
    # Stage 1+ teachers are prior students (v-prediction).
    # When resuming at stage >= 1, --teacher_checkpoint should point to the
    # previous stage's final checkpoint, which is v-prediction.
    teacher_is_v_pred = (start_stage >= 1)
    if teacher_is_v_pred:
        print(f"  Teacher loaded as v-prediction (resume_stage={start_stage})")

    for stage_idx in range(start_stage, len(stages)):
        stage = stages[stage_idx]
        teacher_steps = stage["teacher_steps"]
        student_steps = stage["student_steps"]
        pred_label = "v-pred" if teacher_is_v_pred else "eps-pred"
        print(f"\n{'=' * 60}")
        print(f"STAGE {stage_idx + 1}: {teacher_steps} -> {student_steps} "
              f"steps ({pred_label} teacher, v-pred student)")
        print(f"{'=' * 60}")

        student = create_model_from_config(config).to(device)
        student.load_state_dict(teacher.state_dict())

        start_step = 0
        optimizer = None
        ema = None
        scaler = None

        if args.resume_checkpoint and stage_idx == start_stage:
            print(f"  Resuming from {args.resume_checkpoint}")
            ckpt = torch.load(args.resume_checkpoint, map_location=device)
            student.load_state_dict(ckpt["model"])
            start_step = ckpt.get("step", 0)

            optimizer = torch.optim.AdamW(
                student.parameters(),
                lr=config["training"]["learning_rate"],
                betas=tuple(config["training"]["betas"]),
                weight_decay=config["training"]["weight_decay"],
            )
            optimizer.load_state_dict(ckpt["optimizer"])

            ema = EMA(student, decay=config["training"]["ema_decay"])
            if "ema" in ckpt:
                ema.load_state_dict(ckpt["ema"])

            device_type = "cuda" if device.type == "cuda" else "cpu"
            scaler = GradScaler(
                device_type,
                enabled=config["infrastructure"]["mixed_precision"])
            if "scaler" in ckpt:
                scaler.load_state_dict(ckpt["scaler"])

        global_step_offset = stage_idx * num_iters

        final_path = train_one_stage(
            stage_idx=stage_idx,
            teacher=teacher,
            student=student,
            teacher_steps=teacher_steps,
            student_steps=student_steps,
            dataloader=dataloader,
            config=config,
            device=device,
            log_dir=log_dir,
            start_step=start_step,
            optimizer=optimizer,
            ema=ema,
            scaler=scaler,
            wandb_run=wandb_run,
            global_step_offset=global_step_offset,
            teacher_is_v_pred=teacher_is_v_pred,
        )

        print(f"  Stage {stage_idx + 1} complete -> {final_path}")

        if stage_idx < len(stages) - 1:
            teacher = copy.deepcopy(student)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
            ema_teacher = EMA(teacher, decay=config["training"]["ema_decay"])
            ckpt = torch.load(final_path, map_location=device)
            if "ema" in ckpt:
                ema_teacher.load_state_dict(ckpt["ema"])
                ema_teacher.apply_shadow()
            else:
                teacher.load_state_dict(ckpt["model"])
            # From Stage 2 onward the teacher is always v-prediction
            teacher_is_v_pred = True

    print("\nAll stages complete!")

    if wandb_run is not None:
        try:
            wandb.finish()
        except Exception as e:
            print(f"WARNING: Failed to finish wandb run: {e}")


if __name__ == "__main__":
    main()
