---
name: v-prediction PD distillation
overview: Rewrite the progressive distillation pipeline following Salimans & Ho (ICLR 2022) Algorithm 2 exactly -- plain DDIM teacher, v-prediction student, DDIM-inversion target x-tilde, truncated-SNR loss weighting, clean 2x-halving stages (32->16->8->4->2->1). Retain all existing infrastructure (W&B, checkpoints every 200, NaN safeguards, sample generation every 5000).
todos:
  - id: rewrite-teacher
    content: "Replace teacher_rx_ddim_block + _ddim_step with teacher_two_step_ddim: runs 2 DDIM steps (eps-prediction), returns z_t'' (the final noisy latent)"
    status: completed
  - id: rewrite-student
    content: "Replace student_one_step_ddim with student_x_from_v: student predicts v, converts to x-hat = sqrt(abar)*z_t - sqrt(1-abar)*v (no DDIM step in student forward pass)"
    status: completed
  - id: rewrite-loss
    content: "Implement Algorithm 2 loss: compute distillation target x-tilde via DDIM inversion (Eq 43), loss = w(lambda_t) * MSE(x_hat, x_tilde) with truncated-SNR weighting"
    status: completed
  - id: simplify-blocks
    content: "Simplify train_one_stage: remove gammas/rx_k, use uniform 2-block boundaries with exact 2x halving, sample time as t=i/N per Algorithm 2"
    status: completed
  - id: update-samples
    content: "Update sample generation: student uses v-prediction DDIM, add v-to-x0 conversion for sampling"
    status: completed
  - id: update-config
    content: "Rewrite pd_distill.yaml: 5 stages (32->16->8->4->2->1), remove rx_k, set teacher_type ddim, 100k iters for final stage"
    status: completed
  - id: update-dws-inference
    content: Update dws_rx_ddim_sample in ddpm.py to decode distilled model output as v-prediction instead of epsilon
    status: completed
  - id: verify-syntax
    content: Run py_compile on train_distill.py and check lints
    status: completed
isProject: false
---

# v-Prediction Progressive Distillation (Salimans & Ho Algorithm 2)

Three files change: [train_distill.py](train_distill.py) (core rewrite), [configs/pd_distill.yaml](configs/pd_distill.yaml) (stages + config), and [src/methods/ddpm.py](src/methods/ddpm.py) (DWS inference v-prediction decode).

## Algorithm 2 from the Paper (Our Reference)

The paper's distillation loop per iteration:

```
x ~ D                                    # sample data
t = i/N, i ~ Cat[1,...,N]               # sample discrete time
eps ~ N(0,I)                             # sample noise
z_t = alpha_t * x + sigma_t * eps        # forward process

# Teacher: 2 DDIM steps
t'  = t - 0.5/N
t'' = t - 1/N
z_t'  = alpha_t' * x_hat_teacher(z_t)  + (sigma_t'/sigma_t)(z_t  - alpha_t  * x_hat_teacher(z_t))
z_t'' = alpha_t'' * x_hat_teacher(z_t') + (sigma_t''/sigma_t')(z_t' - alpha_t' * x_hat_teacher(z_t'))

# Distillation target (DDIM inversion, Appendix G Eq 43)
x_tilde = (z_t'' - (sigma_t''/sigma_t) * z_t) / (alpha_t'' - (sigma_t''/sigma_t) * alpha_t)

# Loss on denoising predictions (NOT z-space outputs)
L = w(lambda_t) * ||x_tilde - x_hat_student(z_t)||^2
```

Key: the loss matches **denoising x-predictions**, not DDIM z-outputs. The target x_tilde is the x-value the student would need to predict to make its single DDIM step match the teacher's two steps.

Notation mapping (paper -> our code):

- `alpha_t` (paper) = `sqrt(alphas_cumprod[t])` (code)
- `sigma_t` (paper) = `sqrt(1 - alphas_cumprod[t])` (code)

## 1. Teacher Function

Replace `teacher_rx_ddim_block` + `_ddim_step` (lines 97-195) with:

```python
@torch.no_grad()
def teacher_two_step_ddim(teacher, z_t, t_start, t_mid, t_end, alphas_cumprod):
    """2 DDIM steps: t_start -> t_mid -> t_end. Returns z_t'' (final noisy latent)."""
    # Step 1: t_start -> t_mid
    a_s, a_m = alphas_cumprod[t_start], alphas_cumprod[t_mid]
    sig_s, sig_m = torch.sqrt(1-a_s), torch.sqrt(1-a_m)
    eps1 = teacher(z_t, t_batch_start)
    x0_1 = (z_t - sig_s * eps1) / torch.sqrt(a_s)
    z_mid = torch.sqrt(a_m) * x0_1 + sig_m * eps1

    # Step 2: t_mid -> t_end
    a_e = alphas_cumprod[t_end]
    sig_e = torch.sqrt(1-a_e)
    eps2 = teacher(z_mid, t_batch_mid)
    x0_2 = (z_mid - sig_m * eps2) / torch.sqrt(a_m)
    z_end = torch.sqrt(a_e) * x0_2 + sig_e * eps2

    return z_end
```

The teacher is always the original epsilon-prediction model (or previous stage's student with EMA applied). It returns `z_t''`, not `x_target`.

## 2. Student: v-Prediction x-hat

Replace `student_one_step_ddim` (lines 202-215) with:

```python
def student_x_from_v(student, z_t, t, alphas_cumprod):
    """Student predicts v, returns x-hat. No DDIM step in the forward pass."""
    abar = alphas_cumprod[t]
    t_batch = torch.full((bs,), t, device=z_t.device, dtype=torch.long)
    v_pred = student(z_t, t_batch)
    x_hat = torch.sqrt(abar) * z_t - torch.sqrt(1.0 - abar) * v_pred
    return x_hat
```

This is the critical fix: `x_hat = sqrt(abar)*z - sqrt(1-abar)*v` has no division by small numbers. Gradients are well-conditioned at all noise levels.

## 3. Loss: DDIM Inversion Target + Truncated SNR

The training loop becomes:

```python
# Forward process
z_t = sqrt(abar_s) * x_0 + sqrt(1-abar_s) * noise

# Teacher 2-step DDIM -> z_t''
z_end = teacher_two_step_ddim(teacher, z_t, t_start, t_mid, t_end, alphas_cumprod)

# DDIM inversion target (Appendix G, Eq 43)
sig_ratio = sqrt(1-abar_end) / sqrt(1-abar_start)
x_tilde = (z_end - sig_ratio * z_t) / (sqrt(abar_end) - sig_ratio * sqrt(abar_start))

# Student denoising prediction via v-prediction
x_hat = student_x_from_v(student, z_t, t_start, alphas_cumprod)

# Truncated SNR weighting (Section 4 of paper)
snr = abar_start / (1.0 - abar_start)
w = max(snr, 1.0)   # = max(alpha^2/sigma^2, 1)
loss = w * F.mse_loss(x_hat, x_tilde)
```

This is a **fundamental change** from our current code which matched z-space outputs with uniform weighting.

## 4. Block Logic Simplification

With exact 2x halving, every stage has `N` teacher steps and `N/2` student steps. The teacher schedule has `N+1` boundary timesteps. Each student block spans indices `[2b, 2b+1, 2b+2]`:

- `t_start = schedule[2b]`
- `t_mid = schedule[2b+1]`
- `t_end = schedule[2b+2]`

Randomly sample block index `b ~ Uniform{0, 1, ..., N/2 - 1}`. Remove all gammas, rx_k, and remainder logic.

## 5. Sample Generation During Training

The student uses v-prediction, so we need a v-prediction-aware sampler. Two options:

- Convert v to (x0, eps) and run standard DDIM: `x0 = sqrt(abar)*z - sqrt(1-abar)*v`, `eps = sqrt(1-abar)*z + sqrt(abar)*v`, then `z_next = sqrt(abar_next)*x0 + sqrt(1-abar_next)*eps`
- This is mathematically identical to DDIM but uses v internally

Implement a small `ddim_sample_v_pred` helper for the sample generation block.

## 6. Config: [configs/pd_distill.yaml](configs/pd_distill.yaml)

```yaml
distillation:
  teacher_checkpoint: "logs/ddpm_modal/ddpm_20260127_162352/checkpoints/ddpm_final.pt"
  teacher_type: "ddim"
  save_every: 200
  max_nan_consecutive: 50

  stages:
    - { teacher_steps: 32, student_steps: 16 }
    - { teacher_steps: 16, student_steps:  8 }
    - { teacher_steps:  8, student_steps:  4 }
    - { teacher_steps:  4, student_steps:  2 }
    - { teacher_steps:  2, student_steps:  1 }
```

Paper uses 50k iterations per stage, 100k for final 2->1 stage. We can handle this per-stage via config or hardcode.

## 7. DWS Inference: [src/methods/ddpm.py](src/methods/ddpm.py)

Lines 681-691: decode distilled model output as v instead of epsilon:

```python
v_d = distilled_model(z, t_batch)
x0_hat = torch.sqrt(abar_T) * z - torch.sqrt(1.0 - abar_T) * v_d
```

The RX-DDIM refinement pass still uses the original epsilon-prediction base model, unchanged.

## What Stays the Same

- `save_checkpoint`, `load_teacher_model` -- unchanged
- NaN safeguards -- unchanged
- Checkpoint saving every 200 iterations -- unchanged
- W&B logging -- unchanged (update tags from "rx_ddim" to "v_pred_pd")
- W&B init/finish in main() -- unchanged
- Sample generation every 5000 iterations -- same structure, new v-pred sampler
- Resume logic -- unchanged
- Stage-to-stage teacher promotion -- unchanged
- EMA, AMP, gradient clipping -- unchanged
- All CLI args -- unchanged

