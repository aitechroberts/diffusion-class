# Distillation Concerns: v-prediction Teacher Decode Bug

## Bug Summary

Progressive Distillation (Salimans & Ho, ICLR 2022) transitions the model
parameterization from **epsilon-prediction** (the original DDPM teacher) to
**v-prediction** (every student from Stage 1 onward).  When a trained student
becomes the teacher for the next stage, the teacher's output must be decoded
with the **v-prediction** formula — not the epsilon-prediction formula used for
the original DDPM.

Our initial implementation of `teacher_two_step_ddim` hard-coded the
epsilon-prediction decode for *all* stages, causing Stage 2+ to produce
garbage targets and ~50% NaN loss rates.

## Symptoms

- **Stage 1 (32 -> 16):** Worked perfectly.  Teacher is the original DDPM
  (epsilon-prediction), so the hard-coded decode was correct.  Loss converged
  normally; generated samples showed clear CelebA faces.

- **Stage 2 (16 -> 8):** Immediately broken.  Teacher is the Stage 1 student
  (v-prediction), but the decode still treated its output as epsilon.
  - ~50% of training steps produced NaN/Inf loss (skipped by safeguard).
  - Finite loss values were extremely high (700–2000+).
  - Generated samples showed black/white blocks with scattered color artifacts.

## Root Cause

The epsilon-prediction decode is:

```
x0 = (z_t - sigma_t * eps) / alpha_t
```

The v-prediction decode is:

```
x0 = alpha_t * z_t - sigma_t * v
eps = sigma_t * z_t + alpha_t * v
```

where `alpha_t = sqrt(alpha_bar_t)` and `sigma_t = sqrt(1 - alpha_bar_t)`.

When the model outputs `v` but the code applies the epsilon formula, the
resulting `x0` and `eps` are nonsensical.  The DDIM step then produces a
wildly incorrect `z_end`, which feeds into the DDIM inversion formula
(`compute_x_tilde`) to create an extreme or NaN target.  The student's
v-prediction loss against this target either overflows (NaN) or produces
very large gradients that corrupt the model weights.

## Fix

Added a `teacher_is_v_pred` boolean flag to `teacher_two_step_ddim` and
threaded it through `train_one_stage` and `main()`:

- **Stage 1** (`stage_idx == 0`, original DDPM teacher): `teacher_is_v_pred=False`
- **Stage 2+** (`stage_idx >= 1`, prior student is teacher): `teacher_is_v_pred=True`
- **Resume at Stage 2+**: when `--resume_stage >= 1` and `--teacher_checkpoint`
  points to a prior stage's final checkpoint, it is a v-prediction model, so
  `teacher_is_v_pred=True`.

## Lesson

In any multi-stage distillation pipeline where the parameterization changes
between teacher and student, the teacher decode logic **must** be conditioned
on which parameterization the teacher was trained with.  This is especially
critical when the original pre-trained model (epsilon) differs from the
distillation parameterization (v-prediction).
