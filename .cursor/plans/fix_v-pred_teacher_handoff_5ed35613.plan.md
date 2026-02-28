---
name: Fix v-pred teacher handoff
overview: Fix the critical bug where Stage 2+ teachers (which are v-prediction models) are decoded as epsilon-prediction in `teacher_two_step_ddim`, and enable resuming from the existing Stage 1 checkpoint.
todos:
  - id: write-concerns-md
    content: Create DISTILLATION_CONCERNS.md documenting the v-pred teacher decode bug
    status: completed
  - id: fix-teacher-fn
    content: Add teacher_is_v_pred flag to teacher_two_step_ddim with branching decode logic
    status: completed
  - id: fix-train-one-stage
    content: Thread teacher_is_v_pred through train_one_stage to the teacher call
    status: completed
  - id: fix-main-loop
    content: Set teacher_is_v_pred correctly per stage in main() and handle resume_stage >= 1
    status: completed
  - id: fix-modal-dispatch
    content: Forward resume_stage and resume_checkpoint through modal_app.py --action distill dispatch
    status: completed
  - id: verify-syntax
    content: py_compile both files and check lints
    status: completed
isProject: false
---

# Fix v-prediction Teacher Handoff in Progressive Distillation

## Problem

In Progressive Distillation (Salimans & Ho 2022), the original teacher uses epsilon-prediction, but every student trained from Stage 1 onward uses v-prediction. When the Stage 1 student becomes the teacher for Stage 2, `teacher_two_step_ddim` still decodes its output as epsilon:

```python
# Line 122-123 of train_distill.py — WRONG for v-pred teacher
eps1 = teacher(z_t, t_batch)
x0_1 = (z_t - sig_s * eps1) / torch.sqrt(abar_s)
```

The correct decode for a v-prediction teacher is:

```python
v1 = teacher(z_t, t_batch)
x0_1 = torch.sqrt(abar_s) * z_t - sig_s * v1
eps1 = sig_s * z_t + torch.sqrt(abar_s) * v1
```

This caused Stage 2 to produce black/white block artifacts and ~50% NaN rate.

## Changes

### 1. Create `DISTILLATION_CONCERNS.md`

Document the bug, root cause, symptoms, and fix for future reference.

### 2. Fix `teacher_two_step_ddim` in [train_distill.py](train_distill.py) (lines 98-135)

Add a `teacher_is_v_pred: bool = False` parameter. When `True`, decode the model output as v instead of epsilon:

- For each of the 2 DDIM steps, branch on the flag:
  - **eps-pred**: `x0 = (z - sigma * output) / alpha` (existing code)
  - **v-pred**: `x0 = alpha * z - sigma * output` and `eps = sigma * z + alpha * output`
- The rest of the DDIM step (`z_next = alpha_next * x0 + sigma_next * eps`) is identical

### 3. Pass the flag through `train_one_stage` in [train_distill.py](train_distill.py) (line 204)

Add a `teacher_is_v_pred: bool = False` parameter to `train_one_stage` and forward it to `teacher_two_step_ddim` at line 299.

### 4. Set the flag in the stage loop in `main()` ([train_distill.py](train_distill.py), lines 499-559)

- Stage 1 (`stage_idx == 0` and no resume): `teacher_is_v_pred=False` (original DDPM teacher)
- Stage 2+ (`stage_idx >= 1`): `teacher_is_v_pred=True` (prior student)
- Resume case: if `resume_stage >= 1`, the teacher loaded from checkpoint is v-pred, so set `teacher_is_v_pred=True`

The flag is passed to `train_one_stage(...)`.

### 5. Fix resume path: `main()` and `modal_app.py`

Currently `--resume_stage` causes `main()` to load the **original** DDPM teacher at line 495, then skip ahead to the resume stage. But for Stage 2+, the teacher should be the previous stage's final checkpoint, not the original DDPM.

Add a `--teacher_stage_checkpoint` argument (or reuse `--resume_checkpoint`) so that when resuming at Stage 2, we load the Stage 1 final checkpoint as teacher. Concretely for our case:

```
--resume_stage 1 --teacher_checkpoint logs/pd_distill/pd_distill_20260225_204845/checkpoints/pd_stage1_final.pt
```

When `resume_stage >= 1` and `teacher_checkpoint` is provided, use it as the teacher instead of the original DDPM path from config. The teacher loaded this way is v-prediction.

### 6. Update `modal_app.py` dispatch ([modal_app.py](modal_app.py), lines 763-768)

The `--action distill` handler doesn't forward `resume_stage` or `resume_checkpoint`. Add CLI args `resume_stage` and `resume_checkpoint` to `main()` and pass them through to `distill.remote(...)`.

### Resume Command

After fixes, run from Stage 2 onward:

```bash
modal run modal_app.py --action distill \
  --checkpoint logs/pd_distill/pd_distill_20260225_204845/checkpoints/pd_stage1_final.pt \
  --resume-stage 1
```

This sets the Stage 1 v-pred checkpoint as the teacher and starts the loop at `stage_idx=1` (Stage 2: 16->8).