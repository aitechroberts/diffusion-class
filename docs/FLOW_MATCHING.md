---
name: Flow Matching and DDIM
overview: Implement a Flow Matching method (new training + sampling) and DDIM sampling (no retraining, reuses DDPM checkpoint), then wire both into the training/sampling/Modal pipeline with configs and commands.
todos:
  - id: flow-matching-class
    content: Create src/methods/flow_matching.py with FlowMatching class (compute_loss, sample, from_config)
    status: pending
  - id: ddim-sampling
    content: Add ddim_sample() method to existing DDPM class in src/methods/ddpm.py
    status: pending
  - id: methods-init
    content: Update src/methods/__init__.py to export FlowMatching
    status: pending
  - id: flow-matching-config
    content: Create configs/flow_matching_modal.yaml matching DDPM hyperparameters
    status: pending
  - id: train-py
    content: Update train.py to support flow_matching method
    status: pending
  - id: sample-py
    content: Update sample.py to support flow_matching and ddim sampling
    status: pending
  - id: modal-app
    content: Update modal_app.py to pass through new methods
    status: pending
  - id: modal-commands
    content: Add flow matching and DDIM Modal commands to z_modal_commands.md
    status: pending
isProject: false
---

# Flow Matching + DDIM Implementation Plan

## Background

Flow Matching learns a velocity field `v_theta(x_t, t)` to transport noise to data along straight paths. DDIM is a deterministic sampler that reuses the existing trained DDPM model but generates images in far fewer steps. Neither requires changes to the UNet -- the existing `TimestepEmbedding` already handles continuous time values.

## 1. New File: `src/methods/flow_matching.py`

Create a `FlowMatching` class inheriting from `BaseMethod` with the same interface as DDPM.

**Core algorithm:**

- Convention: `t in [0, 1]`, where `t=0` is data, `t=1` is noise
- Interpolation: `x_t = (1 - t) * x_0 + t * epsilon`, where `epsilon ~ N(0, I)`
- Target velocity: `v_target = epsilon - x_0`
- Training loss: `MSE(v_theta(x_t, t_scaled), v_target)` where `t_scaled = t * 999` (to match UNet's expected timestep range)
- Sampling: Euler integration from `t=1` (noise) to `t=0` (data):
`x_{t-dt} = x_t - dt * v_theta(x_t, t_scaled)`

Key methods:

- `compute_loss(x_0)` -- sample random `t ~ U(0,1)`, compute interpolation and MSE loss against velocity target
- `sample(batch_size, image_shape, num_steps)` -- Euler integration from noise to data
- `from_config(model, config, device)` -- construct from YAML config

The class stores `num_timesteps` (default 1000, used for scaling `t` to integer range for the UNet).

## 2. Add DDIM Sampling to `src/methods/ddpm.py`

Add a `ddim_sample` method to the existing `DDPM` class (no changes to existing methods). The DDIM update rule from the assignment PDF (Algorithm 1):

```python
@torch.no_grad()
def ddim_sample(self, batch_size, image_shape, num_steps=100):
    # Create timestep subsequence evenly spaced from T-1 down to 0
    # tau = [tau_S, tau_{S-1}, ..., tau_1]
    step_indices = torch.linspace(self.num_timesteps - 1, 0, num_steps + 1, dtype=torch.long)
    
    x = torch.randn(batch_size, *image_shape, device=self.device)
    
    for i in range(num_steps):
        t = step_indices[i]
        t_prev = step_indices[i + 1]
        
        t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
        eps = self.model(x, t_batch)
        
        alpha_bar_t = self.alphas_cumprod[t]
        alpha_bar_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)
        
        x0_pred = (x - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)
        x = sqrt(alpha_bar_prev) * x0_pred + sqrt(1 - alpha_bar_prev) * eps
    
    return x
```

## 3. Update `src/methods/__init__.py`

Add `FlowMatching` to the imports and `__all__`.

## 4. New Config: `configs/flow_matching_modal.yaml`

Copy from `ddpm_modal.yaml` and replace the `ddpm:` section with:

```yaml
flow_matching:
  num_timesteps: 1000  # Used for scaling continuous t to UNet range

sampling:
  num_steps: 100       # Euler steps (flow matching needs far fewer)
  sampler: "euler"
```

Keep same model architecture, batch size, training iterations, and learning rate as DDPM.

## 5. Update `train.py`

- Add `'flow_matching'` to `argparse` choices (line ~668: `choices=['ddpm']`)
- Add `FlowMatching` import from `src.methods`
- Add creation branch in `train()` (~line 328):

```python
elif method_name == 'flow_matching':
    method = FlowMatching.from_config(model, config, device)
```

## 6. Update `sample.py`

- Add `'flow_matching'` and `'ddim'` to `argparse` choices for `--method`
- Add `FlowMatching` import
- Add creation branches:
  - `flow_matching` -> create `FlowMatching` method, sample normally
  - `ddim` -> create `DDPM` method from checkpoint, call `method.ddim_sample(...)` instead of `method.sample(...)`
- Add `--sampler` argument with choices `['default', 'ddim']` for flexibility

## 7. Update `modal_app.py`

- Update the `sample()` function to pass the method through to `sample.py` (already works since it passes `--method`)
- The existing CLI already accepts arbitrary `--method` strings, but `sample.py` validates choices -- so updating `sample.py` is sufficient.

## 8. Modal Commands

**Train Flow Matching:**

```bash
modal run modal_app.py --action train --method flow_matching --config configs/flow_matching_modal.yaml
```

**Generate 4x4 grid of 16 flow matching samples:**

```bash
modal run modal_app.py --action sample --method flow_matching --checkpoint <flow_matching_checkpoint_path> --num-samples 16 --num-steps 100
```

**Generate 4x4 grid using DDIM (reuses existing DDPM checkpoint):**

```bash
modal run modal_app.py --action sample --method ddim --checkpoint logs/ddpm_modal/ddpm_20260127_162352/checkpoints/ddpm_final.pt --num-samples 16 --num-steps 100
```

These commands will be appended to `z_modal_commands.md`.

## File Change Summary


| File                               | Action                                                    |
| ---------------------------------- | --------------------------------------------------------- |
| `src/methods/flow_matching.py`     | **Create** -- FlowMatching class                          |
| `src/methods/ddpm.py`              | **Edit** -- Add `ddim_sample()` method                    |
| `src/methods/__init__.py`          | **Edit** -- Add FlowMatching import                       |
| `configs/flow_matching_modal.yaml` | **Create** -- Flow matching config                        |
| `train.py`                         | **Edit** -- Add flow_matching method choice               |
| `sample.py`                        | **Edit** -- Add flow_matching + ddim support              |
| `modal_app.py`                     | **Edit** -- Minor: ensure flow_matching/ddim pass through |
| `z_modal_commands.md`              | **Edit** -- Add new commands                              |


