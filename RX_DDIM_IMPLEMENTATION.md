# RX-DDIM Implementation

Implementation of **RX-DPM** (Richardson Extrapolation for Diffusion Probabilistic Models) applied to DDIM sampling, based on:

> Choi, J., Kang, J., & Han, B. (2025). *Enhanced Diffusion Sampling via Extrapolation with Multiple ODE Solutions*. ICLR 2025.
> Paper: [https://openreview.net/forum?id=xRyD6TPmKi](https://openreview.net/forum?id=xRyD6TPmKi)
> Code: [https://github.com/jin01020/rx-dpm](https://github.com/jin01020/rx-dpm)

## What Is RX-DDIM?

DDIM sampling is equivalent to solving an ODE with the Euler method. Each step introduces a local truncation error of O(h^2). RX-DDIM applies **Richardson extrapolation** every *k* steps to cancel the leading error term, raising accuracy to O(h^3) — all without any additional neural network forward passes.

The idea: over every block of *k* DDIM steps, we already compute a **k-step** solution (the normal DDIM trajectory). We can cheaply obtain a **1-step** solution over the same interval by reusing the noise prediction from the first step. Since both solutions approximate the same ideal value but have different error profiles, a weighted combination cancels the dominant error.

## Core Formula

Given a block of *k* steps from timestep index `i` to `i+k`, two solutions at `t_{i-k}` are:

- `x_k_step` — the result of *k* normal DDIM steps (fine grid)
- `x_1_step` — the result of 1 DDIM step over the whole interval (coarse grid, reuses first eps)

The extrapolated solution (Equation 18 from the paper):

```
x_tilde = (x_k_step - S * x_1_step) / (1 - S)
```

where:

```
S = sum_{j=1}^{k} lambda_j^p

lambda_j = (gamma[t_{i-j+1}] - gamma[t_{i-j}]) / (gamma[t_i] - gamma[t_{i-k}])

gamma(t) = sqrt((1 - alpha_bar_t) / alpha_bar_t)

p = 2   (for DDIM, which is a first-order Euler method)
```

### Why Gamma-Space?

DDIM's deterministic update rule solves an ODE naturally expressed in the coordinate `gamma(t) = sqrt((1 - alpha_bar_t) / alpha_bar_t)`, not in the raw timestep index. Computing the extrapolation coefficients (`lambda_j`) in gamma-space rather than index-space accounts for the non-uniform nature of the noise schedule and yields better results than naive uniform-grid Richardson extrapolation (see paper Fig. 2, "Naive" vs "RX-Euler").

### Why Zero Extra Cost?

The 1-step coarse prediction reuses `eps` (the noise prediction) from the first step of the k-step block, which was already computed. The only extra work is the DDIM formula applied once more plus a linear combination of two tensors — negligible compared to a network forward pass.

## Files Changed

### `src/methods/ddpm.py`

Added the `rx_ddim_sample()` method to the `DDPM` class. This sits alongside the existing `ddim_sample()` and shares the same trained model and noise schedule. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_steps` | 100 | Total number of DDIM steps |
| `k` | 2 | Extrapolation interval (every k steps). k=2 performs best. |

### `sample.py`

- Added `rx_ddim` to the `--method` choices
- Added `--k` argument (default: 2) for the extrapolation interval
- Added routing so `rx_ddim` creates a `DDPM` method instance and calls `rx_ddim_sample()`

## Usage

RX-DDIM uses the exact same DDPM checkpoint as regular DDIM — no retraining needed.

```bash
# Standard DDIM (baseline)
python sample.py --checkpoint ckpt.pt --method ddim --num_steps 50 --num_samples 64 --grid

# RX-DDIM with k=2 (default, recommended)
python sample.py --checkpoint ckpt.pt --method rx_ddim --num_steps 50 --k 2 --num_samples 64 --grid

# RX-DDIM with k=3 (less frequent extrapolation)
python sample.py --checkpoint ckpt.pt --method rx_ddim --num_steps 60 --k 3 --num_samples 64 --grid
```

Note: for best results, `num_steps` should be divisible by `k`. If not, the remaining steps at the end use plain DDIM without extrapolation.

## Expected Improvements

From the paper (Table 1, Stable Diffusion V2 on COCO2014):

| NFEs | DDIM FID | RX-DDIM FID | Improvement |
|------|----------|-------------|-------------|
| 15   | 19.15    | 17.24       | -10.0%      |
| 20   | 18.43    | 17.12       | -7.1%       |
| 30   | 19.00    | 17.62       | -7.3%       |
| 50   | 18.65    | 17.83       | -4.4%       |

The improvement is most pronounced at lower step counts, which is exactly where faster sampling matters most. Qualitatively, RX-DDIM produces sharper textures, more vivid colors, and more realistic details compared to DDIM at the same NFE budget.

## Algorithm Reference

Pseudocode for Algorithm 1 from the paper, specialized for our DDIM implementation:

```
Input: trained noise predictor eps_theta, num_steps N, schedule alpha_bar, k=2, p=2
Output: generated sample x

1. Build timestep subsequence [t_0, t_1, ..., t_N] from T-1 to 0
2. Compute gamma[t] = sqrt((1 - alpha_bar[t]) / alpha_bar[t]) for all t
3. x ~ N(0, I)
4. For i = 0, k, 2k, ...:
   a. x_k_step = x
   b. For j = 0 to k-1:          (k-step fine path)
      - eps = eps_theta(x_k_step, t_{i+j})
      - if j == 0: store first_eps = eps
      - x_k_step = DDIM_step(x_k_step, t_{i+j}, t_{i+j+1}, eps)
   c. x_1_step = DDIM_step(x, t_i, t_{i+k}, first_eps)   (1-step coarse path, no extra NFE)
   d. Compute S = sum_j (lambda_j)^p using gamma-space intervals
   e. x = (x_k_step - S * x_1_step) / (1 - S)            (extrapolation)
5. Return x
```
