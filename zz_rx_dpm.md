```bash
python sample.py --checkpoint ckpt.pt --method rx_ddim --num_steps 100 --num_samples 8 --grid
```

```bash
# k=2 (default, most frequent extrapolation, best quality)
python sample.py --checkpoint ckpt.pt --method rx_ddim --num_steps 100 --k 2 --num_samples 8 --grid

# Fewer steps (where RX-DDIM shines most vs plain DDIM)
python sample.py --checkpoint ckpt.pt --method rx_ddim --num_steps 50 --k 2 --num_samples 8 --grid

# Save individual images instead of a grid
python sample.py --checkpoint ckpt.pt --method rx_ddim --num_steps 100 --k 2 --num_samples 1 --output_dir samples_rx_ddim

# With a specific seed for reproducibility
python sample.py --checkpoint ckpt.pt --method rx_ddim --num_steps 100 --k 2 --num_samples 1 --seed 42 --grid
```

```bash
# Sample with RX-DDIM on Modal
modal run modal_app.py --action sample --method rx_ddim --num-steps 100 --k 2

# Evaluate RX-DDIM FID on Modal
modal run modal_app.py --action evaluate --method rx_ddim --num-steps 100 --k 2 --num-samples 5000

# Compare side by side
modal run modal_app.py --action sample --method ddim --num-steps 50
modal run modal_app.py --action sample --method rx_ddim --num-steps 50 --k 2
```


zip -r hw3.zip \
  src/ \
  configs/ \
  scripts/ \
  train.py \
  sample.py \
  modal_app.py \
  evaluate_rx_ddim.py \
  benchmark_timing.py \
  visualize_metrics.py \
  pyproject.toml \
  setup-uv.sh \
  -x "src/__pycache__/*" "src/*/__pycache__/*"

modal run modal_app.py \
  --action evaluate \
  --method rx_ddim \
  --checkpoint logs/ddpm_modal/ddpm_20260127_162352/checkpoints/ddpm_final.pt \
  --num-steps 5 \
  --k 2 \
  --metrics kid \
  --num-samples 5000


Steps 5: 0.04011031
Steps 100: 0.00327682