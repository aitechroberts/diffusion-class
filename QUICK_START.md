# DDPM Quick Start Guide

## ✅ Implementation Complete!

All DDPM components have been implemented. You're ready to train!

---

## 🚀 Quick Start Commands

### 1. Sanity Check (5 minutes)
Test if everything works by overfitting to a single batch:

```bash
source .venv-cuda121/bin/activate

python train.py \
  --method ddpm \
  --config configs/ddpm_babel.yaml \
  --overfit-single-batch
```

**Expected:** Loss should drop to ~0.001, samples should match training data.

---

### 2. Full Training (10-12 hours)
Train the full model on the entire dataset:

```bash
python train.py \
  --method ddpm \
  --config configs/ddpm_babel.yaml
```

**Progress:**
- Logs saved to: `logs/ddpm_YYYYMMDD_HHMMSS/`
- Checkpoints: `logs/*/checkpoints/ddpm_*.pt` (every 10k iterations)
- Samples: `logs/*/samples/samples_*.png` (every 5k iterations)
- W&B: Check dashboard for real-time metrics (if enabled)

---

### 3. Resume Training
If training is interrupted:

```bash
python train.py \
  --method ddpm \
  --config configs/ddpm_babel.yaml \
  --resume logs/ddpm_20260126_123456/checkpoints/ddpm_0050000.pt
```

---

### 4. Generate Samples
After training completes:

```bash
# Generate 64 samples as a grid
python sample.py \
  --checkpoint logs/ddpm_TIMESTAMP/checkpoints/ddpm_final.pt \
  --method ddpm \
  --num_samples 64 \
  --grid \
  --output final_samples.png

# Or generate 1000 individual images
python sample.py \
  --checkpoint logs/ddpm_TIMESTAMP/checkpoints/ddpm_final.pt \
  --method ddpm \
  --num_samples 1000 \
  --output_dir samples/ \
  --batch_size 64
```

---

## 📊 What to Report (Q4a)

### 1. Model Size
Check the training log at startup:
```
Model parameters: 35,438,659 (35.44M)
```

### 2. Batch Size
```
64
```

### 3. Total Training Iterations
```
100,000 iterations
```

### 4. Training Loss Curve
- Initial loss: ~0.5-1.0
- Final loss: ~0.01-0.05
- Plot from W&B or extract from logs

To get loss values:
```bash
grep "loss" logs/ddpm_*/training.log > loss_curve.txt
```

### 5. Compute Cost
Estimated: **10-15 GPU hours** on RTX 4090 (24GB)

Calculate actual:
```
Training time (hours) × Number of GPUs = GPU hours
```

---

## 🔍 Monitor Training

### Check Progress During Training

```bash
# View latest samples
ls -lt logs/ddpm_*/samples/*.png | head -5

# Check current iteration
tail logs/ddpm_*/training.log

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Sample Quality Timeline

| Iteration | Expected Quality |
|-----------|------------------|
| 5,000 | Blurry colors/shapes |
| 10,000 | Basic structure visible |
| 30,000 | Recognizable as faces |
| 60,000 | Clear faces, good details |
| 100,000 | High quality, diverse faces |

---

## ⚙️ Adjust Settings (If Needed)

### If Out of Memory
Edit `configs/ddpm_babel.yaml`:
```yaml
training:
  batch_size: 32  # or 16
```

Or reduce model size:
```yaml
model:
  base_channels: 96  # instead of 128
```

### If Training is Slow
- Check GPU utilization: `nvidia-smi` (should be >90%)
- Ensure data is on fast storage (SSD, not network drive)
- Mixed precision is already enabled

### If Loss Explodes (NaN)
Reduce learning rate in config:
```yaml
training:
  learning_rate: 1e-4  # instead of 2e-4
```

---

## 📁 File Locations

### After Training, Find:

| File | Location |
|------|----------|
| Checkpoints | `logs/ddpm_TIMESTAMP/checkpoints/` |
| Generated samples | `logs/ddpm_TIMESTAMP/samples/` |
| Configuration | `logs/ddpm_TIMESTAMP/config.yaml` |
| W&B logs | Check W&B dashboard |

### For Your Report:

| Item | Source |
|------|--------|
| Model parameters | First lines of training log |
| Training curve | W&B or grep logs |
| Sample images | `logs/*/samples/samples_0100000.png` |
| Training time | Check log timestamps |

---

## 🐛 Quick Troubleshooting

### "CUDA out of memory"
```bash
# Reduce batch size to 32 or 16
# Or use gradient accumulation
```

### "Dataset not found"
```bash
# Re-download dataset
python download_dataset.py --output_dir ./data/celeba-subset
```

### "ImportError: No module named..."
```bash
# Reinstall dependencies
source .venv-cuda121/bin/activate
uv pip install -e ".[datasets]"
```

### Loss is stuck / not decreasing
- Check if data is normalized correctly
- Verify model is actually training (check W&B gradients)
- Try lower learning rate
- Train longer (DDPM needs many iterations)

### Samples are all noise
- Check if you're using EMA weights (enabled by default after 5k iters)
- Verify sampling loop runs for full 1000 steps
- Check if loss has actually decreased

---

## ✅ Checklist Before Training

- [x] Dataset downloaded (`data/celeba-subset/`)
- [x] Environment activated (`.venv-cuda121`)
- [x] Config filled (`configs/ddpm_babel.yaml`)
- [x] GPU available (`nvidia-smi` shows GPU)
- [x] Enough disk space (~10GB for checkpoints)
- [x] W&B configured (optional, but recommended)

---

## 📚 Additional Resources

### Files to Read:
- `IMPLEMENTATION_SUMMARY.md` - Detailed explanation of implementation
- `CHANGES.md` - What was modified in each file
- `docs/DIRECTORY-STRUCTURE.md` - Project organization

### Test Implementation:
```bash
python test_implementation.py
```

---

## 🎯 Success Criteria

Your training is successful if:

1. ✅ Loss decreases steadily to < 0.05
2. ✅ Samples show clear faces by 60k iterations
3. ✅ No NaN losses or crashes
4. ✅ Final samples are diverse and high quality
5. ✅ EMA improves sample quality visibly

---

## 💡 Tips

1. **Start with overfit test:** Always verify with `--overfit-single-batch` first
2. **Monitor early:** Check samples at 5k, 10k iterations to catch issues early
3. **Use EMA:** It significantly improves sample quality (automatically enabled)
4. **Be patient:** DDPM needs 50k+ iterations for good results
5. **Save often:** Checkpoints are saved every 10k iterations

---

## 🚨 When to Stop Training

Stop training if:
- ✅ Reached 100k iterations (planned stopping point)
- ✅ Sample quality plateaus for 20k+ iterations
- ✅ Loss hasn't improved for 30k+ iterations

Don't stop if:
- ❌ Only 10k-20k iterations (too early!)
- ❌ Samples still improving
- ❌ Loss still decreasing

---

## Good luck with your training! 🎉

Questions? Check:
1. `IMPLEMENTATION_SUMMARY.md` for detailed explanation
2. Inline code comments for implementation details
3. DDPM paper (Ho et al., 2020) for theoretical background
