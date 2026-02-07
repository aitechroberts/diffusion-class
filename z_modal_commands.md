
# DDPM FINAL MODEL
Modal Location: logs/ddpm_modal/ddpm_20260127_162352/checkpoints/ddpm_final.pt



**Evaluate DDPM FINAL MODEL**
```bash
modal run modal_app.py --action evaluate --checkpoint logs/ddpm_modal/ddpm_20260127_162352/checkpoints/ddpm_final.pt --num-steps 1000 --num-samples 1000 --metrics kid

# --override lets you overwrite previously generated samples, otherwise, it'll just use the old ones
```

Or
```bash
chmod +x scripts/evaluate_modal_torch_fidelity.sh
./scripts/evaluate_modal_torch_fidelity.sh
```

**Use DDPM FINAL MODEL to Generate Sample**
```bash
modal run modal_app.py --action sample --checkpoint logs/ddpm_modal/ddpm_20260127_162352/checkpoints/ddpm_final.pt --num-samples 1 --num-steps 1000
```

**Download Sample to Local Env**
```bash
modal volume get cmu-10799-diffusion-data logs/ddpm_modal/ddpm_20260127_162352/checkpoints/samples/sample_1000steps.png ./sample_1000.png
```


modal volume get cmu-10799-diffusion-data logs/ddpm_modal/ddpm_20260127_162352/checkpoints/samples/generated/000999.png ./sample_grid.png


Evaluate Command
modal run modal_app.py --action evaluate --checkpoint logs/ddpm_modal/ddpm_20260127_162352/checkpoints/ddpm_final.pt --num-steps 1000 --num-samples 1000 --metrics kid

---

# FLOW MATCHING

**Train Flow Matching Model**
```bash
modal run modal_app.py --action train --method flow_matching --config configs/flow_matching_modal.yaml
```

**Generate 4x4 Grid of 16 Flow Matching Samples**
```bash
modal run modal_app.py --action sample --method flow_matching --checkpoint <FLOW_MATCHING_CHECKPOINT_PATH> --num-samples 16 --num-steps 100
```

**Evaluate Flow Matching KID (1k samples)**
```bash
modal run modal_app.py --action evaluate --method flow_matching --checkpoint <FLOW_MATCHING_CHECKPOINT_PATH> --num-steps 100 --num-samples 1000 --metrics kid
```

---

# DDIM (uses existing DDPM checkpoint, no retraining needed)

**Generate 4x4 Grid of 16 DDIM Samples (100 steps)**
```bash
modal run modal_app.py --action sample --method ddim --checkpoint logs/ddpm_modal/ddpm_20260127_162352/checkpoints/ddpm_final.pt --num-samples 16 --num-steps 100
```

**Evaluate DDIM KID (1k samples, 100 steps)**
```bash
modal run modal_app.py --action evaluate --method ddim --checkpoint logs/ddpm_modal/ddpm_20260127_162352/checkpoints/ddpm_final.pt --num-steps 100 --num-samples 1000 --metrics kid
```