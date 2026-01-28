
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