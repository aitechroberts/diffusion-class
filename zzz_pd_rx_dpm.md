Saved checkpoint to /data/logs/pd_distill/checkpoints/pd_stage5_step0050000.pt
Saved checkpoint to /data/logs/pd_distill/checkpoints/pd_stage5_final.pt
Stage 5 complete -> /data/logs/pd_distill/checkpoints/pd_stage5_final.pt

All stages complete!
Progressive Distillation complete!
Stopping app - local entrypoint completed.
✓ App completed. View run at https://modal.com/apps/aitechroberts/main/ap-9ErIXoM4cZ66DqZxE9DO7l


CKPT="logs/ddpm_modal/ddpm_20260127_162352/checkpoints/ddpm_final.pt"


Step 1: Run Progressive Distillation (training the 1-step student model)
This is the longest step (~15-20 hours on a single L40S). It runs all 5 stages sequentially:
```bash
modal run modal_app.py --action distill
```
The config already points to your teacher checkpoint at logs/ddpm_modal/ddpm_20260127_162352/checkpoints/ddpm_final.pt. If you need to override it:
```bash
modal run modal_app.py --action distill \  --checkpoint "logs/ddpm_modal/ddpm_20260127_162352/checkpoints/ddpm_final.pt"
```
When done, checkpoints will be at /data/logs/pd_distill/checkpoints/ on the Modal volume. The final 1-step model is pd_stage5_final.pt.


Step 2: Download the distilled checkpoint (to verify it exists)
```bash
modal volume ls cmu-10799-diffusion-data logs/pd_distill/checkpoints/
```

Step 3: Run the tau hyperparameter search (optional but recommended, ~2 hours)

This finds the optimal (tau, n_rx_steps) before you commit to a full evaluation:
**You'd need to add a search_tau Modal function, or run it via a sample-like action.# Simplest approach: run it directly through the existing sample infrastructure.# For now, you can skip this and use the default tau=0.3, n_rx_steps=2.**

# Step 4: Generate DWS-RX-DDIM sample grids
Quick visual check with 16 samples:
```bash
CKPT="logs/ddpm_modal/ddpm_20260127_162352/checkpoints/ddpm_final.pt"
DISTILLED="logs/pd_distill/checkpoints/pd_stage5_final.pt"

# DWS-RX-DDIM with default tau=0.3,  2 RX steps (total 3 NFE)
modal run modal_app.py \  
    --action sample \  --method dws_rx_ddim \  
    --checkpoint "$CKPT" \  
    --num-samples 16 \  
    --num-steps 2
```
Note: The sample Modal function doesn't currently pass through --distilled-checkpoint from the CLI entrypoint. You'd need to either add that parameter to the main() entrypoint, or run sample.py directly on Modal. The quickest workaround is a script like:
```shell
#!/bin/bash
CKPT="logs/ddpm_modal/ddpm_20260127_162352/checkpoints/ddpm_final.pt"
DISTILLED="logs/pd_distill/checkpoints/pd_stage5_final.pt"
mkdir -p samples_comparison
for tau in 0.2 0.3 0.4; do  
    for n_rx in 2 3; do
        echo "=== DWS-RX-DDIM tau=${tau} n_rx=${n_rx} ==="
        output=$(modal run modal_app.py \      
            --action sample \      
            --method dws_rx_ddim \      
            --checkpoint "$CKPT" \      
            --num-samples 16 \      
            --num-steps "$n_rx" 2>&1)
        echo "$output"
        
        vol_path=$(echo "$output" | grep -oP '(?<=Samples saved to /data/)\S+')
        if [ -n "$vol_path" ]; then
            modal volume get cmu-10799-diffusion-data "$vol_path" \ 
            "./samples_comparison/dws_tau${tau}_rx${n_rx}.png"    
            fi
    done
done
```

# Step 5: Download results from the volume
```bash
# Download stage 5 dense checkpoints (250 files for LCSC)
modal volume get cmu-10799-diffusion-data logs/pd_distill/checkpoints/ ./pd_checkpoints/

# Download sample grids
modal volume get cmu-10799-diffusion-data samples/ ./samples_comparison/

# Download any individual sample (just change the sample choice)
modal volume get cmu-10799-diffusion-data samples/dws_rx_ddim_20260226_164130.png dws_test.png
```
Important caveat I spotted: The sample Modal function accepts distilled_checkpoint as a parameter, but the main() CLI entrypoint in modal_app.py doesn't expose it as a CLI argument — it's not in the parameter list of def main(...). So when you run modal run modal_app.py --action sample, there's no --distilled-checkpoint flag available to pass through.
To fix this, you'd need to switch to Agent mode so I can add distilled_checkpoint, dws_tau, n_rx_steps, and extrapolation parameters to the main() entrypoint and wire them into the sample.remote(...) call. Without that fix, you'll need to test DWS sampling locally or add a custom Modal entrypoint script. Want me to do that?




1-Step PD with 1-Step DWS (2 NFE): 5x less NFE same quality
KID: 0.013810
FID: 18.72581 - Roughly Equivalent to 10-NFE RX_DDIM Got on the full CelebA (18.31)

199.78 it/s

1-Step PD (1 NFE):
KID: 0.0151612
FID: 19.3321 - basically what DDIM got on 

257.58 it/s

10 Steps DWS (11 NFE): 29.8% Improvement over RX-DDIM
KID: 0.007869
FID: 13.042

60 it/s

10 Step RX-DDIM:
FID: 18.31

100 Steps DWS (101 NFE) tau 0.5:
KID: 0.007741008
FID: 12.36951

7.74 it/s

100 Steps DWS (101 NFE) tau 0.8:
KID: 0.006386392
FID: 10.74567

7.69 it/s

100 Steps DWS (101 NFE) tau 0.95:
KID: 0.05246
FID: 9.471213

7.72 it/s