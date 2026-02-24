#!/bin/bash
CKPT="logs/ddpm_modal/ddpm_20260127_162352/checkpoints/ddpm_final.pt"
STEPS=(1 3 5 10 25 50 100)

mkdir -p samples_comparison

for steps in "${STEPS[@]}"; do
  for method in ddim rx_ddim; do
    echo "=== Generating ${method} @ ${steps} steps ==="
    output=$(modal run modal_app.py \
      --action sample \
      --method "$method" \
      --checkpoint "$CKPT" \
      --num-samples 16 \
      --num-steps "$steps" \
      --k 2 2>&1)
    echo "$output"

    # Parse the volume path from "Samples saved to /data/samples/..."
    vol_path=$(echo "$output" | grep -oP '(?<=Samples saved to /data/)\S+')

    if [ -n "$vol_path" ]; then
      modal volume get cmu-10799-diffusion-data "$vol_path" "./samples_comparison/${method}_${steps}.png"
      echo "Downloaded -> samples_comparison/${method}_${steps}.png"
    else
      echo "WARNING: Could not parse output path for ${method}_${steps}"
    fi
  done
done

echo "Done! All samples in samples_comparison/"