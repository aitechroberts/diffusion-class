Execute Ablation combinations with the U-Net architecture of:

RX-DPM- Improves Sampler
LCSC- Improves Model Weights
Progressive Distillation- Improves Speed

Along with a completely different execution of F2D2 and the continous flow map model


So it would look like:
1. RX
2. LCSC
3. PD
4. RX-LCSC
5. RX-PD
6. LCSC-PD
7. RX-LCSC-PD

Continuous Flow Map
1. F2D2
2. LCSC-F2D2
3. Try out RX with F2D2 to see if it works
