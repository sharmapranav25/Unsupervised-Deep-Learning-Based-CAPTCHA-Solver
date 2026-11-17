# Architecture

Original lightweight CAPRI core (~3.66M parameters) learns image/background/appearance attributes and internal character count. A fine-tuned DINOv2 ViT-S/14 register branch adds font attribute residuals and a128D font embedding. An independent four-layer128-wide count transformer is trained jointly. The generator consumes pixels only. Three custom/DINO/mixed controllers refine recipes. Current generation uses CAPRI internal count, with a unique leading model-predicted count supported by at least50/100 references, and uses the learned font branch. It does not use the separate counter at generation time.

Exact architecture, losses and limitations: runs/generator_training/protocol.json and src/capri_controller/lightweight. Missing text-ordinal and font width/slant supervision remains a limitation; improvement cannot be attributed solely to font intelligence without ablation.
