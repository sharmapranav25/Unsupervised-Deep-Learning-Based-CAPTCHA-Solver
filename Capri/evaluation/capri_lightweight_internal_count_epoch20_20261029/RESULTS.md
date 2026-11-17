# CAPRI internal count: completed epoch 20

The internal length head was evaluated on the same fixed 1,200 target images as the earlier partial-epoch-16 diagnostic. Predictions use pixels only and were saved and hash-frozen before label scoring.

| Target scheme | Epoch 20 correct | Epoch 20 accuracy | Earlier epoch 16 accuracy | Change (percentage points) |
|---|---:|---:|---:|---:|
| DarkBay | 783/900 | 87.0% | 89.2% | -2.2 |
| CryptBB | 59/100 | 59.0% | 63.0% | -4.0 |
| City | 48/51 | 94.1% | 100.0% | -5.9 |
| SimpleCaptcha | 14/100 | 14.0% | 7.0% | +7.0 |

City: 49 of the 100 images have no matching verified annotation and remain unscored. SimpleCaptcha count ground truth comes from saved publisher metadata. These historical, mostly fixed-length targets do not establish variable-length count generalization.

Internal count remains stronger than the separate transformer on every target. It regressed on DarkBay, CryptBB and City since the earlier diagnostic; SimpleCaptcha improved but remains weak. No checkpoint selection, threshold tuning, deployment or branch removal was performed.

The unchanged evaluator also saved separate-transformer outputs in summary.json for provenance; they do not enter the internal count predictions or the table above. Exact internal count/full model parity passed.

Checkpoint: E:\pranav\CAPRI\runs\capri_lightweight_joint20_100k_20261028\epochs\epoch_0020\checkpoint.pt (EMA, completed epoch 20, step 19206).
Checkpoint SHA256: 28f95fd8daf3831fb860ab59e15b8b65bd4a68bacc434eb7eb7a296e6190b764

Reproduce from E:/pranav/CAPRI with capri.py test-lightweight-count --run runs/capri_lightweight_joint20_100k_20261028 --out <fresh E-drive run path>. Use the E-drive Python environment documented in the training run README.
