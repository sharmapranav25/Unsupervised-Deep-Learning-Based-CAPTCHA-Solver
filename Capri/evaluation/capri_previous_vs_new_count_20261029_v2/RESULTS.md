# Previous CAPRI versus new CAPRI: internal count

Original deployed epoch15 EMA versus new completed epoch20 EMA. Both use the same 1,200 images, labels, CPU FP32 and new-model preprocessing. The old model was rerun; its historical results used different samples and Lanczos/cap960 preprocessing.

| Scheme | Previous CAPRI | New CAPRI | Change (percentage points) |
|---|---:|---:|---:|
| DarkBay | 672/900 (74.7%) | 783/900 (87.0%) | +12.3 |
| CryptBB | 9/100 (9.0%) | 59/100 (59.0%) | +50.0 |
| City | 48/51 (94.1%) | 48/51 (94.1%) | +0.0 |
| SimpleCaptcha | 21/100 (21.0%) | 14/100 (14.0%) | -7.0 |

Historical mostly fixed-length target sets. City49unscored; SimpleCaptcha ground truth from saved publisher count metadata. This is count accuracy, not text recognition. No checkpoint selection or deployment changes.

Baseline predictions were frozen before saved scoring labels were read. Exact image IDs and pixel hashes match the new evaluation. Checkpoint and prediction hashes verified. Reproduction uses capri.py compare-capri-count --reference <reference run> --out <fresh E-drive path>.
