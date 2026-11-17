# Latest DarkBay Model2 available checkpoint accuracy

Family darkbay_v3_feature_20261115. Fixed snapshot of 15 committed checkpoints. Same900 historical real images,100 references excluded. Predictions frozen before labels. GPU IEEE FP32,batch4,7% allocator cap; evaluator reused unchanged from verified Model1 diagnostics. Model2-specific CPU/GPU parity is not claimed. No training or fixed-final20 selection changes.

| Epoch | Model2-A | Model1-A | Difference pp | Model2-B | Model1-B | Difference pp |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0/900 (0.00%) | 0/900 (0.00%) | +0.00 | 0/900 (0.00%) | 0/900 (0.00%) | +0.00 |
| 2 | 359/900 (39.89%) | 374/900 (41.56%) | -1.67 | 72/900 (8.00%) | 406/900 (45.11%) | -37.11 |
| 3 | 444/900 (49.33%) | 544/900 (60.44%) | -11.11 | 487/900 (54.11%) | 448/900 (49.78%) | +4.33 |
| 4 | 587/900 (65.22%) | 655/900 (72.78%) | -7.56 | 516/900 (57.33%) | 632/900 (70.22%) | -12.89 |
| 5 | 565/900 (62.78%) | 643/900 (71.44%) | -8.67 | 528/900 (58.67%) | 627/900 (69.67%) | -11.00 |
| 6 | 616/900 (68.44%) | 604/900 (67.11%) | +1.33 | Not saved at snapshot | — | — |
| 7 | 575/900 (63.89%) | 519/900 (57.67%) | +6.22 | Not saved at snapshot | — | — |
| 8 | 568/900 (63.11%) | 527/900 (58.56%) | +4.56 | Not saved at snapshot | — | — |
| 9 | 545/900 (60.56%) | 579/900 (64.33%) | -3.78 | Not saved at snapshot | — | — |
| 10 | 552/900 (61.33%) | 536/900 (59.56%) | +1.78 | Not saved at snapshot | — | — |
