# CryptBB epoch20-teacher static A report

Updated 2026-11-15T23:20:17.050979+00:00. Status: COMPLETE.

## Reader A

Saved 20/20 epochs; 15640/15640 committed updates. Live status: {"state": "epoch_complete", "completed_epoch": 20, "global_step": 15640, "updated_at": "2026-11-15T23:17:40.760569+00:00"}.
Training wall time: 2.14 hours. Evaluated 20; backlog [].

Original v1 selected epoch 14: 66/251 (26.29%). Top three [14, 20, 19]; k=5; fallback=True. Checkpoint SHA256 `fce1b7d4dab98bfce37e95bfc66d7f592684d02336e3de460d0f3765dbbbe6d0`.
Mask checks: `[{"k": 0, "next_distance_residual": 0.07680707851789383, "far_residual": 0.034487033386374134, "tolerance": 0.008621758346593534, "passes": false}, {"k": 1, "next_distance_residual": 0.0657869788543783, "far_residual": 0.027760241539712637, "tolerance": 0.006940060384928159, "passes": false}, {"k": 2, "next_distance_residual": 0.05575116401835259, "far_residual": 0.02032177785990324, "tolerance": 0.00508044446497581, "passes": false}, {"k": 3, "next_distance_residual": 0.03714149874585036, "far_residual": 0.013731568933719298, "tolerance": 0.005, "passes": false}, {"k": 4, "next_distance_residual": 0.034703589139234665, "far_residual": 0.009609859491150818, "tolerance": 0.005, "passes": false}, {"k": 5, "next_distance_residual": 0.016883909664850316, "far_residual": 0.006800685104388093, "tolerance": 0.005, "passes": false}]`.
All20 mean 20.5976%; Model1 all20 mean 17.9880%. Retrospective best (diagnostic only): {'correct': 66, 'epochs': [14, 19], 'diagnostic_only': True}.

Shared previous-static epochs [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]: cumulative mean 20.5976%, previous static mean 17.4900%.

| Reader | Epoch | Mean loss | Correct/251 | Accuracy | Model1 correct | Delta pp | Previous static correct | Delta pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 1 | 3.820019 | 0/251 | 0.00% | 0 | 0.00 | 0 | 0.00 |
| A | 2 | 1.260896 | 35/251 | 13.94% | 0 | 13.94 | 31 | 1.59 |
| A | 3 | 0.990486 | 33/251 | 13.15% | 28 | 1.99 | 35 | -0.80 |
| A | 4 | 0.711883 | 33/251 | 13.15% | 42 | -3.59 | 41 | -3.19 |
| A | 5 | 0.569669 | 42/251 | 16.73% | 49 | -2.79 | 34 | 3.19 |
| A | 6 | 0.469612 | 53/251 | 21.12% | 46 | 2.79 | 35 | 7.17 |
| A | 7 | 0.374631 | 51/251 | 20.32% | 51 | 0.00 | 51 | 0.00 |
| A | 8 | 0.301600 | 58/251 | 23.11% | 52 | 2.39 | 52 | 2.39 |
| A | 9 | 0.226275 | 51/251 | 20.32% | 49 | 0.80 | 51 | 0.00 |
| A | 10 | 0.175985 | 58/251 | 23.11% | 57 | 0.40 | 46 | 4.78 |
| A | 11 | 0.126249 | 55/251 | 21.91% | 52 | 1.20 | 50 | 1.99 |
| A | 12 | 0.085775 | 62/251 | 24.70% | 61 | 0.40 | 53 | 3.59 |
| A | 13 | 0.056256 | 56/251 | 22.31% | 45 | 4.38 | 47 | 3.59 |
| A | 14 | 0.039375 | 66/251 | 26.29% | 45 | 8.37 | 49 | 6.77 |
| A | 15 | 0.024262 | 64/251 | 25.50% | 53 | 4.38 | 49 | 5.98 |
| A | 16 | 0.016063 | 63/251 | 25.10% | 48 | 5.98 | 47 | 6.37 |
| A | 17 | 0.010499 | 65/251 | 25.90% | 49 | 6.37 | 49 | 6.37 |
| A | 18 | 0.007536 | 59/251 | 23.51% | 58 | 0.40 | 56 | 1.20 |
| A | 19 | 0.006123 | 66/251 | 26.29% | 59 | 2.79 | 52 | 5.58 |
| A | 20 | 0.005361 | 64/251 | 25.50% | 59 | 1.99 | 50 | 5.58 |

Failures: {}. Automatic restarts: 0.

Single seed48 A run; no schedule tuning on clean synthetic validation.
No additional control runs or B training authorized.
Historical Model1 and previous static comparisons do not establish curriculum causality.
Previous static B stopped with eight saved epochs; compare shared epochs only.
Model2 loss is epoch mean; Model1 logged loss may be rolling median.

Execution change after epoch4: user-authorized checkpoint pause/resume. Batch128, sample order, weights/optimizer/scheduler/scaler/RNG preserved.24 actual subsequent updates reproduced bit-for-bit before resume; see fast_execution_v1. Disposable verification updates are not training progress.
