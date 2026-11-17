# CryptBB epoch20-teacher static B report

Updated 2026-11-16T01:18:54.800445+00:00. Status: COMPLETE.

## Reader B

Saved 20/20 epochs; 15640/15640 committed updates. Live status: {"state": "epoch_complete", "completed_epoch": 20, "global_step": 15640, "updated_at": "2026-11-16T01:16:30.836417+00:00"}.
Training wall time: 1.65 hours. Evaluated 20; backlog [].

Original v1 selected epoch 14: 52/251 (20.72%). Top three [14, 20, 17]; k=5; fallback=True. Checkpoint SHA256 `107861015f7366f42827cf8a3a08305dfbe995ccda8be939ea2dc980fcd9b4d1`.
Mask checks: `[{"k": 0, "next_distance_residual": 0.07124929143577624, "far_residual": 0.03456568657520377, "tolerance": 0.008641421643800943, "passes": false}, {"k": 1, "next_distance_residual": 0.060445554196664114, "far_residual": 0.02629678367498603, "tolerance": 0.006574195918746507, "passes": false}, {"k": 2, "next_distance_residual": 0.04862239013616912, "far_residual": 0.017837335356862733, "tolerance": 0.005, "passes": false}, {"k": 3, "next_distance_residual": 0.04220211619211013, "far_residual": 0.011660598437970054, "tolerance": 0.005, "passes": false}, {"k": 4, "next_distance_residual": 0.022652139076575015, "far_residual": 0.007988318778764252, "tolerance": 0.005, "passes": false}, {"k": 5, "next_distance_residual": 0.0153099900998846, "far_residual": 0.006789727737667205, "tolerance": 0.005, "passes": false}]`.
All20 mean 19.0837%; Model1 all20 mean 18.5259%. Retrospective best (diagnostic only): {'correct': 64, 'epochs': [19, 20], 'diagnostic_only': True}.

Shared previous-static epochs [1, 4]: cumulative mean 6.5737%, previous static mean 6.9721%.

| Reader | Epoch | Mean loss | Correct/251 | Accuracy | Model1 correct | Delta pp | Previous static correct | Delta pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B | 1 | 3.835751 | 0/251 | 0.00% | 0 | 0.00 | 0 | 0.00 |
| B | 2 | 1.373541 | 31/251 | 12.35% | 0 | 12.35 | â€” | â€” |
| B | 3 | 0.995229 | 41/251 | 16.33% | 38 | 1.20 | â€” | â€” |
| B | 4 | 0.718973 | 33/251 | 13.15% | 39 | -2.39 | 35 | -0.80 |
| B | 5 | 0.569242 | 47/251 | 18.73% | 50 | -1.20 | â€” | â€” |
| B | 6 | 0.461253 | 47/251 | 18.73% | 42 | 1.99 | â€” | â€” |
| B | 7 | 0.372579 | 45/251 | 17.93% | 50 | -1.99 | â€” | â€” |
| B | 8 | 0.298256 | 44/251 | 17.53% | 51 | -2.79 | â€” | â€” |
| B | 9 | 0.230267 | 47/251 | 18.73% | 35 | 4.78 | â€” | â€” |
| B | 10 | 0.172169 | 48/251 | 19.12% | 45 | 1.20 | â€” | â€” |
| B | 11 | 0.120176 | 55/251 | 21.91% | 52 | 1.20 | â€” | â€” |
| B | 12 | 0.083482 | 56/251 | 22.31% | 55 | 0.40 | â€” | â€” |
| B | 13 | 0.056170 | 52/251 | 20.72% | 62 | -3.98 | â€” | â€” |
| B | 14 | 0.038926 | 52/251 | 20.72% | 53 | -0.40 | â€” | â€” |
| B | 15 | 0.024777 | 55/251 | 21.91% | 58 | -1.20 | â€” | â€” |
| B | 16 | 0.016031 | 57/251 | 22.71% | 59 | -0.80 | â€” | â€” |
| B | 17 | 0.010728 | 60/251 | 23.90% | 58 | 0.80 | â€” | â€” |
| B | 18 | 0.007785 | 60/251 | 23.90% | 62 | -0.80 | â€” | â€” |
| B | 19 | 0.005827 | 64/251 | 25.50% | 61 | 1.20 | â€” | â€” |
| B | 20 | 0.005655 | 64/251 | 25.50% | 60 | 1.59 | â€” | â€” |

Failures: {}. Automatic restarts: 0.

Single seed48 B run; no schedule tuning on clean synthetic validation.
No additional controls or Model3/4 training authorized.
Historical Model1 and previous static comparisons do not establish curriculum causality.
Previous static B stopped with eight saved epochs; compare shared epochs only.
Model2 loss is epoch mean; Model1 logged loss may be rolling median.
