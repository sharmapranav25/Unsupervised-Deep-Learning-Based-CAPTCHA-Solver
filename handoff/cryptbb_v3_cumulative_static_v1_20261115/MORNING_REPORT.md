# CryptBB cumulative static morning report

Updated 2026-11-15T19:09:50.951666+00:00. Status: COMPLETE.

## Reader A

Saved 20/20 epochs; 15640/15640 committed updates. Live status: {"state": "epoch_complete", "completed_epoch": 20, "global_step": 15640, "updated_at": "2026-11-15T13:17:43.540329+00:00"}.
Training wall time: 3.51 hours. Evaluated 20; backlog [].

Original v1 selected epoch 14: 60/251 (23.90%). Top three [14, 13, 12]; k=5; fallback=True. Checkpoint SHA256 `922e30ec7c96dd1267f77c55f578f985013e883e1f3c23adaa9c4923a62ae3e5`.
Mask checks: `[{"k": 0, "next_distance_residual": 0.07859173809963815, "far_residual": 0.041712234513845904, "tolerance": 0.010428058628461476, "passes": false}, {"k": 1, "next_distance_residual": 0.06841409508295654, "far_residual": 0.033793005844168514, "tolerance": 0.008448251461042128, "passes": false}, {"k": 2, "next_distance_residual": 0.05685805526424577, "far_residual": 0.024593154927027173, "tolerance": 0.006148288731756793, "passes": false}, {"k": 3, "next_distance_residual": 0.0468646906714946, "far_residual": 0.015090681989227119, "tolerance": 0.005, "passes": false}, {"k": 4, "next_distance_residual": 0.029959626879343244, "far_residual": 0.010139541573193498, "tolerance": 0.005, "passes": false}, {"k": 5, "next_distance_residual": 0.018514883488590032, "far_residual": 0.008173242474541348, "tolerance": 0.005, "passes": false}]`.
All20 mean 18.9841%; Model1 all20 mean 17.9880%. Retrospective best (diagnostic only): {'correct': 62, 'epochs': [12], 'diagnostic_only': True}.

Shared previous-static epochs [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]: cumulative mean 18.9841%, previous static mean 17.4900%.

## Reader B

Saved 20/20 epochs; 15640/15640 committed updates. Live status: {"state": "epoch_complete", "completed_epoch": 20, "global_step": 15640, "updated_at": "2026-11-15T19:06:04.702511+00:00"}.
Training wall time: 3.56 hours. Evaluated 20; backlog [].

Original v1 selected epoch 20: 51/251 (20.32%). Top three [20, 14, 19]; k=5; fallback=True. Checkpoint SHA256 `73e3278abf02ceb299dad1eed89b14544e2488513ce21896ac428493705dbf07`.
Mask checks: `[{"k": 0, "next_distance_residual": 0.07506393774021765, "far_residual": 0.03534704539342429, "tolerance": 0.008836761348356072, "passes": false}, {"k": 1, "next_distance_residual": 0.05429240114924491, "far_residual": 0.027910517119089828, "tolerance": 0.006977629279772457, "passes": false}, {"k": 2, "next_distance_residual": 0.04871940872730107, "far_residual": 0.02019768801081399, "tolerance": 0.005049422002703498, "passes": false}, {"k": 3, "next_distance_residual": 0.03434146445987242, "far_residual": 0.012823071731930293, "tolerance": 0.005, "passes": false}, {"k": 4, "next_distance_residual": 0.028966829253630092, "far_residual": 0.008828540613271702, "tolerance": 0.005, "passes": false}, {"k": 5, "next_distance_residual": 0.017999040008298432, "far_residual": 0.006046044448754067, "tolerance": 0.005, "passes": false}]`.
All20 mean 17.6892%; Model1 all20 mean 18.5259%. Retrospective best (diagnostic only): {'correct': 60, 'epochs': [14], 'diagnostic_only': True}.

Shared previous-static epochs [1, 4]: cumulative mean 7.5697%, previous static mean 6.9721%.

| Reader | Epoch | Mean loss | Correct/251 | Accuracy | Model1 correct | Delta pp | Previous static correct | Delta pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 1 | 3.806186 | 0/251 | 0.00% | 0 | 0.00 | 0 | 0.00 |
| A | 2 | 1.985870 | 31/251 | 12.35% | 0 | 12.35 | 31 | 0.00 |
| A | 3 | 0.790871 | 32/251 | 12.75% | 28 | 1.59 | 35 | -1.20 |
| A | 4 | 0.563228 | 40/251 | 15.94% | 42 | -0.80 | 41 | -0.40 |
| A | 5 | 0.453716 | 42/251 | 16.73% | 49 | -2.79 | 34 | 3.19 |
| A | 6 | 0.377496 | 48/251 | 19.12% | 46 | 0.80 | 35 | 5.18 |
| A | 7 | 0.324337 | 57/251 | 22.71% | 51 | 2.39 | 51 | 2.39 |
| A | 8 | 0.272361 | 43/251 | 17.13% | 52 | -3.59 | 52 | -3.59 |
| A | 9 | 0.228825 | 49/251 | 19.52% | 49 | 0.00 | 51 | -0.80 |
| A | 10 | 0.192792 | 52/251 | 20.72% | 57 | -1.99 | 46 | 2.39 |
| A | 11 | 0.149051 | 51/251 | 20.32% | 52 | -0.40 | 50 | 0.40 |
| A | 12 | 0.122080 | 62/251 | 24.70% | 61 | 0.40 | 53 | 3.59 |
| A | 13 | 0.086722 | 53/251 | 21.12% | 45 | 3.19 | 47 | 2.39 |
| A | 14 | 0.062657 | 60/251 | 23.90% | 45 | 5.98 | 49 | 4.38 |
| A | 15 | 0.042990 | 59/251 | 23.51% | 53 | 2.39 | 49 | 3.98 |
| A | 16 | 0.029178 | 56/251 | 22.31% | 48 | 3.19 | 47 | 3.59 |
| A | 17 | 0.019361 | 57/251 | 22.71% | 49 | 3.19 | 49 | 3.19 |
| A | 18 | 0.013658 | 53/251 | 21.12% | 58 | -1.99 | 56 | -1.20 |
| A | 19 | 0.010475 | 54/251 | 21.51% | 59 | -1.99 | 52 | 0.80 |
| A | 20 | 0.009744 | 54/251 | 21.51% | 59 | -1.99 | 50 | 1.59 |
| B | 1 | 3.805913 | 0/251 | 0.00% | 0 | 0.00 | 0 | 0.00 |
| B | 2 | 2.413995 | 20/251 | 7.97% | 0 | 7.97 | — | — |
| B | 3 | 0.827081 | 31/251 | 12.35% | 38 | -2.79 | — | — |
| B | 4 | 0.579630 | 38/251 | 15.14% | 39 | -0.40 | 35 | 1.20 |
| B | 5 | 0.461410 | 34/251 | 13.55% | 50 | -6.37 | — | — |
| B | 6 | 0.382795 | 44/251 | 17.53% | 42 | 0.80 | — | — |
| B | 7 | 0.332064 | 50/251 | 19.92% | 50 | 0.00 | — | — |
| B | 8 | 0.281642 | 47/251 | 18.73% | 51 | -1.59 | — | — |
| B | 9 | 0.236069 | 52/251 | 20.72% | 35 | 6.77 | — | — |
| B | 10 | 0.192340 | 44/251 | 17.53% | 45 | -0.40 | — | — |
| B | 11 | 0.154405 | 50/251 | 19.92% | 52 | -0.80 | — | — |
| B | 12 | 0.123864 | 52/251 | 20.72% | 55 | -1.20 | — | — |
| B | 13 | 0.088777 | 55/251 | 21.91% | 62 | -2.79 | — | — |
| B | 14 | 0.066469 | 60/251 | 23.90% | 53 | 2.79 | — | — |
| B | 15 | 0.043635 | 50/251 | 19.92% | 58 | -3.19 | — | — |
| B | 16 | 0.029307 | 58/251 | 23.11% | 59 | -0.40 | — | — |
| B | 17 | 0.019867 | 51/251 | 20.32% | 58 | -2.79 | — | — |
| B | 18 | 0.013733 | 53/251 | 21.12% | 62 | -3.59 | — | — |
| B | 19 | 0.011088 | 48/251 | 19.12% | 61 | -5.18 | — | — |
| B | 20 | 0.010077 | 51/251 | 20.32% | 60 | -3.59 | — | — |

Failures: {}. Automatic restarts: 0.

Single seed48 pair; no schedule tuning on clean synthetic validation.
Uniform/random expansion/exact exposure shuffled controls have manifests only, not training results.
Historical Model1 and previous static comparisons do not establish curriculum causality.
Previous static B stopped with eight saved epochs; compare shared epochs only.
Model2 loss is epoch mean; Model1 logged loss may be rolling median.

Recovery: B stopped at05:49 Pacific after a Windows status-file rename denial. Epoch3 was fully committed; partial epoch4 was discarded. User authorized one resume from epoch3, preserving all training state. A remained complete. Frozen sources unchanged; recovery wrapper retries PermissionError on JSON writes for up to5seconds. Failure evidence: recovery_B_status_lock_20261115. User-authorized restarts:1.
