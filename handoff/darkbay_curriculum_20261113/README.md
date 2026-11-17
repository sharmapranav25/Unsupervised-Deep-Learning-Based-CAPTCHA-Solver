# Authorized DarkBay curriculum study

Read status.json for coordinator phase, active_children.json for child PIDs, and each model2/3/runs/darkbay/A,B/status.json for actual training progress. Status files are saved snapshots; verify the process before assuming a run is live. coordinator.pid can be the environment launcher PID; status.json records the executing Python PID.

The coordinator completes both100k teacher rankings, passes readiness, trains Model2-A/B, then independently initializes Model3-A/B, freezes all four final predictions, scores the historical900-image set, writes REPORT.md and family_report.json, and stops for CryptBB readiness. No other family or STR reader is launched.

Production has782 full batch128 updates per epoch,20epochs. This matches Model1's100096 padded draws per epoch from100k images. Replacement curriculum draws intentionally repeat samples. Two processes prepare inputs concurrently and take turns on one GPU to preserve full-batch behavior. Their wall times overlap; do not add overlapping run wall times as total study time.

From the E: workspace, source environment.ps1, then use the configured Python with `-B Capri/capri.py darkbay-ab-curriculum coordinate`. Do not launch a duplicate coordinator. It requires the existing frozen protocol. New run directories are refused unless resuming a committed latest.pth. Resume is at epoch boundaries: an incomplete epoch is replayed from its last committed checkpoint. Interrupted rankings/readiness require inspection; no automatic destructive cleanup occurs.

To stop gracefully, create STOP in this folder. Each training process checks it at epoch boundaries. Do not remove historical STOP markers elsewhere. Before a deliberate restart, inspect logs and saved states; only remove this new STOP when resumption is authorized.

Checks: readiness.json records transform equivalence with the baseline, complete draw budgets, adaptive replay, deterministic scoring, model/optimizer/scheduler/RNG save-load continuation and paired full-batch GPU pilots. A failed gate stops production. Original Model1, datasets and historical source files remain unchanged.

Outputs remain in the existing hierarchy: model2/rankings/darkbay/{A,B}; model{2,3}/runs/darkbay/{A,B}; model{2,3}/results/darkbay/{A,B}/evaluation_epoch20. Raw adaptive scores and curriculum states are under each Model3 run's adaptive folder. Epoch checkpoints include all resume state. Logs for each process are adjacent A.console.log/B.console.log; configs/history/status/plots are inside each run. This folder holds orchestration logs, protocol/source hashes, source snapshots and final reports.

The generator threshold is50% unique vote agreement, not40% measured length accuracy. Existing DarkBay had89/100 votes for6. No generator run is repeated; future CryptBB/City inputs must be independently voted at generation time.
