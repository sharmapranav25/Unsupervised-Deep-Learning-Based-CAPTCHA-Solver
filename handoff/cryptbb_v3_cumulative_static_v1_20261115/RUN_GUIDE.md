# Cumulative static CryptBB A/B operations

Canonical sources are Capri/src/capri_controller/training/cryptbb_cumulative*.py. The source snapshots and hashes are in this study's protocol.json/source/. Implementation staging and launch records are in E:/pranav/CAPRI/runs/cryptbb_cumulative_implementation_20261115. Old frozen sources and ranking completion receipts remain unchanged. No model registry deployment occurred.

From E:/pranav/Thesis-Pranav-Final, source environment.ps1 and use the configured $thesisPython:

    & $thesisPython -B -u -m capri_controller.training.cryptbb_cumulative_static setup
    & $thesisPython -B -u -m capri_controller.training.cryptbb_cumulative_static check
    & $thesisPython -B -u -m capri_controller.training.cryptbb_cumulative_static coordinate

These stages have ALREADY RUN. Do not repeat setup or coordinate. Both refuse existing production state. Coordinator owns sequential A then B and evaluate --watch. Check PID plus process_created in the locks and launch receipts before any operation. Production worker locks include a shared training_worker lock. CPU evaluator/report uses evaluation_report.lock.json. Follow-on controls are manifests only.

Create STOP in this study directory to stop before the next optimizer update and prevent B launch. Preserve it until a new explicit user resume instruction. Completed checkpoints are immutable checkpoints/epoch_XX.pth plus epoch_XX.json receipts, with all recovery state. A partial epoch is discarded on STOP. There is deliberately no automatic restart. If an understood transient failure is approved for recovery, inspect receipt hashes, frozen sources, STOP and exact identities before using worker --dataset A|B --resume; recover the coordinator separately and never run two trainers. No old Model2 B resume is authorized.

Evaluation uses the existing fixed251 manifest, CPU FP32/two threads/batch16. Each evaluator freezes predictions before hashing/opening scoring labels. The first evaluation saves the label-file identity in evaluation_inputs.json and later evaluations enforce it. The legacy evaluation policy supplies the expected image manifest hash; it did not contain a scoring-label hash. Existing incomplete outputs are retained and require inspection rather than overwrite. An idle evaluator can be recovered with evaluate --watch only after checking its shared lock and STOP.

After all20 predictions for a reader, an isolated selector with a score-file-denying audit hook applies the unchanged original v1 estimator. Report joins accuracy afterward. MORNING_REPORT.md/results.json/results.csv are live progress outputs until report_verification.json says complete=true. Checkpoint selection files are immutable. No report is a final result merely because its filename says morning.

The heartbeat checks every15minutes in the launch chat, quiet on ordinary progress. It delivers a status at/after09:00 America/Los_Angeles on2026-11-15 and final results when complete, then pauses. Keep the host powered and the app available. Coordinator requests system wakefulness while alive, without changing permanent sleep settings. Final completion requires both20epoch runs,40 verified evaluations, two original-v1 selections and report hashes.

Readiness: six exact preprocessing comparisons; all40 complete draw streams/replays; first-two-epoch parity; uniform full-pool RNG parity from3; full128-image finite-loss/gradient pilots for each corpus; bit-exact next update across weights/optimizer/scheduler/scaler/all RNG/sampler; scratch fingerprints; mocked20epoch transition/archive/STOP/failure lifecycle tests. These pilots never initialize production.

Limits: single matched seed48; schedule predeclared, no independent synthetic-validation tuning; no causal control runs; historical Model1 loss may be rolling median while new loss is epoch mean; previous-static B has8 saved epochs only. Canonical Capri/docs/CODE_AND_MODELS.md is absent; available E:/pranav/CAPRI/docs/CODE_AND_MODELS.md was read along with canonical AGENTS/README/CURRENT_STATE and Capri/README.
