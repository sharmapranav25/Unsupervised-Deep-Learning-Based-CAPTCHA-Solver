# CryptBB Model1 preparation only

CURRENT OVERRIDE14November2026UTC: preparation is complete, heartbeat paused, no training started. Use datasets_ready_v2.json, prepared_fixed32x128.yml, preflight_fixed32x128/receipt.json and agreement_protocol.json. The old synthetic-validation launcher gate has been replaced, and upstream resume is explicitly blocked pending validation. See handoff/cryptbb_model1_preparation/REPORT.md. Historical progress/gap descriptions below are preserved and superseded where this override applies. CryptBB Model2/3 adapters remain pending; learning-progress/SFL discussion is not implementation authorization.


Run/resume after checking preparation.lock.json and active_child.json: source environment.ps1, then use the configured E-drive Python with -B Capri/capri.py prepare-cryptbb-model1 --stage run. Never duplicate an active process. STOP in this folder prevents subsequent stages.

No reader training is implemented or launched by this workflow. Two100k corpora and matched20epoch from-scratch Model1 configs are prepared. Real filename labels are documented historical scoring labels; they are isolated under evaluation_data/cryptbb/scoring_labels and excluded from generator inputs. Original600 pixels contain all100 references; the551 historical test and100 historical selftrain images have zero exact pixel overlap with them.

The preliminary50x15 RGB resized similarity check found no close pair: minimum normalized RMSE0.1074 for original/test and0.1367 for original/selftrain; this is a heuristic, not a proof against all transformed duplicates.

Future launch requires new reader-training authorization and a frozen CryptBB synthetic validation set for epoch selection. The present preparation does not establish validation readiness. Check completion.json and every preflight receipt before reporting datasets ready.

Prepared future launcher: model1/src/train_cryptbb.py --dataset A (or B) checks readiness without training. --start-training is for a later explicitly authorized launch only, after all preparation and a frozen validation_protocol.json are present. --resume is explicit. The validation protocol must record manifest (workspace-relative), manifest_sha256, training_overlap:0, real_labels_used:false, and selection:highest_synthetic_exact_match_earliest_tie. This workflow does not create it from real data. The retained upstream checkpoint archive schedule matches DarkBay and saves epochs2-20.
