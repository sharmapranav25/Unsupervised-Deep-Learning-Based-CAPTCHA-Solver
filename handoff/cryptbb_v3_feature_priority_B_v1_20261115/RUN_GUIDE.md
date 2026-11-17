# CryptBB Model3-B feature-priority run

User requested preparation, verification and start from MODEL3B_FEATURE_PRIORITY_HANDOFF.md. This separate B adapter retains the completed Model3-A policy and frozen A source bytes.

Run: `model3/runs/cryptbb_v3_feature_priority_B_v1_20261115/B`.
Study: `handoff/cryptbb_v3_feature_priority_B_v1_20261115`.
Entry: `model3/src/feature_priority_b_v1/capri.py`.
Maintained modules: `Capri/src/capri_controller/training/cryptbb_feature_b_v1.py` and `cryptbb_feature_b_v1_metadata.py`. Sampler: unchanged `cryptbb_feature_v1_sampler.py`.

Existing 100000-image CryptBB v3 B corpus; frozen Model1-A epoch20 ranking from `model2/rankings/cryptbb_v3_epoch20_static_B_20261115/B`. Teacher SHA256 `a599a42d9a33e991cc7ae1f22ab92851bd3495e7d9f9fce517fdb1feb81cc8c3`. Student starts from scratch, seed48. No regeneration or new split. Batch128, 782 updates/100096 replacement draws each epoch, 20 epochs/15640 updates. Original architecture, RGB bicubic32x128 preprocessing, AdamW, OneCycle, BF16 and clipping retained.

Teacher thirds remain immutable. Epoch1 easy-only and epoch2 equal easy/medium reproduce static B image IDs exactly. From epoch3, both bucket mass and conditional sampling adapt. Three highest-priority unlearned families are active; accuracy strictly >75% marks learned, learned attention .25, <=75% restores eligibility. Geometry/effect/content importance uses saved pixel-reference prevalence. Fitted font-set importance uses distinct reference coverage, not verified real font identity. B derives its own metadata settings and quintile bins by A's unchanged rules.

Cell demand is prevalence * attention * (1-accuracy) * min(1,n/100) / number_of_family_settings. Bucket masses mix 20% uniform coverage with normalized demand; conditional image probabilities spread cell demand across its images, sum overlapping demand and mix 20% uniform coverage. Deterministic draw RNG, rounding and tie-breaking unchanged. See A's RUN_GUIDE.md for the full shared method and protocol.json for frozen B identities.

After every epoch, frozen-student FP32 GPU inference assesses all100000 synthetic images in ascending ID order, batch256 and final160, with ordered CPU prefetch and no answer tokens or lengths. All training RNG and weights are preserved. Epoch2 assessment sets epoch3 sampling. Training and assessment seconds are reported separately. Fixed256 repeatability is checked; universal cross-batch invariance is not claimed. Read readiness.json for actual B early/mature probes.

Primary selection remains fixed final20. No real labels or diagnostic accuracy influence training, thresholds or checkpoint selection. This launch queues no evaluator or heartbeat and no follow-on experiment. A's paused heartbeat remains unchanged. Future real diagnostics must use separately adapted B paths, freeze predictions before labels and verify the same251-image identity before comparisons with Model1-B and static Model2-B. No real-transfer advantage is established.

Checks: installation.json verifies A frozen sources and final checkpoint; adapter_verification.json records exact declared source adaptation and disposable worker lifecycle/STOP/failure checks; scenario_verification.json records independent20epoch policy scenarios; features/manifest.json records B corpus/ranking hashes and groups; readiness.json records all100k labels, scratch parity, source/config parity, full128 exact model/optimizer/scheduler/scaler/RNG/sampler replay, fixed256 inference repeatability and mature A20 comparison. protocol.json/source freeze all production dependencies and check scripts before launch. launch_verified.json confirms repeated real optimizer updates and process creation identity.

Inspect live_monitor.json in this study and status.json/history.json in the run. A committed epoch requires its checkpoint JSON receipt and matching checkpoint SHA; an assessment-in-progress epoch is not yet committed. The hidden coordinator keeps the system awake and stops after B20. No automatic restart.

STOP in this study is checked before training/inference batches. Do not remove STOP or recover without explicit user authorization. Explicit `worker --resume` uses only a verified committed epoch and archives partial next-epoch artifacts; the coordinator's launch is fresh-only. Never rerun prepare or launch against this existing protocol/output.

Python: `E:/pranav/Final sem/sourcefree_synth/experiments_xgb/.venv/Scripts/python.exe`, with `-B -u`; canonical PYTHONPATH is `E:/pranav/Thesis-Pranav-Final/Capri/src`. All project artifacts stay on E:.
