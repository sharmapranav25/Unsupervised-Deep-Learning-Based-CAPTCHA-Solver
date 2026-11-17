# CryptBB full vertical: next-session handoff

CURRENT OVERRIDE14November2026UTC: preparation is complete, heartbeat paused, no training started. Use datasets_ready_v2.json, prepared_fixed32x128.yml, preflight_fixed32x128/receipt.json and agreement_protocol.json. The old synthetic-validation launcher gate has been replaced, and upstream resume is explicitly blocked pending validation. See handoff/cryptbb_model1_preparation/REPORT.md. Historical progress/gap descriptions below are preserved and superseded where this override applies. CryptBB Model2/3 adapters remain pending; learning-progress/SFL discussion is not implementation authorization.


Prepared 13 November 2026. Workspace: E:/pranav/Thesis-Pranav-Final. Read AGENTS.md, README.md and handoff/CURRENT_STATE.md before acting. All outputs, caches and temporary files remain on E:. Paths below are workspace-relative unless absolute.

## Scope and authorization

The intended full vertical is PARSeq Model1-A/B -> opposite-Model1 ranking -> Model2-A/B (fixed curriculum) -> Model3-A/B (adaptive curriculum) -> agreement-selected real evaluation and family report. All six readers use SVTRv2LNConvTwo33 + PARSeq, approximately20.1M parameters.

CURRENT AUTHORIZATION: finish CryptBB datasets and Model1 setup only. This request creates a handoff; it does not authorize reader optimization or launching the full vertical. Obtain a new training instruction before starting readers. Do not start City, DAN, ASTER or other STR models. GLM-OCR is a separately authorized external inference baseline, not a CAPRI component. Do not add pseudo-labeling or additional training images to this comparison.

## Live preparation: inspect before doing anything

Coordinator was PID58132, child wrapper59928 / interpreter139336 for generationB at the latest process verification. PIDs are historical observations, not permanent identities: check live command lines and creation times. Never duplicate or interrupt a live job.

- handoff/cryptbb_model1_preparation/status.json
- active_child.json, preparation.lock.json, events.jsonl in that folder
- coordinator.stdout.log / coordinator.stderr.log and generate_A.log / generate_B.log / pack_A.log
- datasets/cryptbb/{A,B}/corpus_status.json, corpus_summary.json, corpus_audit.json, data_receipt.json

At handoff preparation: A has100000 unique decoded images, packed LMDB, zero exact overlap with original/reference/historical real pixels. B is generating (6000 images at the sampled read; use live status for current count). After B, the coordinator packs it, checks A/B overlap, writes configs and performs forward-only preflight. Do not claim completion from image count alone.

The preparation coordinator owns status and active_child files; do not overwrite them. Do not modify frozen generation code/configs while it runs: protocol.json records their hashes. Its STOP file stops subsequent stages. Inspect failures before resuming; partial fits/packing must not be overwritten blindly.

Read-only checks after sourcing environment.ps1:

```powershell
Set-Location 'E:/pranav/Thesis-Pranav-Final'
. ./environment.ps1
Get-Content handoff/cryptbb_model1_preparation/status.json
Get-Content handoff/cryptbb_model1_preparation/active_child.json
& $thesisPython -B model1/src/train_cryptbb.py --dataset A
& $thesisPython -B model1/src/train_cryptbb.py --dataset B
```

The last two commands do not train, but currently check an OBSOLETE synthetic-validation gate described below. Their success alone is insufficient.

Only if no preparation process is active and unfinished work can safely resume, the existing preparation command is:
`& $thesisPython -B Capri/capri.py prepare-cryptbb-model1 --stage run`

Completion requires completion.json, datasets_ready.json, both data_receipt.json files, both model1/runs/cryptbb/{A,B}/preflight/receipt.json passed, and200000 unique pixels with zero A/B overlap. Verify config/LMDB/charset hashes and source mappings. This is dataset/setup completion, not full training readiness. Pause the prepare-cryptbb-after-darkbay heartbeat once preparation finishes; do not broaden that automation to training.

## Data and provenance

Selected CAPRI registry: Capri/models/registry.json. Selected checkpoint: Capri/models/capri/checkpoint.pt, SHA25628f95fd8daf3831fb860ab59e15b8b65bd4a68bacc434eb7eb7a296e6190b764. Epoch20 EMA, internal count head; separate counter is not used for generation. Legacy model is calibration provenance, not the selected generator.

Fresh100-reference pixel-only vote: length2:1,3:5,4:20,5:70,6:3,7:1. Length5 was accepted as the unique leader with70/100 votes under the >=50/100 rule. This is vote agreement, not accuracy or a40% threshold. Never manually choose a family length or use real strings to generate.

Each fixed pool contains100000 images:33334 custom,33333 DINO,33333 mixed. A/B use different render seeds but the same controller recipes/mixture. All model variants reuse these exact pools and shared LMDBs; do not regenerate data for Models2/3 or create copies inside reader folders.

Paths:
- Generator: Capri/runs/cryptbb_model1_preparation_20261113 (inputs, selected_generator, generator, previews).
- Corpora: datasets/cryptbb/{A,B}/corpus; shared training LMDBs: datasets/cryptbb/{A,B}/train_lmdb.
- Charset: datasets/cryptbb/charset.txt; per-pool sample_provenance.jsonl and data_receipt.json.
- References:100 opaque images under the generator run's inputs, all drawn from original600 pixels.
- Real test: evaluation_data/cryptbb/images and image_manifest.json,551 images.
- Isolated scoring labels: evaluation_data/cryptbb/scoring_labels/scoring_labels.json and provenance.json. Historically documented filename labels, not independently reannotated.
- Historical selftrain100 is excluded. Reference/test/selftrain exact pixel overlaps are zero; resized similarity checks are heuristic, not proof against all transformed duplicates.
- Source audit: handoff/cryptbb_model1_preparation/source_audit.json, source_pixels.json, setup_receipt.json and fresh_count_verification.json.

Original collection: E:/pranav/CryptBBCaptchas(unsolved)07272023-20260501T023431Z-3-001/CryptBBCaptchas(unsolved)07272023.
Historical reference: E:/pranav/CAPRI/runs/capri1_vs_capri2_controller_examples_20261029/inputs/CryptBB.
Historical selftrain/test: E:/pranav/cryptbb_sourcefree/data/cryptbb_split.
Preserve originals and existing directory names.

## Critical implementation gaps before reader launch

1. Current model1/src/train_cryptbb.py still requires validation_protocol.json with highest_synthetic_exact_match_earliest_tie. Preparation receipts also contain this old text because the job was frozen before the user adopted agreement selection. Do NOT fabricate that file to bypass the gate. After preparation completes, add a versioned agreement protocol/addendum and update the launch gate, launch receipts and readiness wording to the adopted rule. Preserve original preparation protocol and receipts.
2. Capri/src/capri_controller/training/ab_curriculum.py is DARKBAY-SPECIFIC: STUDY, paths, datasets, teachers, evaluations and coordinator assume DarkBay. Do not run its coordinate command for CryptBB or edit historical protocol files. Create a family-aware implementation or a reviewed CryptBB adapter under Capri/src/capri_controller/training and thin adapters in model2/src and model3/src. Register a new explicit capri.py route. These CryptBB Model2/3 commands do not exist yet; never present old commands as ready.
3. The Model1 launcher uses the upstream trainer and lacks the curriculum runner's GPU arbitration. Do not blindly run two full readers concurrently on the16GB GPU. Use sequential runs or validate safe batch-level arbitration preserving batch128 and BatchNorm. The DarkBay pair serialized GPU batches; parallel processes did not imply2x throughput.
4. Audit Model1 --resume before promising deterministic continuation: it reseeds module NumPy generators after Trainer construction. Verify restored optimizer/scheduler/scaler/global RNG and decoder RNG, or repair/document it before any resume. Curriculum resume must restore sampler and adaptive state as well.
5. Freeze teacher selection separately from reporting selection: match DarkBay's opposite-Model1 EPOCH20 teachers unless an explicit protocol change is approved. Do not silently replace ranking teachers with agreement-selected reporting checkpoints.

## Prospective checkpoint-selection protocol

User replaced synthetic-validation selection with A/B same-epoch exact full-string agreement. Authoritative precedent: handoff/darkbay_curriculum_20261113/agreement_selection_policy.json; implementation reference: Capri/tools/analyze_ab_epoch_agreement.py (adapt family paths, counts and checkpoint inventory; do not execute unchanged).

Before CryptBB reader training/results, save a CryptBB selection protocol with frozen pixel manifest/hash, prediction normalization/decoder, candidate epochs, tie handling and all exclusions. For each model type, compare A and B predictions on identical unlabeled images at the same epoch; maximize exact full-string agreement, earliest epoch on a tie. One selected epoch per model type shared by its A/B pair. Raw full-string agreement is the adopted rule: do not tune confidence thresholds, exclude disagreements, or replace it with temporal/consecutive-epoch agreement.

The existing Model1 checkpoint archive saves epochs2-20. Recommended matched candidate window is2-20 for all models; freeze it in the new protocol. Alternatively save epoch1 consistently for all six BEFORE starting, explicitly documenting that change. Never silently compare different candidate sets.

For a direct DarkBay-style replication, freeze the551 staged real images as the unlabeled agreement pool and subsequent scoring pool. State explicitly that this is transductive checkpoint selection: real pixels are used for selection, never training/gradients. It is not an untouched independent test. A separate unlabeled selection pool would be a protocol change requiring a documented prospective split and provenance audit; do not silently substitute the100 generator references or excluded selftrain images.

Produce/freeze predictions and selection choices without loading scoring labels; score only frozen choices with the separate labels. Real epoch diagnostics, if requested, must never change the rule or training. Shared wrong outputs can increase agreement; agreement does not guarantee the hindsight-best real epoch. Do not optimize the method against CryptBB test accuracy. No need to generate a fourth-controller synthetic validation set to satisfy this superseded launch condition.

## Training sequence and matched budget

All six runs start FROM SCRATCH, seed48 including module NumPy RNGs. No Model1->Model2 or Model2->Model3 weight transfer, and no DarkBay reader weights. Same architecture, charset, preprocessing, optimizer and100k pool per A/B comparison. A/B are dataset replicates, not three seed repeats.

Budget per run:20 epochs, batch128,782 full steps/epoch,100096 draws/epoch,15640 optimizer steps. The original sampler pads the final batch; the budget is not exactly100000 draws. Baseline config: RGB PIL bicubic32x128, ToTensor/Normalize(.5,.5), no added augmentation; AdamW lr0.00035 weight_decay0.05 with original exclusions, OneCycle warmup2/20, BF16 AMP, clip20. Derive exact settings from model1/configs/cryptbb/{A,B}/prepared.yml and audit against model1/configs/darkbay/{A,B}/relocated.yml.

1. Finish data/setup and freeze the corrected protocols. Once newly authorized, train Model1-A and Model1-B for20 epochs with no real validation loader. Save immutable epoch checkpoints and completion receipts. Evaluate agreement to choose reporting checkpoints, but retain epoch20 for ranking teachers.
2. Rank all100000 images in A using CryptBB Model1-B epoch20; rank B using CryptBB Model1-A epoch20. Freeze teacher checkpoint hashes, pool identity hashes and scores. Model1 must not rank its own training pool for initial curriculum. BOTH Models2 and3 reuse the identical initial opposite-Model1 ranking.
3. Train Model2-A/B from scratch on those same pools using fixed ranks. DarkBay static masses: epoch1 easy/medium/hard=(1,0,0); epoch2=(0.5,0.5,0); epochs3-20=(0.15,0.25,0.60). Sampling with replacement preserves the matched step budget. Do not regenerate images or change ranks.
4. After both Model2 runs complete, train Model3-A/B from scratch using the SAME initial rankings. After each epoch, its own reader rescores the entire corresponding synthetic pool and updates its adaptive sampler, including the final epoch20 rescore for matching accounting. Initial teacher is Model1; later adaptive updates are Model3's own predictions against known SYNTHETIC labels, never real labels or newly generated data.
5. Preserve ds3_data.AdaptiveSampler.update behavior:0.5 rank smoothing, equal thirds, risk1-exp(-min(NLL,50)), masses bounded0.2-0.5, maximum0.1 change and original uniform reference. Inspect implementation rather than reimplementing from this prose. Keep rank updates/weights/draw traces per epoch; separately report training and full-pool rescoring time.
6. Complete all20 epochs regardless of interim real results, evaluate frozen agreement rule for each pair, freeze selected epochs, score/report separately for A/B and optionally their arithmetic average. Average accuracy is not ensemble accuracy. Stop after CryptBB report; City requires its own readiness/authorization.

Ranking/scoring must match existing ab_curriculum.py: FP32 free autoregressive decoding with normal refinement/early stop, character-only renormalization and summed synthetic-label NLL; ordinal tertiles with image-ID tie-break. If output horizon is too short, retry fixed architecture max_length25 as already implemented. Do not pass real labels or manually chosen target length to decoding.

## Verification before production and during runs

Check all paths resolve to CryptBB; no accidental DarkBay checkpoint/LMDB/evaluation output. Validate sample identity/charset/preprocessing parity, synthetic rank coverage and finite scores, exact sampler draw budget, Model2/3 identical first-epoch ranking basis, adaptive replay over20 updates, and deterministic saved-state continuation. Existing DarkBay readiness() and readiness.json provide the reference checks. Production-optimization pilots require training authorization; current preparation preflight is forward-only.

Save checkpoints only as committed epochs with config/code/pool/protocol hashes and optimizer/scheduler/scaler/RNG/sampler state. Resume only a known committed checkpoint, not partially written files. Check actual processes and locks before every launch/resume; failures stop the vertical until diagnosed. Preserve failed artifacts and never report a failed/partial model as complete.

## Output locations and report

Existing/prepared: model1/configs/cryptbb/{A,B}/prepared.yml; model1/runs/cryptbb/{A,B}; datasets/cryptbb/{A,B}; evaluation_data/cryptbb.
Planned (create when implementing): model2/rankings/cryptbb/{A,B}; model{2,3}/configs/cryptbb/{A,B}; model{2,3}/runs/cryptbb/{A,B}; model{1,2,3}/results/cryptbb/{A,B}; handoff/cryptbb_curriculum_<date>; thesis/figures/cryptbb. Model3 adaptive epoch artifacts remain within its run. Reuse shared pools, do not flatten or move existing folders.

Report selected epoch, agreement count/denominator, A/B exact correct/551, exact accuracy and CER, average when requested, training vs rescoring time, fixed20 diagnostics separately, checkpoint paths/hashes, all failures and limits. Record raw curves; any smoothing is descriptive and does not select checkpoints. Colors for plots: Model1 green, Model2 blue, Model3 red.

DarkBay primary reference: agreement-selected Model1 e17 mean66.28%, Model2 e12 mean72.50%, Model3 e13 mean71.22%. Hindsight peaks mean69.89/73.67/72.83% are diagnostic only. One family does not establish adaptive/static equivalence. Fixed image pools and matched compute budget isolate curriculum from extra-data effects; no claim of equal wall-clock cost.

GLM-OCR CryptBB inference finished: Capri/runs/glm_ocr_cryptbb_20261113/summary.json reports188/551=34.119782%, CER28.752719%, no token-limit hits. Revision2e85a62840ccac27daa451df36c736c4636b8628, official processor/prompt, no family tuning. Prediction hash3052c10a4ff05a2ee12c64cd9a9da8356198d4d3f5fbabfba2144cc63085958d. Independent post-run integrity/score verification and completion documentation remain outstanding; inspect existing artifacts before any rerun. Historical DarkBay GLM baseline645/900=71.67%. Do not use GLM outputs for generation or reader labels.

## End-of-session maintenance

Update handoff/CURRENT_STATE.md, CHANGELOG.md, SOURCE_MAP.json and workspace inventory with concrete receipts. Record code/protocol adaptations in Capri/docs/MAINTENANCE_LOG.md. Preserve the old selection artifacts and frozen preparation sources; add versioned superseding records rather than rewriting history. Leave exact active commands/PIDs, committed stages, failures and next safe action. User-facing wording: all readers train exclusively on synthetic images; unlabeled real prediction agreement selects reporting checkpoints.
