# CryptBB datasets and Model1 setup complete

Completed14 November2026 UTC. Reader training has NOT started. Preparation-only follow-up prepare-cryptbb-after-darkbay is paused.

- Dataset A:100000 images; Dataset B:100000 images;200000 unique pixels combined; zero A/B pixel overlap.
- Each pool:33334 custom,33333 DINO,33333 mixed. Both audited/decode-checked and packed as shared LMDBs. No exact overlap with audited original/reference/historical real pixels.
- Length5 came from fresh pixel-only70/100 votes, unique leader meeting>=50/100. Not a manually selected length or40% accuracy threshold.
- Real evaluation:551 staged images; historical selftrain100 excluded; scoring labels isolated.

## Authoritative locations

All paths relative to E:/pranav/Thesis-Pranav-Final.

- Generator/controller provenance: Capri/runs/cryptbb_model1_preparation_20261113.
- Corpora, audits and LMDBs: datasets/cryptbb/{A,B}; train_lmdb/data.mdb; data_receipt.json; sample_provenance.jsonl.
- Shared charset: datasets/cryptbb/charset.txt.
- Real pixels/manifest: evaluation_data/cryptbb/images and image_manifest.json.
- Separate scoring labels: evaluation_data/cryptbb/scoring_labels.
- Corrected configs: model1/configs/cryptbb/{A,B}/prepared_fixed32x128.yml.
- Forward-only checks: model1/runs/cryptbb/{A,B}/preflight_fixed32x128/receipt.json.
- Current setup mapping: handoff/cryptbb_model1_preparation/datasets_ready_v2.json.
- Completion and selection: handoff/cryptbb_model1_preparation/completion.json and agreement_protocol.json.
- Future launch/default read-only readiness helper: model1/src/train_cryptbb.py.

## Failure and repair

The original prepared.yml copied DarkBay's ratio-dependent loader settings. CryptBB's200x60 pixels fall in ratio3, whose default resize is40x112; preflight required32x128 and correctly stopped. No training occurred. Original config, data and failure log preflight_A.log remain unchanged.

New configs add only loader min_ratio=4 alongside max_ratio=4, forcing the intended32x128 transform without changing original pixels or LMDBs. Maintained repair implementation: Capri/src/capri_controller/cryptbb_preflight_repair.py, entry Capri/capri.py finish-cryptbb-preparation. Original frozen preparation module remains unchanged. Both checks passed batch128,782batches/epoch,20076927parameters, finite forward outputs, no backward/optimizer steps, unchanged parameters/buffers and eight exact transform-parity comparisons per pool.

The original preparation protocol's synthetic-validation requirement was superseded by the user's adopted agreement rule. agreement_protocol.json fixes same-epoch exact A/B string agreement, earliest tie, epochs2-20, and the551-image pixel manifest. Labels are not opened for selection. The same pixel pool is subsequently scored, so this is transductive selection, not an untouched independent test. Ranking teachers remain opposite Model1 epoch20, distinct from reporting checkpoints.

## Limits and next action

No new reader-training authorization. New Model2/3 family-aware adapters and their training readiness checks are still future work, documented in CRYPTBB_VERTICAL_HANDOFF.md. The Model1 launch helper now blocks resume until upstream decoder RNG/optimizer continuation is validated; its default invocation only checks readiness. A whole-run lock prevents simultaneous unarbitrated Model1 A/B training. Do not run the original preparation coordinator again merely to replace the historical failed event; use current completion receipts.

Labels follow documented historical filename conventions and were not independently reannotated. Exact pixel audits do not prove absence of all transformed near-duplicates. Agreement can be jointly wrong. No learning-progress or SFL code was implemented; those remain discussion-only.
