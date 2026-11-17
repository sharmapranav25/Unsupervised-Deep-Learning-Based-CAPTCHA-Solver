# Current plan: Dataset B, one reader run

User cancelled repeated training runs. Only the single seed_48 configuration is executed; 48 is a recorded reproducibility setting, not a multi-seed experiment. Both A and B must pass audits before either reader trains. Use the A/B coordinator, not the historical three-run launcher. Study: E:\pranav\CAPRI\runs\darkbay_ab_study_20261112

The earlier setup guide below is historical and superseded where it specifies three runs.

# CAPRI2 DarkBay PARSeq experiment

Authorized full run:100 references ->300 custom/DINO/mixed recipes ->100,000 unique synthetic images ->three PARSeq seeds48,123,456 for20epochs each ->same900-image real evaluation.

Read pipeline_status.json for the orchestrator stage, generator/status.json for controller progress, generator/corpus_status.json for corpus generation, and seed_<seed>/train.log for actual reader progress. Preparation and training are separate stages. Final completion requires experiment_completion.json. All artifacts are under this E-drive directory.

The CAPRI1 reader configurations are copied with output/data/charset paths relocated only. Initialization is from scratch. AdamW lr0.00035,weight_decay0.05,batch128,OneCycle with2-epoch warmup,mixed precision. Fixed final epoch20 checkpoints. Baseline reports782 batches/epoch and15,640 total steps per seed; the new loader must match before training. No curriculum or new augmentation. All100k generated images are retained; strict readability is diagnostic, not a filter.

CAPRI2 uses its final epoch20 EMA, internal count and trained DINO residual font heads. Full100-input fitting uses all three existing controllers with original CAPRI1 calibration features reconstructed from its first24 recipes. Count threshold50%, no manually supplied family length. Frozen checkpoints, source snapshots and provenance are in protocol.json and provenance/. Nothing is deployed by this experiment.

Preflight on existing baseline synthetic data tests the unchanged reader without optimizer updates, then a second preflight tests the actual CAPRI2 corpus. Dataset audits decode and hash all PNGs, verify100k unique images, controller balance and synthetic text labels, and exclude exact overlap with100 fit and900 test pixels. Evaluation reads real labels only after all final predictions are hash-frozen. Read-only baseline remains runs/parseq_darkbay_100k_20261005.

Start/resume from E:/pranav/CAPRI using the configured E-drive Python: capri.py train-capri2-parseq --out <this directory> --resume. A live pipeline lock prevents duplicate runs. Incomplete controller or LMDB stages stop for inspection; they are not silently overwritten. Corpus shards resume only if their hashes and recipe manifests match. Reader recovery uses the existing baseline checkpoint mechanism; an interrupted epoch may be replayed and bit-exact resume is not promised. Create STOP to stop after the current stage; it does not interrupt a reader epoch or kill another job.

Initial budget: roughly7-9hours total based on the old three-reader duration (~5h43m), plus full controller fitting, corpus generation and audits. Actual ETA depends on measured throughput and other GPU use. Weak/faint generator outputs remain a known limitation; success does not imply better reader accuracy.
