# Model3 B feature priority training handoff

## Next chat

Read this handoff, then prepare, verify and start CryptBB Model3-B using the same feature-priority adaptive method as completed Model3-A. The user explicitly requested this handoff so the next chat can start B. This turn writes documentation only; B was not launched. Do not redesign the curriculum in response to A real-test results. Older A-only authorization statements describe the historical A run; this user request establishes the next-chat B scope.

Canonical workspace: E:/pranav/Thesis-Pranav-Final. Read its AGENTS.md, README.md and handoff/CURRENT_STATE.md, plus applicable nested instructions. Before controller changes, read Capri/README.md and Capri/docs/CODE_AND_MODELS.md where present. All project outputs and temporary artifacts stay on E:. Follow maintained-code, changelog and inventory requirements.

## Critical launch limitation

The existing model3/src/feature_priority_v1/capri.py entry imports Capri/src/capri_controller/training/cryptbb_feature_v1.py. That implementation is hard-coded for A: output path, configuration, dataset, static baseline, readiness initial tensor comparison, locks, logs and teacher checks. cryptbb_feature_v1_metadata.py also hard-codes A data, A ranking and teacher B. There is no verified dataset-B CLI switch. Do not simply run the current launch command or assume --dataset B works.

A source hashes are frozen in its protocol and checked by diagnostics. Preserve those original source files and artifacts. Implement a separately versioned B adapter/metadata builder or another isolated solution that leaves A frozen source bytes intact. Reuse the unchanged sampler policy where appropriate. Keep maintained implementation under Capri/src/capri_controller/training and a versioned adapter under model3/src. Choose fresh descriptive B study/output paths, document them, and freeze B sources and protocol before launch. Do not overwrite the existing A protocol, features, readiness, launch receipts or outputs.

## Completed A evidence

Family: cryptbb_v3_feature_priority_v1_20261115.
Run: model3/runs/cryptbb_v3_feature_priority_v1_20261115/A.
Study: handoff/cryptbb_v3_feature_priority_v1_20261115.
Read RUN_GUIDE.md and protocol.json there for the exact implementation, source manifest and policy.

A completed epoch20, global step15640. Coordinator status is complete. Final checkpoint SHA256: fc6df1515ca82526d4dce774f5d10672809fcbe894cd4a7d6c0e603ddab55f52.
A protocol SHA256: cb6c931940cc5bbf2a1781a3789e874f31746300804fb2d50a1b602dbabafa7a.
Total training seconds6502.485; assessment seconds3383.905, reported separately.

All20 A real CryptBB diagnostics have passed verification under model3/runs/cryptbb_v3_feature_priority_v1_20261115/diagnostics/real_cryptbb/epoch_XX. Correct counts out of251, epochs1 through20: 0,35,30,26,38,50,49,49,52,44,66,52,53,56,54,59,64,64,64,65.
Final20 A: Model1 59/251 (23.51%), static Model2 64/251 (25.50%), adaptive Model3 65/251 (25.90%). A best observed66/251 at11 is retrospective diagnostic only; primary selection remains fixed final20.
Use individual summary.json and verification.json as authoritative. Aggregate all_checkpoints_summary.json may be stale because later epochs were evaluated directly.

The heartbeat measure-new-model3-a-checkpoints is PAUSED after completion; monitor_reported.json records20. Do not repurpose it silently for B. No new B schedule was created by this handoff.

## B data and teacher direction

Reuse existing datasets/cryptbb_v3/B, including train_lmdb, sample_provenance.jsonl and data_receipt.json. No regeneration, duplicate corpus or new split.
B must use the mature Model1-A epoch20 teacher ranking, not teacher B and not Model3-A weights. Expected ranking location to verify: model2/rankings/cryptbb_v3_epoch20_static_B_20261115/B. Verify its completion receipt, teacher identity, hashes, IDs, NLL ordering and immutable bucket membership before use. Verify the actual existing B paths rather than trusting naming alone.
Use model1/configs/cryptbb_v3/B/prepared.yml and the matching static B study handoff/cryptbb_v3_epoch20_static_B_20261115 for configuration, warmup and readiness comparisons. Start the student from scratch with seed48, not a teacher or A checkpoint.

## Preserve the exact method

100000 images;20epochs;batch128;782updates and100096draws per epoch;15640updates total. Preserve matched architecture, preprocessing, AdamW, OneCycle, BF16 and clipping configuration from the frozen implementation and static B baseline.

Teacher thirds are immutable. Epoch1 easy only; epoch2 equal easy/medium; verify the actual warmup image-ID sequences against static B. Adapt both bucket masses and within-bucket weights starting epoch3. The older fixed-outer-schedule handoff is superseded for this experiment.

Importance comes from prevalence in the100 saved pixel-reference predictions, not a manually invented ranking. Preserve the metadata feature definitions and reference mapping. Build per-image metadata from B provenance; never reuse A row assignments. Distinct fitted-font reference counts are representation evidence, not verified real font identity.

Three highest-priority unlearned families are active. Strictly greater than75% synthetic exact-string accuracy advances a family. Learned attention remains0.25; at75% or below it becomes eligible again. Weak subgroups retain demand. Active attention1; uniform coverage0.20; sparse-cell support100. Exact settings if at most8, otherwise metadata-only quintiles. Keep the same formulas and tie-breaking from RUN_GUIDE and sampler source. B metadata-dependent bins/counts must be derived by the same rules, not copied blindly from A.

Cell demand = reference prevalence * attention * (1 - accuracy) * min(1, cell images/100) / number of family settings. Bucket mass =0.20/3 +0.80*share of total demand; equal thirds if zero total demand. Within buckets, distribute cell demand over images, sum overlaps and mix20% uniform coverage. Preserve deterministic epoch RNG and rounding.

Full-pool assessment after every epoch: frozen student, FP32 GPU inference batch256, ascending IDs, final160 batch, ordered CPU prefetch. No target tokens or answer lengths supplied to inference. Preserve all training RNG and weights. Epoch2 assessment drives epoch3. Fixed256 reproducibility is required; universal cross-batch invariance was explicitly not established for early students.

## Required B preparation and launch checks

Inspect existing B artifacts, live process identities, GPU ownership, locks and STOP before creating or starting anything. Do not duplicate an active worker; preserve partial artifacts and investigate failures. Do not remove STOP or perform recovery without the relevant explicit user request.

Verify corpus/provenance hashes, all IDs and synthetic labels, cross-teacher ranking, initial model parity with static B, exact warmup streams, sampling budget, policy scenarios, source hashes and deterministic full-state resume behavior. Follow A readiness checks with correct B identities, including reproducible fixed-batch inference and mature-reader comparison. Do not weaken checks to obtain readiness.

Freeze a separate B protocol/source snapshot and passing readiness evidence. Launch through the new versioned entry using hidden coordinator/worker processes. Verify repeated optimizer-step progress and save process creation identities plus launch evidence; spawning a process alone is not a verified training start. No automatic follow-on experiments or restarts.

Python: E:/pranav/Final sem/sourcefree_synth/experiments_xgb/.venv/Scripts/python.exe. Use -B -u and canonical PYTHONPATH E:/pranav/Thesis-Pranav-Final/Capri/src where needed. Request tool filesystem escalation when necessary. Record implementation/location changes in handoff/CHANGELOG.md and applicable maintenance log; refresh inventory when relevant.

## Evaluation and scientific interpretation

Real labels are scoring-only. Freeze predictions before opening labels, hash-verify committed checkpoint receipts, reuse results and avoid duplicate evaluators. Do not use real accuracy to alter curriculum, thresholds, data, source code or checkpoint selection. The current A diagnostic module/launcher may also contain A-specific paths; inspect and adapt separately for B rather than assuming it supports the new run.

For B comparisons use same-epoch Model1-B summaries under model1/results/cryptbb_v3/B/requested_cpu_epochXX and static Model2-B under model2/results/cryptbb_v3_epoch20_static_B_20261115/B/epoch_XX, after verifying identity and the same251 image set. User cares primarily about real exact-string accuracy. Distinguish running epoch from fully committed and evaluated checkpoint. Report correct/251, accuracy and matched static comparison; keep ordinary automated progress quiet if later scheduling is authorized.

Do not interpret A's one-image final advantage as established superiority. The user has tried several settings and wants a scientifically honest result, not repeated tuning until adaptive wins. Discussion of held-out synthetic probes, controlled branches or alternative learning-value signals was conceptual only, not authorization to alter B or launch extra experiments.

The user requested these exact notes, saved in thesis/THESIS_NOTES.md:

Under our synthetic-only curriculum feedback and unlabeled target-reference setting, the tested adaptive approach did not demonstrate a clear real-accuracy advantage over static curriculum.

The limited improvement may reflect a mismatch between the adaptation signals and target-domain learning needs. Reference-feature frequency estimates prevalence, while synthetic group accuracy estimates synthetic difficulty; neither directly establishes which features contribute to real recognition errors or benefit from additional practice.

This explanation remains a hypothesis. The discussed positive literature used different feedback conditions, including labeled validation; it does not prove adaptation cannot help without target labels. Preserve that qualification in future thesis writing.
