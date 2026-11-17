# CryptBB cumulative static curriculum overnight handoff

## Next chat

Implement, verify, and launch a NEW CryptBB v3 Model2-A/B experiment from scratch using the cumulative static curriculum below. Run A first and B second, with one GPU training worker at a time. Monitor overnight and prepare per-epoch results, independent original-v1 checkpoint selections, and comparisons with Model1 and the previous static run. The user's latest request authorizes this next-session work; the earlier stop still applies to the OLD experiment. Do not resume its B checkpoint.

This handoff was prepared on 15 November 2026. No new training was started while preparing it. Preparation of a handoff is not proof of implementation, readiness, launch, or completion. Verify live state before acting.

Canonical training workspace: `E:/pranav/Thesis-Pranav-Final`. This file also lives at `E:/pranav/CAPRI/handoff/CRYPTBB_CUMULATIVE_STATIC_OVERNIGHT_HANDOFF.md`. Read canonical `AGENTS.md`, `README.md`, `handoff/CURRENT_STATE.md`, and `Capri/docs/CODE_AND_MODELS.md`; newest specific instructions here supersede historical README selection/scheduling descriptions.

Python: `E:/pranav/Final sem/sourcefree_synth/experiments_xgb/.venv/Scripts/python.exe`. Source canonical `environment.ps1` for maintained environment/PYTHONPATH configuration. Canonical workspace writes may require tool escalation when the current workspace is E:/pranav/CAPRI. Keep project artifacts on E:. Use hidden background processes and exact PID plus creation-time identity checks.

## Method in simple words

The teacher grades every synthetic image once. Split the sorted images into easy, medium, and hard thirds. Start with easy examples, add medium examples, then train equally on the entire dataset. Earlier examples stay available. No new training images are added, and difficulty scores never change during the run.

The previous run used the same equal thirds but, from epoch 3 through 20, sampled 15% easy, 25% medium, and 60% hard. The new run removes that persistent preference for hard examples. Equal thirds have not been shown to be wrong. The teacher's roughly 2% whole-string accuracy in A's hard third does not prove those images are unlearnable or irrelevant.

| Epoch | Available examples | Frozen sampling rule |
|---|---|---|
| 1 | Easy third | 100096 draws with replacement from easy |
| 2 | Easy and medium | Preserve original 50048 easy + 50048 medium draws; shuffle |
| 3 through 20 | All 100000 images | 100096 draws uniformly with replacement from the entire pool |

The two first epochs retain the original sampler streams and behavior to minimize differences. Epoch 2 is approximately uniform over its union because the two groups differ by one image. From epoch 3, use exact equal per-image probabilities over the whole pool. Bucket counts will fluctuate around one-third; do not mistakenly require identical realized counts or reuse the old 15/25/60 assertion. Preserve the existing seed-derived per-epoch RNG scheme and freeze the exact implementation and draw hashes.

This is a fixed, predeclared candidate schedule, NOT a schedule already optimized on synthetic validation. Freeze it before launch. Do not choose transitions, feature weights, stopping points, or curriculum parameters using the 251 real answers or accuracy reports.

## Scope and validation decision

Overnight priority is TWO primary training runs: cumulative-static A and cumulative-static B, seed 48. This is not authorization to silently expand to eight full training runs or a multi-seed search. Preserve the full existing 100000-image A and B training pools for comparability with the historical runs. Do not silently resurrect the old 95k/5k split, hold out A using B as validation, or regenerate the corpus.

Our general methodology recommends reference/style-disjoint synthetic validation when choosing a schedule. The existing feature-analysis split is NOT a clean validation set for these readers: teacher B already learned shared recipe families, and full-corpus diagnostics were inspected. For this overnight pair, perform no schedule search; use the predeclared schedule above. A future schedule-selection study must define a genuinely separate synthetic validation protocol and apply identical training/validation membership to every compared method. Do not claim the overnight pair completes that study.

Feature metadata is diagnostic here. Do not add feature-combination weighting, target-similarity reweighting, quality filtering, new difficulty definitions, or adaptive rescoring. Preserve the pixel-only controller workflow, provenance and controller mixture. No VLM, real target strings, or manually chosen family length.

## Reuse the verified teachers and rankings

The selected Model1 checkpoints are ranking teachers only, never initialization weights:

| New reader | Dataset | Ranking teacher | Teacher SHA256 |
|---|---|---|---|
| Model2-A | CryptBB v3 A | Model1-B epoch 13 | `556d95df5b4b0f471dbebafab85d6a1bb6aee2cfbc78a1a5b29d78e3c4c9a002` |
| Model2-B | CryptBB v3 B | Model1-A epoch 12 | `feb37b75d99890f30f66d5545ca661306aaa1949f496dc16333db37c89adacfc` |

Checkpoints: `model1/runs/cryptbb_v3/B/epoch_13.pth` and `model1/runs/cryptbb_v3/A/epoch_12.pth`.

Selection manifest: `handoff/cryptbb_v3_model2_rank1_v1_20261115/teacher_selection.json`; SHA256 `82546ce9b99494f7268553802014b5070be4004a2764fbd4bdfd6443f9687051`.

Existing compatible rankings: `model2/rankings/cryptbb_v3_rank1_v1/{A,B}/ranking.npz`.

- A ranking SHA256: `9bd75fa67421da57a51976e7229bfc91efa25f8756e36d4836a1f7c80952b752`.
- B ranking SHA256: `313fe5177989eae33d0dd028d1c770d976e7973b1fc3f2d7ca38166fbd7ec53c`.
- Verify teacher checkpoint bytes, ranking bytes, completion receipts, data identities, finite nonnegative scores, IDs 1..100000 exactly once, and ordinal thirds of 33334/33333/33333 with ID tie-breaking.
- Ranking method stays the original free-AR plus one-refinement, synthetic-only summed character NLL. Do not replace it with teacher forcing, length-normalized loss, correctness bins, or a new feature score.
- Reuse valid rankings read-only; avoid wasting overnight time rescoring the same unchanged inputs. Write a NEW import/provenance receipt referencing their OLD protocol. Do not rewrite old completion receipts to claim they were generated under the new protocol.

Data: `datasets/cryptbb_v3/{A,B}/train_lmdb`; shared `datasets/cryptbb_v3/charset.txt`. Original receipt LMDB hashes: A `837cdf8826b539812872dc51bd7849e93f5142f8e800d945c13119349f1d53c2`, B `5824df6983de37b02d7659bf286fbbdd1f3316a8005975aec064fe8042d9f13a`. Use receipt-defined hashing, not a guessed hash of a directory.

## Preserve training settings and old evidence

- Independent scratch initialization, seed 48; no Model1 or previous Model2 weights or optimizer state.
- Same PARSeq architecture, charset and model configs from `model1/configs/cryptbb_v3/{A,B}/prepared.yml`.
- 20 epochs, 782 updates per epoch, batch 128, 100096 presentations per epoch: 15640 updates and 2001920 image presentations per reader. Epoch here means a fixed update budget, not one traversal of the currently admitted subset.
- AdamW learning rate 0.00035, weight decay 0.05; existing OneCycle warmup 2/20, BF16, gradient clip 20. Preserve all remaining optimizer/scheduler settings from the source config.
- Existing RGB PIL bicubic 32x128 preprocessing, ToTensor and Normalize(.5,.5); no new augmentation.
- Checkpoints must retain optimizer, scheduler, scaler, Python/NumPy/Torch/CUDA/module RNG and sampler state for exact recovery.

Old experiment: `model2/runs/cryptbb_v3_rank1_v1/{A,B}`. A completed 20; B stopped during epoch 9 with epoch 8 fully committed. Stop receipt: `handoff/cryptbb_v3_model2_rank1_v1_20261115/user_stop_20261115T092052Z/receipt.json`. Keep its STOP marker, checkpoints, source snapshots and all frozen receipts intact. Do not restart its coordinator.

Use separate new paths, after checking for an existing valid run of this exact protocol:

- Run family: `model2/runs/cryptbb_v3_cumulative_static_v1_20261115/{A,B}`.
- Study/protocol/status: `handoff/cryptbb_v3_cumulative_static_v1_20261115/`.
- Evaluations: `model2/results/cryptbb_v3_cumulative_static_v1_20261115/{A,B}/epoch_XX/`.
- Final report: study directory `MORNING_REPORT.md`, with machine-readable results and selection receipts beside it.

## Implementation and required launch checks

Relevant existing modules under canonical `Capri/src/capri_controller/training/`:

- `cryptbb_v3_model2_selected_v1.py`: existing selected-teacher adapter and readiness patterns.
- `ab_curriculum.py`: model construction, training, checkpointing and fixed-budget draws. Its `draws_for` HARDCODES the old 15/25/60 rule.
- `ds3_data.py`: sampler state container; its adaptive-policy invariants are not a new static curriculum specification.
- `cryptbb_v3_model2_sequential.py`: historical A-then-B observer. Its B-running loop does not honor STOP promptly; do not copy that defect.
- `cryptbb_v3_model2_checkpoint_evaluation_v2.py`: established requested evaluation path, with hardcoded old run locations that require careful adaptation.

Create a separately named maintained adapter/entry for the new experiment. Do not edit source files covered by an old frozen source manifest merely to redirect them. A small new adapter may reuse unchanged helpers, but all worker, coordinator, pilot, verifier and evaluator paths must refer to the new study. Use the maintained CLI where available; otherwise add a clearly documented entry with setup/check/worker/coordinator/evaluate stages. Do not invent a runnable command in the handoff before it exists.

Before production launch:

1. Inspect exact live process identities, old stop records, GPU use and new destination contents. Never duplicate a worker or overwrite an unreviewed partial run.
2. Freeze protocol, input/teacher/ranking provenance, code hashes, sampling seed and schedule; no real labels read by setup/ranking/training/selection.
3. Generate and verify all 20 epochs of draw IDs for each reader. Check permitted pools, draw counts, zero dropped batches, exact first-two-epoch equivalence to the old sampler, reproducibility, and uniform full-pool probabilities from epoch 3. Preserve the exact complete draw sequence for later exposure controls.
4. Verify preprocessing parity on both datasets, scratch initialization fingerprints, no pretrained/resume config, a disposable full-batch training pilot, finite loss/gradient, and exact next-step checkpoint replay including RNG/sampler/optimizer/scheduler. Pilot weights must never become production initialization.
5. Verify epoch transition and completion behavior, immutable archives, prompt STOP handling while A or B runs, and coordinator failure propagation. Honor STOP before launching B or restarting anything.
6. Launch A and its durable coordinator in hidden processes. Record wrapper and interpreter identities plus creation times. Verify repeated actual optimizer updates, not just process creation. Queue fresh B automatically only after A has a valid epoch-20 completion and its worker exits.
7. Update canonical CURRENT_STATE/CHANGELOG and applicable maintenance log; record executable commands and source mappings. The new run does not deploy a selected generator checkpoint, so do not change the model registry unnecessarily.

## Controls needed for a causal claim

Document these four conditions with identical data membership, scratch initialization per matched seed, preprocessing, optimizer/LR schedule, batch size and update budget:

1. Normal uniform training: all images available from the beginning.
2. Random expansion: fixed random thirds, same availability schedule as curriculum.
3. Difficulty expansion: the overnight cumulative-static experiment.
4. Exact exposure, shuffled: take condition 3's entire sampled image-ID sequence and permute it across the full training horizon; retain every per-image count exactly. Freeze permutation seed and verify all counts match. This tests temporal placement under the same LR schedule, not a context-free property of ordering.

Difficulty versus random expansion tests ranking value; difficulty versus exact-exposure shuffled tests temporal order beyond cumulative image frequency. Difficulty versus ordinary training tests practical benefit. Available Model1 results can serve as historical comparisons only after verifying settings and sampling comparability; they are not automatically exact-exposure controls. The previous static run is another historical comparison, not a randomized expansion control.

Prepare the manifests/draw evidence needed for these comparisons, but prioritize only the requested A/B curriculum pair overnight. The remaining controls are a follow-up experiment, not completed evidence. Do not report curriculum causality from the overnight pair alone.

## Overnight supervision and results

Use one training worker at a time, A then B, following the user's resource preference. The durable coordinator should handle the transition without depending on a chat being active. Check status/log freshness and actual step advancement every 15 minutes; distinguish saved completed epoch from current in-progress epoch. Keep ordinary heartbeat progress quiet; report a blocking failure or completion.

Existing heartbeat ID: `watch-cryptbb-model1-and-model2-training`. Updated and verified ACTIVE during handoff preparation, every 15 minutes, attached to the preparation thread `01a1dd61-244d-72b2-9a18-66df76eb2563`. Inspect its current state rather than creating a duplicate; monitor updates will appear in that thread unless deliberately retargeted. It must wait for a verified new launch receipt; it must not implement/start training on its own before the next session launches. Once launched, it may supervise only this new pair and recover a verified same-protocol checkpoint if a transient failure is understood. Never resume old static B, start Model3/4, regenerate data, change hyperparameters, or use real accuracy as a recovery/stop decision. Respect any new user stop immediately. Repeated native GPU crashes require diagnosis, not blind restart loops. The heartbeat requests one morning status at or after 09:00 local on November 15, even if unfinished, and pauses if still unlaunched at noon that day. If launch happens later, update those dates when activating monitoring.

The launch session owns training and evaluations. Heartbeat should check their receipts and avoid competing processes. Establish a single process lock for report/evaluation work. If an evaluator has failed, recover only the new run's missing immutable checkpoints. Verify write permissions and monitoring operation before leaving the run unattended. Keep the machine awake and plugged in; a scheduled monitor is not proof that the host will remain available.

Predict each fully committed checkpoint on the existing fixed 251 CryptBB images, preferably using the established CPU FP32/two-thread/batch16 inference pipeline so GPU training stays exclusive. No forced real length or case normalization. Hash checkpoints, freeze predictions before opening scoring labels, and reuse valid completed evaluations. CPU evaluation may trail training; report any backlog. Real labels are for requested diagnostic scoring only and must not feed training, curriculum, stopping or selection.

Unlabeled manifest: `evaluation_data/model4_reservation_20261114/cryptbb/evaluation_pool.json`. Scoring labels: `evaluation_data/model4_reservation_20261114/scoring_only/cryptbb/evaluation_pool_labels.json`. These are canonical-workspace-relative paths, verified from the existing evaluator. Verify their hashes using its protocol before evaluation.

After all 20 prediction sets exist for each reader, select independently with ORIGINAL v1 masked rank-one exact-string agreement over epochs 1..20, masks abs(epoch_i-epoch_j)>k, k=0..5, original choose_k, smallest passing k or unchanged fallback 5. Highest factor wins; earliest epoch breaks exact ties. Report failed mask checks. No common A/B epoch requirement and no substitution of v2 or best real accuracy.

Original selector: `analysis/checkpoint_selection/ckpt_select/estimator.py`, SHA256 `dee14c99919ffb4d51ba47b2104f766cea584dc113a995ab857a037539421150`. Reuse the isolated selection pattern in `E:/pranav/CAPRI/runs/cryptbb_v3_model2_checkpoint_measurements_20261115/final_selector_a.py`, but remove its hardcoded old A paths in a NEW utility. Selection reads predictions only and freezes choices before joining scores.

Morning report must contain:

- A/B completion or exact remaining progress, wall time, update counts, any failures/restarts and evaluation backlog.
- Every completed epoch's mean training loss and real correct/251 with percentage; Model1-A/B same-epoch comparisons and previous static A/B where available.
- Independent selected epoch, top three, mask/fallback diagnostics, selected checkpoint hash and selected real accuracy.
- Average real accuracy across all 20 epochs when complete, plus observed best epoch clearly labeled retrospective diagnostic, never the chosen checkpoint.
- Prior static B stopped at epoch 8: compare only shared completed epochs, not an invented 20-epoch mean. Model1 logs may be rolling median loss while Model2 history is epoch mean; do not compare them as identical statistics.
- State that controls and synthetic-validation tuning were not completed in the primary overnight pair.

Prior static A: final v1 choice epoch 14, 49/251 (19.52%); observed best epoch 18, 56/251 (22.31%); all-20 average 17.4900%. Current Model1-A/B all-20 means 17.9880% / 18.5259%. Recount from source reports before copying into new outputs. Previous A final evidence: `handoff/cryptbb_v3_model2_measurements_20261115/A_all20_2026-11-15T084159_284787+0000/`.

Recent B epochs took about 646 seconds each: approximately 3.6 hours per 20-epoch reader, or 7.2 hours for the pair, before implementation/readiness and any evaluation backlog. Budget about 8-10 hours from next-session start, reassess from new-run timings, and do not promise completion by an unspecified morning deadline. Stop and pause the monitor after the pair, evaluations and report are verified. Do not start controls or another model automatically.

## Research basis and feature evidence

- Bengio et al. (2009): https://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf
- Hacohen and Weinshall (2019): https://proceedings.mlr.press/v97/hacohen19a/hacohen19a.pdf
- Platanios et al. (2019): https://aclanthology.org/N19-1119/
- Wu et al. (2021), random-expansion comparisons: https://arxiv.org/abs/2012.03107

These motivate expanding pools and controls; they do not establish our exact schedule as optimal for CryptBB.

Existing synthetic feature work: `E:/pranav/CAPRI/runs/cryptbb_feature_curriculum_side_20261115/`, especially `combinations_v1/robustness/report.html`. B13 solved 65424/100000 A images. Feature combinations are observational; 99 reference groups and 383 recipes are not 100000 independent styles. Five-fold reference-group analysis favored an additive feature predictor over the shallow interaction tree. Do not turn those diagnostic bands into new curriculum weights in this experiment.

## Suggested opening prompt

Read E:\pranav\CAPRI\handoff\CRYPTBB_CUMULATIVE_STATIC_OVERNIGHT_HANDOFF.md starting at Next chat. Implement and verify the new CryptBB cumulative static Model2-A/B experiment, then start both from scratch sequentially, A then B. Reuse the verified selected opposite-model rankings. Preserve all other training settings and the fixed update budget. Keep the old runs stopped. Verify overnight monitoring, measure all saved checkpoints, apply original v1 selection independently, and prepare the morning report. Do not launch the extra comparison conditions tonight.
