# Claude handoff: Model 3 curriculum and Model 4 pseudo-labeling

Prepared 13 November 2026 local / 14 November UTC. Workspace: E:/pranav/Thesis-Pranav-Final. This is the latest discussion handoff; read it alongside AGENTS.md, README.md and handoff/CURRENT_STATE.md. The original E:/pranav/CAPRI workspace preserves historical experiments. All project outputs and caches stay on E:.

## User intent and authorization

The user requested two separate topics to continue with Claude:

1. Model 3: the rolling practice, mastery and forgetting curriculum developed in this conversation.
2. Model 4: pseudo-labeling, as a separate research track.

This request is to prepare a handoff. It does not authorize implementing either new strategy, pseudo-label generation, new training data, Model 2/3/4 training, or changes to the active Model 1 runs. The user previously corrected an assistant for treating conceptual questions as implementation approval. Discuss and freeze each design before proceeding under a later implementation instruction. Do not combine learning progress, SFL, rolling curriculum and pseudo-labeling into one experiment.

A separate explicit instruction DOES authorize the ongoing CryptBB Model One A/B training. Leave it running. A and B train sequentially for 20 epochs each, then the coordinator stops with agreement reporting pending. Model 2 and Model 3 do not automatically launch.

## Track 1: proposed Model 3 rolling curriculum

### Motivation and agreed direction

The user wants adaptive training to respond to what a reader has mastered, what it gets wrong and what it forgets, with the aim of improving transferable recognition rather than just synthetic performance. Their analogy: a child repeatedly answering 2+2 correctly may have memorized that question; practice different questions, return to earlier ones later, and check both retention and broader learning.

The current historical Model 3 already is adaptive: after each epoch it re-scores all images, updates relative difficulty buckets and changes sampling proportions. But it does not explicitly model sustained mastery or forgetting. The proposed strategy is a NEW version, not a description of that existing implementation. Preserve all historical code/protocols/checkpoints and name new outputs distinctly.

Agreed conceptual choices:

- Reuse the existing fixed 100,000-image synthetic pool for each A/B direction. No new synthetic images, real training images or pseudo-labels in this Model 3 track.
- Drop Model One's easy/medium/hard ranking completely for this proposed strategy. The user explicitly agreed. Model 1 teacher rankings remain part of the existing static Model 2 and historical Model 3 protocols; this decision does not retroactively change those.
- Start epoch 1 from a representative spread of synthetic appearances, using generator metadata to define candidate groups. There is no teacher-driven easy-first requirement for the new strategy.
- Keep a rotating practice subset. Other images temporarily rest from gradient updates but remain available for evaluation and future practice.
- Give more practice to weaknesses; reduce repeated practice for sustained successes; restore practice when previously successful examples deteriorate.
- Periodically rotate in low-exposure examples so temporary resting does not become permanent exclusion.
- Recheck the entire synthetic pool after each epoch, including rested examples. Track each image's exposure and correctness history.
- Maintain the intended training-step/image-presentation budget while redistributing repetitions. Fewer UNIQUE images may be practiced in one epoch even though presentation count remains matched. All 100,000 remain eligible over the run.
- Use separate fixed synthetic validation for a genuinely never-trained check. The user pointed out the existing fourth-controller SynthTIGER validation set; no additional permanent split from the training pool is needed for that purpose.

A permanent one-third holdout was discussed and then replaced by the rolling idea. Neither one-third nor one-half is a finalized rotation fraction. No mastery threshold, forgetting threshold, allocation formula, exploration floor, cap, cooldown, rotation interval or validation-driven sampling rule has been selected. Do not invent them as previously agreed.

### Appearance grouping: available evidence

Teacher difficulty is not visual similarity. Generation metadata provides a separate description of appearance.

Inspected DarkBay evidence:
- datasets/darkbay/A/sample_provenance.jsonl links image IDs to shard/controller/entry_id.
- Capri/controllers/darkbay/entries.json contains recipes, font/layout/effect specifications and final controller controls.
- Corpus ZIP labels.jsonl records image ID, render seed, entry ID, input index, text_spec and effects_spec.
- Capri/src/capri_controller/corpus.py documents those records.

Recipe settings include font, spacing, rotation/jitter, background/effect choices and colors. Treat final controls and actual per-render settings carefully: they can override base recipe descriptions. Different recipe IDs can produce similar appearances; one recipe can contain variation. Do not call 300 entries 300 independent visual families. Controller arm alone is not a style group. Use metadata to propose groups, then inspect representative pixels to check whether the grouping captures meaningful appearance differences. Group by appearance rather than by literal target strings.

For CryptBB, inspect its own sample_provenance schema and corpus shard metadata rather than assuming DarkBay's exact fields. Its provenance records shard/controller/seed; per-image labels.jsonl and entries.json carry the additional links/settings. No grouping/clustering has been built yet.

### Corrected first-three-epoch illustration

This is an explanation, not a frozen algorithm:

1. Epoch 1: draw a representative practice set across appearance groups, keeping other images temporarily rested. Repeat selected examples to fill the matched training budget. Evaluate the full pool and the separate validation set afterward. A single correct prediction is not sufficient evidence of stable mastery.
2. Epoch 2: emphasize mistakes, retain some successful examples, and rotate in previously rested/low-exposure images. Recheck the whole pool. Previously correct -> now wrong supplies a forgetting signal. Never-practiced -> now correct supplies evidence of improvement beyond directly practiced images.
3. Epoch 3: return forgotten examples to practice, maintain attention to weaknesses, reduce emphasis on consistently correct images and continue rotation. Bound repeated attention to persistent failures so they cannot consume the entire budget. Repeat the checks.

Once an image has entered training, later testing after rest measures retention or improvement without recent practice, not genuinely unseen performance. Even success on never-practiced images from the same generator does not prove real transfer. Synthetic generalization and real transfer must remain distinct claims.

### Fixed validation and reporting

Existing SynthTIGER validation is DARKBAY only: datasets/darkbay/validation/controller4_synthtiger_1000, with evidence under Capri/runs/darkbay_controller4_synthtiger_1000_20261113 and handoff/controller4_darkbay_validation.json. CryptBB has no prepared SynthTIGER validation set at this handoff. Its preparation would be a separate prospective design/action; do not claim the DarkBay set is CryptBB validation.

The adopted primary reporting rule remains maximum same-epoch exact full-string A/B agreement on unlabeled real pixels, earliest tie. Discussing SynthTIGER monitoring did not revoke that rule. Whether its scores should affect the new sampler is OPEN; if they do, it becomes development feedback, not an untouched final check. Real labels never guide sampling or checkpoint selection.

### Comparison and unresolved design work

The user wants to compare the complete proposed method with the existing static curriculum at the same training presentation/update budget. A matching permanent holdout in static is NOT mandatory for that complete-method comparison. A controlled ablation would be needed to attribute a gain specifically to rotation, forgetting feedback or removing teacher initialization. Say which question each comparison answers.

Original matched budget is 20 epochs, batch 128, 782 full steps and 100,096 presentations per epoch (100,000 unique pool plus sampler padding), 15,640 steps per reader. Rescoring overhead is separate compute. Keep A/B, architecture, optimizer, preprocessing and initialization comparable unless explicitly revised. Exactly one run per model/pool, not repeated seed trials.

Next discussion should specify: appearance grouping; practice/rest allocation; how many successes establish mastery; forgetting detection; treatment of persistently wrong/noisy samples; minimum exploration; rotation cadence; sampling budget; validation's role; and saved history/resume state. Freeze before implementation/training. Model 3 should not silently acquire pseudo-labeling.

Learning progress and SFL were discussed earlier but were NOT adopted or implemented. SFL p(1-p) is not the same as prioritizing all errors or tracking forgetting. See the older ADAPTIVE_TRANSFER_DISCUSSION_HANDOFF.md for background, but its teacher-initialized proposal and preparation-only CryptBB status are superseded where this document says so.

## Track 2: Model 4 with pseudo-labeling

The ONLY established new decision is to discuss pseudo-labeling as a separate Model 4 track. No detailed Model 4 protocol was developed in this conversation. No teacher, target pool, label-generation method, confidence threshold, loss weight, training schedule or acceptance policy has been approved. No pseudo-labels have been generated and no Model 4 code/training has been added.

Questions for Claude to resolve with the user:

- Which reader(s) produce pseudo-labels: a Model 1/2/3 checkpoint, A/B consensus, a teacher ensemble, or another explicitly chosen teacher? Do not assume GLM-OCR is the teacher merely because its benchmark was discussed.
- Which UNLABELED real images are eligible, and where do they come from? Audit reference, prior selftrain, selection and evaluation membership before using them.
- Is the experiment transductive, or does it reserve an independent target test? The current 551 CryptBB images are already the unlabeled agreement-selection and labeled scoring pool. Training on their pseudo-labels would change that protocol and would not be untouched-test evaluation. Never silently reuse them for training or hide the overlap.
- How are pseudo-labels accepted and refreshed? Confidence/consensus are imperfect; shared errors can be reinforced. Thresholds must not be chosen by sweeping real test accuracy.
- Is Model 4 a continuation/fine-tune or a fresh model? Which checkpoint initializes it, and what synthetic/real mixture and matched-compute comparator make the claim meaningful?
- How will original synthetic labels, teacher predictions, accepted/rejected decisions and image/teacher hashes be preserved for reproducibility?
- What prospective checkpoint selection and final evaluation rules apply?

Using pseudo-labeled real images changes the training information available compared with synthetic-only Models 1/2/3. Report it as an additional adaptation method, not a curriculum-only improvement. Preserve the pixel-only Capri generation pipeline and custom/DINO/mixed provenance. No human real strings enter training, generation or pseudo-label filtering. VLMs remain excluded from the Capri generation workflow; an external pseudo-label teacher would need an explicit separate decision, not an inferred exception.

## Active CryptBB Model One: do not interrupt or duplicate

Authoritative directory: handoff/cryptbb_model1_training_20261113.
- status.json, coordinator.lock.json, active_child.json, launch_receipt.json, launch_verification.json.
- protocol.json and sources/: frozen implementation/config identity.
- readiness/receipt.json: full-batch GPU/continuation checks passed; 7.51 GiB peak; zero next-update model difference after save/load; sampler replay preserved 100,096 draws and full-pool coverage.
- train_A.log / train_B.log and coordinator stdout/stderr.
- model1/runs/cryptbb/{A,B}/progress.json: actual reader PID, optimizer count and current epoch.
- epoch_N.pth / epoch_N_receipt.json: committed immutable checkpoints.
- training_completion.json per reader and study completion.json establish completion, not a stale status/PID alone.

At the last read, A had entered epoch 4, with 2,400 optimizer steps; B was queued. This is a snapshot, not a future live-status claim. Re-read progress and receipts. Earlier launch PIDs: coordinator123152, A interpreter69180; verify current process identity before acting.

Actual entry point: source environment.ps1, then run the configured thesisPython with Capri/capri.py train-cryptbb-model1 coordinate. Recovery uses --resume only after confirming no live coordinator and matching committed checkpoint receipts. Do not issue a second launch now. The old model1/src/train_cryptbb.py remains a readiness dependency/direct-launch helper; the new coordinator is the recovery-aware execution path. No active source edits while training.

The source code adds only a process-local checkpoint save hook to upstream Trainer.train; optimizer/training math is reused. Epoch saves include optimizer/scheduler/scaler, global/decoder RNG, dataset indexing and sampler state. Recovery smoke tested same-process continuation; do not promise cross-device numerical identity. An interruption between archive/latest/receipt writes may need explicit reconciliation. The coordinator stops on failure; inspect instead of blindly restarting.

Preparation completed with both 100,000-image pools, 200,000 unique pixels and zero A/B or audited real-source exact overlaps. Corrected configs are model1/configs/cryptbb/{A,B}/prepared_fixed32x128.yml. Only min_ratio=4 was added to max_ratio=4; original ratio3 produced40x112 and failed the intended32x128 preflight. Failure artifacts and original configs remain preserved. Read handoff/cryptbb_model1_preparation/datasets_ready_v2.json, completion.json and agreement_protocol.json. Preparation heartbeat was paused after completion.

Training: scratch independently for A/B, seed48, SVTRv2LNConvTwo33 + PARSeq (~20.1M parameters), RGB bicubic32x128, no extra augmentation, AdamW lr0.00035, weight decay0.05, OneCycle two-epoch warmup, BF16, batch128,20epochs. Sequential because of16GB GPU capacity. Initial observed speed about275images/s suggested approximately4hours total, not a guarantee.

Primary reporting: freeze A/B predictions for candidate epochs2-20 on the same551 unlabeled real images, select shared maximum raw full-string agreement with earliest tie, then read labels to score. Epoch1 is also saved for recovery but excluded from these reporting candidates. The coordinator currently stops after training with selection/evaluation pending; it does not implement automatic report selection. Opposite Model1 epoch20 remains the existing static Model2 teacher rule. The NEW proposed Model3 has explicitly dropped that dependency.

## Results and cautions relevant to the discussion

- CryptBB Model1-A completed epoch2:16/551=2.9038% exact whole-string accuracy; CER48.0783%. CPU diagnostic, predictions frozen before labels, verification passed, training continued unchanged. Evidence: model1/results/cryptbb/A/requested_cpu_epoch02/{summary,verification,protocol,prediction_freeze}.json. Evaluator: Capri/src/capri_controller/training/cryptbb_checkpoint_evaluation.py. B has no result yet.
- DarkBay Model1 epoch2:A380/900=42.22%, B349/900=38.78%. Saved table: handoff/darkbay_curriculum_20261113/model1_epoch_accuracy.json.
- Current DarkBay agreement-selected results: Model1 e17 A69.00/B63.56%; Model2 e12 A73.22/B71.78%; historical Model3 e13 A71.78/B70.67%. Agreement was adopted retrospectively on DarkBay. Its prospective CryptBB rule was frozen before training/scoring.
- GLM-OCR CryptBB:188/551=34.1198%, CER28.7527%, no target-specific tuning. Source: Capri/runs/glm_ocr_cryptbb_20261113/summary.json. It is an external benchmark, not a generator or approved pseudo-label teacher.
- The assistant offered a speculative30-35% CryptBB forecast only after the user requested a numerical analogy: DarkBay's epoch2 A/B mean40.50% to mean individual-run hindsight maxima69.89% is +29.39pp; adding that to CryptBB A2.90% gives32.29%. This is NOT measured evidence, a fitted learning curve, a statistical prediction interval, a training target or a reason to change selection. It mixes a DarkBay two-run summary with one early CryptBB run and may be badly wrong. Do not propagate it as an expected outcome or promise.

## Suggested next action for Claude

Acknowledge the two separate tracks and ask which the user wants to develop first. For Model3, turn the agreed loop into explicit candidate rules for discussion without launching it. For Model4, first clarify teacher and unlabeled target data/evaluation separation. Keep active CryptBB Model1 running and preserve all historical evidence. Update shared handoff/CHANGELOG.md and source/evidence mappings for any later authorized changes.

2026-11-14T02:28:29.403490+00:00: User cancelled automatic Model1-B launch and requested generator/controller examples. Terminated only verified coordinator PID 123152; Model1-A interpreter 69180 remains alive and continues to20epochs. No STOP marker and no frozen source edits. Durable STOP_AFTER_A.json and stop_after_A_verification.json prohibit restarting the pair coordinator without new authorization. A completion is model1/runs/cryptbb/A/training_completion.json; no pair completion is expected. Existing coordinator lock is now stale, retained as evidence. B has no launch/checkpoint.
