# Next chat

Continue the design of a NEW Model3 variant: keep the mature teacher's difficulty buckets fixed, and adapt sampling only WITHIN each bucket according to controller-derived appearance groups the student is not learning well.

This handoff was requested on 15 November 2026 (America/Los_Angeles). It records an agreed concept, not an implementation or training launch. Do not run the old Model3 launcher or resume an old experiment on the strength of this document. The next session should make the proposed protocol concrete and reviewable before any new launch.

## User's intended design

The user said:

> use the mature teachers ranking throughout but within easy medium hard we rearrange based on what the controllers identified as important and the model isn't learning

They explicitly confirmed:

> exactly and the data remains the same

Agreed interpretation:

- Reuse the existing 100,000 images per corpus, existing synthetic labels, IDs and metadata. A/B remain their existing separate corpora. No new generation, pixel modification, corpus duplication or synthetic train/validation split is authorized.
- Use the mature Model1 epoch-20 cross-teacher: Model1-B ranks A, Model1-A ranks B. Reuse verified saved rankings where available.
- Freeze each image's easy/medium/hard membership for the entire run. Student performance must never move images between buckets.
- Preserve the matched Model2 schedule and overall draw/update budget. Only conditional sampling within a bucket changes.
- Within a bucket, preferentially sample existing images from appearance groups on which the student shows persistent weaknesses, while retaining coverage of the rest.
- Example: medium clean text is learned, but medium distorted text with cluttered backgrounds is not. Increase practice on the latter, keeping those images in medium and keeping medium's total share unchanged.
- The intended experimental question is: does adaptive practice within a fixed teacher curriculum improve on Model2's uniform sampling within that same curriculum?

Use conditional sampling: P(image at epoch t) = fixed schedule mass of its teacher bucket x adaptive probability inside that bucket. Within-bucket probabilities must sum to one. Preserve the baseline's actual bucket-count rounding and draw semantics rather than silently implementing a different sampler.

## What is fixed versus still to decide

Fixed concept: mature teacher buckets, same data, same outer schedule, same training budget, adaptation only inside buckets, no real-label feedback.

The concrete family/run scope (DarkBay, CryptBB, or both) has NOT been chosen for this new variant. Do not infer a launch order.

For the current matched static experiments, the reference schedule is epoch1 easy only; epoch2 50% easy / 50% medium; epochs3-20 15% easy / 25% medium / 60% hard. Reference training: seed48, batch128, 782 updates and100096 sampled draws per epoch,20epochs/15640updates, same scratch initialization, preprocessing, optimizer and scheduler as the corresponding Model2. Verify frozen per-family protocols before implementation. The previous discussion of a new ladder was not implemented and is not the design to silently substitute here.

Still open, requiring a concrete proposal based on available evidence:

1. Metadata: what per-image controller/recipe/appearance attributes actually exist and are reliably linked to IDs? Controller branch alone is not automatically an interpretable visual feature. Do not invent metadata or claim identified attributes are causally important.
2. Grouping: how to represent font appearance, distortion, background and their overlaps without sparse or confounded groups; minimum group size and treatment of unknown metadata.
3. Weakness signal: synthetic error/NLL, persistent error, learning progress and/or forgetting. Normalize for group size and avoid interpreting unequal sampling exposure as intrinsic weakness. Synthetic labels are permitted; real labels are not.
4. Measurement: a deterministic probe over existing data or full-pool rescoring, update cadence, startup delay and score smoothing. A held-out synthetic probe was suggested in discussion but NOT adopted. Do not remove images from the100k pool or regenerate a probe set without authorization. An in-training-pool probe measures mastery, not held-out generalization.
5. Sampling rule: weight formula, cap, minimum coverage/exploration and normalization separately within each immutable bucket. Highest loss alone can concentrate sampling on defective or unlearnable images. Safeguards were suggested, but no numerical constants were agreed.
6. Selection/reporting: user currently favors a fixed final epoch for a consistent comparison. Freeze the new run's reporting policy explicitly; do not retroactively replace official historical selections. Loss<0.1 is NOT an agreed new sampling, warmup, overfitting or selection rule.

Avoid adding an independent font/augmentation/generation intervention: that would obscure the within-bucket sampling comparison.

## What was wrong with the earlier interpretation

The user suspected Model3 could not improve a mature epoch-20 ranking by using earlier student epochs. This is a plausible mechanism, not an established causal finding. Student-relative difficulty can also be useful.

Verified historical DarkBay Model3 setup:

- Started with exactly the same mature epoch-20 cross-teacher rankings as Model2.
- Rescored the entire existing pool using the current student after each epoch.
- Changed bucket membership using smoothed ranks AND changed bucket sampling masses.
- Therefore the old experiment did not isolate within-bucket adaptation. Do not reuse its AdaptiveSampler.update unchanged for this proposal.

Do not state that Model3 universally underperformed Model2. At fixed epoch20 on historical900 DarkBay images, seed48:

| Model | A correct/900 | B correct/900 | A/B mean accuracy |
|---|---:|---:|---:|
| Model1 | 615 (68.33%) | 573 (63.67%) | 66.00% |
| Model2 static | 600 (66.67%) | 627 (69.67%) | 68.17% |
| Model3 historical adaptive | 629 (69.89%) | 610 (67.78%) | 68.83% |

Model3 beat Model2 on A and lost on B; its final-epoch mean was higher. Results under other checkpoint-selection rules differ and must be named separately.

## Current CryptBB context and selector discussion

The relevant NEW Model2 pair uses epoch-20 teachers and the DarkBay static schedule, NOT the older selected-teacher or cumulative-static experiments. Both new runs completed20epochs/15640updates and all20 checkpoint evaluations. A used verified faster execution after epoch4; B used verified faster execution from the start. No new Model3 implementation or training was performed in this discussion. The previous overnight heartbeat remains paused; do not reactivate it.

Current families: cryptbb_v3_epoch20_static_A_20261115 and cryptbb_v3_epoch20_static_B_20261115. Current Model1 family: cryptbb_v3. Real CryptBB denominator251.

| Reader | Final20 correct/251 | Original unrestricted rank-one selection |
|---|---:|---|
| Model1-A | 59 (23.51%) | epoch12:61 (24.30%) |
| Model1-B | 60 (23.90%) | epoch13:62 (24.70%) |
| Model2-A | 64 (25.50%) | epoch14:66 (26.29%) |
| Model2-B | 64 (25.50%) | epoch14:52 (20.72%) |

Final20 Model2 improves on Model1 by5images on A and4 on B; A/B mean improvement1.79 percentage points. DarkBay final20 mean improvement is2.17points but A falls and B rises. These support average gains in these experiments, not universal or proven causal gains from curriculum alone.

Exploratory selector history:

- Full20 rank-one scores with loss<0.1 winner eligibility selected14 for all4. All automatic mask checks failed; k5 fallback was used. Without the loss gate choices are12,13,14,14 as above.
- Original whole-string mode with all20 voters and loss<0.1 winner eligibility selected M1A19,M1B20,M2A20,M2B20. Character mode differed only at M2A19. Choices froze before label scoring.
- Equal-weight average of eligible rank-one and exact-mode ranks gave19,20,20,20. This was an exploratory calculation, not an adopted selector.
- User subsequently asked to remove the0.1 filter and discussed fixed-final20 as the more consistent comparison. Do not carry0.1 forward as an agreed requirement.
- Model1 loss is logged end-of-epoch rolling median; Model2 loss is full-epoch mean. They are not calibrated equivalent statistics. Low training loss alone does not demonstrate overfitting; held-out synthetic validation deterioration would be needed for that claim. Synthetic-to-real mismatch is distinct.
- Real labels/curves have been repeatedly inspected. Any new method development is retrospective; do not claim a fresh blind test. Freeze choices before scoring and never tune weights/thresholds by real accuracy.

## Evidence and code map

Canonical workspace: E:/pranav/Thesis-Pranav-Final. Local writable working workspace: E:/pranav/CAPRI. Read canonical AGENTS.md, README.md and newest handoff/CURRENT_STATE.md; older sections contain superseded running/launch instructions.

Historical DarkBay:

- handoff/darkbay_curriculum_20261113/protocol.json: teachers, fixed schedule, budget and historical adaptive rule.
- handoff/darkbay_curriculum_20261113/model3_launch_verification.json: identical initial ranking/sampling and independent scratch initialization.
- handoff/darkbay_curriculum_20261113/REPORT.md and interim_comparison_epoch20.json: fixed20 counts above.
- handoff/darkbay_curriculum_20261113/source/Capri/src/capri_controller/training/ab_curriculum.py: frozen score_pool and after-epoch sampler.update call.
- Capri/src/capri_controller/training/ds3_data.py: old sampler; it changes both buckets and masses.
- datasets/darkbay/A,B and model2/rankings/darkbay/A,B: shared corpora and teacher ranks; inspect manifests rather than assuming metadata schema.

Current CryptBB:

- handoff/cryptbb_v3_epoch20_static_A_20261115 and handoff/cryptbb_v3_epoch20_static_B_20261115: frozen protocols, rankings/provenance, readiness, completion, MORNING_REPORT.md, results.json, report_verification.json.
- model2/runs/<family>/A or B and model2/results/<family>/A or B: checkpoints and saved epoch evaluations.
- model1/runs/cryptbb_v3/A,B and model1/results/cryptbb_v3/A,B: current baseline; do not confuse with historical cryptbb/A.
- Capri/src/capri_controller/training/cryptbb_epoch20_static_a*.py and cryptbb_epoch20_static_b*.py: maintained matched static adapters and verified fast execution wrappers. Inspect protocol/source maps for actual data/ranking paths.
- handoff/cryptbb_loss_gate_all20_20261115/{selection,scored_results,verification}.json: frozen all20 rank-one diagnostic and verified score table.
- E:/pranav/CAPRI/runs/cryptbb_mode_loss01_20261115/{analyze.py,selection.json,scored_results.json}: original modes with corrected0.1 gate. cryptbb_mode_loss001_20261115 contains an abandoned0.01 draft, not completed results.

## Next-session work

First inspect existing per-image metadata and frozen rankings. Report which appearance groups can actually be formed, then specify one simple within-bucket adaptive rule with deterministic updates and bounded weights. Keep open decisions visible; do not silently turn suggestions into approved settings.

If implementation is subsequently authorized, use a separate versioned module/protocol/output path. Preserve original frozen sources, checkpoints and reports. Verify immutable bucket membership, unchanged outer draw counts/full batches, deterministic draws, correct group statistics, resume of all sampler/optimizer/scheduler/scaler/RNG state, and exclusion of real-label data. Record rescoring overhead separately. Fast execution must preserve the new rule's semantics; historical equivalence checks do not automatically verify a new adaptive implementation.

No new training, dataset generation, model selection adoption, automation, or controller changes were authorized or performed by this handoff request.
