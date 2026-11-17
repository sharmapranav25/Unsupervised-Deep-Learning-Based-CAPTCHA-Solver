# CryptBB Model3-A reference-priority curriculum

This is the user-authorized Model3-A experiment, family `cryptbb_v3_feature_priority_v1_20261115`. It uses the existing CryptBB v3 A corpus and Model1-B epoch20 ranking. No Model3-B launch is included.

## Fixed parts

- Scratch seed48, batch128, 20epochs, 782optimizer updates and100096draws per epoch;15640updates total. Same architecture, preprocessing, optimizer, scheduler and BF16 training as the matched Model2-A.
- All100000 existing synthetic images remain available. Teacher-assigned easy/medium/hard membership is immutable.
- Epoch1 easy only; epoch2 50/50 easy/medium. Actual image-ID sequences match the saved Model2 sequences exactly.
- Primary checkpoint is fixed final epoch20. No real labels, real-image scores or checkpoint-selection diagnostics enter training.

## Feature importance and progression

Importance is the proportion of100 real reference images exhibiting a feature in the saved pixel-derived predictions. Width combines condensed/expanded predictions without double-counting references; spacing combines wide/tight spacing. Fitted font sets use the number of distinct reference IDs whose fitted recipes emit that font set. These font counts describe fitted representation, not verified real font identity. Geometry, mapped effects and content categories with positive reference prevalence are included. Unsupported/zero-prevalence features have no targeted priority; their images remain eligible through coverage sampling.

Families are sorted by descending prevalence, with lexical name breaking equal-prevalence ties. Three highest-ranked unlearned families are active together. Exact synthetic full-string accuracy strictly greater than75% marks a family ready to advance. An active family receives attention1; a learned family retains attention0.25. If its accuracy falls to75% or below it becomes unlearned and eligible for the ranked active frontier again. Other unlearned families receive uniform coverage until their turn.

Each family is subdivided using actual saved synthetic settings: exact values when there are at most8, otherwise metadata-only quintile ranges. Weak subgroups and weak buckets retain extra demand even after the family's overall accuracy passes75%. Accuracy is an association with the images carrying a setting, not proof of an isolated feature cause. Features overlap.

## Adaptive sampling

After each epoch, one frozen student state predicts all100000 images in FP32 GPU batches of256, ascending image-ID order, with ordered CPU prefetch and a final160-image batch. No synthetic answer tokens or answer length are supplied to inference. Predictions are saved, then compared with synthetic answers. Epoch1 is diagnostic only; the epoch2 assessment sets epoch3 sampling. Subsequent assessments update every next epoch. This fixed batching convention is part of the protocol: different batch sizes are not universally equivalent for an early student. A disposable early-model probe changed2/512 predicted strings between256 and64, with0correct under either; repeated256 predictions were exact. The mature B20 probe matched all512strings and348correct under both. Weights and all training RNG remained unchanged. No batch-size change is allowed during the run.

For each family/setting/bucket cell, demand is:

`reference_prevalence * attention * (1 - accuracy) * min(1, cell_images/100) / number_of_family_settings`

The support factor limits influence of sparse cells. Dividing by the number of settings prevents finer binning from multiplying a family's importance. Sum cell demands separately for easy, medium and hard. Each bucket's mass is `0.20/3 + 0.80 * its_share_of_total_demand`. If all demands are zero, masses are equal. Every bucket therefore has at least6.6667% of training draws after warmup; no fixed split of the remainder exists.

Within a bucket, distribute each cell's demand equally over its images and add overlapping contributions. Mix the normalized image demands with20% uniform per-image coverage. Use uniform sampling if there is no demand. Round bucket counts using the original Model2 rounding rule and preserve full batches. All draws use deterministic epoch-specific NumPy RNG and are saved.

The active-family count3, learned-attention0.25, coverage0.20 and sparse-cell support100 are declared implementation choices, not user-specified values or constants fitted to real test accuracy. This is a retrospective experimental method; no transfer gain is claimed.

## Execution and recovery

Entry point: `model3/src/feature_priority_v1/capri.py`. Shared maintained implementation: `Capri/src/capri_controller/training/cryptbb_feature_v1*.py`. The historical CAPRI launcher and baseline sources are preserved.

Stages: `prepare`, `check`, `launch`. Launch starts one hidden coordinator and one GPU worker. It stops after Model3-A20; no automatic restarts, follow-on B run or real evaluation is queued. The coordinator prevents system sleep while active. `STOP` in this handoff directory is checked before training and inference batches. Removing STOP and recovery require an explicit user request.

Each committed checkpoint includes weights, optimizer, scheduler, scaler, all RNG states, epoch, history and adaptive sampler state. Resume is only from a hash-verified completed epoch using `worker --resume`; uncommitted next-epoch artifacts are preserved in a recovery directory. Do not use this command without a new explicit recovery request. The coordinator is intentionally fresh-launch only.

Training outputs: `model3/runs/cryptbb_v3_feature_priority_v1_20261115/A`. Inspect `status.json`, `history.json`, `sampling_epoch_XX.json`, `assessments/epoch_XX/feedback.json` and committed checkpoint receipts. Training and full-pool assessment time are recorded separately. `launch_verified.json` in this handoff confirms repeated production updates; preparation/readiness alone does not establish a launch.

Initial prelaunch revision1 was preserved separately before adding the content-category groups and mature-reader inference parity test. It is not an earlier training run.

The failed early batch-size-invariance check and its protocol/source snapshot are preserved under `prelaunch_revision2`. Readiness requires reproducible fixed256 scoring and the recorded mature-reader comparison, not an untrue claim of universal cross-batch invariance. No production training had started during either prelaunch revision.
