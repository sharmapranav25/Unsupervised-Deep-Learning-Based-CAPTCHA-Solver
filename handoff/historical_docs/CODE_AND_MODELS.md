# Code and model management

The maintained controller implementation lives in `src/capri_controller`. This separates reusable code from old run directories without relocating evidence or breaking historical scripts. Use the root `capri.py` entry point for new work.

## Code structure

| Module | Responsibility |
|---|---|
| `cli.py` | Explicit inspection, fit, generation, audit and reporting commands |
| `runtime.py` | Central configuration, model resolution, JSON and streaming hashes |
| `pipeline.py` | Per-image hybrid inference, cleanup, model-voted length and three-controller refinement |
| `controllers.py` | Custom, DINO and calibrated mixed scoring |
| `features/` | Shared pixel descriptors, inferred regions, custom and DINO comparisons |
| `networks/` | Frozen custom encoder and mask architectures; checkpoint-compatible output dimensions |
| `rendering/` | Generic renderer controls and complete glyph controls |
| `artifacts.py` | Input RGB-hash verification and saved render labels |
| `corpus.py` | Balanced, resumable ZIP-shard generation and full image audit |
| `reporting.py` | Lightweight HTML overview from saved artifacts |
| `inventory.py` | Metadata catalog for folders and historical checkpoints |
| `vendor/dino/` | Existing official DINO source with its original Apache license |

Package-local imports are explicit and relative. `runtime.py` is the only path bridge to the existing external renderer and feature-model source repositories. Those dependencies and the font bank remain external; this cleanup does not make the project standalone or portable to another computer automatically. Edit `configs/local.json` on another host, verify all dependencies and review absolute image/font paths in saved recipes.

## Selected models

| Registry name | Purpose | Selection |
|---|---|---|
| `hybrid` | Initial feature labels and length votes | Original simple hybrid, epoch 15 EMA |
| `custom_encoder` | Appearance features for the custom controller | Existing frozen custom CNN encoder |
| `text_mask` | Infer text regions from pixels | Existing synthetic-trained mask network |
| `dino` | Attention and appearance features | Official original DINO ViT-S/8 |

The phrase “DINO controller v2” describes our controller iteration. It does **not** mean DINOv2 pretrained weights. The mixed controller combines custom and DINO scores; it has no additional trained checkpoint. No VLM or XGBoost model is loaded by this workflow. The hybrid architecture has a CTC branch, but the controller does not decode target strings.

`models/registry.json` is authoritative for active weight paths and expected hashes. Copies in `models/active` are byte-identical to the recorded originals. Retaining the originals is deliberate: old experiments still depend on them. Do not infer model quality from filenames such as `best.pt`, timestamps or file size.

`models/historical_inventory.json` lists workspace checkpoint locations and sizes. Except for the active copies, its hashes are not computed and its contents have not been audited for duplication or fitness. External training checkpoints are not recursively inventoried; the selected external source checkpoint is recorded in the active registry. The inventory is not authorization to delete training states, specialist teachers or frozen releases.

## Operation and provenance

Use `capri.py verify` after changing machines or selected weights. It checks active file hashes and the two external source directories; the regression tests provide stronger evidence of actual inference and rendering behavior. It does not automatically download models or install packages.

Use a fresh `runs/<name>` directory for controller fitting. The CLI rejects an existing output directory. Corpus generation resumes completed shards only when their stored checkpoint-independent recipe manifest hash and ZIP hash match the current run. Each image keeps its chosen synthetic text, render seed, controller and recipe ID, and basic/strict validity flags.

The model decides the default target length through raw vote aggregation. The user-requested 50% vote threshold and renderer cleanup are engineering rules, not human target labels. A unique leading count with at least50/100 votes qualifies; a50/50 tie does not select an arbitrary winner. Standard full runs stop if no count qualifies; bounded previews keep the existing equal-observed-count fallback. Target strings, positions and texture labels are never supplied.

The opt-in `controllers --preview` path refines the first6 inputs against all100 references. Only this bounded path implements the documented no-majority fallback: seeded shuffled cycles of distinct observed model lengths, with one assigned length shared by all controller candidates/arms for each input. It changes no comparator, objective, weights or candidate search. `generate` refuses preview runs. `smaller_hybrid_preview.py` provides the dated `preview-smaller-hybrid` orchestration, exact-input/checkpoint-verified DarkBay reuse, per-scheme audits and a comparison report. Raw inference evidence is now saved before the standard no-majority failure as well.

The 100,000-image allocation is 333 or 334 images per entry. Rejections are retried within the same entry. The stricter readability flag is diagnostic, not an export requirement. Filtering to strict-only images would change the mixture and must be reported.

The new `report` command writes `overview.html`; it does not rerender or replace the detailed historical `report.html`. Re-run `inventory` to refresh the catalog after adding experiments or moving artifacts. Update [MAINTENANCE_LOG.md](MAINTENANCE_LOG.md) whenever maintained code, selected models, locations or reproduction commands change.

## Historical reproduction

Original MVP folders and the original three DarkBay scripts remain in place. They reproduce the old run and are not the maintenance target for new features. Their hashes are preserved in `maintenance/extraction_20261005.json`. Changes in the new package include direct imports, registry-based paths, a new-run overwrite guard, removal of the old DarkBay-specific vote assertion, a generic gallery heading, and exclusion of sampled text from the parameter-diversity count. Rendering and comparator calculations were preserved and regression checked.

Root-level logs were moved to `archive/root_logs_20261005`; `maintenance/root_log_moves_20261005.json` records every original path, destination, byte size and SHA256. Old scripts can create new logs at their old destinations on rerun. No historical model, corpus, source directory, release or ZIP package was deleted.

## Model 3 adaptive reader experiment

`capri.py darkbay-ds3` runs the separate curriculum-only experiment on the existing Model2 LMDB and split. Numeric protocol: [MODEL_3_ADAPTIVE_PROTOCOL.md](MODEL_3_ADAPTIVE_PROTOCOL.md). Implementation is under `src/capri_controller/training/ds3*.py`; outputs and state are under `runs/Darkbay_model3_adaptive_20261006`. Inspect status before invoking; the run lock prevents duplicate orchestration. Readiness receipts are required before production. All baseline artifacts and selected controller models remain unchanged.

## Shared CAPRI + independent-count dataset preparation (27 October 2026)

`capri.py prepare-architecture-data --source runs/capri_decoder_500k_20261007 --out runs/<fresh-output>` builds a portable20k TRAIN/2k DEV initial learning package from frozen paired pilot indices and saved font libraries. `capri.py verify-architecture-data --dataset runs/<output>` verifies all artifact hashes and loader isolation. Code: `src/capri_controller/architecture_data.py`; preparation uses no GPU/model/renderer and does not open reserved calibration/final-test archives. Runtime manifests use package-relative paths. Historical source provenance retains original paths and is never executed. TRAIN references are optimization support; DEV reference reader rejects optimization use. Count-only reader loads RGB and a separate count target; full scene reader preserves masks/background/annotations for the other heads. See the dataset README for native-resolution preprocessing requirements, unknown/absent-label masking and duplicate-reference safeguards. Existing controller model selection is unchanged.


## User-requested 100k architecture dataset (27 October 2026)

Use `capri.py prepare-architecture-100k --source runs/capri_decoder_500k_20261007 --base runs/capri_shared_count_dataset_20261027 --out runs/<fresh-output> --workers 4` for the deterministic100k extension. Implementation: `src/capri_controller/architecture_data_100k.py`. The completed package is `runs/capri_shared_count_dataset_100k_20261027`; verify with `capri.py verify-architecture-data --dataset runs/capri_shared_count_dataset_100k_20261027`. Use the configured E-drive Python with `-B`.

All100,000 TRAIN scenes are eligible;2,000 reused DEV scenes and auxiliary font specimens are additional. Existing20k records, paired identities, DEV selection and alias-aware font libraries are preserved. Four CPU workers validate/copy the additional80k from original hash-verified archives; no renderer, GPU or model training runs. Package-relative paths allow relocation. Original archives, selected models and controller mixture remain unchanged. This dataset-size change does not implement or launch the new count transformer or alter the old trainer; full100k training runtime requires measurement.


## 100k architecture training (27 October 2026)

`capri.py check-architecture-training --dataset runs/capri_shared_count_dataset_100k_20261027 --out runs/<fresh-readiness> --checkpoint runs/capri_decoder_500k_20261007/diagnostics/timed_pilot_20261008/epochs/epoch_0008/checkpoint.pt` verifies package integrity, gradient behavior, duplicate/alias exclusions, tiny-batch learning and GPU speed. Then `capri.py train-architecture-100k --dataset runs/capri_shared_count_dataset_100k_20261027 --readiness runs/<passed-readiness> --out runs/<fresh-training>` launches the fixed8-count/4-main-epoch schedule; every full epoch covers exactly100k scenes. Use configured Python with `-B`, working directory E:/pranav/CAPRI.

Code: `count_transformer.py`, `architecture_training_data.py`, `architecture_training.py`; additional evaluation/resume smoke checks in `architecture_training_checks.py`. Old decoder sources remain unchanged. Count is a4-layer128-wide image transformer trained from scratch; main heads start from the old epoch8 EMA with CNN/DINO/fusion/localization frozen. Positive font pairs require different glyph content; duplicate specimen/glyph pairs and known specimen-alias family negatives are excluded without merging families. DEV references never optimize; gallery evaluation preserves candidate ties and reports them separately.

Training saves protocol/source copies, stage progress, raw/EMA/full-state checkpoints every1,000 updates and each epoch, count logits and main-font candidate/consistency records. To request a graceful stop, create `STOP` inside the training output. After removing it, continue with the same command plus `--resume`; protocol/code hashes must match. New output directories are refused unless explicit resume is used. No automatic controller deployment; registry and existing mixture stay unchanged. Count selection uses synthetic DEV NLL; the main selection is the fixed fourth epoch. Read run status/ETA rather than assuming completion from elapsed time.


## Joint20 new CAPRI (28 October 2026)

The user explicitly requests all components together for20epochs on100k scenes, including font intelligence and the separate count transformer. `joint_training_20.py` supplies a paired-model wrapper and combined training loop; all CNN/DINO/fusion/localization/task-head/count parameters are trainable. This differs from the stopped frozen-core experiment and the unlaunched8epoch oldstyle adapter. Use `capri.py check-joint20 --dataset runs/capri_shared_count_dataset_100k_20261027 --out runs/<fresh-readiness> --batch-size 8`, then `capri.py train-joint20 --dataset runs/capri_shared_count_dataset_100k_20261027 --readiness runs/<passed-readiness> --out runs/<fresh-run>`. Use configured E-drive Python with `-B`. Resume uses the latter command plus `--resume` and matching source hashes; create STOP in the run directory for a checkpointed stop. Dataset and model inputs remain pixel-only; labels are separate loss targets. Current active controller registry is not replaced automatically. The shared handoff records full initialization, trainability, losses and schedule.


## Lightweight replacement following user correction,28October2026

The heavy joint20 production run was explicitly deleted by the user. Its commands remain historical code, not the active training recommendation. Use capri.py prepare-lightweight --dataset runs/capri_shared_count_dataset_100k_20261027 --out runs/<fresh-cache>; check-lightweight --dataset runs/<cache> --out runs/<fresh-check>; train-lightweight --dataset runs/<cache> --readiness runs/<passed-check> --out runs/<fresh-run>. Explicit --resume only after the worker stops. src/capri_controller/lightweight contains the original3.66M core and added fine-tuned DINO font and separate count branches. Read the shared handoff for data-label differences, optimizer and validation gates. No active controller selection changes automatically.

## Latest lightweight target-count diagnostic

Use capri.py test-lightweight-count --run runs/capri_lightweight_joint20_100k_20261028 --out runs/<fresh-diagnostic>. It captures the latest committed checkpoint once,loads EMA,and scores both count heads on the existing fixed target manifests. CPU inference avoids GPU memory interference with training; count-only outputs must match the full model exactly in CPU FP32. Predictions are hash-frozen before labels are read for scoring. City unmatched annotations remain unscored. This read-only diagnostic does not select checkpoints or alter training. Sources and outputs are retained for handoff.


## Matched previous-CAPRI count comparison

Use capri.py compare-capri-count --reference runs/capri_lightweight_internal_count_epoch20_20261029 --out runs/<fresh-comparison>. Maintained source: src/capri_controller/count_baseline_comparison.py. Loads the registry hybrid checkpoint and compares its internal length head on exactly the new diagnostic image set, using the same preprocessing. Freezes baseline predictions before scoring. No active model changes. Verified result: runs/capri_previous_vs_new_count_20261029_v2/RESULTS.md.


## Paired controller previews

capri.py compare-capri-controllers --out runs/<fresh-name> produces first6 fixed examples per scheme for CAPRI1 and CAPRI2 (DarkBay,CryptBB,City), retaining custom,DINO,mixed outputs. Implemented in controller_comparison.py using preview-only pipeline hooks. Uses same input pixels, RGB96/bilinear/cap480 preprocessing, seed schedule and shared CAPRI1 calibration for both versions. CAPRI2 includes trained DINO residual font heads and internal count; the separate counter is unused. Registry remains unchanged. Run outputs contain source snapshots, copied input pixels, logs, recipes, masks, labels, audit.json, overview.png and report.html. No target text labels are loaded.


## CAPRI2 synthetic-data PARSeq experiment

Use `capri.py train-capri2-parseq --out <fresh E-drive runs directory>` for full controller fitting, 100k audited synthetic images, matched CAPRI1 PARSeq training (three seeds, 20 epochs each), and fixed final evaluation. Existing prepared output requires `--resume`; inspect the pipeline lock before starting. Current authorized experiment: `runs/parseq_darkbay_capri2_100k_20261112`. See its README, protocol, source snapshots and pipeline_status.json for handoff. Production adapter hooks require explicit `production_model=True`; paired controller previews retain their previous defaults. Selected active models are unchanged.

## Current A/B reader study, superseding the three-run plan

User requests two100k corpora A/B and exactly one20epoch Model1 reader per corpus, with both datasets audited first. The coordinator is tools/run_ab_study.py (implementation src/capri_controller/training/ab_study.py); it invokes capri.py for all dataset and reader stages without editing the existing frozen launch sources. Current state: runs/darkbay_ab_study_20261112/status.json; see that run's README for paths and resume instructions. The previous three-seed launcher is cancelled after generation. Use the A/B coordinator for recovery, not train-capri2-parseq --resume's historical three-run loop. Model1-A will rank B and Model1-B will rank A for later fixed/adaptive curricula. Model2/3 are not launched by this coordinator.


## CPU final-reader diagnostic

With PYTHONPATH set to E:/pranav/CAPRI/src, run the configured Python with -B -m capri_controller.training.evaluate_model1_cpu --run <completed single-reader run> --out <fresh child output>. It scores the same900 historical DarkBay pixels after freezing predictions. CPU-only execution uses an isolated device-lookup adapter; no upstream training source is edited. The ordinary scheduled GPU evaluator remains unchanged. See the12November maintenance entry for the verified Model1-A result and limitations.

For a specifically requested interim CPU diagnostic, evaluate_model1_cpu also accepts --checkpoint <run>/seed_48/epoch_N.pth. Capture the latest completed epoch once; do not use the actively rewritten latest.pth. The summary records the tested epoch and preserves the separate final-checkpoint reference.
