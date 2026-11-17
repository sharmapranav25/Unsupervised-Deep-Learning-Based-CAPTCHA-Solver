# CAPRI research workspace

Completed DarkBay experiment record: [DarkBay through Model 2](docs/DARKBAY_THROUGH_MODEL_2.md), covering the 100-image hybrid/controller pipeline, both datasets, reader training, validation, real evaluation and synthetic-calibrated shoot-rate analysis through 6 October 2026. This later record supersedes the preliminary outcome statements below; historical protocols remain preserved.

Start here for the maintained pixel-only controller workflow. The active code is now in `src/capri_controller`, with four selected checkpoints in `models/active` and one entry point, `capri.py`. Existing experiment folders are historical evidence and remain at their recorded paths.

## Current training replacement: original small CAPRI with DINO fonts and separate count

On28October2026 the user ordered deletion of the heavy joint20 decoder run. That run and launcher are deleted. The replacement uses the actual3.66M original CAPRI core, a fine-tuned DINO font branch and a separate count transformer, jointly on the same100k images for20epochs. Checks passed and the replacement launched at12:26PDT28October2026. [Run guide](runs/capri_lightweight_joint20_100k_20261028/README.md), [live status](runs/capri_lightweight_joint20_100k_20261028/status.json). Initial measured speed82.3images/s; estimate8-10hours. See [shared handoff](docs/HYBRID_FONT_CONTROLLER_REDESIGN_HANDOFF.md) and maintenance/joint20_deletion_20261028.json.

## Where things belong

| Location | Purpose |
|---|---|
| [src/capri_controller](src/capri_controller/) | Maintained hybrid, custom, DINO and mixed controller integration |
| [models/registry.json](models/registry.json) | Selected checkpoint paths, roles, original locations and SHA256 hashes |
| [configs/local.json](configs/local.json) | Machine-specific Python, renderer, feature-model source and run paths |
| [configs/targets](configs/targets/) | Pixel input manifests; no target text labels |
| `runs/` | Destination for future runs; never put new experiment outputs inside `src/` |
| [capri_darkbay_workflow/report.html](capri_darkbay_workflow/report.html) | Completed DarkBay run and comparisons, preserved in place |
| [docs/MAINTENANCE_LOG.md](docs/MAINTENANCE_LOG.md) | What the cleanup changed, validation and remaining limitations |
| [docs/WORKSPACE_CATALOG.md](docs/WORKSPACE_CATALOG.md) | Generated index of all workspace folders and their sizes |
| [docs/CODE_AND_MODELS.md](docs/CODE_AND_MODELS.md) | Architecture, model selection and operating instructions |
| [archive/root_logs_20261005](archive/root_logs_20261005/) | Archived root-level logs; exact move manifest in `maintenance/` |

## Commands

Run from this directory using the existing environment. The environment folder happens to be named `experiments_xgb`; the maintained controllers do not use XGBoost.

```powershell
$capriPython = 'E:/pranav/Final sem/sourcefree_synth/experiments_xgb/.venv/Scripts/python.exe'
& $capriPython capri.py verify
& $capriPython capri.py status
& $capriPython capri.py models
& $capriPython capri.py report --run capri_darkbay_workflow
& $capriPython capri.py audit --run capri_darkbay_workflow
```

For a new run, use a new output directory. Controller fitting is expensive; `status`, `models`, `verify`, `inventory` and `report` do not train or fit controllers.

```powershell
& $capriPython capri.py controllers --out runs/darkbay_next
& $capriPython capri.py generate --run runs/darkbay_next --pilot --workers 8
& $capriPython capri.py generate --run runs/darkbay_next --workers 8
& $capriPython capri.py report --run runs/darkbay_next
& $capriPython capri.py inventory
```

The default target manifest contains the same 100 DarkBay pixel references. `controllers --targets path/to/manifest.json` selects another manifest with a `fit` list of `source` and `pixel_sha256` objects. Standard full runs require 100 unique inputs and a unique model-derived leading length with at least 50/100 votes; they stop if no length qualifies, including a 50/50 tie. The bounded preview exception is described below.

For a visual preview, `controllers --preview --out runs/<new-name> --targets <manifest>` uses all100 references but refines only the first6 inputs, preserving custom/DINO/mixed outputs. In this preview mode only, a missing majority uses seeded, equally weighted cycles across the distinct model-predicted lengths, as specified in `CAPRI_WORKFLOW.md`. This never uses known scheme lengths or target strings. Preview runs cannot export a corpus. `capri.py preview-smaller-hybrid` orchestrates the user-approved first6 examples from the existing five smaller-hybrid sets and reuses the verified exact-input DarkBay outputs; it refuses to overwrite its dated output. Standard full controller/corpus runs still require a majority.

## Current evidence

DarkBay: 100,000 distinct labeled images, 300 recipe entries, 179 parameter combinations, balanced across the three controllers. The default length of 6 came from 72/100 raw hybrid votes. Only 36.3% pass the strict readability check; the corpus is experimental, with no established downstream solver benefit. See [CAPRI_WORKFLOW.md](CAPRI_WORKFLOW.md).

The four active weight copies have verified hashes. Original checkpoints and experiment scripts are retained for reproduction. Historical XGBoost, SAM, Meta-Sim2 and DINOv3 experiments are not selected by the maintained workflow. Frozen `CAPRI` packages are separate research releases, not this entry point.

## PARSeq reader experiment

DS2 curriculum job: `capri.py darkbay-ds2`. Its dataset, protocol, launch receipt, stage logs and `status.json` are under `runs/Darkbay_ds2_noFont-intel`. It generates100k fresh renders, ranks with the DS1 epoch20 ensemble, holds out5000 synthetic images, and trains seeds48/123/456 for20epochs with Model2's adapted curriculum and augmentation. It saves each epoch and selects by synthetic validation, then evaluates on historical real DarkBay. Check status before starting; the running lock prevents duplicate orchestration. See the maintenance log for protocol differences from DS1.

The training dataset is named **Darkbay_ds1_noFont-intel** (100,000 synthetic images, without font intelligence). Its identity and existing storage paths are recorded in `capri_darkbay_workflow/dataset_identity.json`.

The 100k DarkBay corpus has a fixed20-epoch PARSeq run with seeds48/123/456 and final-checkpoint selection. Data/configs/status are in `runs/parseq_darkbay_100k_20261005`. Training was launched on 2026-10-05 with random initialization after the user requested a start with the pretrained checkpoint unavailable. This initialization differs from the thesis's pretrained recipe. Read the run's `status.json` for live state; do not launch a duplicate job. The guarded pretrained launcher, for a separately planned run, is:

```powershell
& $capriPython capri.py train-parseq --checkpoint 'path/to/scene-text-base.pth'
```

`--from-scratch` is a separate explicit alternative and changes the initialization protocol. See the maintenance log and saved `protocol.json` before launching.

All three reader runs have now completed20epochs. Real DarkBay evaluation on900 images excluding the100 controller inputs gives54.67%,48.78%,52.67% exact case-sensitive accuracy (mean52.04%). This is historical-data evaluation, not a newly untouched benchmark. [Evaluation report](runs/parseq_darkbay_100k_20261005/evaluation_darkbay/report.html). Predictions were frozen before labels were scored; no checkpoint was selected based on these results.

## Initial data for shared CAPRI + independent count (27 October 2026)

Prepared [dataset package](runs/capri_shared_count_dataset_20261027/README.md):20k TRAIN/2k DEV scenes plus36,492 TRAIN and3,060 DEV font-library specimens. [Current architecture and handoff](docs/HYBRID_FONT_CONTROLLER_REDESIGN_HANDOFF.md) records the accepted design and data limits, including cross-family reference pixel aliases. Verify with `capri.py verify-architecture-data --dataset runs/capri_shared_count_dataset_20261027`. Training remains stopped.


## Current training dataset: 100,000 images (27 October 2026)

The user-requested [100k training package](runs/capri_shared_count_dataset_100k_20261027/README.md) supersedes the20k selection above. It contains100,000 unique TRAIN CAPTCHA images,2,000 separate DEV images and the preserved auxiliary font libraries. All625 artifacts passed hash verification; [reader checks](runs/capri_shared_count_dataset_100k_checks_20261027/checks.json) and [selection checks](runs/capri_shared_count_dataset_100k_checks_20261027/selection_checks.json) passed. The original corpus and20k package remain preserved. See the [active shared handoff](docs/HYBRID_FONT_CONTROLLER_REDESIGN_HANDOFF.md) for architecture, reference-alias safeguards and next steps. Training has not started; the previous20k runtime budget is not a100k ETA.


## Stopped 100k architecture training

The user-authorized [training run](runs/capri_shared_count_training_100k_20261027/README.md) launched27October2026 at20:35PDT. It trains the independent count transformer for8 full100k epochs, then font/background/appearance heads for4 full100k epochs with the saved CNN/DINO/localization core frozen. Read [live status](runs/capri_shared_count_training_100k_20261027/status.json) for progress; earlier stopped/preparation-only descriptions are historical. Initial ETA7–10hours including validation/checkpoints. Model registry remains unchanged.

Training was explicitly stopped by the user at22:27PDT27October2026. Main epoch1 is partial (70,987images); final recovery step9,050 is preserved. The earlier ETA is canceled; no restart authorized. See the active handoff and run user_stop.json.



## CAPRI2 DarkBay reader experiment, 12 November 2026

The new generator's 20-epoch training is complete. A downstream PARSeq experiment has now launched: full three-controller fitting, 100k synthetic images, then three readers trained for 20 epochs each. [Shared run guide](runs/parseq_darkbay_capri2_100k_20261112/README.md), [current pipeline stage](runs/parseq_darkbay_capri2_100k_20261112/pipeline_status.json). Preparation is separate from reader optimization. All run outputs and caches are on E:.

## Latest reader plan: Dataset A and Dataset B, one run each

The author cancelled repeated seed runs. The active coordinator prepares and audits two100k synthetic corpora, then trains one20epoch PARSeq reader on A and one on B. [Study guide](runs/darkbay_ab_study_20261112/README.md), [current stage](runs/darkbay_ab_study_20261112/status.json). The previous three-run plan is superseded. Cross-dataset teachers will support later Model2/3 experiments in both directions; those curriculum stages are not started yet.
