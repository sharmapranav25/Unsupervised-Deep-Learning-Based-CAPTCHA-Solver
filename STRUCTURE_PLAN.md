# Thesis Pranav Final structure plan

Revised 12 November 2026: one CAPRI baseline, plus model1, model2 and model3. This supersedes the earlier proposal for CAPRI version folders. All project files belong under E:/pranav/Thesis-Pranav-Final. This is a proposed structure; no existing training or research artifacts have been moved.

## Structure

```text
Thesis-Pranav-Final/
├── README.md
├── AGENTS.md
├── STRUCTURE_PLAN.md
├── Capri/
│   ├── README.md
│   ├── architecture.md
│   ├── training_setup.md
│   ├── baseline_selection.md
│   ├── src/
│   ├── configs/
│   ├── models/
│   ├── controllers/
│   ├── renderer/
│   ├── training_data/
│   └── evaluation/
├── model1/
│   ├── README.md
│   ├── training_setup.md
│   ├── src/
│   ├── configs/
│   ├── data_manifest.json
│   ├── models/
│   │   └── parseq/
│   │       ├── A/
│   │       └── B/
│   ├── runs/
│   └── results/
├── model2/
│   ├── README.md
│   ├── training_setup.md
│   ├── curriculum.md
│   ├── src/
│   ├── configs/
│   ├── data_manifest.json
│   ├── models/
│   │   └── parseq/
│   ├── runs/
│   └── results/
├── model3/
│   ├── README.md
│   ├── training_setup.md
│   ├── adaptive_curriculum.md
│   ├── src/
│   ├── configs/
│   ├── data_manifest.json
│   ├── models/
│   │   └── parseq/
│   ├── runs/
│   └── results/
├── datasets/
│   └── <scheme>/
│       ├── A/
│       └── B/
├── evaluation_data/
│   ├── images/
│   ├── scoring_labels/
│   └── split_manifests/
├── thesis/
│   ├── chapters/
│   ├── figures/
│   ├── tables/
│   └── bibliography/
└── handoff/
    ├── CURRENT_STATE.md
    ├── SOURCE_MAP.json
    ├── CLAIMS_AND_EVIDENCE.md
    └── CHANGELOG.md
```

## Meaning of each component

Capri is the single selected generator baseline for the thesis. It has no v1/v2 subfolders. The author specifies that if the newer candidate beats the previous candidate, it becomes the CAPRI baseline; otherwise the previous baseline remains. Selection is pending the completed comparison, not established by this folder plan. baseline_selection.md records the compared checkpoints, evaluation protocol, metric, results and selection reason, with links to preserved historical evidence. The selected generator has one architecture, training setup, source, configuration and checkpoint registry. Controller code and weights, renderer and assets have one authoritative location. Training data used to learn CAPRI belongs here; datasets generated for reader training live once under datasets/<scheme>/ and are referenced by the reader experiments. Data manifests specify actual membership, roles and hashes without duplicate copies.

Proposed naming assumption: model1 means the standard PARSeq reader experiment, model2 means the fixed-curriculum reader experiment, and model3 means the adaptive-curriculum reader experiment. The author explicitly confirmed that model3 belongs in this workspace. These names must be confirmed against the author's intended thesis terminology before migration because the older E:/pranav/model 1 and model 2 repositories refer to different experiments from CAPRI's DarkBay DS1 and DS2 readers.

model1, model2 and model3 are reader training approaches built around the selected CAPRI baseline. There are no capri_v1/capri_v2 subfolders in their final layout. Preserve the actual generator checkpoint and dataset hashes in every run record: an old reader trained on previous-generator data must not be relabeled as using the newly selected baseline. A matched comparison of reader approaches requires the appropriate data and protocol for each approach. Every reader approach has one run on A and one on B, with no repeated seed trials. Store these under models/parseq/A and B, inserting a scheme folder first if multiple schemes are included. Record a random initialization setting internally for reproducibility without treating it as an extra experiment.

training_setup.md records the precise dataset and split, generator checkpoint identity, reader architecture, initialization, epochs, batch size, reproducibility settings, optimizer, learning rate schedule, preprocessing, augmentation, checkpoint selection, environment and reproduction commands. curriculum.md additionally records difficulty scoring, teacher identity, sampling schedule and validation rules. For model3, adaptive_curriculum.md records how current reader performance changes difficulty rankings and sampling, including rescoring frequency, smoothing, constraints and saved curriculum state.

runs/<run_id> keeps immutable protocol, exact configurations, source snapshot, logs, generation recipes, audits and completion receipts. models/parseq holds the selected reader checkpoints and their manifests; run records refer to those checkpoints by path and hash. Recovery checkpoints remain in their run when needed. results holds frozen predictions, metrics, plots and error analysis, each linked to its run, split and selected checkpoint. Do not duplicate large model files to satisfy both layouts.

## Historical sources for selection and provenance

- CAPRI implementation and controller assets: E:/pranav/CAPRI/src/capri_controller and models/registry.json. Preserve package imports and record dependencies when packaging.
- CAPRI 1 generator identity: active hybrid registry checkpoint and its original training provenance.
- CAPRI 2 generator: E:/pranav/CAPRI/runs/capri_lightweight_joint20_100k_20261028.
- Standard reader using CAPRI 1 data: E:/pranav/CAPRI/runs/parseq_darkbay_100k_20261005.
- Curriculum reader using CAPRI 1 data: E:/pranav/CAPRI/runs/Darkbay_ds2_noFont-intel.
- Standard reader using CAPRI 2 data: E:/pranav/CAPRI/runs/parseq_darkbay_capri2_100k_20261112; launched, final completion must be checked.
- Adaptive reader documentation: E:/pranav/CAPRI/docs/MODEL_3_ADAPTIVE_CURRICULUM_HANDOFF.md; reconcile its historical planning state with actual run artifacts before migration. E:/pranav/model3 is a separate candidate source whose identity must be checked.
- Earlier standalone model repositories remain candidates, not automatically merged with the DarkBay studies.

## Organization rules

Everything is shared on E:, including caches and temporary outputs. README and handoff/CURRENT_STATE.md are the starting points for the author or any assistant. Machine paths must be configurable; portable manifests should use relative paths. Record hashes, original sources and relocation mappings. Package required external renderer/model dependencies and fonts, with their license information, before describing the workspace as self-contained.

Keep the three-controller mixture and pixel-only target conditioning. Target strings and manually selected target-family lengths do not enter generation. Keep scoring labels separate from training inputs. Learn rejection/attempt policies using synthetic validation and freeze decisions before real-label scoring. Document repeated historical-test use and any historical real-test confidence sweeps separately.

Only include studies that support the thesis, including relevant negative results. Keep generator-count accuracy, image-quality diagnostics and reader recognition accuracy distinct. Every manuscript claim should link to the actual experiment evidence.

Implementation should copy the selected baseline and verify completed artifacts, preserving originals and frozen records. Historical comparison sources remain evidence; they do not become separate CAPRI versions in the final folder structure. Leave running experiments at their current paths until complete; then copy verified outputs. First settle the model1/model2/model3 terminology, then create the agreed skeleton, inventory assets, copy with checksums and validate reproduction from the new workspace.


## Dataset A and Dataset B with both curriculum directions

Author decisions, 12 November 2026: generate A and B once per CAPTCHA scheme and have both ready before reader optimization. Each main corpus contains 100000 synthetic training images. Reuse the same frozen controller recipes and mixture, but use disjoint rendering seed ranges and check that the corpora have no identical RGB images. Store one physical copy of each corpus under datasets/<scheme>/A and B. References and real scoring data remain separate.

| Experiment | Optimization images | Initial curriculum teacher |
|---|---|---|
| Model 1-A | A | None; standard training |
| Model 1-B | B | None; standard training |
| Model 2-A | A | Model 1-B ranks A; keep ranking fixed |
| Model 2-B | B | Model 1-A ranks B; keep ranking fixed |
| Model 3-A | A | Start from Model 1-B ranking of A, then adapt during learning |
| Model 3-B | B | Start from Model 1-A ranking of B, then adapt during learning |

The explicit choice to run both directions supersedes the earlier B-only rule for Models 2 and 3. There is exactly one training run per model/dataset combination; the user cancelled repeated seed runs. The full proposed study therefore has six reader runs per scheme, not eighteen. Only the two Model 1 runs are being launched now. Each uses 20 epochs, batch128 and the existing standard training settings. Adaptive rescoring changes order/sampling without generating another corpus. Initial teacher weights and curriculum ranking use the other dataset; later adaptive scoring uses the current Model 3 reader as it learns.

Do not interpret a teacher's scores on its own training data as cross-dataset difficulty. Use synthetic labels to compute difficulty on the opposite corpus. Hold validation membership and optimization budgets constant across reader approaches within each direction. The earlier DarkBay DS2 experiment used a95000/5000 split; do not silently apply that historical split to this newly specified100000-training-image study. The validation and checkpoint-selection protocol for Models 2/3 remains to be fixed before those stages launch. Current Model 1 uses fixed final epoch20 selection without validation selection.

Current execution uses the CAPRI 2 candidate, pending its comparison with the previous CAPRI baseline. Dataset A is the already-generating corpus in E:/pranav/CAPRI/runs/parseq_darkbay_capri2_100k_20261112. Dataset B is E:/pranav/CAPRI/runs/parseq_darkbay_dataset_b_100k_20261112. Coordinator and shared receipts: E:/pranav/CAPRI/runs/darkbay_ab_study_20261112. The original three-run launcher is held after generation and replaced by this coordinator. Both corpora must pass full individual audits and cross-dataset duplicate checks before it trains Model 1-A once and Model 1-B once, sequentially.

Historical filenames may contain seed_48; this is the sole retained run's reproducibility identifier. No seed123 or456 training is scheduled. Preserve historical provenance and do not rename live paths. The coordinator does not launch Models 2/3 or deploy a generator. Dataset A is the matched comparison against the previous generator's single corresponding reader; report Dataset B separately instead of selecting whichever outcome is better.

## Migration readiness while Model 1 B trains

Checked 12 November 2026: Model1-B is active; the study STOP marker requests stopping after its final checkpoint is saved. No files were migrated by this assessment.

We can create the agreed folder skeleton and copy stable artifacts now: architecture/training documentation, selected CAPRI generator and controller checkpoint snapshots, renderer/source/config snapshots, completed Model1-A checkpoint and its frozen evaluation outputs, manuscript drafts, figures, bibliography and historical experiment evidence. Keep original paths and checkpoint identities in SOURCE_MAP.json; copied code is a staging snapshot until its external dependencies and path settings are verified. Historical Model2/3 results must remain marked historical, not presented as new A/B experiments.

Leave all original source, model, environment and run paths in place while the current job runs. Source files and dependencies are hash-checked by the coordinator and may be needed again. Do not physically move the live B LMDB, its output/checkpoint directory, logs/status/STOP/lock files, the coordinator, Python environment, OpenOCR, or runtime dependencies. Do not copy an actively rewritten latest.pth as a final artifact.

Both synthetic corpora are immutable, so their image archives can be copied, but defer bulk corpus/LMDB transfers and large upstream training-data copies until the job stops to avoid competing with training disk reads. Prefer documents and completed artifact snapshots first. After B completes and the coordinator stops, verify checkpoints, copy the remaining data/artifacts, test reproduction from the new root and only then consider retiring original locations. This is copy-first staging, not authorization to delete historical sources.
