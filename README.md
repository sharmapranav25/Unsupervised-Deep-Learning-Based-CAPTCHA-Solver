# Thesis-Pranav-Final

Current checkpoint-selection rule (user adopted13November2026): maximize A/B exact full-string agreement on the same unlabeled real images at a shared epoch, earliest tie; freeze choices then score labels. Model1 selects17, Model2 selects12, Model3 selects13. Primary results: handoff/darkbay_curriculum_20261113/family_report_agreement_selected.json. Policy: handoff/darkbay_curriculum_20261113/agreement_selection_policy.json. Adopted retrospectively on DarkBay; earlier synthetic-validation and fixed20 results remain preserved historical comparisons.

This is the shared thesis workspace on E:. Start with handoff/CURRENT_STATE.md. All original experiments are preserved; the files here were copied and SHA-256 verified. Use workspace-relative paths in current manifests. Immutable historical files retain their original paths and statements; they are not current instructions.

## Current study

One selected CAPRI generator: original lightweight core, fine-tuned DINO font branch and separate count transformer. Generation uses its internal count head; the separate transformer is retained as training provenance. A and B each contain 100,000 synthetic DarkBay training images. Their pixels are disjoint; their controller recipes and mixture are shared. The archive and LMDB are two representations of each corpus, shared across all reader approaches.

Model 1-A and 1-B are complete at epoch20. Same900 historical real DarkBay images: A615/900 (68.33%), B573/900 (63.67%). The user has now authorized the remaining DarkBay training. The new runner and protocol are under Capri/src/capri_controller/training/ab_curriculum.py and handoff/darkbay_curriculum_20261113. Read that folder's status.json and readiness.json for current progress; launch is gated on completed rankings and readiness. Model2 A/B runs first, then Model3 A/B from scratch, then final evaluation and stop pending CryptBB inputs.

## Layout

- Capri/: generator, shared maintained package, controller dependencies, renderer, font assets, training data, code and provenance.
- datasets/darkbay/A and B: the only working copies of these corpora in this workspace.
- model1/: completed standard readers, OpenOCR, configs, checkpoints, evaluations.
- model2/: completed static curriculum, frozen rankings and results.
- model3/: completed adaptive curriculum, updates and results.
- evaluation_data/: real image pixels and separate scoring labels; never training inputs.
- thesis/: empty writing, figures, tables and bibliography destinations. No manuscript was invented.
- handoff/: current state, source mappings, copy receipts, evidence and original documentation.
- runtime/: host Python location and E-drive caches. Python itself remains in the existing E: environment.

From PowerShell, source .\environment.ps1, then use `& $thesisPython -B workspace.py status`, `verify`, or `inventory`. `verify --full` rehashes all copied files; modified working sources will be reported against the original migration snapshot. These commands do not train. Use model1/src/verify_migrated_readers.py for the migration inference check.

Historical scripts have fixed run names and old orchestration assumptions. Do not launch them unchanged: the new A/B curriculum adapters are still pending. Canonical common implementation is Capri/src/capri_controller; model folders refer to it rather than maintaining divergent copies. The copied Capri/capri.py is preserved historical source, not an approved new-study launcher. Read Capri/README.md before using it.

This workspace is usable by another assistant on this machine. It is not yet a standalone Python environment for another computer: see runtime/local.json and handoff/DEPENDENCIES.md.
