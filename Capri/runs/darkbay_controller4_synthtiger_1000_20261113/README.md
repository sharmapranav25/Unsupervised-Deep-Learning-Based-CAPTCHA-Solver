# DarkBay controller 4: SynthTIGER validation prototype

Completed 1,000 new 200x50 RGB images with generated ground-truth labels. These are validation-only artifacts and are not added to any reader training configuration. No reader optimization or reader evaluation was performed for this controller.

## Framework and fitting

SynthTIGER1.2.1 renders, rotates and composites glyphs in a custom template. Backgrounds are freshly synthesized stripe layers. The maintained implementation is Capri/src/capri_controller/synthtiger_validation.py; the frozen source and hash are saved here. This is a new appearance controller and rendering framework, not an additional pretrained neural network.

The controller measures the existing100 unlabeled reference images only. Anonymous component shape/stroke statistics select64 fonts from314 fonts with valid rendering for every charset character. A seeded random search compares64 appearance recipes, each rendered on24 calibration seeds, using energy distance between pixel-descriptor distributions. The eight lowest-distance recipes produce125 final images each. Export seeds are separate from calibration and A/B training. No real image is pasted into an output.

The initial uniform-font pilot and calibration are preserved as prototype evidence. Font-shape matching was introduced after visual inspection of that pilot. The final shape-matched calibration was frozen before export and before any reader evaluation. Pilot/calibration images are not validation members.

This renderer shares reference pixels, font assets, the synthetic charset and the existing CAPRI count vote with the original pipeline. The count is6 because the frozen pixel-only vote was89/100, satisfying the unique-leader >=50/100 rule. No human target strings, manually chosen family length, real scoring labels, reader predictions or reader accuracy entered fitting. Existing custom/DINO/mixed training recipes and selected weights remain unchanged.

## Files

Dataset: datasets/darkbay/validation/controller4_synthtiger_1000 (relative to workspace root). images/ contains1000 PNGs; labels.tsv and manifest.jsonl contain synthetic labels, seeds, recipe IDs, font hashes, glyph boxes and image hashes. validation_lmdb/ is a second representation of this same validation set. It is not a training corpus.

This run contains calibration_shape_matched.json, font_shape_matching.json, reference_measurements.json, generation_protocol.json, dependency_lock.json, frozen_controller.source.py, audit.json and validation_contact_sheet.png. All assets and caches are on E:.

## Verification and limits

All1000 PNGs were loaded from LMDB and checked against their labels and RGB hashes. All pixels and labels are unique. There is zero pixel overlap with A/B, reference pixels or real evaluation pixels; zero synthetic-string overlap with A/B. Three independently rerendered samples exactly reproduce their saved hashes. Glyph visibility and frame bounds were checked programmatically. The final32-image contact sheet was visually inspected; this does not certify all fonts as unambiguous to humans.

Pixel similarity and fresh synthetic labels do not establish real-domain fidelity or prove that synthetic validation chooses the best real-transfer epoch. The set is held out from reader optimization but was designed after real DarkBay results had been inspected, so any checkpoint-selection experiment on this study is retrospective. Existing fixed epoch20 primary results remain unchanged. A later validation experiment should freeze metric, tie handling and checkpoint eligibility before scoring readers, and must not tune using real labels.

## Execution and dependencies

From the workspace, source environment.ps1. The new route is Capri/capri.py controller4-validation --stage fit or --stage generate. Both stages protect existing frozen outputs and are not commands to rerun this completed set. The isolated package directory is runtime/controller4_packages; dependency_lock.json records versions and downloaded archive hashes. A basic-Latin TextLayer adapter uses Pillow getlength because this environment lacks RAQM and Pillow11 removed getsize. The adapter is limited to this ASCII alphanumeric charset.

Official framework: https://github.com/clovaai/synthtiger (Apache2.0). Existing font licenses remain with their source assets; no relicensing is implied.
