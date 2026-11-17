# Codex handoff: controller change, then CryptBB retraining

2026-11-14. Workspace `E:/pranav/Thesis-Pranav-Final`. Background detail, run list and scores: `handoff/CODEX_CONTROLLER_V2_DETAILS.md`.

## 1. What changed

The CAPRI recipe controllers were replaced with a new version, `Capri/src/capri_controller/controller_v2.py`. It searches wider ranges of size/width/spacing/rotation/slant/weight/bold, tries alternative fonts, replaces wrong CAPRI backgrounds, takes per-letter colours from the reference pixels, rejects low-contrast text, and adds a fourth arm (`synthtiger`) that runs the same search with SynthTIGER glyphs. The user chose `--mode free` (default). The old controllers (`pipeline.py`) and all existing entries, corpora, datasets and Model1 runs are unchanged and must be preserved.

## 2. What happens next

The user will retrain CryptBB on data produced by the new controllers. Before any launch, confirm with the user which readers to train (Model1-A/B) and that launch is approved. Steps:

1. **Full controller fit (CryptBB, all 100 references):**
   ```bash
   export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="model1/src/parseq_packages;model1/src/OpenOCR;Capri/src"
   "/e/pranav/Final sem/sourcefree_synth/experiments_xgb/.venv/Scripts/python.exe" -B -m capri_controller.controller_v2 \
     --dataset cryptbb --inputs 100 --workers 14 --out Capri/runs/cryptbb_controller_v2_free_<date>
   ```
   About 2.5 min per input, ~4 h. Outputs `entries.json` with 400 entries (custom, dino, mixed, synthtiger × 100).
2. **Corpus generation needs an adapter.** `corpus.py` assumes 300 entries/3 arms, reads `entry['recipe']`, and renders with the CAPRI glyph path only. v2 entries have no `recipe` key (join on `input_index` from the source run's `refined_recipes.json`), carry `engine`, and the `synthtiger` entries must render through `controller_v2.render_job` (includes the contrast check and NumPy-2 shim). Write a new corpus module rather than editing `corpus.py`; allocate 100,000 images over 400 entries (250 each) and log the rule.
3. **New dataset names**, e.g. `datasets/cryptbb_v2/{A,B}`, with new render seed bases recorded; verify zero pixel overlap with each other, with the existing `datasets/cryptbb/{A,B}`, and with all real CryptBB pixels (`Capri/runs/cryptbb_model1_preparation_20261113/excluded_pixels.json`).
4. **Pack, configs, preflight**: follow `Capri/src/capri_controller/cryptbb_preparation.py` (`pack`, `configs`, `preflight`) with the new paths; keep the corrected `min_ratio=4/max_ratio=4` 32x128 loader setting from `model1/configs/cryptbb/{A,B}/prepared_fixed32x128.yml`.
5. **Train** under new run directories (e.g. `model1/runs/cryptbb_v2/{A,B}`), same recipe as the existing CryptBB Model1 (scratch, seed 48, SVTRv2+PARSeq, batch 128, 20 epochs, AdamW 3.5e-4, sequential A then B). Report with the frozen agreement rule (max same-epoch A/B exact agreement on the 551 unlabeled images, epochs 2-20, earliest tie) before reading labels.

## 3. Do not

- Overwrite or restart the existing CryptBB Model1 runs. Model1-A (20 epochs, selection pending) stays as the comparison baseline; the old pair coordinator is blocked by `handoff/cryptbb_model1_training_20261113/STOP_AFTER_A.json`.
- Edit `Capri/capri.py` (frozen in reader provenance); invoke new modules with `-m`.
- Use real labels, the 551 scoring images or `JoeyRetitling` in controller fitting or corpus generation.

Log every step in `handoff/CHANGELOG.md`.
