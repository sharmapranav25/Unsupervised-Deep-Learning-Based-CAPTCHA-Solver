# Codex handoff: CryptBB data generation with the v3 split controllers

2026-11-14, from Claude. Workspace `E:/pranav/Thesis-Pranav-Final`. This **replaces** `handoff/CODEX_CONTROLLER_V2_HANDOFF.md` for the next CryptBB generation.

## 1. What the user decided

- The CryptBB v2 run (`handoff/cryptbb_v2_20261114`) was stopped by the user at 28/100 references (STOP + stop_receipt.json). Nothing was generated or trained from it.
- The v2 controllers produced near-identical images across arms. After previews, the user chose **v3 "split" controllers**: each controller fits one job.
- The user asked Codex to start the database generation again with v3, correctly and fast.

| arm | fits | render engine | judged by |
|---|---|---|---|
| custom | thickness (weight/bold) + text colour | CAPRI | custom |
| dino | background/effects + size/width/spacing | CAPRI | DINO |
| synthtiger | font | SynthTIGER glyphs | custom |
| mixed | the three winners combined, then a +/-1-step refine | CAPRI | mixed |

Slant, rotation and wobble stay CAPRI's for all arms. Neither judge reacted to them above seed noise. The job assignment is evidence-based: `Capri/runs/controller_split_sensitivity_20261114/`. Approved previews: `Capri/runs/controller_A_vs_B_20261114/{cryptbb,darkbay}_v3_split_large.png`, runs `Capri/runs/controller_v3split_diverse4_*_20261114`.

## 2. Code (new, ready; v2 untouched)

`Capri/src/capri_controller/controller_v3_split.py`, sha256 `87993051b97a49704b03d0fe912b967a20913e8ab1f780ed622144deb1a92097`. Record this hash in the receipts and do not edit the file while jobs run.

- **FastMixed, the speed fix:** the v2 fit was slow (164 s/reference) because ~90% of the time was per-image NumPy/SciPy feature extraction on one parent core, with the 14 workers idle (`handoff/cryptbb_v2_20261114/performance_investigation/FINDINGS.md`).
  - FastMixed runs that work in the process pool, keeps the GPU passes batched, and skips a discarded feature pass.
  - It is verified bitwise identical to `controllers.Mixed`: max abs diff 0.0 on 84 rendered images, 7.1x faster.
  - Re-run the check with `-m capri_controller.controller_v3_split --dataset cryptbb --verify`.
- **Same as v2:** rendering (`controller_v2.render_job`, including the SynthTIGER engine switch and contrast >= 40), seeds, objective, calibration, reference pixels and CAPRI recipes.
- **Outputs, written after every reference:**
  - `entries.json` with fields entry_id, controller, engine, judge, input_index, controls, validation_score, validation_all_judges, unvalidated, candidates_tried. There is no `recipe` key; join on `input_index` with `refined_recipes.json`.
  - `controller_history.json`, one record per reference.
  - `status.json`, which ends with `state=complete` and `inputs`.
  - `samples/` PNGs, written at the end.
- **Measured runtime:**
  - 1-reference smoke: 44 s.
  - 4-reference previews: 125-143 s.
  - Expected full fit: about 65-70 min for 100 references, vs about 4.5 h for v2.
- **No resume:** the fit refuses an existing output directory.

## 3. Steps

1. **Full fit (CryptBB, all 100 references):**
   ```bash
   export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="model1/src/parseq_packages;model1/src/OpenOCR;Capri/src"
   # plus the environment.ps1 variables (CAPRI_CONFIG, CAPRI_PROJECT, CAPRI_RENDERER, caches on E:, HF offline)
   "/e/pranav/Final sem/sourcefree_synth/experiments_xgb/.venv/Scripts/python.exe" -B -u -m capri_controller.controller_v3_split \
     --dataset cryptbb --inputs 100 --workers 14 --out Capri/runs/cryptbb_controller_v3split_full_<date>
   ```
   Use a new study folder, e.g. `handoff/cryptbb_v3_<date>/`, with launch and status receipts (PID + create time) as before. The completion gate is:
   - status `state=complete` with 100 inputs;
   - 400 entries, 100 per arm, each arm covering inputs 0-99;
   - 100 history records.
2. **Entry admission.** Unvalidated entries block export, as in v2. If any appear, stop and report them to the user with the entry ids. Do not relax the rule silently.
   - CryptBB previews and the smoke run had 0/20 unvalidated.
   - DarkBay #21 had 3/4 unvalidated (low contrast), so CryptBB is not guaranteed to be clean.
3. **Corpus adapter.** Create `cryptbb_v3_corpus.py` as a copy of `cryptbb_v2_corpus.py`. Do not edit the v2 file. Changes:
   - `FIT` = the new v3 fit directory;
   - dataset paths `datasets/cryptbb_v3/{A,B}`;
   - version string `controller_v3_split`;
   - **new seed bases**, not 3e10/4e10 (e.g. A=5e10, B=6e10), recorded in the protocol;
   - the forbidden/overlap set covers `datasets/cryptbb/{A,B}`, all real CryptBB pixels (`Capri/runs/cryptbb_model1_preparation_20261113/excluded_pixels.json`), and v3 A when building v3 B.

   The arm names are the same four, so the 400-entry / 250-per-entry / 25,000-per-arm checks still hold. Generation is render-only and already runs on 12 workers, so the fit bottleneck does not apply; expect minutes for rendering per 100k, plus PNG, decode and LMDB packing.
4. **Pack, configs, preflight** as in v2, with the corrected 32x128 `min_ratio=4/max_ratio=4` loader setting.
5. **Workflow and observer:** make v3 copies of `cryptbb_v2_workflow.py` and the observer, pointed at the v3 paths, with the gate from step 1. The v2 observer's history check (100 records) is satisfied by v3.
6. **Training: ask the user first.** The earlier authorization (parallel Model1-A/B after generation) was for v2 and ended with the user's stop. Before any reader launch, confirm with the user whether to train Model1-A/B on v3 now. If yes, use the same recipe as before:
   - SVTRv2+PARSeq from scratch, seed 48, batch 128, 20 epochs, AdamW 3.5e-4, corrected 32x128;
   - output at `model1/runs/cryptbb_v3/{A,B}`;
   - the frozen agreement reporting rule.

## 4. Do not

- Modify, resume or delete the stopped v2 run or its STOP/receipts (`handoff/cryptbb_v2_20261114`, `Capri/runs/cryptbb_controller_v2_free_full_20261114`). They are evidence.
- Edit `controller_v2.py`, `pipeline.py`, `corpus.py`, `cryptbb_v2_*.py` or `Capri/capri.py` (frozen). Invoke new modules with `-m`.
- Restart the old Model1 pair coordinator (`STOP_AFTER_A.json` still applies) or overwrite existing datasets or runs.
- Use real labels, the 551 scoring images or `JoeyRetitling` in fitting or generation.

Log every step in `handoff/CHANGELOG.md` and `Capri/docs/MAINTENANCE_LOG.md`.
