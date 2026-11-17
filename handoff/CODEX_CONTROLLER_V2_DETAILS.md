# Codex handoff: Controller v2 (CAPRI recipe controllers)

Prepared 2026-11-14 (local 2026-11-13 evening PST). Workspace: `E:/pranav/Thesis-Pranav-Final`. Read `AGENTS.md`, `README.md` and `handoff/CURRENT_STATE.md` first; this file covers only the controller v2 work done in the Claude session of 2026-11-14. Full chronological detail is in `handoff/CHANGELOG.md` (entries headed "2026-11-14 Controller v2 preview" and the dated lines after it).

## 1. Authorization boundary (read first)

- **Authorized and done:** audit of what the v1 controllers add; a new controller module with gap fixes and a SynthTIGER-rendered fourth arm; preview runs on 4 inputs per dataset; mode comparison; setting the default mode.
- **Not authorized:** full 100-input controller fits with v2, any 100k corpus from v2, LMDB packing, reader training (Model1/2/3/4), retraining CAPRI, edits to `Capri/capri.py`, `pipeline.py`, existing `entries.json`, corpora or datasets. Ask the user before any of these.
- **Current default chosen by the user:** `--mode free` (option A). The user moved through B → B-widened → A on the same day after looking at images; treat A as the current choice, not a frozen protocol.
- **CryptBB Model1:** A finished 20 epochs (`model1/runs/cryptbb/A/training_completion.json`, completed 2026-11-14T03:02:49Z, `selection_pending: true`). B was cancelled earlier by the user; `handoff/cryptbb_model1_training_20261113/STOP_AFTER_A.json` forbids restarting the pair coordinator without new authorization. The controller work did not touch training.

## 2. Why v2 exists (audit findings on v1)

v1 = `Capri/src/capri_controller/pipeline.py`, three arms (custom / DINO / mixed) that differ only in their similarity score. Measured on the DarkBay run (`Capri/controllers/darkbay`):

- CAPRI's recipe builder (`Capri/dependencies/sourcefree_synth/feature_bank/iter2/controller.py::initial_recipe`) **hard-codes** size 0.62, background `random`, text palette `contrast`. CAPRI predicts font class, slant/rotation/spacing/width levels, effects, length and charset.
- Mixed score: CAPRI recipe 3.94 → pixel colours 3.55 → v1 pick 2.92. About 40% of the gain is colour, the rest geometry.
- v1 winners sat at grid edges (size 0.8, width 1.1, spacing 0.15, rotation 20 = max values) while the score still improved: search too narrow.
- Font never searched (`font_pool` unused); effects only all-on/all-off; absolute overrides discarded CAPRI's per-image rotation; 3 arms gave only 187/300 distinct recipes; 2 seeds per candidate.

## 3. The module

`Capri/src/capri_controller/controller_v2.py` (new file; sha256 at handoff `c8b4ca641dbfcd62e44a7c82a7dba8791ac655f0cc91061bb8145b13f5ca7f83`). Reuses, unchanged, each source run's reference pixels, CAPRI refined recipes, calibration features, `controllers.Objective` and model-derived length:

- DarkBay source: `Capri/controllers/darkbay` (length 6)
- CryptBB source: `Capri/runs/cryptbb_model1_preparation_20261113/generator` (length 5)

What it does per input image:

1. Candidate groups: base, geometry, font (CAPRI's next-ranked fonts + best of every other category), colour (pixel ink palette; one median colour per letter blob of the real image), background, surface/effect strength.
2. Evaluate each candidate on 3 render seeds; score = 0.5 × (whole-100-image distribution + this image), same objective as v1.
3. Merge step: leader + best font/colour/background, pairs of top replacement backgrounds.
4. Three rounds of coordinate refinement on size, width, spacing, rotation, jitter, size-jitter, slant, weight, bold, plus effect strength.
5. Validate top 4 per arm on 4 fresh seeds (all must be usable); cross-arm de-dup within 5%.
6. Fourth arm `synthtiger`: identical search, every candidate drawn with SynthTIGER glyphs via `synthtiger_recipe_adapter.glyph_mask`, scored with the mixed objective.
7. Text-contrast rejection: median text colour vs 3-px ring around it < 40 RGB → sample unusable (applies everywhere, including the v1 re-score).
8. v1's chosen recipe is re-scored on the same 4 validation seeds for like-for-like comparison (`v1_validation_score`).

Modes (`--mode`):

| mode | background | text (font, size, style, colour) |
|---|---|---|
| `free` **(default, option A)** | may replace CAPRI's background (drop its under-text effects, try each of the 15 alone/pairs) | full search |
| `bounded` (B widened) | CAPRI's effects kept, strength ±2, pixel background colour, no replacement | full search |
| `bounded_strict` (original B) | as `bounded` | within one step of CAPRI's values, CAPRI top-2 fonts |
| `textonly` | CAPRI's background verbatim, including its `random` colour | full search |

Other flags: `--inputs N`, `--diverse` (most typical input + farthest-point picks on custom whole-image descriptors; no labels, no hand-picking), `--workers`, `--out` (must be a new E: path; existing outputs are never overwritten).

## 4. How to run

From `E:/pranav/Thesis-Pranav-Final` (Git Bash shown; `environment.ps1` sets the same PYTHONPATH):

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="model1/src/parseq_packages;model1/src/OpenOCR;Capri/src"
"/e/pranav/Final sem/sourcefree_synth/experiments_xgb/.venv/Scripts/python.exe" -B -m capri_controller.controller_v2 \
  --dataset darkbay --inputs 4 --diverse --workers 14 --out Capri/runs/<new_name>
```

- Invoke with `-m`; do **not** add a subcommand to `Capri/capri.py` (frozen in active reader provenance hashes).
- Cost: about 2.5 min per input image on 14 CPU workers (≈650 renders + feature scoring); GPU use is light. DarkBay and CryptBB run fine in parallel on this 32-thread machine. A full 100-input fit ≈ 4 h per dataset; not authorized yet.
- SynthTIGER workers need a NumPy-2 shim (`np.sctypes`) for imgaug; it is applied inside `render_job` only.
- If no candidate passes all 4 validation seeds, the best fit-phase record with most usable renders is used and flagged `unvalidated: true` in `entries.json` (seen only in `textonly`).

Outputs per run: `protocol.json`, `entries.json` (per input × arm: controls, selected phase, `validation_score`, `v1_validation_score`, flags), `controller_history.json` (every candidate and score), `samples/` (4 fresh images per entry), `<dataset>_v1_vs_v2.png`, `<dataset>_v2_samples.png`.

## 5. Evidence (all previews, 4 diverse inputs unless noted)

Inputs chosen by `--diverse`: DarkBay 86, 45, 21, 26; CryptBB 28, 13, 94, 85.

| Run directory (`Capri/runs/…`) | Mode / version |
|---|---|
| `controller_v2_smoke_darkbay_20261114` | v2, 1 input smoke test |
| `controller_v2_preview_darkbay_20261114` | v2, first-12 run **stopped by user at 4/12**; partial, keep as evidence |
| `controller_v2_diverse4_{darkbay,cryptbb}_20261114` | v2 (before background replacement/letter colours/contrast) |
| `controller_v2b_diverse4_{darkbay,cryptbb}_20261114` | **A** (`free`), v2b |
| `controller_v2b_bounded_diverse4_{…}` | `bounded_strict` semantics (run when it was called `bounded`) |
| `controller_v2b_textonly_diverse4_{…}` | `textonly` |
| `controller_v2c_Btext_diverse4_{…}` | current `bounded` (B + full text search, weight/bold added) |
| `controller_A_vs_B_20261114` | comparison sheets (`*_A_vs_B.png`, `*_A_B_textonly.png`, `*_A_Bstrict_Btext.png`, `*_A_only.png`) and `scores*.json` |

Mean objective (lower = more similar to the 100 references; this is the controllers' own appearance score, **not** reader accuracy):

| | v1 (re-scored) | A free | B strict | B + text | textonly |
|---|---|---|---|---|---|
| DarkBay | ≈2.1–4.3 per entry | **1.82** | 2.28 | 2.05 | 2.66 |
| CryptBB | ≈1.4–3.2 per entry | **1.83** | 1.95 | 1.94 | 1.88 (12/16 unvalidated) |

Note: the A images were generated before `weight_absolute`/`bold_absolute` joined the numeric search. A new `free` run searches those two extra controls, so it will not reproduce the A preview pixels exactly.

## 6. What the user saw and concluded (visual, not measured)

- DarkBay A: replaced CAPRI backgrounds with layered rectangles + some scribbles/grid; closest to real. Occasional black scribble clutter (input 86) and SynthTIGER colour clutter (21, 26).
- CryptBB: the objective disagrees with the eye. A swapped input 13's real grid for blobs while scoring it better; input 28 kept CAPRI's pink blob background in every CAPRI-glyph mode (the judge never preferred a white background); B + text gave the best-looking CryptBB 13/94/85.
- `textonly` failed because CAPRI's background colour is literally `random`.
- The controllers now change almost everything CAPRI decides except length and charset (font changed 12–18 of 16–32 entries; size 23/32; colours 30/32). The user asked whether this means retraining CAPRI; three levels were discussed (pixel-measured recipe builder, head-only fine-tune, full retrain). **None is authorized.**

## 7. Open items, in suggested order (ask before each)

1. **Judge diagnosis (no new run):** decompose `Objective` components (`custom_detail`/`dino_detail`: glyph_shape, ink_shape, layout_v2, text_color, background_structure, background_color) for CryptBB input 28 (pink vs plain/white/blocks candidates) and input 13 (grid vs blobs) from `controller_history.json` / a re-score. Hypothesis: the 50% whole-set distribution term favours busy coloured backgrounds. Do not tune weights by looking at real reader accuracy.
2. **Contrast threshold:** 40 lets thin light-grey text through (DarkBay 21 in B + text); 60 was proposed, not applied.
3. **Generic-start ablation:** same controllers from a default recipe (keep CAPRI length/charset) to measure what CAPRI's starting recipe is worth. Proposed, not run.
4. **Full 100-input fits → corpus:** only with explicit authorization; would create a new, separately named corpus version alongside the existing A/B datasets and needs its own `corpus.py` adaptation (v2 entries contain `engine` and new control keys; the SynthTIGER arm needs the adapter in corpus workers).
5. CAPRI improvements (level 1 recipe builder that measures size/colour from pixels; level 2/3 retraining) remain discussion-only.

## 8. Rules that still apply

Pixel-only generation: no VLMs, no human target strings, no real labels in fitting. The 551 CryptBB scoring images and `JoeyRetitling` must never enter controller fitting. Preserve all run directories listed above. Record every change in `handoff/CHANGELOG.md`. Keep outputs and caches on E:.
