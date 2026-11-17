# CAPRI 1 versus CAPRI 2: same-controller examples

Completed: 108 example entries, six inputs x three controllers x two CAPRIs x three schemes. Open report.html, or DarkBay.png, CryptBB.png and City.png. overview.png shows the first two fixed inputs of each scheme. No examples were selected by visual quality.

CAPRI 1: active original epoch-15 EMA, models/active/hybrid/hybrid.pt. CAPRI 2: final epoch-20 EMA, runs/capri_lightweight_joint20_100k_20261028/epochs/epoch_0020/checkpoint.pt. CAPRI 2 uses the trained DINO residual font heads and internal count. Its separate counter is unused by generation. All three controller branches remain unchanged and retained: custom, DINO and mixed. Registry is unchanged.

Inputs: 100 original historical fit images per scheme copied with opaque filenames under inputs. Only the first six initialize example recipes. Target labels/strings were not loaded. Both models use RGB height96, bilinear, widthcap480, edgepadding and FP32. This common preprocessing differs from historical CAPRI1 controller Lanczos/cap960 inference. Both use the same candidate search, seed schedules and calibration derived once from CAPRI1 first24 recipes. Saved calibration constants match exactly.

Length policy is unchanged: 60/100 raw internal-count votes establish the common count; otherwise preview uses seeded equal cycles of observed predicted lengths. DarkBay votes for six: CAPRI1 75, CAPRI2 89. City votes for five: CAPRI1 91, CAPRI2 96. CryptBB: CAPRI1 has no majority and uses two to five; CAPRI2 has 70 votes for five. Vote agreement is not accuracy.

Every scheme/version directory contains raw model predictions/logits, cleaned recipes, length decisions, controller search history, selected entries, synthetic labels/images/masks and adapter full-model parity evidence. Calibration features and model provenance are saved. Top-level protocol.json, audit.json, verification.json, results.json and worker logs cover reproduction. Exact code/config snapshots are under provenance.

All108 entries passed the existing complete/usable render checks. Strict readability passes were CAPRI1/CAPRI2: DarkBay10/18 versus6/18, CryptBB9/18 versus8/18, City15/18 versus15/18. These are renderer diagnostics, not target similarity or reader accuracy. Unique pixel outputs across the36 entries per scheme: DarkBay21, CryptBB27, City17. Identical arms remain in the mixture. Some new DarkBay samples lose background structure or have faint glyphs. Text ordinal and font width/slant supervision were missing from CAPRI2 training; those initialization outputs remain a limitation. These examples are not evidence that CAPRI2 improves all aspects of generation.

Reproduce from E:/pranav/CAPRI with the E-drive Python environment:

    & 'E:/pranav/Final sem/sourcefree_synth/experiments_xgb/.venv/Scripts/python.exe' -B capri.py compare-capri-controllers --out 'E:/pranav/CAPRI/runs/<fresh-name>'

Set TEMP, TMP and PYTHONDONTWRITEBYTECODE to the shared E-drive settings. The command configures its child workers and cache locations on E. External renderer/font and Python dependency paths are recorded in configs/local.json and its snapshot; they must remain available. Do not overwrite this run or infer permission to deploy/retrain from these previews.
