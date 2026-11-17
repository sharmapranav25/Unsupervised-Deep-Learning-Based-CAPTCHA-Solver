# Checkpoint selection analysis

Standalone analysis package. It reads frozen prediction files from the shared thesis workspace and never changes training. Historical Model1-A uses epochs1-20; v3 Model1-B is frozen at1-16. See [REPORT.md](REPORT.md) for findings and limitations.

## CLI

Run from this directory with Python containing numpy, pandas, scipy and matplotlib:

```powershell
python -m ckpt_select.run --run Model1-A --kernel exact --k auto
python -m ckpt_select.run --run all --kernel all --k auto --evaluate --out results/new-replication
```

The default command does NOT open real labels. `--evaluate` explicitly scores frozen selections afterward. `--root` overrides the thesis root. Existing output directories are rejected. All six k values are always fitted/reported for robustness; `--k auto` chooses by the frozen residual criterion, while `--k 0` through `5` reports a requested fixed choice. Modes use exact/char only; edit applies to pairwise and factor methods.

The simulation gate always runs before real predictions are loaded. `loader`, `agreement`, `baselines`, and `estimator` contain no label-loading paths. Only `evaluate.py` opens labels, after verifying selection and source hashes. `simulate.py` has synthetic truth solely for its tests. Nonnegative rank2 is diagnostic and never chooses between its unidentified components using labels.

## This machine

The shared Python dependencies already include numpy2.0.2, pandas2.3.3, scipy1.13.1 and matplotlib. Pytest8.4.2 was installed ONLY in this folder's `test_site`, leaving shared training dependencies unchanged. The first install needed the existing Anaconda DLL search path. This command runs tests:

```powershell
. ../../environment.ps1
$env:PATH = 'C:/ProgramData/Anaconda3/Library/bin;' + $env:PATH
$env:PYTHONPATH = (Join-Path $PWD 'test_site') + ';' + $env:PYTHONPATH
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
& $thesisPython -B -m pytest -q -p no:cacheprovider
```

Use the same process environment for the CLI. CPU-only; no Torch/CUDA imported. `results/cryptbb_A20_B16_verified` is the completed study. `results/cryptbb_A20_B16` preserves an initial pre-label freeze stopped by a plotting syntax error; decisions were identical after correction. `write_report.py` exports the human report and flat factor/loading CSVs from the verified artifacts, without changing selections.

## Epoch ranking only (latest user scope)

`python -m ckpt_select.ranking_only --results results/cryptbb_A20_B16_verified` verifies the frozen prediction-only artifacts and exports only selected/top3 epochs, mask choices and mask-criterion status. It never opens labels, accuracy summaries or calibration output. Internal factor values are ranking scores, not probabilities. Historical absolute-accuracy diagnostics remain preserved.

Input windows can now be set with `--a-through` and `--b-through` (defaults20/16 preserve the original experiment). The requested18-epoch refresh uses `--b-through 18`. Prediction-only execution writes ranking_only.json, summary_rankings.csv and preference-rank plots, without absolute-accuracy interpretation. Prior successful source is snapshotted in source_versions/A20_B16; new runs snapshot their code inside source_snapshot.

## Version 2: capped-mask and two-regime analysis

Use `python -m ckpt_select.run_regimes --kernel all --evaluate --out results/new-v2-run`. Omit `--evaluate` to keep the whole run label-free. This command uses historical A20 and v3 B20, capped masks, 2000-image-bootstrap refits per configuration, with/without degenerate partners, no-mask exponential coupling, plateau selection and regional diagnostics. `run.py` is preserved as the historical v1 CLI. See results/cryptbb_A20_B20_regimes_v2/REPORT.md for exact definitions, failed simulation cases and conclusions.
