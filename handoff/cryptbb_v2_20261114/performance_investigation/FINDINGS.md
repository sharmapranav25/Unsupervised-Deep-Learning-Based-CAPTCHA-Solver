# CryptBB controller-v2 fit: why it is slow (2026-11-14, Claude)

Read-only on the production run; no process signalled, no source edited. Evidence in this folder:
`sample_fit.py` + `sample2.json` (90 s, 1 Hz CPU per process of PID 137140's tree, verified create time, plus nvidia-smi),
`profile_v2.py` + `profile_result.json` (isolated profile, same code/objective/seeds, CryptBB input 0).
Contention note: the profile ran while production was live. Production used about 1 CPU core of 32 and brief GPU bursts.

## Measured
- Live sampling (90 s): the parent process used 105 core-s, the 14 workers 33 core-s out of 1260 available (2.7% occupancy).
  Tree CPU was about 100% (one core) most of the time. All workers were busy only in sub-second bursts. GPU sat at 1-7% with 1-s spikes to 50-99%.
- Isolated, 42 images (= 14 candidates x 3 fit seeds):
  - render: CAPRI 0.053 s/img, SynthTIGER 0.045 s/img serial median. SynthTIGER's first call takes 1.7 s (import). Spread across 14 workers this is about 0.15 s per batch.
  - feature extraction in the parent: custom.extract_pair 3.64 s plus dino.extract 1.92 s, i.e. 0.133 s/img.
  - objective scoring: 0.034 s per candidate (global plus local).
- Profile of extract: mostly single-threaded NumPy/SciPy per image: regions.extra, histograms, sobel/gaussian, erosion, distance transforms, quantiles.
  The DINO ThreadPool(4) waits on the GIL (1.56 s in lock acquire). GPU forward passes are a small share.
- `extra()` runs 84 times for 42 images. custom.extract_pair first computes the full regions.Perception "old" feature set,
  including extra() on the old masks (about 1.1 s/42 imgs). It then keeps only old['whole'].

## Accounting per reference (estimate from the measurements)
About 1119 images/reference: 1027 fit, about 64 validation, 12 v1 re-score, 16 display.
- extraction: 1119 x 0.133 = about 149 s
- scoring: about 342 candidates x 0.034 = about 12 s
- rendering: about 4 s

That sums to about 165 s, against the observed 164 s/reference. Rendering is not the bottleneck. Serial parent-side CPU feature extraction is about 90% of the runtime.
Search volume matters only through the image count; there is no hang and no slow-effect outlier.

## Recommendation
Let this run finish (no restart). At 27/100 and 4503 s, about 3.3 h remain. A replacement now would save at most about 2 h. It would also need a new versioned run, exactness re-validation, and new receipts and gates for the coordinator.
Corpus generation afterwards is render-only, so this bottleneck does not apply to it.

For the NEXT fit (e.g. DarkBay v2), these speedups are exact (same features and scores; verify bitwise on a batch before use):
1. Move the per-image CPU feature post-processing (extra, ink_shape, descriptors, dino pixel_features) into the existing process pool.
   GPU passes stay batched in the parent. Expected 3-4x on the fit.
2. Skip the discarded old extra() pass in custom.extract_pair (keep old['whole']). About 20% of extraction.
3. Or, with no feature-code change, shard references across 3-4 parent processes (search is per-reference independent; seeds depend only on the index), then merge. Expected about 3-4x. GPU memory per parent must be checked first.

Not recommended: fewer seeds, candidates or references. That changes the experiment.
