# Controller4 validation sweep results

Verified 112 checkpoint evaluations, each on the same1000 images.

| Reader | Highest validation accuracy | Epoch(s) | Latest evaluated | Latest accuracy | Late-epoch real/validation Spearman rho |
|---|---|---|---|---|---|
| M1-A | 14.9% (149/1000) | 12 | 20 | 13.1% | 0.237 |
| M1-B | 13.7% (137/1000) | 8 | 20 | 13.4% | -0.376 |
| M2-A | 12.6% (126/1000) | 16 | 20 | 12.0% | 0.501 |
| M2-B | 12.3% (123/1000) | 18 | 20 | 12.2% | 0.356 |
| M3-A | 12.2% (122/1000) | 16 | 17 | 11.5% | 0.043 |
| M3-B | 12.4% (124/1000) | 14 | 17 | 11.2% | 0.103 |

The primary metric is case-sensitive whole-string exact match with standard saved free autoregressive decoding plus refinement. Every prediction identity, hash and count was verified. Model1 epoch1 is unavailable, not imputed. Model3 uncommitted epochs were excluded.

Correlations are descriptive, over epochs6 onward with already-saved real diagnostics. Missing real diagnostics are omitted. These serial epoch measurements are not independent trials. The real test was previously inspected and this validation set was designed retrospectively. These associations do not establish prospective checkpoint-selection reliability.

Full per-epoch table: epoch_accuracy.md. Frozen protocol and inputs: protocol.json. Predictions and summaries: modelN/results/darkbay/{A,B}/controller4_validation/epoch_NN. Charts: thesis/figures/darkbay/controller4_validation (overview0-100% and detail0-20% PNG/SVG).

No training, curriculum, generator settings or primary fixed-epoch20 selection changed. One early empty decoder output caused an unused confidence-mean warning; exact-match scoring uses strings only and all1000 predictions were retained.
