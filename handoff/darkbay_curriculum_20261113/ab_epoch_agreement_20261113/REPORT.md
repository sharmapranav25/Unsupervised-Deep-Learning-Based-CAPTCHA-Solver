# A/B agreement checkpoint-selection diagnostic

Selections were frozen using predictions only, then real labels scored them. Exact whole-string agreement, same-epoch A/B pairs, earliest tie. Nonempty agreement is a predeclared secondary rule; its denominator is all900 images. No confidence or family-length threshold.

| Model | Rule | Epoch | Agreement | A real | B real | Mean real | Change vs synthetic selection |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | raw | 17 | 66.78% | 69.00% | 63.56% | 66.28% | -0.22pp |
| 1 | nonempty | 17 | 66.78% | 69.00% | 63.56% | 66.28% | -0.22pp |
| 2 | raw | 12 | 65.44% | 73.22% | 71.78% | 72.50% | +3.50pp |
| 2 | nonempty | 12 | 65.44% | 73.22% | 71.78% | 72.50% | +3.50pp |
| 3 | raw | 13 | 67.00% | 71.78% | 70.67% | 71.22% | +2.89pp |
| 3 | nonempty | 13 | 67.00% | 71.78% | 70.67% | 71.22% | +2.89pp |

## Association after initial learning

| Model | Epoch window | Spearman agreement vs mean real accuracy |
|---|---|---:|
| 1 | 2-20 | 0.673 |
| 1 | 6-20 | 0.340 |
| 2 | 2-20 | 0.851 |
| 2 | 6-20 | 0.696 |
| 3 | 2-20 | 0.852 |
| 3 | 6-20 | 0.698 |

## Limits

Retrospective hypothesis developed after real curves had been viewed. The same900 unlabeled real images determine agreement then receive label scoring: transductive diagnostic, not an untouched external test. A/B share architecture and recipes; agreement can contain correlated errors. No rule was retuned after scoring. Model1 epoch1 is unavailable. Same-epoch joint selection constrains both runs; the existing synthetic-validation policy selects runs independently. Original study selections remain unchanged.

Checked 118 prediction files, hashes, checkpoint identities against the verified sweep, and900 unique matched image IDs each.

Saved: protocol.json, prediction_inventory.json, agreement_curves.json, selection_freeze.json, scored_curves.json, results.json. Plot: thesis/figures/darkbay/ab_epoch_agreement_20261113/ agreement_vs_real.png and .svg.
