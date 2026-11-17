# Epoch-to-epoch accuracy variation

Same contiguous epoch window for all six readers within each comparison. Real2-19 and6-19; synthetic2-17 and6-17. No interpolation, smoothing, A/B averaging, checkpoint selection or training changes.

Descriptive variation along one training trajectory per dataset, not independent-run variance or statistical proof of curriculum effects. Real results are repeatedly inspected historical diagnostics.

## real_darkbay: epochs2-19

All deviations are percentage points. Lower means less observed variation.

| Reader | Accuracy SD | Mean absolute epoch change | SD of signed changes | Largest change |
|---|---:|---:|---:|---:|
| Model1-A | 7.29 | 3.55 | 4.49 | 11.22 |
| Model1-B | 7.59 | 3.88 | 4.54 | 11.89 |
| Model2-A | 7.09 | 5.18 | 7.09 | 18.78 |
| Model2-B | 9.13 | 5.12 | 6.24 | 15.78 |
| Model3-A | 8.71 | 4.99 | 6.70 | 19.22 |
| Model3-B | 8.47 | 5.31 | 7.15 | 18.44 |

## real_darkbay: epochs6-19

All deviations are percentage points. Lower means less observed variation.

| Reader | Accuracy SD | Mean absolute epoch change | SD of signed changes | Largest change |
|---|---:|---:|---:|---:|
| Model1-A | 3.42 | 2.92 | 4.07 | 11.22 |
| Model1-B | 2.21 | 2.85 | 3.19 | 5.33 |
| Model2-A | 4.04 | 4.10 | 5.31 | 10.00 |
| Model2-B | 3.87 | 4.30 | 5.83 | 15.78 |
| Model3-A | 3.41 | 3.72 | 4.82 | 9.00 |
| Model3-B | 3.29 | 3.80 | 5.41 | 13.22 |

## synthetic_validation: epochs2-17

All deviations are percentage points. Lower means less observed variation.

| Reader | Accuracy SD | Mean absolute epoch change | SD of signed changes | Largest change |
|---|---:|---:|---:|---:|
| Model1-A | 2.57 | 1.43 | 1.54 | 3.00 |
| Model1-B | 2.63 | 0.99 | 1.21 | 2.60 |
| Model2-A | 2.19 | 1.13 | 1.36 | 3.70 |
| Model2-B | 2.22 | 1.37 | 1.67 | 4.00 |
| Model3-A | 2.01 | 1.35 | 1.62 | 3.90 |
| Model3-B | 1.96 | 1.12 | 1.39 | 2.90 |

## synthetic_validation: epochs6-17

All deviations are percentage points. Lower means less observed variation.

| Reader | Accuracy SD | Mean absolute epoch change | SD of signed changes | Largest change |
|---|---:|---:|---:|---:|
| Model1-A | 1.33 | 1.31 | 1.55 | 2.70 |
| Model1-B | 0.67 | 0.65 | 0.84 | 1.70 |
| Model2-A | 1.08 | 0.90 | 1.16 | 2.10 |
| Model2-B | 1.28 | 1.35 | 1.73 | 4.00 |
| Model3-A | 0.81 | 0.95 | 1.15 | 2.10 |
| Model3-B | 0.94 | 0.75 | 1.04 | 2.10 |

