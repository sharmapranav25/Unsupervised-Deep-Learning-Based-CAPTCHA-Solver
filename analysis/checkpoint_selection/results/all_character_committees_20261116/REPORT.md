# Frozen20-epoch character committee across CryptBB and DarkBay

Same11/20 rule as the initial DarkBay Model3-A test. Each run is its own committee. All20 epochs eligible, no loss gate; case-insensitive ASCII letters, exact digits; skip disputed/missing-majority positions; earliest tie. Real evaluation remains case-sensitive exact-string. All choices frozen together before labels were read. Repeatedly inspected historical tests: retrospective diagnostic, no official selection or training changes.

Compared with final20: 5 improved, 11 unchanged, 6 worse. Runs are correlated; this tally is descriptive, not an independent-trial significance test.

| Domain / family | Model / arm | Selected epoch | Selected accuracy | Final20 accuracy | Difference pp | Key coverage |
|---|---|---:|---:|---:|---:|---:|
| CryptBB / cryptbb | M1-A | 19 | 47/251 (18.73%) | 49/251 (19.52%) | -0.80 | 88.92% |
| CryptBB / cryptbb_v3 | M1-A | 19 | 59/251 (23.51%) | 59/251 (23.51%) | +0.00 | 89.56% |
| CryptBB / cryptbb_v3 | M1-B | 19 | 61/251 (24.30%) | 60/251 (23.90%) | +0.40 | 89.72% |
| CryptBB / cryptbb_v3_cumulative_static_v1_20261115 | M2-A | 20 | 54/251 (21.51%) | 54/251 (21.51%) | +0.00 | 89.32% |
| CryptBB / cryptbb_v3_cumulative_static_v1_20261115 | M2-B | 19 | 48/251 (19.12%) | 51/251 (20.32%) | -1.20 | 92.11% |
| CryptBB / cryptbb_v3_epoch20_static_A_20261115 | M2-A | 20 | 64/251 (25.50%) | 64/251 (25.50%) | +0.00 | 89.21% |
| CryptBB / cryptbb_v3_epoch20_static_B_20261115 | M2-B | 20 | 64/251 (25.50%) | 64/251 (25.50%) | +0.00 | 91.39% |
| CryptBB / cryptbb_v3_rank1_v1 | M2-A | 17 | 49/251 (19.52%) | 50/251 (19.92%) | -0.40 | 90.76% |
| CryptBB / cryptbb_v3_feature_priority_B_v1_20261115 | M3-B | 19 | 51/251 (20.32%) | 48/251 (19.12%) | +1.20 | 89.32% |
| CryptBB / cryptbb_v3_feature_priority_v1_20261115 | M3-A | 20 | 65/251 (25.90%) | 65/251 (25.90%) | +0.00 | 88.05% |
| DarkBay / darkbay_v3_feature_20261115 | M1-A | 19 | 544/900 (60.44%) | 550/900 (61.11%) | -0.67 | 98.69% |
| DarkBay / darkbay_v3_feature_20261115 | M1-B | 20 | 503/900 (55.89%) | 503/900 (55.89%) | +0.00 | 98.52% |
| DarkBay / darkbay | M2-A | 20 | 600/900 (66.67%) | 600/900 (66.67%) | +0.00 | 98.46% |
| DarkBay / darkbay | M2-B | 19 | 627/900 (69.67%) | 627/900 (69.67%) | +0.00 | 98.44% |
| DarkBay / darkbay_v3_feature_20261115 | M2-A | 19 | 528/900 (58.67%) | 523/900 (58.11%) | +0.56 | 98.91% |
| DarkBay / darkbay_v3_feature_20261115 | M2-B | 19 | 549/900 (61.00%) | 536/900 (59.56%) | +1.44 | 98.63% |
| DarkBay / historical_Darkbay_ds2_noFont-intel | M2-seed123 | 20 | 520/900 (57.78%) | 520/900 (57.78%) | +0.00 | 96.98% |
| DarkBay / historical_Darkbay_ds2_noFont-intel | M2-seed48 | 20 | 577/900 (64.11%) | 577/900 (64.11%) | +0.00 | 97.83% |
| DarkBay / darkbay | M3-A | 19 | 621/900 (69.00%) | 629/900 (69.89%) | -0.89 | 98.44% |
| DarkBay / darkbay | M3-B | 18 | 609/900 (67.67%) | 610/900 (67.78%) | -0.11 | 98.65% |
| DarkBay / darkbay_v3_feature_20261115 | M3-A | 18 | 542/900 (60.22%) | 532/900 (59.11%) | +1.11 | 99.09% |
| DarkBay / historical_Darkbay_model3_adaptive_20261006 | M3-seed48 | 20 | 569/900 (63.22%) | 569/900 (63.22%) | +0.00 | 97.06% |

## Missing20-epoch prediction sets

- CryptBB_M2_cryptbb_v3_rank1_v1_B: missing epochs [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]; no19-epoch/partial committee substituted.
- DarkBay_M1_darkbay_A: missing epochs [1]; no19-epoch/partial committee substituted.
- DarkBay_M1_darkbay_B: missing epochs [1]; no19-epoch/partial committee substituted.
- DarkBay_M2_historical_Darkbay_ds2_noFont-intel_seed456: missing epochs [1, 2, 3, 4, 5, 6, 8, 9, 11, 12]; no19-epoch/partial committee substituted.
- DarkBay_M3_darkbay_v3_feature_20261115_B: missing epochs [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]; no19-epoch/partial committee substituted.
- DarkBay_M3_historical_Darkbay_model3_adaptive_20261006_seed123: missing epochs [19]; no19-epoch/partial committee substituted.
- DarkBay_M3_historical_Darkbay_model3_adaptive_20261006_seed456: missing epochs [1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 19]; no19-epoch/partial committee substituted.

## Limits

Letter-case mistakes do not distinguish epochs in selection, but do count against real exact accuracy. Shared errors, correlated checkpoints, self-votes, positional shifts and skipped tails can all bias this selector. No target-derived lengths or scoring-label thresholds. Earlier archived experiments outside the canonical A/B series are inventoried separately; no extra training or inference was started. Full per-epoch scores, ties, input hashes and coverage are in the JSON artifacts.
