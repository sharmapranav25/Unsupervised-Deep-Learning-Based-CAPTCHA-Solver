# Checkpoint selection v2: capped mask and temporal coupling

Completed on historical CryptBB Model1-A epochs 1-20 and v3 Model1-B epochs 1-20, using the same 251 images. Historical A in this analysis is not the newly resumed v3 A training run. All choices were frozen before this evaluation opened labels. V2 was requested after previous results and labels had been inspected: this is retrospective method development, not a new blind test.

## Selected epochs

Primary exact-string results, all partners:

| Run | Distant-only | Combined CI + plateau | No-mask coupling |
|---|---:|---:|---:|
| Model1-A | 14 | 19 | 20 |
| Model1-B | 20 | 17 | 20 |

For B, the new combined rule selects **17**, the capped distant-only rule selects **20**, and the no-mask coupling fit selects **20**. This supersedes neither the adopted reporting policy nor the preserved original k5 experiment, which selected13. Excluding degenerate epochs as partners does not change any of these choices.

Char and edit kernels both select A14/B14 for distant-only, A16/B17 for combined, and A20/B19 for no-mask coupling. They measure partial string similarity and are not exact accuracy models.

## Frozen implementation choices

- Mask search uses k=0..floor((n-2)/4), giving0..4 here and0..3 at n15. Original epoch distances are retained after excluding partners. All mask fits and partner counts/mean factors are saved in partner_diagnostics.csv. No majority-low-factor warning fired at the frozen .05 threshold.
- Residual tolerance is unchanged: next-distance mean absolute residual must be at most far mean plus max(.005, .25*far mean). Far means distance at least max(k+2, ceil(epoch-span/2)). **Every real-data configuration fails this criterion. All use explicit fallback k4**, not successful band detection.
- Degeneracy means more than50% empty outputs, more than50% identical outputs across images, or initial unmasked factor below.01. Exact kernel flags A1 and B1-2. Removal means those epochs cannot influence the fit; their factors are inferred afterward from active partners. Flags and k are fixed in bootstrap resamples. Empty/identical means across different images within an epoch, not stability on the same image over time.
- Each of12 configurations uses2000 paired image bootstrap resamples and refits the distant factors. All epoch predictions for a sampled image move together. Intervals are conditional percentile95% intervals for agreement factors, not accuracy intervals or selection-adjusted confidence guarantees.
- Candidates have CIs overlapping the CI of the largest point-estimate factor. Plateau is the mean factor over existing original epochs t+-k including self, truncated at boundaries. Choose highest plateau among candidates, then lower mean neighbour kappa, then earlier epoch. No label-based tie breaking.
- No-mask fit optimizes all off-diagonal pairs jointly with four deterministic starts, a and kappa0 in[0,1], positive tau parameterized by log tau. Numerical tau range.05..100*n is recorded. No main exact fit reaches that tau boundary.
- Neighbour residuals and kappa retain negative or >1 values; denominators <=1e-6 produce an undefined diagnostic rather than division explosions. Per-epoch kappa averages the available +-d partners at each d and then averages those distances. First/second-half decay fits include only pairs wholly within that half.
- Pairwise neighbour/distant weighted contributions add to the eligible off-diagonal mean. **Mode is nonlinear:** regional modes are recomputed using the neighbour or distant voter pool and are not falsely presented as an additive decomposition of the original mode.
- Exact next-epoch disagreement is the flip rate. For char/edit,1-A is fractional dissimilarity, not the frequency of changed full strings. Exact flip rates are available independently in the exact results.

## Uncertainty and coupling findings

- Model1-A: candidates [14, 15, 19, 20]; joint kappa0=0.2040, tau=6.7969. Raw neighbour kappa ranges -0.741..4.803; 27/70 pair estimates lie outside[0,1].
  - first_half: fitted kappa0=0.2140, tau=6.1726, residual MSE=0.0085.
  - second_half: fitted kappa0=1.0000, tau=191.3226, residual MSE=2.2546.
- Model1-B: candidates [12, 13, 14, 15, 16, 17, 18, 19, 20]; joint kappa0=0.2091, tau=4.7311. Raw neighbour kappa ranges -0.488..4.940; 25/70 pair estimates lie outside[0,1].
  - first_half: fitted kappa0=0.2636, tau=4.5996, residual MSE=0.0127.
  - second_half: fitted kappa0=1.0000, tau=131.8772, residual MSE=2.1511.

Later epochs show more excess agreement, but late-half kappa0 hits its upper bound and raw kappa can exceed1. This is evidence of poor calibration/model mismatch, not a literal wrong-error probability or proof that LR decay caused the pattern. Removing degenerate epochs changes joint tau substantially (A6.80 to10.76; B4.73 to15.99), despite unchanged selected epochs.

## Does the old A19 choice come from neighbours?

The proposed expectation is **not established**. The original self-inclusive whole-string and character modes select A19. Recomputed neighbour-only modes, neighbour mean agreement, and distant mean agreement select A20. Distant fractional whole-string mode selects A14, while distant any-tie mode selects A20. These regional diagnostics do not isolate A19 as a neighbour-driven winner. The new combined rule does select A19, but that alone is not evidence for the proposed explanation.

## Evaluation after freezing

| Run | Method | Epoch | Correct/251 | Accuracy | Regret (percentage points) |
|---|---|---:|---:|---:|---:|
| Model1-A | distant_pick | 14 | 47/251 | 18.73% | 1.99 |
| Model1-A | combined_pick | 19 | 47/251 | 18.73% | 1.99 |
| Model1-A | joint_pick | 20 | 49/251 | 19.52% | 1.20 |
| Model1-B | distant_pick | 20 | 60/251 | 23.90% | 0.80 |
| Model1-B | combined_pick | 17 | 58/251 | 23.11% | 1.59 |
| Model1-B | joint_pick | 20 | 60/251 | 23.90% | 0.80 |

B13 and B18 remain observed joint best at62/251. The new combined pick B17 has58/251; distant and joint B20 have60/251. Thus the added plateau rule does not improve this observed B result. A best remains17 at52/251. Differences are small on251 repeatedly inspected images; these results do not establish superiority.

For exact agreement, each neighbour pair is checked to equal P(both correct)+P(both wrong, same string). The residual kappa proxy underestimates observed shared-wrong agreement: mean observed/proxy A=.2461/.0805, B=.2358/.0747. These errors are saved per pair. Char/edit have a separately reported partial-similarity remainder.

B transitions illustrate the late stability:

| Transition | Right-right | Right-wrong | Wrong-right | Same wrong | Different wrong |
|---|---:|---:|---:|---:|---:|
| 17 to 18 | 52 | 6 | 10 | 131 | 52 |
| 18 to 19 | 57 | 5 | 4 | 150 | 35 |
| 19 to 20 | 58 | 3 | 2 | 174 | 14 |

## Validation and limits

18 pytest tests passed, covering original simulations, exact known-model recovery, mask caps, excluded-partner isolation, vectorized-bootstrap parity with independent refits, deterministic resampling, coupling/flip algebra, candidate tie breaking, zero-neighbour handling and label-boundary/tampered-freeze rejection. All12 saved bootstrap arrays have shape2000x20 and finite values. All frozen artifact/source hashes verified. One representative evaluation figure visually checked.

The original three required simulation tests pass. **The revised automatic methods still fail two challenging simulations:** both capped-auto and no-mask fits choose19 instead of6 for the localized late error cluster, and20 instead of8 for the systematic trap. They recover13 in the independent-error case. A stationary exponential coupling term cannot necessarily represent a late-only cluster or systematic traps. These failures were reported, not hidden by threshold tuning.

The method is still exploratory. Keep distant-only, combined and joint choices separate; do not silently choose the one closest to real-label performance. Two runs are insufficient to validate a rule. Freeze the next candidate method and compare on independent runs/families;20-30 additional runs across several seeds and at least3 families would be a useful pilot, not a statistical power guarantee.

## Files and reproduction

Run from analysis/checkpoint_selection with the existing thesis Python environment:
```text
python -m ckpt_select.run_regimes --kernel all --evaluate --out results/new-v2-replication
```
Omit --evaluate for a fully prediction-only run. Existing output directories are rejected. Original run.py and earlier study artifacts are preserved as the v1 implementation.

- summary_rankings.csv: all12 configurations and the three distinct picks.
- partner_diagnostics.csv: every epoch/k, partner count, mean factor and low-partner warning.
- epoch_diagnostics.csv: per-epoch factor, bootstrap CI/SE, candidate and plateau/coupling diagnostics.
- regime_details.json: all mask choices, decay parameters by training half, regional-mode picks and raw residuals.
- eval_transitions.csv and eval_neighbour_agreement.csv: label-only transition and agreement decompositions.
- *_eval.png: true accuracy, uncalibrated distant factor/CI and neighbour kappa on a shared epoch axis.
- pre_real_protocol.json, selection_freeze.json, regime_evaluation_receipt.json, source_snapshot/: protocol and audit trail.
