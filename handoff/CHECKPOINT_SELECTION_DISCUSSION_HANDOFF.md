# Next-session handoff: checkpoint-selection analysis before and after updates

Workspace: E:/pranav/Thesis-Pranav-Final. Prepared 2026-11-15 UTC.

## Purpose and immediate answer

The user wants to DISCUSS all checkpoint-selection experiments tried so far, including original modes, A/B agreement, the first mathematical experiment and the updated capped-mask/two-regime version. Read AGENTS.md, the newest handoff/CURRENT_STATE.md, this handoff, and the [latest detailed report](../analysis/checkpoint_selection/results/cryptbb_A20_B20_regimes_v2/REPORT.md).

Latest exact-string results for current B: **v2 combined rule picks17; v2 distant-only picks20; v2 no-mask coupling picks20**. Original v1 masked rank1 picked13. Original whole/character modes picked18 using epochs1-18, then20 using epochs1-20. Always name method, version and prediction window. Do not answer “the selected epoch” without these distinctions.

This is a discussion handoff, not authorization to change training or adopt whichever rule scores best.

## Motivation and constraints

Training uses synthetic CAPTCHAs with substantial synthetic-to-real shift. Every epoch predicts the same real images. The user hypothesized that wrong answers can take many forms while a correct answer has one form, so repeated temporal whole-string/character agreement might identify a transferable checkpoint without labels. Temporal dependence, shared systematic mistakes and item difficulty break that intuition. Consensus does not guarantee a local or global accuracy optimum.

The user explicitly rejected estimating absolute real accuracy: **only predict which epoch**. Fitted a-hat values are uncalibrated agreement factors. The latest request includes their bootstrap CIs and eval-only plots against accuracy; this does not turn them into accuracy probabilities or a measurement of the synthetic-to-real gap.

- Select from predictions only; freeze choices before opening real labels for evaluation.
- Do not tune k, tolerance, weights, burn-in or CI thresholds by sweeping the251 labels.
- Labels/results have already been repeatedly inspected. Code-level separation is enforced, but overall method development is retrospective, not a new blind validation.
- No new experimental selector was adopted for training/reporting. Do not use diagnostics to change curriculum, budget, stopping or selected production checkpoints.
- Preserve historical outputs and code snapshots. New methods/windows require new output folders.

## Which A and B?

| Analysis name | Actual experiment | Frozen predictions |
|---|---|---|
| Model1-A | Historical CryptBB A, previous generator/corpus | Epochs1-20 on251 images |
| Model1-B | Current CryptBB v3 B, now trained to completion | Epochs1-20 on the same251 |
| Current v3 Model1-A | Separate run resumed from paused epoch1 | Still training; NOT historical A above |

Historical A/current B are not controlled independent seeds of the same corpus version. Cross-run agreement here differs from same-epoch A/B agreement on DarkBay.

CryptBB651 real images were split into100 controller references,300 Model4 reserve and251 evaluation. Manifest: evaluation_data/model4_reservation_20261114/cryptbb/evaluation_pool.json. Labels: evaluation_data/model4_reservation_20261114/scoring_only/cryptbb/evaluation_pool_labels.json.

Exact accuracy is case-sensitive correct whole strings/251. Reported character accuracy is100-CER using Levenshtein edits and1258 reference characters. Character voting used1255 observed predicted positions (251 five-character predictions), a different denominator. Five-character decoding was not forced.

## 1. Original whole-string and character modes

For each real image, count epoch predictions; keep all most frequent strings. Score an epoch by how often it matches a mode. The original rule includes its own vote and gives full credit for matching ANY tied mode. A mode need not be a majority and can be tied.

Character mode votes independently at each character position, then scores each epoch against positional modes. It can assemble a string no epoch produced. The general implementation derives lengths from predictions and uses explicit missing-position votes; never label-derived lengths.

Early B results:

| Prediction window | Whole-string mode pick(s) | Character-mode pick(s) |
|---|---|---|
| 1-12 | [11] | not run for this window |
| 1-13 | [13] | [13] |
| 1-15 | [13] | [13] |

At12 epochs:189 unique whole-string modes,62 ties,49 strict majorities. At13:189 unique,62 ties,60 strict majorities. At15:211 unique,40 ties,72 strict majorities. Character voting at15:1228 unique positional modes,27 tied positions,225 images with unique consensus; six unique consensus strings never appeared in any single epoch.

Artifacts: model1/results/cryptbb_v3/B/{prediction_modes_epochs01_12,prediction_modes_epochs01_13,prediction_modes_epochs01_15,character_modes_epochs01_13,character_modes_epochs01_15}. Each preserves protocol, modes, epoch coverage, summary and verification.

Historical A1-20 modes selected19 rather than observed best17. A19 had221/251 whole-mode matches and1189/1255 character-mode matches; A17 had194/251 and1175/1255. These are agreement counts, NOT correct answers. Artifact: model1/results/cryptbb/A/character_modes_epochs01_20.

## 2. DarkBay temporal consensus plus A/B agreement

Separate experiment on118 frozen files and the historical900 DarkBay images: Model1 epochs2-20; Model2/3 epochs1-20, A and B each. Compared temporal character consensus averaged across A/B, same-epoch exact A/B agreement, and50/50 average of those scores.

| DarkBay model | Temporal pick | A/B pick | 50/50 combined |
|---|---:|---:|---:|
| 1 | 20 | 17 | 17 |
| 2 | 20 | 12 | 18 |
| 3 | 20 | 13 | 13 |

No consistent improvement from mixing scores. Model2 worsened from A73.22%/B71.78% at A/B-selected12 to A66.00%/B69.44% at combined18. Shared agreement can be wrong: Model1 epoch17 had601/900 A/B matches,494 correct and107 shared wrong. The user rejected the simple combination as the desired solution.

Evidence: handoff/darkbay_curriculum_20261113/temporal_ab_consensus_20261114/{analysis.py,protocol.json,selection_freeze.json,results.json,curves.json,character_votes.json,vote_statistics.json,source_inventory.json}.

The existing adopted policy is still handoff/darkbay_curriculum_20261113/agreement_selection_policy.json: shared-epoch exact A/B agreement, earliest tie. No replacement was formally adopted in these diagnostic discussions.

## 3. Mathematical experiment v1

Let A_tu be average prediction similarity between epochs. Under strong independent/diverse-error assumptions, A_tu approximately a_t*a_u. Initialize from median valid triplets sqrt(A_tu*A_tv/A_uv), then fit bounded rank1 with Gauss-Seidel coordinate updates. Shared mistakes and image-specific difficulty violate the assumptions.

Implemented kernels:
- exact: full-string equality;
- char: positional equality for equal lengths, otherwise edit;
- edit:1-Levenshtein/max length.

Implemented baselines: self-inclusive whole/character modes; fractional1/k tie credit; leave-one-out (LOO); leave-neighbourhood-out; all/masked mean pairwise agreement; cross-run mode/consensus and mean agreement.

Rank1 masks |t-u|<=k, including diagonal. V1 searches0..5. Choose smallest k with next-distance mean absolute residual <= far mean + max(.005,.25*far mean). Far distance >= max(k+2,ceil(epoch-span/2)). If none pass, explicit fallback5. A fallback is not successful band detection.

Nonnegative rank2 diagnostic uses four fixed-seed starts, flags extra structure if relative MSE improvement>.35 AND absolute RMSE improvement>.003. Components cannot be identified as truth/trap without anchors. Optional scalar collision term was not fitted because it adds confounding.

Eval-only outputs: Spearman/Kendall, regret, true-best in top3, McNemar,10000 paired-bootstrap resamples, calibration by mode vote share and majority wrong-string attractors. Initial broad task also produced factor MAE/calibration; user subsequently narrowed scope to epoch ranking. Preserve that history but do not describe factors as accuracy estimates.

### V1 B choices as the prediction window grew

| B window | Whole mode | Character mode | Whole LOO fractional | Exact rank1 | k / criterion | Rank1 top3 | Cross-run mean exact |
|---|---:|---:|---:|---:|---|---|---:|
| 1-16 | 15 | 13 | 13 | 13 | 4 / passed | [13, 16, 11] | 14 |
| 1-18 | 18 | 18 | 17 | 13 | 5 / FAILED, fallback | [13, 12, 11] | 14 |
| 1-20 | 20 | 20 | 19 | 13 | 5 / FAILED, fallback | [13, 14, 20] | 20 |

A remains fixed at20: original modes19, whole LOO fractional20, exact rank1=14 with k5 fallback, cross-run mean exact14. Its own-run predictions do not change when B's window grows; cross-run source pool does.

At B20, exact rank1 chooses20 for every fixed k0..4, but13 at k5. Char/edit rank1 auto picks14 with fallback5 (earlier B16/B18 picked12). At B18, modes chose18, tied for best actual accuracy, while rank1 top3 excluded18. This motivated the wide-mask concern and v2 request.

### V1 findings and limits

- Independent-error simulation,2000 images: MAE .00868, correct best13.
- Late six-epoch shared-error cluster: unmasked picks19; fixed k5 recovers true6. However automatic-k picks0 and misses6. Do not equate the passing fixed-mask test with successful automatic selection.
- Systematic trap: rank2 flags extra structure (96.32% relative MSE improvement). Automatic rank1 still picks20 instead of true8.
- Initial A20/B16 exact rank2 flags both, approximately89.75% and67.37% relative MSE improvement. Majority wrong-attractors: A61 images, B44 (B16 window only).
- Calibration fails: B13 factor .64288 versus actual .24701. Not a measurement of absolute real accuracy.
- A17 vs A19:52 vs47 correct. Discordants8 favour17,3 favour19; McNemar p=.2265625. A19-minus17=-1.992pp, paired-bootstrap95% CI[-4.7809,+.3984]pp. Not distinguishable at5%; oracle/multiple-comparison selection not adjusted.
- Unique modes with vote share>80% were correct7/11 for A and9/15 for B in the initial study; high agreement is not calibrated confidence.
- Cross-run scoring did not consistently help. Different generation versions are not independent unbiased voters.

Outputs: analysis/checkpoint_selection/results/{cryptbb_A20_B16_verified,cryptbb_A20_B18_rankings,cryptbb_A20_B20_rankings}. Initial cryptbb_A20_B16 is preserved pre-label output interrupted by a plotting syntax error; use *_verified for completed original results. Choices matched after plotting repair. Successful original source preserved in source_versions/A20_B16 and later per-run source_snapshot folders.

## 4. Mathematical experiment v2: latest update

The user requested this AFTER v1 results were inspected. Implemented and completed with unchanged residual tolerance, no label-tuned thresholds. Full report: analysis/checkpoint_selection/results/cryptbb_A20_B20_regimes_v2/REPORT.md.

1. Cap k_max=floor((n-2)/4):4 at20,3 at15. Search0..cap. Per epoch/k: distant partner count, mean factor, warning if majority partner factors<.05. All real auto criteria fail; fallback4. No majority-low-factor warning fired.
2. No-mask model jointly fits all off-diagonal pairs: a_t*a_u + kappa0*exp(-distance/tau)*(1-a_t)*(1-a_u). a,kappa0 in[0,1]; tau positive via log parameter, numerical bounds.05..100*n. Four deterministic starts, boundary flags.
3. Burn-in/degeneracy: >50% empty outputs or >50% identical outputs ACROSS DIFFERENT IMAGES in an epoch, or initial unmasked factor<.01. Exact flags A1/B1-2. Remove them as partners, fit active epochs, then infer excluded factors from active partners; never renumber epochs. Report both variants.
4.2000 paired IMAGE bootstrap resamples per run/kernel/variant, full factor refits. All epoch predictions for one image move together. k and flags fixed.95% percentile CIs/SE are conditional factor uncertainty, not accuracy intervals or selection-adjusted guarantees.12 configurations,24000 refits.
5. Neighbour residual R=A-a_t*a_u and kappa=R/[(1-a_t)(1-a_u)], averaged over available +/-d. Preserve negative/>1 values; denominator<=1e-6 is undefined. Exact flip rate1-A_t,t+1; char/edit1-A is fractional dissimilarity. Fit decay globally and separately within each training half.
6. Decompose pairwise agreement into weighted neighbour/distant contributions. Recompute regional modes separately. Mode is nonlinear: regional mode scores are NOT additive parts of global mode.
7. Combined candidates:95% factor CIs overlap CI of largest point-estimate factor. Pick highest mean factor across t+-k including self, truncated at epoch boundaries. Ties: lower mean neighbour kappa, then earlier epoch. Keep distant-only, combined and joint choices separate.
8. Freeze then eval: both-correct and both-wrong-same neighbour agreement, kappa comparisons, five next-epoch transition types and shared-x-axis true accuracy/factor/kappa plots. Exact A decomposes exactly into both-correct+same-wrong; char/edit need a partial-similarity remainder.

### V2 choices on all20 epochs

| Kernel | Run | Distant-only | Combined | No-mask coupling |
|---|---|---:|---:|---:|
| Exact | Historical A | 14 | 19 | 20 |
| Exact | Current B | 20 | 17 | 20 |
| Character | Historical A | 14 | 16 | 20 |
| Character | Current B | 14 | 17 | 19 |
| Edit | Historical A | 14 | 16 | 20 |
| Edit | Current B | 14 | 17 | 19 |

Excluding degenerate partners changes no choices. Exact candidates: A[14,15,19,20], B[12,13,14,15,16,17,18,19,20]. B17 is a plateau choice within a broad noise band, not a sharply identified optimum.

Joint exact (all partners): A kappa0=.2040,tau=6.7969; B .2091,4.7311. Excluding flagged partners changes tau to10.7578 and15.9919 without changing picks. Late-half neighbour fits hit kappa0=1; raw kappas outside[0,1] occur in27/70 A and25/70 B pairs. The probability model does not fit those residuals well. Do not interpret kappa as calibrated inherited-error probability or claim LR decay caused the pattern.

The expected explanation “old A19 is neighbour-driven” was NOT established. Original global modes pick19, but neighbour-only modes and neighbour mean agreement pick20. Distant fractional whole-string mode picks14; distant any-tie mode20. Combined v2 selecting19 does not prove the explanation.

V2 exact B: distant20=60/251; combined17=58/251; joint20=60/251. Best13/18=62/251. Added plateau did not improve the observed B outcome. Historical A14/A19=47/251; A20=49/251; observed best A17=52/251.

Late B stability includes increasing shared wrong answers:

| Transition | Right-right | Right-wrong | Wrong-right | Same wrong | Different wrong |
|---|---:|---:|---:|---:|---:|
| 17 to18 | 52 | 6 | 10 | 131 | 52 |
| 18 to19 | 57 | 5 | 4 | 150 | 35 |
| 19 to20 | 58 | 3 | 2 | 174 | 14 |

18 tests pass, including known exponential-model recovery, excluded-partner isolation and bootstrap parity. This does NOT mean all revised methods succeed: capped-auto/no-mask both pick19 instead of6 in the late-cluster simulation, and20 instead of8 in the trap simulation. Both recover13 in independent errors. Failures are saved, not tuned away.

## 5. All comparable real accuracies (evaluation only)

Same251 images, labels not used by selection. Character%=100-CER.

| Epoch | Historical A correct | A exact % | A char % | Current B correct | B exact % | B char % |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0.00 | 1.75 | 0 | 0.00 | 1.51 |
| 2 | 3 | 1.20 | 52.07 | 0 | 0.00 | 9.22 |
| 3 | 25 | 9.96 | 63.67 | 38 | 15.14 | 66.06 |
| 4 | 46 | 18.33 | 70.19 | 39 | 15.54 | 67.09 |
| 5 | 34 | 13.55 | 67.25 | 50 | 19.92 | 72.42 |
| 6 | 31 | 12.35 | 65.74 | 42 | 16.73 | 67.41 |
| 7 | 37 | 14.74 | 66.30 | 50 | 19.92 | 73.29 |
| 8 | 33 | 13.15 | 67.33 | 51 | 20.32 | 71.62 |
| 9 | 47 | 18.73 | 69.48 | 35 | 13.94 | 67.49 |
| 10 | 50 | 19.92 | 69.16 | 45 | 17.93 | 70.35 |
| 11 | 46 | 18.33 | 70.67 | 52 | 20.72 | 71.94 |
| 12 | 47 | 18.73 | 70.75 | 55 | 21.91 | 72.73 |
| 13 | 42 | 16.73 | 70.59 | 62 | 24.70 | 72.42 |
| 14 | 47 | 18.73 | 71.14 | 53 | 21.12 | 72.89 |
| 15 | 45 | 17.93 | 70.59 | 58 | 23.11 | 73.05 |
| 16 | 47 | 18.73 | 70.83 | 59 | 23.51 | 71.86 |
| 17 | 52 | 20.72 | 71.78 | 58 | 23.11 | 73.13 |
| 18 | 48 | 19.12 | 71.07 | 62 | 24.70 | 74.09 |
| 19 | 47 | 18.73 | 71.14 | 61 | 24.30 | 73.61 |
| 20 | 49 | 19.52 | 71.38 | 60 | 23.90 | 73.21 |

A best17=52/25120.72%. B best13/18=62/25124.70%; B18 character74.09%, B13 character72.42%. Latest question: v2-selected B17 gets58/25123.11%, character73.13%.

## 6. Implementation and evidence map

Package: analysis/checkpoint_selection/ckpt_select/.

- loader.py: only opaque_id/prediction fields; verifies frozen hashes and identical251 IDs; read-only predictions.
- agreement.py, baselines.py, estimator.py: preserved v1 kernels/baselines/math.
- simulate.py and tests/test_selection.py: original simulations/tests.
- regimes.py: v2 prediction-only math; no label path/evaluate imports.
- run_regimes.py: v2 orchestration, source snapshots/protocol, bootstrap/plots, freezes before optional eval.
- evaluate.py: ONLY module allowed to open real labels, after artifact/source verification. Original evaluator plus evaluate_regimes.
- tests/test_regimes.py: nine new tests, total18. Includes cap, support, exclusion, known-model fit, bootstrap equivalence/determinism, kappa/flip algebra, plateau ties, absent neighbours and label/freeze boundary.
- run.py remains historical v1 (k0..5). DO NOT use it expecting v2. ranking_only.py reads frozen choices without labels.
- results/cryptbb_A20_B20_regimes_v2: REPORT.md; summary_rankings.csv; partner_diagnostics.csv; epoch_diagnostics.csv; decomposition.csv; residual_by_distance.csv; regime_details.json (allk picks, decay fits/halves, raw residuals);12 *_bootstrap.npy arrays (2000x20 each); heatmaps, residual and CI plots; *_eval.png.
- Eval files: eval_selected.csv, eval_transitions.csv, eval_neighbour_agreement.csv. Audit: simulation_gate.json, pre_real_protocol.json, selection_freeze.json, regime_evaluation_receipt.json, verification.json, source_snapshot/.
- Root REPORT.md links latest and retains historical original report. Old absolute-calibration text is history, not current objective.

Reproduce v2 from shared thesis PowerShell only if needed; do not rerun merely to discuss saved results:

```powershell
. ./environment.ps1
$env:PATH='C:/ProgramData/Anaconda3/Library/bin;'+$env:PATH
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
Push-Location analysis/checkpoint_selection
& $thesisPython -B -m ckpt_select.run_regimes --kernel all --evaluate --out results/new-v2-replication
Pop-Location
```

Omit --evaluate for prediction-only. Fresh output folders required. Pytest installed only under analysis/checkpoint_selection/test_site; no shared training dependency changes. Reproducing historical freezes requires their source snapshots because current evaluate.py includes new v2 code.

## 7. Questions for next discussion (undecided)

- Which objective matters: consensus resemblance, stability, independence from shared mistakes or transferable correctness? They are not equivalent.
- Should failure of every residual criterion cause abstention instead of capped fallback? Not adopted.
- Is plateau averaging justified when late consensus grows while measured accuracy decreases? Boundary truncation changes plateau support.
- How could localized/nonstationary coupling and systematic traps be separated without anchors? The stationary exponential model fails those simulations.
- Are overlapping individual CIs the right candidate rule, or should a prospectively frozen paired-difference/selection-aware alternative be tested? Current version uses overlapping95% CIs.
- Once current v3 A finishes, how to test a truly matched A/B comparison without silently selecting a rule based on the known251 labels?
- Two historical runs are too few. Freeze prospective methods and test independent runs/families.20-30 more independent runs across seeds/conditions and at least3 families is a pilot suggestion, not a power guarantee or authorization to launch new runs.

## 8. Live training instructions (read fresh state next session)

Current v3B completed20 epochs/15640 updates. Epoch20 SHA b530ebe4ec3a0a312d71b6eafa9e3452b8c0e0198df795b557cd8c25d2486022. All B1-20 real251 evaluations saved.

User explicitly said “Start model 1 -A training”, superseding A's earlier epoch1 pause. Current v3A resumed verified epoch1 with optimizer/scheduler/RNG, same seed48,batch128,20epoch total. Actual workerPID132152/create1794700519.2510707; launcherPID75364. Confirm against training.lock.json; a PID alone is not proof.

At handoff creation: {"updated_at": "2026-11-15T00:09:18.647818+00:00", "pid": 132152, "optimizer_steps": 1700, "epoch": 3, "step_in_epoch": 136}. Latest committed checkpoint: epoch2. Re-read live state; progress is not a saved checkpoint.

Evidence: handoff/cryptbb_v3_training_20261114/resume_A_after_B_20261114/{authorization.json,launch.json,verified_running.json,worker_A.log}. Heartbeat watch-cryptbb-model1-and-model2-training is ACTIVE, updated to monitor A alone every5minutes and pause after verified completion. Routine heartbeat progress is quiet.

Do not launch Model2/3/4; Model4 explicitly stopped. Do not invoke old paired coordinator or old pause watcher. B must not restart. Full100000 synthetic GPU evaluation was explicitly canceled and must not resume; partial8128 predictions are not full accuracy. Training is separate from this CPU analysis.

## Suggested next-session opening

> Read AGENTS.md, handoff/CURRENT_STATE.md and handoff/CHECKPOINT_SELECTION_DISCUSSION_HANDOFF.md. I want to discuss all versions of label-free checkpoint selection: original whole-string/character modes and A/B agreement, v1 masked rank1, and v2 capped-mask/coupling/plateau. Keep versions and historical A versus current v3 A distinct. Compare why choices changed, what the simulations and real diagnostics support, and what remains unproven. Do not change training or adopt a new selector just from this discussion.
