## 2026-11-14: full20-epoch prediction-only selection refresh

User requested both analyses after epoch20 measurement. Both runs now use epochs1-20 on the same251 image IDs. Simulation gate passed; frozen artifact hashes verified; estimator, baselines and agreement source unchanged from through18. No real labels opened by this rerun and no thresholds changed.

For v3 B, original whole-string and character modes both select20, top3 [20,19,18]. Whole-string leave-one-out fractional mode selects19, top3 [19,20,18]; character leave-one-out fractional mode selects20. Exact masked rank1 auto selects13, top3 [13,14,20], using fallback k5 because every automatic-mask criterion fails. Exact rank1 at k0 through4 selects20. Character/edit rank1 auto select14, also fallback5 with criterion failure. Results are diagnostic, not a newly adopted checkpoint policy. A remains paused and B training is complete; monitoring paused.

Evidence: analysis/checkpoint_selection/results/cryptbb_A20_B20_rankings/{ranking_only.json,summary_rankings.csv,selection_freeze.json,simulation_gate.json}.
