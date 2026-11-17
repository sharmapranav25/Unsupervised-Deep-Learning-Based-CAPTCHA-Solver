# Dataset A/B study

Two100k corpora from identical frozen controllers, independent rendering seeds, one20epoch PARSeq run per corpus. All data must audit before either reader trains. No repeated training runs. The folder seed_48 records reproducibility only.

A: E:\pranav\CAPRI\runs\parseq_darkbay_capri2_100k_20261112

B: E:\pranav\CAPRI\runs\parseq_darkbay_dataset_b_100k_20261112

Model1-A will rank B and Model1-B will rank A. Model2 uses a frozen ranking; Model3 starts from that ranking and adapts. Curriculum validation/selection rules must be fixed before those later experiments; this coordinator launches only the two Model1 readers.

Read status.json and active_child.json, then dataset corpus_status.json or seed_48/train.log. datasets_ready.json is the200k-pixel audit. completion.json requires both readers and evaluation. Original artifacts and pre-revision protocol copies remain preserved. No generator deployment.

Resume with tools/run_ab_study.py and the same --a --b --out arguments plus --resume. Do not resume the historical three-run launcher. A study STOP file stops between stages.

## User stop request, 12 November 2026

Finish the currently running Model1-B through epoch20 and save its final checkpoint. Then stop before evaluate_A or any later evaluation/training. STOP and stop_after_model1_b.json enforce and document this instruction. Do not remove STOP or resume without a later user instruction. The existing coordinator labels its STOP-triggered exit failed; this intentional stop is distinguished by the request receipt and B's completion.json.
