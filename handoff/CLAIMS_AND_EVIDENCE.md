# Claims and evidence

| Claim | Evidence | Limit |
|---|---|---|
| A/B each100k,200k unique RGB images, zero cross overlap | datasets/darkbay/manifests/audit.json | Shared controller recipes; not independent schemes |
| Model1-A final615/900,68.33% | model1/results/darkbay/A/evaluation_darkbay_cpu_requested_v2/summary.json | Historical repeatedly examined test |
| Model1-B final573/900,63.67% | model1/results/darkbay/B/evaluation_darkbay_cpu_epoch20_requested/summary.json | One reader run |
| Later checkpoints improved over15 | handoff/study_records/epoch15_comparison.json and epoch20_comparison.json |15checkpoint is part of20epoch LR schedule |
| New generator chosen as thesis baseline | Capri/baseline_selection.md | No font-only ablation; not proof for every scheme |
| New Model2/3 ready to train | NOT established | Rankings, adapters, protocol and checks pending |
| A and B equivalent | NOT established | Similar generation process does not prove equal reader outcomes |
| Migration preserved files | handoff/COPY_MANIFEST.jsonl, VALIDATION.json | External Python runtime still required |
