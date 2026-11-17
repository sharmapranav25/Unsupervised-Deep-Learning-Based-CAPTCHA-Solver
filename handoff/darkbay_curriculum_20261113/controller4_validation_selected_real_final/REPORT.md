# DarkBay final results: synthetic-validation-selected checkpoints

All readers completed20training epochs. Each epoch choice uses only the frozen1000-image synthetic validation accuracy; earliest epoch breaks ties. Real labels score the frozen choice. The adopted rule is retrospective for this existing study.

| Reader | Selected epoch | Validation correct/1000 | Real correct/900 | Real accuracy |
|---|---:|---:|---:|---:|
| Model1-A | 12 | 149 | 585 | 65.00% |
| Model1-B | 8 | 137 | 612 | 68.00% |
| Model2-A | 16 | 126 | 617 | 68.56% |
| Model2-B | 18 | 123 | 625 | 69.44% |
| Model3-A | 19 | 123 | 621 | 69.00% |
| Model3-B | 14 | 124 | 609 | 67.67% |

## Threshold

The40% length-accuracy claim is not established. DarkBay used the unique leading count with89/100 votes for6; the rule is vote agreement >=50/100 with ties rejected. CryptBB must recompute its own vote from reference pixels.

## Saved outputs

- Training checkpoints, metrics, logs and loss plots: model{1,2,3}/runs/darkbay/{A,B}; Model1 final20 archive also under model1/models/parseq/darkbay/{A,B}.
- Frozen ranking: model2/rankings/darkbay/{A,B}; Model3 references the same opposite-Model1 rankings and saves updates under its run adaptive/ folder.
- All validation predictions: model{1,2,3}/results/darkbay/{A,B}/controller4_validation/epoch_NN.
- Complete validation table/protocol/verification: handoff/darkbay_curriculum_20261113/controller4_validation_sweep_final.
- Final selection freeze and verified selected real results: handoff/darkbay_curriculum_20261113/controller4_validation_selected_real_final.
- Charts: thesis/figures/darkbay/controller4_validation_final.
- Real prediction paths and hashes for each selected reader are listed in results.json.
- Historical fixed20 real results: model{2,3}/results/darkbay/{A,B}/evaluation_epoch20 and handoff/darkbay_curriculum_20261113/family_report.json. Earlier validation-through17 reports remain preserved.

## Limits and issues

No training failure was observed. Model1 epoch1 is unavailable. This is one trajectory per dataset, and the real test has been repeatedly inspected. Synthetic accuracy is low and validation/real trajectories do not align consistently; selections nevertheless remain fixed by the adopted synthetic-only rule. Source, checkpoint, prediction and image identities were verified.

The user-authorized next stage is CryptBB dataset and Model1 setup preparation only. CryptBB training, City and other STR models remain unauthorized at this stage.
