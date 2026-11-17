# Real DarkBay accuracy at synthetic-validation-selected epochs

Selection rule: highest exact-match accuracy on the frozen1000 synthetic images per reader; earliest epoch on ties. Selected epochs were frozen before this evaluation.

| Reader | Selected epoch | Synthetic validation | Real DarkBay | Correct /900 |
|---|---|---|---|---|
| Model1-A | 12 | 14.9% | 65.00% | 585/900 |
| Model1-B | 8 | 13.7% | 68.00% | 612/900 |
| Model2-A | 16 | 12.6% | 68.56% | 617/900 |
| Model2-B | 18 | 12.3% | 69.44% | 625/900 |
| Model3-A | 16 | 12.2% | 67.00% | 603/900 |
| Model3-B | 14 | 12.4% | 67.67% | 609/900 |

Model1 eligible epochs2-20 (epoch1 unavailable); Model2 epochs1-20; Model3 epochs1-17 captured at the validation sweep. Model3 selections may change when its remaining epochs are evaluated.

Five existing frozen real evaluations were reused. Missing Model3-B epoch14 was evaluated using the existing paired diagnostic; its companion Model3-A epoch14 result is auxiliary and does not replace the selected Model3-A epoch16. All selected prediction hashes, checkpoint identities and exact900-image sets were verified, then counts were recomputed against scoring labels.

The real test was previously inspected and this synthetic validation design is retrospective. This is a separate selection experiment; original fixed epoch20 primary results and ongoing training are unchanged.
