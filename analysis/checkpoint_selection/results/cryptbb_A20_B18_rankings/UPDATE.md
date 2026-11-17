# Update: CryptBB Model1-B through epoch18

Requested real evaluations on the same251 images: epoch17=58/251 (23.1076%), character accuracy73.1320%; epoch18=62/251 (24.7012%), character accuracy74.0859%. Epoch18 ties epoch13 on complete-CAPTCHA accuracy. Both prediction freezes, checkpoint identities and independent recounts passed.

The selection rerun used predictions only, with A1-20 and B1-18. Existing kernels, baselines, optimizer, thresholds, fixed seeds and k0-5 grid were unchanged. Source snapshots and all162 epoch-only rankings are saved in this directory. No labels, evaluation summaries or absolute-accuracy calculations were used in this rerun.

| Method for B | Selected epoch | Top3 |
|---|---:|---|
| Masked rank1, exact, auto |13|13,12,11|
| Whole-string mode, original any-tie |18|18,17,15|
| Character mode, original any-tie |18|18,17,15|
| Whole-string mode, LOO fractional ties |17|17,18,16|
| Character mode, LOO fractional ties |18|18,17,15|
| Cross-run mean exact agreement |14|14,18,16|

**Mask warning:** Unlike the through16 result, B's exact automatic criterion now fails every tested k. Epoch13 comes from the documented fallback k5; do not present this as a successfully validated mask. Char/edit auto passes at5 and selects12. A remains exact14 with failed-mask fallback5. Choice stability is conditional on the input epoch window, and the highest observed whole-string accuracy is currently tied13/18.

No change to training or adopted checkpoint policy. A remains paused. The cancelled full synthetic evaluation stays stopped. The latest CLI is:

```powershell
python -m ckpt_select.run --run all --kernel all --k auto --b-through 18 --out results/new-A20-B18
```

Nine pytest tests passed. Frozen output/source hashes and equality of estimator/baseline/kernel source against the prior16-epoch code were verified. The updated input-window and ranking-output code is snapshotted here; earlier successful source is preserved at ../../source_versions/A20_B16. See ranking_only.json, summary_rankings.csv and Model1-B_rankings.png. Prior through16 results remain unchanged.
