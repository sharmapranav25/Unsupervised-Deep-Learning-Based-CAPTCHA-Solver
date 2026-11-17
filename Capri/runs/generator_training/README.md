# Original lightweight CAPRI + fine-tuned DINO fonts + separate count

Started 28 October 2026 at12:26:53PDT, after the user explicitly deleted the heavy decoder run and selected this architecture. Shared workspace:E:/pranav/CAPRI. This guide, protocol.json, status.json and docs/HYBRID_FONT_CONTROLLER_REDESIGN_HANDOFF.md allow another collaborator to continue.

## What trains

- Original Iter2Model core:3,662,050parameters. Same CNN/effect branch and five-layer160-wide text transformer as the deployed original CAPRI. Local core matches original outputs bit-for-bit at equal weights and inputs.
- DINOv2 ViT-S/14 with registers:all12blocks fine-tuned, specifically for font attributes and128-dimensional font-reference embeddings. Its font predictions add to the original core font heads. Stochastic backbone behavior stays in evaluation mode; gradients remain enabled.
- Separate four-layer128-wide pixel-only count transformer, updated on every joint logical batch. It receives no answer strings, font labels, boxes or counts as inputs.
- Combined26,724,122parameters. No heavy multiscale decoder, instance queries, masks or localization matching.
- Core/count/new heads start from seed0; DINO starts from official pretrained weights. No checkpoint from the deleted run is reused.

## Data and training

Exactly the same100,000selected TRAIN scenes and2,000separate DEV images as the prepared package.20complete epochs,2,000,000scene presentations. Cached RGB is resized with the original preprocessing:height96,width cap480,bilinear; original width bucketing and edge padding. Data:../capri_lightweight_cache_100k_20261028_v2. Cache generation verified original artifacts and each raw pixel hash. Train/DEV pixels and font groups are disjoint. The cached and original manifests match exactly.

Original seven equally weighted task groups, plus independent-count CE and font-reference contrastive at weight1 each. The prepared corpus differs from the old dataset:legacy text-ordinal and font width/slant ordinal labels are unavailable and masked; text presence derives from stored recipe/glyph values; multi-row CTC is masked; unknown font attributes stay masked. This is the original small-model training method with the requested new branches, not identical historical labels or weights. No labels are passed as model inputs.

Original AdamW settings:lr5e-4,betas(.9,.95),weight decay.05 with bias/embedding/norm exclusions,300-update warmup,cosine to1e-5,gradient clip1,EMA.999,bfloat16 andTF32. DINO has lower lr5e-6. Logical batches are up to128 with pixel budget128*352,959logical updates in epoch1. GPU chunks of32 accumulate into each logical update; all branches share the batch. Gradient accumulation uses microbatch-averaged masked task losses, so rare-label weighting is not mathematically identical to one unsplit batch.

Font support:up to8distinct TRAIN reference specimens chosen per logical batch, forwarded with relevant microbatches; a specimen can support multiple microbatches containing its family. Duplicate pixels and known alias-family negative pairs are excluded. Three deterministic pages/file give6,156TRAIN references and564DEV references. DEV references are evaluation only. Whole-image font retrieval is scored exactly on single-family scenes; ambiguous candidates are retained. Mixed-font images still supervise available attributes but are excluded from single-family retrieval accuracy. This does not reproduce the deleted per-character localization pipeline.

Initial and per-epoch synthetic DEV reporting. Final epoch20 EMA is designated; all epoch checkpoints preserved. No real/test/calibration labels or automatic deployment. Active three-controller registry and original model remain unchanged.

## Readiness and estimate

Passed:core equivalence,exact100kmanifest and split checks,cache/pixel/count/label checks,reference duplicate/alias gradient checks,all component gradient groups and all12DINO blocks,finite learning on a fixed TRAIN batch,combined checkpoint restoration and evaluation smoke. Readiness:../capri_lightweight_readiness_20261028_v3/readiness.json. Tests:../capri_lightweight_checks_20261028/. All probe weights discarded before production.

Measured82.30images/s,peak5.94GiBallocated on widest microbatch32. Pure update projection6.75hours; initial budget8-10hours including evaluation/I/O/checkpoint allowances, about20:30-22:30PDT28October2026 if uninterrupted. Short profiling is provisional; use sustained production measurements. This is a runtime estimate, not a guarantee of convergence or model quality.

## Status and recovery

status.json/events.jsonl contain worker identity and live progress. Initial evaluation precedes training. protocol.json and provenance/ bind exact settings and source hashes. resume.json identifies the latest full checkpoint (raw,EMA,optimizer,scheduler,RNG,sampler/execution state). BF16 does not need a GradScaler. Save at update10,every250updates,and every complete epoch. Each committed checkpoint has a SHA256 receipt.

To stop gracefully, create the empty file STOP here and wait for state=stopped. To resume, first confirm the worker has exited, archive/remove STOP, then run from E:/pranav/CAPRI with unchanged sources/configuration:

```powershell
& 'E:/pranav/Final sem/sourcefree_synth/experiments_xgb/.venv/Scripts/python.exe' -B capri.py train-lightweight --dataset runs/capri_lightweight_cache_100k_20261028_v2 --readiness runs/capri_lightweight_readiness_20261028_v3 --out runs/capri_lightweight_joint20_100k_20261028 --resume
```

Never launch a duplicate worker. ../capri_lightweight_launch_20261028 contains guarded launch.ps1,launch.json,stdout.log,stderr.log. This run,cache,logs,temp,checkpoints and handoff are on E. The deleted heavy run cannot be resumed; its minimal deletion receipt is maintenance/joint20_deletion_20261028.json. Earlier unrelated experiments remain preserved.

Live-launch verification at12:29PDT:initial evaluation completed,training reached step61/image6,286 of epoch1. Step10 full combined checkpoint428,259,851bytes independently SHA256 verified (2e0611ff01ce8fe70ded464e42cadb0cff69cd9b913f67cd0767d1dee45d264a). Worker2268 at launch; current status remains authoritative. Deleted heavy-run folder confirmed absent. Receipt:launch_verification.json. Early production70.7images/s,projected around9-10hours total including overhead; estimate provisional.

## Completion verified29October2026

All20epochs completed,each100,000images. Finished28October2026 at19:53PDT,about7hours26minutes after launch. Final epoch20 EMA checkpoint independently SHA256 verified on29October:28f95fd8daf3831fb860ab59e15b8b65bd4a68bacc434eb7eb7a296e6190b764. Receipt:completion_verification_20261029.json. Synthetic DEV:internal count1620/2000(81.0%),separate counter654/2000(32.7%),font category1163/1870(62.19%),exact single-family font retrieval148/1677(8.83%). Previous target-scheme diagnostic used step14,750 during epoch16,not this final checkpoint. No deployment or counter removal has occurred.
