# Adaptive curriculum for synthetic-to-real transfer: discussion handoff

Prepared14 November2026 UTC (13 November local time). Work in E:/pranav/Thesis-Pranav-Final. Read AGENTS.md, README.md and handoff/CURRENT_STATE.md first. All project outputs/caches remain on E:.

## Start here: user intent and authorization

The user wants to DISCUSS a new research direction in the next session, not implement it yet. Their latest idea:

> With an adaptive curriculum, the goal should be to keep the model from overfitting to synthetic data, rather than simply making it learn synthetic data better.

Agreed interpretation: design a curriculum that encourages transferable learning while limiting specialization to the synthetic generator. The open question is what observable signal could guide this without real training labels. No signal, algorithm, threshold or new experiment has been selected.

Important correction from the user: when they asked whether learning-progress and SFL could be added, the assistant began treating the question as implementation authorization. The user explicitly corrected this: "IT was a question" and "I will not add both together else I will not know which one worked." Only source/literature reads occurred. NO learning-progress or SFL implementation was written, no new sampler was integrated, and no training was launched. Do not infer implementation approval from the discussion.

The existing CryptBB dataset/setup-only authorization and background preparation continue independently. This handoff does not cancel that job or authorize CryptBB reader training, City, other STR architectures, pseudo-labeling or new datasets. See handoff/CRYPTBB_VERTICAL_HANDOFF.md for the technical training plan, implementation gaps and output locations.

## Core distinctions to preserve

1. Better synthetic optimization is not necessarily better real transfer. A curriculum can accelerate learning transferable character shapes OR generator-specific shortcuts.
2. Memorizing synthetic training images and over-specializing to a synthetic distribution are different. A model can generalize to unseen samples from the same generator yet fail on real images.
3. Learning progress measured on synthetic samples is still a synthetic-domain signal. It is not automatically progress toward real transfer.
4. An adaptive curriculum changes example/group emphasis based on feedback over training. Regularization or early stopping alone is not an adaptive curriculum, though they can accompany one.
5. Reducing synthetic specialization is a valid design objective, not an established property of a proposed algorithm. Do not claim a method prevents overfitting before testing it.
6. The synthetic-to-real gap is a plausible contributor to remaining errors, not proven to be the sole cause. Do not claim that curriculum can never reduce the gap, or that all generalization error is overfitting.
7. A/B prediction agreement on unlabeled real images is an imperfect checkpoint-selection proxy: both readers can agree on wrong strings. Higher agreement is not proof of higher target accuracy.
8. One CAPTCHA family is insufficient to establish that adaptive and static curriculum are equivalent. The prospective CryptBB replication remains important.

The proposed research question is:

> Can we adapt sampling within a fixed synthetic pool to improve real-world generalization, without using real labels to guide training?

A useful next conversation should identify the proposed signal, explain why it could indicate transferable learning rather than synthetic mastery, and design a controlled test. Do not jump from the objective to an unvalidated algorithm.

## What the current models actually do

All variants use the same approximately20.1M-parameter SVTRv2LNConvTwo33 encoder + PARSeq decoder. Each dataset A/B is a fixed100000-image synthetic pool. One run per model/pool, seed48;20 epochs, batch128,782 steps/epoch,100096 draws/epoch and15640 steps/run. Each starts from scratch. Same-pool comparisons are essential; no extra images for an adaptive variant.

Initial ranks: Model1-B epoch20 ranks A; Model1-A epoch20 ranks B. This avoids using the initial teacher's own training pool. Model2 freezes those ranks. Model3 starts from those SAME ranks and later uses its own current reader to rescore its synthetic training pool after every epoch. Updating its own difficulty scores is intentional, not the initial cross-teacher protocol.

Model2 static sampling: epoch1 easy/medium/hard=(1,0,0), epoch2=(.5,.5,0), epochs3-20=(.15,.25,.60).

Model3 implementation: Capri/src/capri_controller/training/ds3_data.py, invoked through ab_curriculum.py. It uses a free-decoding character-NLL score against known synthetic strings, not real labels and not exactly the training objective. It smooths rank percentiles with .5 old/.5 new, rebuilds equal thirds, calculates risk=1-exp(-min(NLL,50)), and increases sampling mass for groups with larger mean risk. Later group masses are bounded.2-.5 with a.1 slew limit (first transition special-cased). It does not measure learning progress or estimate repeated success rates.

Observed DarkBay behavior in model3/runs/darkbay/{A,B}/history.json:
- After epoch1, all three group risks were almost1, so epoch2 sampling was approximately uniform.
- Later hard-group mass approaches/reaches50%; A reaches the cap in the update after epoch9, B after epoch10. These updates determine the NEXT epoch's draws.
- The ranking continues to adapt even while mass is capped. This is not evidence of a coding failure.

IMPORTANT INTERPRETATION CORRECTION: Model2 and Model3 differ in BOTH ranking updates and sampling proportions. Therefore their accuracy difference compares two curriculum packages; it does not isolate the effect of updating ranks every epoch. The previous phrasing "updating every epoch has no extra benefit" was too strong. A clean dynamic-ranking ablation would keep the same mass schedule and other settings and vary only frozen vs updated ranks.

## Methods discussed, not implemented

### Learning-progress teacher

Closest proposed next experiment by conceptual fit, not a guarantee of superior target accuracy. Track changes in the same student's performance across time; prioritize groups with meaningful improvement or deterioration/forgetting. Do not compare the pretrained teacher's NLL directly with the freshly initialized student's NLL and call that learning progress.

Suggested discussion example: freeze initial Model1 difficulty groups, then track their smoothed performance across epochs so changing group composition does not masquerade as progress. More practice for loss2.0->1.2 (improving), less priority for2.0->1.99 (stalled), revisit.2->.5 (forgetting), retain coverage for.1->.1 (stable). These numbers were illustrative, not chosen thresholds or an approved formula.

Open choices: per-example vs fixed group signal; loss change vs exact-match change; signed/absolute progress; smoothing/window; normalization; exploration/coverage; treatment of noise/stagnation. Fixed groups plus a new allocation rule would change multiple components relative to current Model3, so name it a separate strategy and use ablations if attributing a specific mechanism. A full TSCL reproduction is not promised; any adaptation must be described accurately.

### SFL / learnability

Uses estimated success probability p and priority p(1-p). For deterministic greedy OCR, repeating identical inference gives the same answer; estimating intermediate success rates requires a declared stochastic decoding procedure or other explicitly defined trials. No inference method or number of trials has been selected. Model confidence is not automatically a measured success frequency.

Published RL theory/speedups do not automatically apply to supervised PARSeq. Even always-wrong synthetic examples can give useful supervised gradients because their correct strings are known. Pure p(1-p) gives those examples zero priority; uniform exploration may preserve their coverage. Multiple trials also increase scoring cost. Test this independently if ever authorized; do not combine it with learning progress in the first comparison.

### Current classification

Current Model3 is best described as teacher-initialized, self-assessed difficulty-based adaptive sampling. It resembles self-paced learning in using student feedback, but it is not classical easy-first threshold self-paced optimization, TSCL, SFL, PPO or a learned RL teacher. No new models or policy implementations have been added.

## Fixed-data and selection constraints

Keep exactly the same A/B images, synthetic labels, architecture, optimizer, training budget and comparable selection policy across variants. Do not add different synthetic examples, augmentations or pseudo-labeled real images only to one variant and attribute gains solely to curriculum. Different exposure frequencies are the intended curriculum intervention; exact draw sequences need not match across strategies.

All readers train exclusively on synthetic images. Current reporting selects each model pair's shared epoch by maximum same-epoch full-string A/B agreement on unlabeled real pixels, earliest tie. This is not agreement between consecutive epochs. Rule: handoff/darkbay_curriculum_20261113/agreement_selection_policy.json.

For future CryptBB, freeze the selection rule/pool/candidate epochs before looking at reader real accuracy. Same real pool for selection and scoring is transductive, not an untouched test. Keep labels isolated. Using real-image agreement to alter training sampling would be a NEW intervention beyond checkpoint selection; do not silently repurpose it or claim no real inputs guide the procedure. Discuss explicitly if proposed.

Current checkpoint selection does not reliably find the best real epoch. Hindsight-best real results are diagnostic benchmarks, not deployable choices or selection targets. Any new proxy evaluated retrospectively on repeatedly inspected DarkBay is exploratory and needs prospective testing. Do not sweep thresholds against real test accuracy.

## Results grounding this discussion

DarkBay900-image real evaluation, current agreement-selected averages across A/B:
- Model1:66.28%, selected epoch17; A69.00%, B63.56%.
- Model2:72.50%, selected epoch12; A73.22%, B71.78%.
- Model3:71.22%, selected epoch13; A71.78%, B70.67%.

Hindsight individual-run real maxima, then averaged: Model1 69.89% (A e14/B e12), Model2 73.67% (A e6/B e7), Model3 72.83% (A/B e6). These do not establish a global optimum beyond observed checkpoints.

Separate GLM-OCR DarkBay baseline71.67%, approximately900M parameters (~45x larger). Model2 mean exceeds it by0.83pp; no significance claim. Earlier synthetic-validation selection remains historical, superseded for primary reporting. Current evidence does not establish adaptive as universally more stable; which metric/window is used matters.

## Operational snapshot and evidence

At the handoff read, CryptBB prep status says generate_B/running, reader_training_started=false. B corpus_status reported71004/100000 images,213/300 shards. A already packed/audited100000 unique images. This is a snapshot, not a fresh process-health verification; consult actual processes and completion receipts next session. Do not duplicate preparation. Last heartbeat verified coordinator58132 and generation139336 alive around00:02UTC. Preparation-only automation: prepare-cryptbb-after-darkbay.

CryptBB count5 was inferred from70/100 votes under unique leader >=50/100; it was not manually chosen or based on40% accuracy. All datasets/setup and immutable-source details are in CRYPTBB_VERTICAL_HANDOFF.md.

Code and evidence:
- Capri/src/capri_controller/training/ds3_data.py
- Capri/src/capri_controller/training/ab_curriculum.py (DarkBay-specific runner; do not launch for CryptBB unchanged)
- model3/runs/darkbay/{A,B}/history.json
- handoff/darkbay_curriculum_20261113/protocol.json (original protocol preserved)
- handoff/darkbay_curriculum_20261113/agreement_selection_policy.json
- handoff/darkbay_curriculum_20261113/family_report_agreement_selected.json
- handoff/cryptbb_model1_preparation/status.json and completion receipts

Primary research consulted for discussion (not implementations adopted):
- Self-Paced Learning for Latent Variable Models: https://proceedings.neurips.cc/paper_files/paper/2010/hash/e57c6b956a6521b28495f2886ca0977a-Abstract.html
- Teacher-Student Curriculum Learning: https://arxiv.org/abs/1707.00183
- Automated Curriculum Learning for Neural Networks: https://proceedings.mlr.press/v70/graves17a.html
- LILO / Learning to Reason at the Frontier of Learnability: https://arxiv.org/abs/2502.12272 and https://proceedings.neurips.cc/paper_files/paper/2025/hash/45d5823950cab9a88e10c6e8dbff351c-Abstract-Conference.html
- SFL official repository: https://github.com/amacrutherford/sampling-for-learnability
- Shortcut Learning in Deep Neural Networks: https://www.nature.com/articles/s42256-020-00257-z
- A Theory of Learning from Different Domains: https://www.alexkulesza.com/pubs/adapt_mlj10.pdf

Suggested next-session opening: clarify which failure the user wants to target (memorization, generator-specific features, unstable training or checkpoint selection), then discuss candidate observable proxies and a single controlled experiment. Discuss first; do not implement or train unless the user explicitly asks to proceed.
