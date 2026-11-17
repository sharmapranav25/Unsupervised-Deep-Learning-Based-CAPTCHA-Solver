# Claude handoff: Model 4 pseudo-labeling discussion

Prepared 2026-11-14 by Claude for a NEW Claude session where the user will discuss Model 4 (pseudo-labeling). Workspace `E:/pranav/Thesis-Pranav-Final`; all outputs and caches stay on E:. Background: `AGENTS.md`, `README.md`, `handoff/CURRENT_STATE.md`. The older joint handoff `handoff/CLAUDE_MODEL3_MODEL4_HANDOFF.md` has the original Model 4 question list. This file supersedes it for Model 4 and updates its state. Model 3 (rolling curriculum) is a separate discussion; do not merge the two.

## 1. Scope and authorization

- **Discussion only.** Nothing for Model 4 is designed, approved, implemented or generated: no teacher, unlabeled pool, threshold, loss weight, schedule or acceptance rule, and no pseudo-labels or code.
- The user has corrected assistants before for treating conceptual questions as approval. Discuss, write down candidate rules, and freeze a design only when the user says so. Implementation and training need a separate explicit instruction.
- Model 4 adds REAL (pseudo-labeled) images to training. Report it as a different adaptation method, not as a curriculum improvement over Models 1-3.

## 2. Current state (re-read receipts; this is a snapshot)

### CryptBB Model1
- **Model1-A, old data** (`datasets/cryptbb/A`): finished all 20 epochs (`model1/runs/cryptbb/A/training_completion.json`, `selection_pending: true`).
- **Model1-B was cancelled** by the user. `handoff/cryptbb_model1_training_20261113/STOP_AFTER_A.json` blocks the old pair coordinator. So the frozen A/B agreement selection cannot run for this pair; no B exists.
- **User-requested CPU diagnostics** of A on the 551 labeled images (exact full-string accuracy, CER), in `model1/results/cryptbb/A/requested_cpu_epochNN/summary.json`. Labels were read for these. **Do not use them to choose any Model 4 checkpoint, teacher or threshold.** That would be selection on the test labels.

| epoch | accuracy | CER |
|---|---|---|
| 2 | 2.9% | 48.1 |
| 9 | 17.6% | 30.1 |
| 10 | 18.3% | 30.2 |
| 12 | 18.7% | 28.7 |
| 13 | 19.1% | 28.3 |
| 17 | 20.7% | 27.6 |
| 20 | 20.0% | 28.1 |

### CryptBB synthetic data is being redone
- The v2 controller run was stopped by the user at 28/100 and preserved (`handoff/cryptbb_v2_20261114`, STOP and stop_receipt.json).
- The user then chose **v3 split controllers** (one job per controller). Codex has a handoff to generate new CryptBB A/B data with them: `handoff/CODEX_CONTROLLER_V3_SPLIT_HANDOFF.md`, code `Capri/src/capri_controller/controller_v3_split.py`.
- Training on v3 data needs the user's confirmation. Model 4's starting checkpoint may therefore be a future v3 Model1 that does not exist yet. Ask; do not assume.

### Other results
- **External benchmark:** GLM-OCR scores 188/551 = 34.1% exact, CER 28.8% (`Capri/runs/glm_ocr_cryptbb_20261113/summary.json`). It is not an approved teacher.
- **DarkBay agreement-selected results:**

| | A | B |
|---|---|---|
| Model1 | 69.0% | 63.6% |
| Model2 | 73.2% | 71.8% |
| historical Model3 | 71.8% | 70.7% |

## 3. Real CryptBB image pools (from `handoff/cryptbb_model1_preparation/source_audit.json`)

| pool | count | labels? | current use |
|---|---|---|---|
| original unsolved | 600 | none | `E:/pranav/CryptBBCaptchas(unsolved)07272023-.../`. 100 of these are the **generator references** (their pixels shaped the synthetic data). The other 500 are not used by the current protocol. |
| reference | 100 | none | subset of original; controller fitting targets |
| historical selftrain | 100 | none | excluded from the current protocol; 0 overlap with the others; earlier self-training use, so audit before reuse |
| test (scoring) | 551 | yes (`evaluation_data/cryptbb/scoring_labels`) | unlabeled A/B agreement selection AND final scoring (transductive) |

- All pairwise pixel overlaps between the pools are 0, except reference ⊂ original.
- **`JoeyRetitling` is the CryptBB eval answer key** (filenames = labels). Never use it as unlabeled input. `auto_synth/data/joey_retitling*` artifacts are derived from it.
- **DarkBay has no clean unlabeled pool**: its 900 images are both selection and scoring. Pseudo-labeling on DarkBay would be transductive on the test set.

## 4. Questions to settle with the user (open; none decided)

1. **Goal.** Is the claim "pseudo-labels on unlabeled real images improve the reader on unseen real images" (needs a held-out test), or transductive adaptation on the scoring pool (weaker claim, must be stated)?
2. **Unlabeled pool.**
   - The natural candidate is the 500 unused original + 100 selftrain = 600 images disjoint from the 551.
   - Should the 100 reference images be included? They are unlabeled but already shaped the generator.
   - Is 600 images enough? It is small compared with the 100k synthetic images.
3. **Teacher.** Options:
   - Model1-A at a prospectively fixed epoch (e.g. 20, by rule, not by the diagnostics above);
   - a v3 Model1 if trained;
   - A/B consensus (needs both readers);
   - several epochs of one reader (self-ensemble);
   - an external OCR like GLM-OCR. That is an explicit separate decision because it brings in outside knowledge. VLMs remain excluded from the Capri generation workflow; a teacher is a different question.
4. **Acceptance.** What counts as a usable pseudo-label:
   - confidence or likelihood threshold;
   - A/B or cross-epoch agreement;
   - test-time augmentation consistency;
   - length or charset sanity;
   - cap per round.

   Thresholds must be fixed without real test labels, e.g. calibrated on held-out synthetic data. Agreement can reinforce shared errors.
5. **Training form.**
   - Fine-tune the teacher or start a fresh student;
   - mix of synthetic and pseudo-labeled real, and loss weight;
   - one round or iterative refresh (re-label each round);
   - fixed compute budget so a matched control (same steps, synthetic only) is possible.
6. **Checkpoint selection and reporting.**
   - The frozen rule is max same-epoch A/B exact agreement on the 551, earliest tie. Does Model 4 keep A/B pairs so that rule still applies?
   - If the 551 are not used for training, the rule stays valid.
   - Pseudo-label acceptance must not use the 551.
7. **Reproducibility.** Store per image: pixel hash, teacher identity/checkpoint hash, raw prediction and confidence, accept/reject reason, round. Synthetic labels stay unchanged.

## 5. Hard constraints

- No human real strings in training, generation or pseudo-label filtering. Labels are read only for final scoring after predictions are frozen.
- Never silently train on the 551 scoring images or JoeyRetitling. If the user chooses transductive use, write it into the protocol and the report.
- Do not interrupt, restart or overwrite existing runs:
  - Model1-A CryptBB;
  - the stopped v2 run;
  - the Codex v3 generation, if running (check `handoff/CURRENT_STATE.md` and receipts first).
- Do not edit `Capri/capri.py` (frozen) or running sources. New code goes in new files invoked with `-m`.
- Log changes in `handoff/CHANGELOG.md`.

## 6. Suggested opening

1. Confirm the goal: held-out vs transductive.
2. Agree the unlabeled pool from section 3.
3. Then choose the teacher and the acceptance rule.

Keep the forecast cautions from the older handoff: no promised accuracy numbers.

## 7. Update 2026-11-14 08:05 UTC: teacher step 1 authorized and queued

- Discussed with user: a team of pretrained STRs, each given a short (~10 min) fine-tune on synthetic data, is the
  intended teacher family. The user has further plans beyond pseudo-labeling; do not assume what they are.
- Authorized and queued: fine-tune 10 STRs on DarkBay synthetic A+B (200k). Nothing after that is decided
  (teacher choice, acceptance rule, student, CryptBB). See `model4/README.md` and `handoff/CHANGELOG.md`.
- Outside pretrained knowledge (MJ/ST, Union14M, CLIP, documents, Chinese text) now enters through these
  teachers. This must be stated in the thesis.
- The Model4 reservation `evaluation_data/model4_reservation_20261114` (300 pseudo-label / 600 eval for DarkBay)
  exists. The teacher step reads none of it.
