# Model4 — step 1: light fine-tune of 10 pretrained STRs on DarkBay synthetic A+B

**Status 2026-11-14 08:20 UTC: queue stopped before start (STOP marker). The user will supervise the fine-tunes; launch only when they say so. Purpose (user): fixed STR judges that weigh in on the top-5 A/B-agreement epoch candidates, plus later uses.**

Authorized by the user 2026-11-14: "have these models fine tuned to the synthetic data for darkbay that we
already have". Only this step is authorized. No pseudo-labeling, no real images, no student training.

## What runs
`model4_teachers.gpu_queue` waits until the CryptBB v3 data workflow is no longer `running` and GPU memory in
use is < 3000 MiB. It then runs each model in its own process: `probe` (40 steps, freezes step budget) then
`train`. Order: parseq, abinet, trba, vitstr, crnn, clip4str_b, mgp_str_base, trocr_small_printed, svtrv2_ch,
doctr_master.

## Uniform recipe (fixed before any GPU run)
- Data: DarkBay synthetic A+B LMDBs (200k, 62-char, length 6), one shared seeded order (20261114), no augmentation.
- Budget: steps = floor50(min(1562, max(200, 540 s / mean probe step time))). 1562 steps is one pass. Batch 128
  (micro-batch + accumulation where memory requires; halves automatically on OOM).
- AdamW, wd 0.01 (none on bias/1-d), peak LR = 0.1 x the model's native from-scratch LR. Fallback 5e-5 for
  MGP-STR (native Adadelta) and TrOCR (published LR already a fine-tune LR). 5% warmup, cosine to 0. Clip 20, bf16.
- Final step kept. SynthTIGER controller4 validation (1000, synthetic) reported zero-shot and after; it is
  not used for selection.

## Where things go
- Fine-tuned weights + receipt: `Capri/fine-tuned-STRs/<model>/{model.pt,receipt.json}` (user-designated folder).
- Logs, probe, validation predictions: `model4/runs/darkbay_AB_teachers_20261114/<model>/`.
- Queue status: `model4/runs/darkbay_AB_teachers_20261114/queue_status.json`; create `STOP` there to halt
  before the next model. Lock: `queue.lock.json` (PID + create time).
- Pretrained weights (sha256 in `weights/sha256.txt`): `model4/weights/`. Pinned sources: `model4/third_party/`
  (parseq@1902db0, CLIP4STR@9730b64, doctr v0.8.1@62d94ff). Small deps installed with `--target` into
  `model4/site` (shared venv untouched); `model4/shims` holds nltk.edit_distance and a weasyprint stub.

## Adapter notes (verified on CPU smoke)
- ABINet: forced full-model fine-tune; upstream would re-initialise the pretrained alignment module at its
  first joint step. Disabled.
- MGP-STR: char head extended 38->64 (uppercase rows copied from lowercase); BPE/WordPiece heads unused.
- TrOCR: explicit seq2seq loss with the checkpoint's own convention (start </s>=2, no leading <s>). Teacher-forced
  loss on its own generation for a clean word = 0.014. transformers 4.57's default would shift labels twice.
- SVTRv2 (openocr_svtrv2_ch.pth): GTC guidance branch dropped, encoder + CTC head loaded strictly; 48x192 input;
  greedy CTC restricted to the 62-char charset.
- CLIP4STR: two import-time compatibility patches (Lightning 2 EPOCH_OUTPUT alias; hard-coded author CLIP path
  assert); real CLIP weights path passed explicitly. Encoder LR x1, decoder x19 (native coef_lr).
- docTR: v0.8.1 because the venv is Python 3.9; `doctr/version.py` written (normally generated at install).
- Windows: `sentencepiece` must import before torch, or `Load` segfaults; done at the top of run.py.

Launch: `powershell -ExecutionPolicy Bypass -File model4/launch_teachers.ps1`
