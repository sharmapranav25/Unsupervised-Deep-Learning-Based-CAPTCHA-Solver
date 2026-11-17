"""Fixed-budget fine-tune of one pretrained STR on DarkBay synthetic A+B (200k). Synthetic data only.

python -m model4_teachers.run --model parseq --mode smoke|probe|train
  smoke : CPU-or-GPU, 2 tiny steps + 16-image eval, nothing saved to the final folder
  probe : GPU, 40 steps, measures throughput, freezes the step budget into probe.json
  train : GPU, reads the frozen budget, fine-tunes, evaluates (synthetic val) before/after, saves
"""
from __future__ import annotations

try:  # Must load before torch on this Windows setup: sentencepiece.Load segfaults if torch's DLLs load first.
    import sentencepiece  # noqa: F401
except ImportError:
    pass

import argparse
import hashlib
import json
import math
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from . import adapters
from .data import ROOT, LmdbUnion, ValSet, collate, data_identity, fixed_order

RUN_DIR = ROOT / 'model4/runs/darkbay_AB_teachers_20261114'
FINAL_DIR = ROOT / 'Capri/fine-tuned-STRs'
BATCH = 128
TARGET_SECONDS = 540          # training time target (~9 min), leaving room inside a 10-min budget
MAX_STEPS = 200_000 // BATCH  # never more than one pass over A+B
MIN_STEPS = 200
WARMUP_FRAC = 0.05
GRAD_CLIP = 20.0
SEED = 48


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(x, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, p)


def cer(a, b):
    import rapidfuzz.distance.Levenshtein as L
    return L.distance(a, b), max(len(b), 1)


def evaluate(ad, m, dev, limit=None, bs=64):
    ds = ValSet(ad.transform())
    if limit:
        ds = Subset(ds, range(limit))
    dl = DataLoader(ds, batch_size=bs, num_workers=0 if limit else 2, collate_fn=collate)
    m.eval()
    rows, exact, ci, ed, n_ch, oov = [], 0, 0, 0, 0, 0
    allowed = set(adapters.CHARSET)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
        for x, labels in dl:
            preds = ad.predict(m, x.to(dev))
            for p, g in zip(preds, labels):
                exact += p == g
                ci += p.lower() == g.lower()
                d, n = cer(p, g)
                ed += d
                n_ch += n
                oov += any(c not in allowed for c in p)
                rows.append({'label': g, 'pred': p})
    n = len(rows)
    return {'n': n, 'exact': exact / n, 'case_insensitive': ci / n, 'cer': ed / n_ch, 'out_of_charset_rate': oov / n}, rows


def make_opt(ad, m, total):
    opt = torch.optim.AdamW(ad.param_groups(m), betas=(0.9, 0.999))
    warm = max(1, int(WARMUP_FRAC * total))

    def f(s):
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm))))
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, f)


def train_loop(ad, m, dev, steps, micro, log, stop_after=None):
    ds = LmdbUnion(ad.transform())
    order = fixed_order(len(ds))[: steps * BATCH]
    dl = DataLoader(Subset(ds, order.tolist()), batch_size=micro, shuffle=False, drop_last=True,
                    num_workers=0 if dev.type == 'cpu' else 6, persistent_workers=dev.type != 'cpu',
                    collate_fn=collate, pin_memory=dev.type == 'cuda')
    accum = BATCH // micro
    opt, sched = make_opt(ad, m, steps)
    m.train()
    it = iter(dl)
    t0, times, step = time.time(), [], 0
    while step < steps:
        ts = time.time()
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(accum):
            x, labels = next(it)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
                loss = ad.loss(m, x.to(dev, non_blocking=True), labels) / accum
            loss.backward()
            tot += float(loss)
        gn = float(torch.nn.utils.clip_grad_norm_(m.parameters(), GRAD_CLIP))
        opt.step()
        sched.step()
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.time() - ts)
        step += 1
        if not math.isfinite(tot):
            raise RuntimeError(f'non-finite loss at step {step}')
        if step % 25 == 0 or step == steps or dev.type == 'cpu':
            mem = torch.cuda.max_memory_allocated() / 2**30 if dev.type == 'cuda' else 0
            log({'step': step, 'loss': round(tot, 4), 'grad_norm': round(gn, 3), 'lr': sched.get_last_lr()[0],
                 'elapsed_s': round(time.time() - t0, 1), 'peak_mem_gib': round(mem, 2)})
        if stop_after and step >= stop_after:
            break
    return {'steps_done': step, 'train_seconds': time.time() - t0, 'step_times': times}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=adapters.ALL)
    ap.add_argument('--mode', required=True, choices=['smoke', 'probe', 'train'])
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    torch.manual_seed(SEED)
    dev = torch.device(a.device if torch.cuda.is_available() or a.device == 'cpu' else 'cpu')
    ad = adapters.get(a.model)
    out = RUN_DIR / a.model
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / f'{a.mode}.log.jsonl', 'a', encoding='utf-8')

    def log(d):
        d = {'t': now(), **d}
        print(json.dumps(d), flush=True)
        logf.write(json.dumps(d) + '\n')
        logf.flush()

    m = ad.build().to(dev)
    n_params = sum(p.numel() for p in m.parameters())
    log({'event': 'built', 'model': a.model, 'params': n_params, 'device': str(dev), 'lr': ad.lr})

    if a.mode == 'smoke':
        micro = 4
        ad_batch = globals()['BATCH']
        globals()['BATCH'] = micro
        r = train_loop(ad, m, dev, steps=2, micro=micro, log=log)
        globals()['BATCH'] = ad_batch
        metrics, rows = evaluate(ad, m, dev, limit=16, bs=8)
        dump(out / 'smoke.json', {'at': now(), 'ok': True, 'params': n_params, 'metrics_16': metrics,
                                  'sample': rows[:4], 'device': str(dev)})
        log({'event': 'smoke_ok', **metrics})
        return

    assert dev.type == 'cuda', 'probe/train need the GPU'
    if a.mode == 'probe':
        if (out / 'probe.json').exists():
            raise SystemExit('probe.json already frozen; refusing to overwrite')
        micro, init = ad.micro_batch, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        while True:
            try:
                torch.cuda.reset_peak_memory_stats()
                r = train_loop(ad, m, dev, steps=MAX_STEPS, micro=micro, log=log, stop_after=40)
                break
            except torch.cuda.OutOfMemoryError:
                if micro <= 8:
                    raise
                micro //= 2  # keep effective batch 128 via more accumulation
                log({'event': 'oom_retry', 'micro_batch': micro})
                m.load_state_dict(init)
                torch.cuda.empty_cache()
        per_step = sum(r['step_times'][10:]) / len(r['step_times'][10:])
        steps = int(min(MAX_STEPS, max(MIN_STEPS, (TARGET_SECONDS / per_step) // 50 * 50)))
        dump(out / 'probe.json', {'at': now(), 'seconds_per_step': per_step, 'frozen_steps': steps,
                                  'rule': f'floor50(min({MAX_STEPS}, max({MIN_STEPS}, {TARGET_SECONDS}s / mean step time of probe steps 11-40)))',
                                  'micro_batch': micro, 'peak_mem_gib': torch.cuda.max_memory_allocated() / 2**30,
                                  'probe_weights_discarded': True})
        log({'event': 'probe_done', 'seconds_per_step': per_step, 'frozen_steps': steps})
        return

    # train
    probe = json.loads((out / 'probe.json').read_text(encoding='utf-8'))
    final = FINAL_DIR / a.model
    if (final / 'model.pt').exists():
        raise SystemExit(f'{final}/model.pt exists; refusing to overwrite')
    steps = probe['frozen_steps']
    before, rows0 = evaluate(ad, m, dev)
    log({'event': 'zero_shot_val', **before})
    torch.cuda.reset_peak_memory_stats()
    ad.micro_batch = probe['micro_batch']
    r = train_loop(ad, m, dev, steps=steps, micro=ad.micro_batch, log=log)
    after, rows1 = evaluate(ad, m, dev)
    log({'event': 'finetuned_val', **after})
    final.mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), final / 'model.pt')
    with open(out / 'val_predictions.jsonl', 'w', encoding='utf-8') as f:
        for r0, r1 in zip(rows0, rows1):
            f.write(json.dumps({'label': r0['label'], 'zero_shot': r0['pred'], 'finetuned': r1['pred']}) + '\n')
    receipt = {
        'completed_at': now(), 'model': a.model, 'family': ad.family, 'source': ad.source,
        'base_weights': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p)} for p in ad.weight_files],
        'finetuned_weights': {'path': str((final / 'model.pt').relative_to(ROOT)), 'sha256': sha(final / 'model.pt'),
                              'format': 'torch state_dict of the adapter-built module'},
        'params': n_params, 'steps': steps, 'images_seen': steps * BATCH, 'batch': BATCH, 'micro_batch': ad.micro_batch,
        'optimizer': 'AdamW(0.9,0.999), wd 0.01 (no decay on bias/1-d)', 'peak_lr': ad.lr, 'lr_rule': ad.lr_rule,
        'schedule': f'{WARMUP_FRAC:.0%} linear warmup, cosine to 0', 'grad_clip': GRAD_CLIP, 'precision': 'bf16 autocast',
        'seed': SEED, 'train_seconds': r['train_seconds'], 'peak_mem_gib': torch.cuda.max_memory_allocated() / 2**30,
        'data': data_identity(), 'adapter_notes': ad.notes(),
        'synthetic_validation': {'zero_shot': before, 'finetuned': after,
                                 'use': 'reported only; not used to select steps or checkpoints'},
        'selection': 'final step, no checkpoint selection', 'real_images_used': False,
        'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'python': platform.python_version(),
        'probe': probe,
    }
    dump(out / 'receipt.json', receipt)
    dump(final / 'receipt.json', receipt)
    log({'event': 'done', 'saved': str(final / 'model.pt')})


if __name__ == '__main__':
    main()
