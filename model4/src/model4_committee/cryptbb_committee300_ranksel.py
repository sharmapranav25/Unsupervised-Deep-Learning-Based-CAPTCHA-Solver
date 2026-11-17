"""User-requested diagnostic: 5-model committee on the CryptBB 300 pool, each member at its rank-selected epoch.

Rank method = unchanged masked_rank1_v1_exact_auto (analysis/checkpoint_selection/ckpt_select/estimator.py,
sha dee14c99...), applied per run to its own epoch-by-epoch exact-string agreement on the 251 unlabeled eval images.
Existing frozen selections are reused: Model1-A 12, Model1-B 13 (teacher_selection.json), Model2-A 14, Model2-B 14
(epoch20_static selection_{A,B}.json). Model3-A had none, so:

  --stage m3_predict : GPU predictions of all 20 Model3-A checkpoints on the 251 (no labels), frozen by sha
  --stage m3_select  : estimator on those predictions, in a process whose audit hook blocks any label file
  --stage committee  : 300-pool predictions at the selected epochs, frozen, THEN labels opened for the report
No training, no pseudo-label export, no change to any official selection.
"""
import argparse, copy, gc, hashlib, json, os, sys, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'model4/runs/cryptbb_committee300_ranksel_20261115'
RES = ROOT / 'evaluation_data/model4_reservation_20261114/cryptbb'
M3RUN = ROOT / 'model3/runs/cryptbb_v3_feature_priority_v1_20261115/A'
ESTIMATOR = ROOT / 'analysis/checkpoint_selection/ckpt_select/estimator.py'
ESTIMATOR_SHA = 'dee14c99919ffb4d51ba47b2104f766cea584dc113a995ab857a037539421150'
FROZEN = {  # name: (selection file, key path, checkpoint pattern)
    'M1A': ('handoff/cryptbb_v3_model2_rank1_v1_20261115/teacher_selection.json', ('selected', 'A'),
            'model1/runs/cryptbb_v3/A/epoch_{e}.pth'),
    'M1B': ('handoff/cryptbb_v3_model2_rank1_v1_20261115/teacher_selection.json', ('selected', 'B'),
            'model1/runs/cryptbb_v3/B/epoch_{e}.pth'),
    'M2A': ('handoff/cryptbb_v3_epoch20_static_A_20261115/selection_A.json', (),
            'model2/runs/cryptbb_v3_epoch20_static_A_20261115/A/checkpoints/epoch_{e:02d}.pth'),
    'M2B': ('handoff/cryptbb_v3_epoch20_static_B_20261115/selection_B.json', (),
            'model2/runs/cryptbb_v3_epoch20_static_B_20261115/B/checkpoints/epoch_{e:02d}.pth'),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(x, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, p)


def load_pool(role, n):
    import numpy as np
    from PIL import Image
    rows = read(RES / f'{role}.json')
    assert len(rows) == n and len({r['opaque_id'] for r in rows}) == n
    pix = []
    for r in rows:
        im = Image.open(ROOT / r['image_path']).convert('RGB')
        assert hashlib.sha256(np.asarray(im).tobytes()).hexdigest() == r['pixel_sha256']
        pix.append(im)
    return rows, pix


def predictor():
    sys.path[:0] = [str(ROOT / 'model1/src/parseq_packages'), str(ROOT / 'model1/src/OpenOCR')]
    import torch
    from torchvision import transforms as T
    from openrec.modeling import build_model
    from openrec.postprocess import build_post_process
    tr = T.Compose([T.Resize((32, 128), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(.5, .5)])

    def run(ckpt, pixels, epoch):
        digest = sha(ckpt)
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        assert ck['epoch'] == epoch and ck['global_step'] == 782 * epoch and ck['config']['Global']['seed'] == 48
        cfg = copy.deepcopy(ck['config'])
        dec = build_post_process(cfg['PostProcess'], cfg['Global'])
        cfg['Architecture']['Decoder']['out_channels'] = dec.get_character_num()
        model = build_model(cfg['Architecture'])
        model.load_state_dict(ck['state_dict'], strict=True)
        del ck
        gc.collect()
        model.cuda().eval()
        x = torch.stack([tr(im) for im in pixels])
        out = []
        with torch.inference_mode():
            for i in range(0, len(pixels), 16):
                out.extend(dec(model(x[i:i + 16].cuda())))
        del model
        torch.cuda.empty_cache()
        assert len(out) == len(pixels) and sha(ckpt) == digest
        return [(t, float(c)) for t, c in out], digest
    return run


def m3_predict():
    run = predictor()
    rows, pix = load_pool('evaluation_pool', 251)
    assert read(M3RUN / 'completion.json') is not None
    for e in range(1, 21):
        d = OUT / 'model3A_251' / f'epoch_{e:02d}'
        if (d / 'prediction_freeze.json').exists():
            continue
        ck = M3RUN / f'checkpoints/epoch_{e:02d}.pth'
        preds, digest = run(ck, pix, e)
        write(d / 'predictions.json', [{'opaque_id': r['opaque_id'], 'prediction': t} for r, (t, _) in zip(rows, preds)])
        write(d / 'prediction_freeze.json', {'frozen_at': now(), 'sha256': sha(d / 'predictions.json'),
                                             'checkpoint': str(ck.relative_to(ROOT)), 'checkpoint_sha256': digest,
                                             'manifest_sha256': sha(RES / 'evaluation_pool.json'), 'device': 'cuda FP32',
                                             'labels_read': False})
        print('M3A epoch', e, 'frozen', flush=True)


def m3_select():
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            p = str(args[0]).lower().replace('\\', '/')
            if any(x in p for x in ('scoring_only/', 'labels.json', 'summary.json', 'per_image_results.json',
                                    'results.json', 'all_measurements', 'hit_rate')):
                raise RuntimeError('Scoring data forbidden during selection: ' + p)
    sys.addaudithook(audit)
    import numpy as np
    assert sha(ESTIMATOR) == ESTIMATOR_SHA
    sys.path.insert(0, str(ESTIMATOR.parent.parent))
    from ckpt_select.estimator import rank1, choose_k, CONFIG
    dest = OUT / 'selection_M3A.json'
    assert not dest.exists()
    ids = sorted(r['opaque_id'] for r in read(RES / 'evaluation_pool.json'))
    epochs = np.arange(1, 21)
    preds, sources = [], []
    for e in epochs.tolist():
        d = OUT / 'model3A_251' / f'epoch_{e:02d}'
        fr = read(d / 'prediction_freeze.json')
        assert sha(d / 'predictions.json') == fr['sha256']
        assert sha(M3RUN / f'checkpoints/epoch_{e:02d}.pth') == fr['checkpoint_sha256']
        v = {r['opaque_id']: r['prediction'] for r in read(d / 'predictions.json')}
        assert sorted(v) == ids
        preds.append([v[i] for i in ids])
        sources.append({'epoch': e, 'sha256': fr['sha256'], 'checkpoint_sha256': fr['checkpoint_sha256']})
    P = np.array(preds, object)
    A = (P[:, None, :] == P[None, :, :]).mean(2)
    fits = {k: rank1(A, epochs, k) for k in range(6)}
    assert all(f['converged'] for f in fits.values())
    k, criteria, passed = choose_k(A, epochs, fits)
    s = fits[k]['scores']
    order = sorted(range(20), key=lambda i: (-float(s[i]), int(epochs[i])))
    res = {'created_at': now(), 'model': 3, 'dataset': 'A', 'selector': 'masked_rank1_v1_exact_auto',
           'estimator_sha256': ESTIMATOR_SHA, 'estimator_config': CONFIG, 'candidate_epochs': epochs.tolist(),
           'selected_epoch': int(epochs[order[0]]), 'top3': [int(epochs[i]) for i in order[:3]],
           'scores': s.tolist(), 'resolved_k': k, 'fallback_used': not passed, 'mask_diagnostics': criteria,
           'labels_read': False, 'prediction_sources': sources, 'device_note': 'predictions on GPU FP32',
           'scope': 'Model4 committee diagnostic only; not an official Model3 reporting selection'}
    dest.write_text(json.dumps(res, indent=2, allow_nan=False, default=float) + '\n', encoding='utf-8')
    print({k: res[k] for k in ('selected_epoch', 'top3', 'resolved_k', 'fallback_used')})


def committee():
    members = {}
    for n, (sel, keys, pat) in FROZEN.items():
        d = read(ROOT / sel)
        for k in keys:
            d = d[k]
        assert d.get('labels_read', read(ROOT / sel).get('labels_read')) is False
        members[n] = {'epoch': int(d['selected_epoch']), 'selection_file': sel}
    s3 = read(OUT / 'selection_M3A.json')
    members['M3A'] = {'epoch': s3['selected_epoch'], 'selection_file': str((OUT / 'selection_M3A.json').relative_to(ROOT))}
    pats = {n: FROZEN[n][2] for n in FROZEN}
    pats['M3A'] = 'model3/runs/cryptbb_v3_feature_priority_v1_20261115/A/checkpoints/epoch_{e:02d}.pth'
    if (OUT / 'prediction_freeze.json').exists():
        raise SystemExit('committee predictions already frozen')
    write(OUT / 'protocol.json', {'created_at': now(), 'members': members, 'pool_sha256': sha(RES / 'pseudo_label_pool.json'),
                                  'device': 'cuda FP32 batch16', 'script_sha256': sha(Path(__file__)),
                                  'labels': 'opened only after prediction_freeze.json'})
    run = predictor()
    rows, pix = load_pool('pseudo_label_pool', 300)
    preds = {}
    for n, m in members.items():
        ck = ROOT / pats[n].format(e=m['epoch'])
        out, digest = run(ck, pix, m['epoch'])
        preds[n] = out
        m.update(checkpoint=str(ck.relative_to(ROOT)), sha256=digest)
        print(n, m, flush=True)
    names = list(members)
    per = [{'opaque_id': r['opaque_id'], **{n: {'text': preds[n][i][0], 'conf': preds[n][i][1]} for n in names}}
           for i, r in enumerate(rows)]
    write(OUT / 'predictions.json', per)
    write(OUT / 'prediction_freeze.json', {'frozen_at': now(), 'sha256': sha(OUT / 'predictions.json'), 'members': members})

    # ---- labels opened only now ----
    lab = read(RES.parent / 'scoring_only/cryptbb/pseudo_label_pool_labels.json')
    assert set(lab) == {r['opaque_id'] for r in rows}
    from rapidfuzz.distance import Levenshtein
    rep = {'n': 300, 'members': {}}
    for n in names:
        rep['members'][n] = {'epoch': members[n]['epoch'],
                             'exact': sum(r[n]['text'] == lab[r['opaque_id']] for r in per),
                             'case_insensitive': sum(r[n]['text'].lower() == lab[r['opaque_id']].lower() for r in per),
                             'cer': sum(Levenshtein.distance(r[n]['text'], lab[r['opaque_id']]) for r in per) / 1500}
    for tag, key in (('exact', lambda s: s), ('case_insensitive', lambda s: s.lower())):
        tab = {}
        for k in (2, 3, 4, 5):
            for t in (0.0, 0.9, 0.95):
                kept = cor = 0
                for r in per:
                    text, cnt = Counter(key(r[n]['text']) for n in names).most_common(1)[0]
                    if cnt >= k and min(r[n]['conf'] for n in names if key(r[n]['text']) == text) >= t:
                        kept += 1
                        cor += text == key(lab[r['opaque_id']])
                tab[f'>={k}/5, min_conf>={t}'] = {'kept': kept, 'correct': cor,
                                                  'precision': round(cor / kept, 3) if kept else None}
        rep[f'agreement_{tag}'] = tab
    rep['plurality_correct'] = sum(Counter(r[n]['text'] for n in names).most_common(1)[0][0] == lab[r['opaque_id']] for r in per)
    rep['any_member_correct'] = sum(any(r[n]['text'] == lab[r['opaque_id']] for n in names) for r in per)
    write(OUT / 'per_image_results.json', [{**r, 'label': lab[r['opaque_id']]} for r in per])
    write(OUT / 'summary.json', {'at': now(), **rep, 'diagnostic_only': True})
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['m3_predict', 'm3_select', 'committee'])
    {'m3_predict': m3_predict, 'm3_select': m3_select, 'committee': committee}[ap.parse_args().stage]()
