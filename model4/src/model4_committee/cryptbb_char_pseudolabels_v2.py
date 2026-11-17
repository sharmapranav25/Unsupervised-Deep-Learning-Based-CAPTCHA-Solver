"""v2 character pseudo-labels: case rule derived ONLY from held-out synthetic data (no real labels anywhere).

See model4/runs/cryptbb_char_pseudolabels_v2_synthcase_20261115/preregistration.json (written and hashed first).
Stage 1: each epoch-20 member reads 10000 images of the synthetic corpus it never trained on -> case-swap rates.
Stage 2: letter pairs with swap rate >= 0.02 (>= 200 occurrences) are case-ambiguous.
Stage 3: rebuild per-position modes from the frozen 400-image committee predictions.
"""
import copy, gc, hashlib, io, json, os, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def _audit(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        p = str(args[0]).lower().replace('\\', '/')
        if any(x in p for x in ('scoring_only', 'scoring_labels', 'accuracy_check', 'per_image_results',
                                'label_quarantine', 'hit_rate')):
            raise RuntimeError('real-label/diagnostic access forbidden: ' + p)


sys.addaudithook(_audit)
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'model1/src/parseq_packages'), str(ROOT / 'model1/src/OpenOCR')]
OUT = ROOT / 'model4/runs/cryptbb_char_pseudolabels_v2_synthcase_20261115'
PREREG_SHA = 'c318b46809b2bcf2fdf66495af2b7844d2c2d1990b53120b7d8477b6eab66686'
SRC = ROOT / 'model4/runs/cryptbb_pseudolabels_unanimous_epoch20_20261115'
MEMBERS = {  # name: (checkpoint, held-out corpus)
    'M1A': ('model1/runs/cryptbb_v3/A/epoch_20.pth', 'B'),
    'M1B': ('model1/runs/cryptbb_v3/B/epoch_20.pth', 'A'),
    'M2A': ('model2/runs/cryptbb_v3_epoch20_static_A_20261115/A/checkpoints/epoch_20.pth', 'B'),
    'M2B': ('model2/runs/cryptbb_v3_epoch20_static_B_20261115/B/checkpoints/epoch_20.pth', 'A'),
    'M3A': ('model3/runs/cryptbb_v3_feature_priority_v1_20261115/A/checkpoints/epoch_20.pth', 'B'),
}
N_SYNTH, SEED, RATE, MIN_OCC = 10000, 20261115, 0.02, 200


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(p, x):
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(x, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, p)


def load_synth(corpus):
    import lmdb, numpy as np
    from PIL import Image
    env = lmdb.open(str(ROOT / f'datasets/cryptbb_v3/{corpus}/train_lmdb'), readonly=True, lock=False, readahead=False)
    with env.begin() as t:
        n = int(t.get(b'num-samples'))
        idx = sorted((np.random.default_rng(SEED).choice(n, N_SYNTH, replace=False) + 1).tolist())
        ims, labs = [], []
        for i in idx:
            ims.append(Image.open(io.BytesIO(t.get(b'image-%09d' % i))).convert('RGB'))
            labs.append(t.get(b'label-%09d' % i).decode())
    return idx, ims, labs


def stage1():
    mpath = OUT / 'synthetic_case_measurement.json'
    if mpath.exists():
        return json.loads(mpath.read_text(encoding='utf-8'))
    import torch
    from torchvision import transforms as T
    from openrec.modeling import build_model
    from openrec.postprocess import build_post_process
    tr = T.Compose([T.Resize((32, 128), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(.5, .5)])
    data = {c: load_synth(c) for c in ('A', 'B')}
    pair = Counter()  # (letter_lower, 'n'|'swap')
    per_member = {}
    for name, (rel, corpus) in MEMBERS.items():
        ck = torch.load(ROOT / rel, map_location='cpu', weights_only=False)
        assert ck['epoch'] == 20 and corpus not in ck['config']['Train']['dataset']['data_dir_list'][0].split('cryptbb_v3')[1][:3]
        cfg = copy.deepcopy(ck['config'])
        dec = build_post_process(cfg['PostProcess'], cfg['Global'])
        cfg['Architecture']['Decoder']['out_channels'] = dec.get_character_num()
        model = build_model(cfg['Architecture'])
        model.load_state_dict(ck['state_dict'], strict=True)
        del ck
        gc.collect()
        model.cuda().eval()
        idx, ims, labs = data[corpus]
        preds = []
        with torch.inference_mode():
            for i in range(0, len(ims), 64):
                preds.extend(t for t, _ in dec(model(torch.stack([tr(im) for im in ims[i:i + 64]]).cuda())))
        del model
        torch.cuda.empty_cache()
        exact = sum(p == g for p, g in zip(preds, labs))
        m = Counter()
        for p, g in zip(preds, labs):
            if len(p) != len(g):
                continue
            for a, b in zip(p, g):
                if b.isalpha() and a.lower() == b.lower():
                    m[(b.lower(), 'n')] += 1
                    m[(b.lower(), 'swap')] += a != b
        pair.update(m)
        per_member[name] = {'held_out_corpus': corpus, 'n': len(labs), 'exact_string': exact / len(labs)}
        print(name, per_member[name], flush=True)
    letters = sorted({k for k, _ in pair})
    rates = {l: {'n': pair[(l, 'n')], 'swaps': pair[(l, 'swap')],
                 'rate': pair[(l, 'swap')] / pair[(l, 'n')] if pair[(l, 'n')] else None} for l in letters}
    res = {'created_at': datetime.now(timezone.utc).isoformat(), 'n_per_corpus': N_SYNTH, 'seed': SEED,
           'members': per_member, 'case_swap_by_letter': rates, 'real_labels_read': False}
    write(mpath, res)
    return res


def main():
    assert sha(OUT / 'preregistration.json') == PREREG_SHA
    meas = stage1()
    amb = sorted(l for l, r in meas['case_swap_by_letter'].items() if r['n'] >= MIN_OCC and r['rate'] >= RATE)
    R = json.loads((SRC / 'all_predictions.json').read_text(encoding='utf-8'))
    assert json.loads((SRC / 'receipt.json').read_text(encoding='utf-8'))['labels_read'] is False and len(R) == 400
    names = list(MEMBERS)
    out, modes, per_img = [], Counter(), Counter()
    for r in R:
        texts = [r[n]['text'] for n in names]
        assert all(len(t) == 5 for t in texts)
        tgt, md = [], []
        for i in range(5):
            ch = [t[i] for t in texts]
            top = Counter(ch).most_common(1)[0][0]
            same, same_ci = len(set(ch)) == 1, len({c.lower() for c in ch}) == 1
            if same and not (top.isalpha() and top.lower() in amb):
                m = 'exact'
            elif same_ci:
                m = 'case_insensitive'
            else:
                m = 'ignore'
            tgt.append(top)
            md.append(m)
            modes[m] += 1
        k = sum(m != 'ignore' for m in md)
        per_img[k] += 1
        if k:
            out.append({'opaque_id': r['opaque_id'], 'pool': r['pool'], 'image_path': r['image_path'],
                        'pixel_sha256': r['pixel_sha256'], 'context_string': ''.join(tgt), 'modes': md,
                        'labelled_positions': k, 'member_predictions': texts})
    dest = OUT / 'char_pseudo_labels.json'
    assert not dest.exists()
    write(dest, out)
    write(OUT / 'receipt.json', {
        'created_at': datetime.now(timezone.utc).isoformat(), 'preregistration_sha256': PREREG_SHA,
        'case_ambiguous_letters_from_synthetic': amb, 'threshold': {'rate': RATE, 'min_occurrences': MIN_OCC},
        'source_predictions_sha256': sha(SRC / 'all_predictions.json'), 'images_with_labels': len(out),
        'positions': dict(modes), 'per_image_labelled_positions': dict(per_img), 'real_labels_read': False,
        'char_pseudo_labels_sha256': sha(dest), 'script_sha256': sha(Path(__file__)),
        'supersedes': 'model4/runs/cryptbb_char_pseudolabels_epoch20_20261115 (v1, label-informed case list)'})
    print('case-ambiguous (synthetic):', amb)
    print(dict(modes), len(out), dict(per_img))


if __name__ == '__main__':
    main()
