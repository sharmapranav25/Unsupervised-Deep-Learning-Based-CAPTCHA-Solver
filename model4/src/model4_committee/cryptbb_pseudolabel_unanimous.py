"""CryptBB Model4 pseudo-labels: 5-model epoch-20 committee, unanimous exact agreement. NO labels are read.

Pool: reference_pool (100) + pseudo_label_pool (300) = 400 unlabeled images (user instruction 2026-11-15).
Members (epoch 20): Model1-A, Model1-B, Model2-A, Model2-B (epoch20-teacher static), Model3-A (feature priority).
Rule: keep an image iff all 5 predicted strings are identical (case-sensitive); that string is its pseudo-label.
An audit hook blocks opening any scoring/label file for the whole process.
"""
import copy, gc, hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path


def _audit(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        p = str(args[0]).lower().replace('\\', '/')
        if 'scoring_only' in p or 'scoring_labels' in p or 'per_image_results' in p or p.endswith('/summary.json'):
            raise RuntimeError('label/scoring file access forbidden: ' + p)


sys.addaudithook(_audit)

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'model1/src/parseq_packages'), str(ROOT / 'model1/src/OpenOCR')]
RES = ROOT / 'evaluation_data/model4_reservation_20261114/cryptbb'
OUT = ROOT / 'model4/runs/cryptbb_pseudolabels_unanimous_epoch20_20261115'
MEMBERS = {
    'M1A': 'model1/runs/cryptbb_v3/A/epoch_20.pth',
    'M1B': 'model1/runs/cryptbb_v3/B/epoch_20.pth',
    'M2A': 'model2/runs/cryptbb_v3_epoch20_static_A_20261115/A/checkpoints/epoch_20.pth',
    'M2B': 'model2/runs/cryptbb_v3_epoch20_static_B_20261115/B/checkpoints/epoch_20.pth',
    'M3A': 'model3/runs/cryptbb_v3_feature_priority_v1_20261115/A/checkpoints/epoch_20.pth',
}


def now():
    return datetime.now(timezone.utc).isoformat()


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


def main():
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms as T
    from openrec.modeling import build_model
    from openrec.postprocess import build_post_process
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'pseudo_labels.json').exists():
        raise SystemExit('pseudo labels already written; refusing to overwrite')
    rows = []
    for role in ('reference_pool', 'pseudo_label_pool'):
        rr = json.loads((RES / f'{role}.json').read_text(encoding='utf-8'))
        rows += [{**r, 'pool': role} for r in rr]
    assert len(rows) == 400 and len({r['pixel_sha256'] for r in rows}) == 400
    pix = []
    for r in rows:
        im = Image.open(ROOT / r['image_path']).convert('RGB')
        assert hashlib.sha256(np.asarray(im).tobytes()).hexdigest() == r['pixel_sha256']
        pix.append(im)
    tr = T.Compose([T.Resize((32, 128), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(.5, .5)])
    x = torch.stack([tr(im) for im in pix])
    preds, info = {}, {}
    for name, rel in MEMBERS.items():
        p = ROOT / rel
        digest = sha(p)
        ck = torch.load(p, map_location='cpu', weights_only=False)
        assert ck['epoch'] == 20 and ck['global_step'] == 15640
        cfg = copy.deepcopy(ck['config'])
        dec = build_post_process(cfg['PostProcess'], cfg['Global'])
        cfg['Architecture']['Decoder']['out_channels'] = dec.get_character_num()
        model = build_model(cfg['Architecture'])
        model.load_state_dict(ck['state_dict'], strict=True)
        del ck
        gc.collect()
        model.cuda().eval()
        out = []
        with torch.inference_mode():
            for i in range(0, len(rows), 16):
                out.extend(dec(model(x[i:i + 16].cuda())))
        del model
        torch.cuda.empty_cache()
        assert len(out) == 400 and sha(p) == digest
        preds[name] = [(t, float(c)) for t, c in out]
        info[name] = {'checkpoint': rel, 'sha256': digest}
        print(name, 'done', flush=True)
    names = list(MEMBERS)
    per, keep = [], []
    for i, r in enumerate(rows):
        texts = [preds[n][i][0] for n in names]
        row = {'opaque_id': r['opaque_id'], 'pool': r['pool'], 'image_path': r['image_path'],
               'pixel_sha256': r['pixel_sha256'], **{n: {'text': preds[n][i][0], 'conf': preds[n][i][1]} for n in names},
               'n_distinct': len(set(texts))}
        per.append(row)
        if len(set(texts)) == 1:
            keep.append({'opaque_id': r['opaque_id'], 'pool': r['pool'], 'image_path': r['image_path'],
                         'pixel_sha256': r['pixel_sha256'], 'pseudo_label': texts[0],
                         'min_conf': min(preds[n][i][1] for n in names)})
    write(OUT / 'all_predictions.json', per)
    write(OUT / 'pseudo_labels.json', keep)
    by_pool = {p: sum(k['pool'] == p for k in keep) for p in ('reference_pool', 'pseudo_label_pool')}
    write(OUT / 'receipt.json', {
        'created_at': now(), 'rule': 'all 5 epoch-20 members predict the identical case-sensitive string',
        'members': info, 'pool': {'reference_pool': 100, 'pseudo_label_pool': 300},
        'kept': len(keep), 'kept_by_pool': by_pool, 'labels_read': False, 'audit_hook': 'scoring/label opens blocked',
        'device': 'cuda FP32 batch16', 'preprocessing': 'RGB bicubic 32x128 Normalize0.5',
        'pseudo_labels_sha256': sha(OUT / 'pseudo_labels.json'), 'script_sha256': sha(Path(__file__))})
    print(json.dumps({'kept': len(keep), 'of': 400, 'by_pool': by_pool}))


if __name__ == '__main__':
    main()
