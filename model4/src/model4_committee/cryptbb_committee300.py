"""User-requested diagnostic: 5-model epoch-20 committee on the CryptBB 300 Model4 pseudo-label pool.

Members (all epoch 20): Model1-A, Model1-B, Model2-A, Model2-B (epoch20-teacher static), Model3-A (feature priority).
GPU FP32 batch 16, one model resident at a time. All predictions are frozen (sha256) before labels are opened.
Labels are used only to report accuracy; no training, selection or pseudo-label export happens here.
"""
import copy, gc, hashlib, inspect, json, os, sys, textwrap, time, types
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'model1/src/parseq_packages'), str(ROOT / 'model1/src/OpenOCR')]
OUT = ROOT / 'model4/runs/cryptbb_committee300_epoch20_20261115'
POOL = ROOT / 'evaluation_data/model4_reservation_20261114/cryptbb/pseudo_label_pool.json'
LABELS = ROOT / 'evaluation_data/model4_reservation_20261114/scoring_only/cryptbb/pseudo_label_pool_labels.json'
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


def predict_all(rows, pixels):
    import torch
    from torchvision import transforms as T
    from openrec.modeling import build_model
    from openrec.postprocess import build_post_process
    tr = T.Compose([T.Resize((32, 128), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(.5, .5)])
    x_all = torch.stack([tr(im) for im in pixels])
    preds, info = {}, {}
    for name, rel in MEMBERS.items():
        p = ROOT / rel
        digest = sha(p)
        ck = torch.load(p, map_location='cpu', weights_only=False)
        assert ck['epoch'] == 20 and ck['global_step'] == 15640 and ck['config']['Global']['seed'] == 48
        cfg = copy.deepcopy(ck['config'])
        dec = build_post_process(cfg['PostProcess'], cfg['Global'])
        cfg['Architecture']['Decoder']['out_channels'] = dec.get_character_num()
        model = build_model(cfg['Architecture'])
        model.load_state_dict(ck['state_dict'], strict=True)
        del ck
        gc.collect()
        model.cuda().eval()
        out, t0 = [], time.time()
        with torch.inference_mode():
            for i in range(0, len(rows), 16):
                out.extend(dec(model(x_all[i:i + 16].cuda())))
        assert len(out) == len(rows) and sha(p) == digest
        preds[name] = [{'text': t, 'conf': float(c)} for t, c in out]
        info[name] = {'checkpoint': rel, 'sha256': digest, 'seconds': round(time.time() - t0, 2),
                      'peak_mem_gib': round(torch.cuda.max_memory_allocated() / 2**30, 3)}
        print(name, info[name], flush=True)
        del model
        torch.cuda.empty_cache()
    return preds, info


def main():
    import numpy as np
    from PIL import Image
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'prediction_freeze.json').exists():
        raise SystemExit('predictions already frozen; refusing to overwrite')
    rows = json.loads(POOL.read_text(encoding='utf-8'))
    assert len(rows) == 300 and len({r['opaque_id'] for r in rows}) == 300
    pixels = []
    for r in rows:
        im = Image.open(ROOT / r['image_path']).convert('RGB')
        assert hashlib.sha256(np.asarray(im).tobytes()).hexdigest() == r['pixel_sha256']
        pixels.append(im)
    write(OUT / 'protocol.json', {
        'created_at': now(), 'purpose': 'user-requested committee accuracy diagnostic on the 300 pseudo-label pool',
        'members': MEMBERS, 'epoch_rule': 'epoch 20 for every member (user instruction)', 'device': 'cuda FP32',
        'batch_size': 16, 'preprocessing': 'RGB PIL bicubic 32x128 ToTensor Normalize0.5',
        'decoding': 'saved config ARLabelDecode, no normalization', 'pool_sha256': sha(POOL),
        'labels': 'opened only after prediction_freeze.json is written', 'script_sha256': sha(Path(__file__)),
        'no_training_no_export': True})
    preds, info = predict_all(rows, pixels)
    per_image = [{'opaque_id': r['opaque_id'], **{n: preds[n][i] for n in MEMBERS}} for i, r in enumerate(rows)]
    write(OUT / 'predictions.json', per_image)
    write(OUT / 'prediction_freeze.json', {'frozen_at': now(), 'sha256': sha(OUT / 'predictions.json'), 'members': info})

    # ---- labels opened only now ----
    labels = json.loads(LABELS.read_text(encoding='utf-8'))
    assert set(labels) == {r['opaque_id'] for r in rows}
    names = list(MEMBERS)
    res = {'n': 300, 'members': {}}
    for n in names:
        res['members'][n] = sum(r[n]['text'] == labels[r['opaque_id']] for r in per_image)
    rules = {}
    for k in (5, 4, 3):
        acc = cor = 0
        for r in per_image:
            text, cnt = Counter(r[n]['text'] for n in names).most_common(1)[0]
            if cnt >= k:
                acc += 1
                cor += text == labels[r['opaque_id']]
        rules[f'>={k}/5 agree'] = {'labelled': acc, 'correct': cor, 'precision': cor / acc if acc else None,
                                   'coverage': acc / 300}
    plural = sum(Counter(r[n]['text'] for n in names).most_common(1)[0][0] == labels[r['opaque_id']] for r in per_image)
    oracle = sum(any(r[n]['text'] == labels[r['opaque_id']] for n in names) for r in per_image)
    res.update(rules=rules, plurality_vote_correct=plural, any_member_correct=oracle,
               distribution_of_agreement=dict(Counter(Counter(r[n]['text'] for n in names).most_common(1)[0][1]
                                                       for r in per_image)))
    write(OUT / 'per_image_results.json', [{**r, 'label': labels[r['opaque_id']]} for r in per_image])
    write(OUT / 'summary.json', {'at': now(), **res, 'diagnostic_only': True})
    assert sha(OUT / 'predictions.json') == json.loads((OUT / 'prediction_freeze.json').read_text())['sha256']
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
