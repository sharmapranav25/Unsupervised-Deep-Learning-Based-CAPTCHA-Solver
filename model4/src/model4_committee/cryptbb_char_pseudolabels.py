"""CryptBB Model4 character-level pseudo-labels from the 5-model epoch-20 committee. NO labels are read.

Input: frozen all_predictions.json of cryptbb_pseudolabels_unanimous_epoch20_20261115 (400 images = reference 100 +
pseudo pool 300; members M1A, M1B, M2A, M2B, M3A at epoch 20). Rule (user-approved 2026-11-15), per position:
  * all 5 identical (case-sensitive) and the letter is NOT case-ambiguous  -> mode 'exact'
  * all 5 identical ignoring case, OR identical but the letter is case-ambiguous -> mode 'case_insensitive'
  * otherwise -> mode 'ignore' (no loss; context token = most common prediction)
Case-ambiguous letters fixed a priori (same glyph shape in both cases): C K O P S U V W X Z.
"""
import hashlib, json, os, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def _audit(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        p = str(args[0]).lower().replace('\\', '/')
        if 'scoring_only' in p or 'scoring_labels' in p or 'accuracy_check' in p or 'per_image_results' in p:
            raise RuntimeError('label/scoring file access forbidden: ' + p)


sys.addaudithook(_audit)
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'model4/runs/cryptbb_pseudolabels_unanimous_epoch20_20261115'
OUT = ROOT / 'model4/runs/cryptbb_char_pseudolabels_epoch20_20261115'
MEMBERS = ['M1A', 'M1B', 'M2A', 'M2B', 'M3A']
CASE_AMBIGUOUS = set('CKOPSUVWXZ')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, x):
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(x, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, p)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'char_pseudo_labels.json').exists():
        raise SystemExit('already written; refusing to overwrite')
    rc = json.loads((SRC / 'receipt.json').read_text(encoding='utf-8'))
    assert rc['labels_read'] is False
    R = json.loads((SRC / 'all_predictions.json').read_text(encoding='utf-8'))
    assert len(R) == 400
    out, modes, by_pool = [], Counter(), Counter()
    for r in R:
        texts = [r[n]['text'] for n in MEMBERS]
        assert all(len(t) == 5 for t in texts)
        target, mode = [], []
        for i in range(5):
            chars = [t[i] for t in texts]
            top = Counter(chars).most_common(1)[0][0]
            if len(set(chars)) == 1 and top.upper() not in CASE_AMBIGUOUS:
                m = 'exact'
            elif len({c.lower() for c in chars}) == 1:
                m = 'case_insensitive'
            else:
                m = 'ignore'
            target.append(top)
            mode.append(m)
            modes[m] += 1
            if m != 'ignore':
                by_pool[r['pool']] += 1
        n_lab = sum(m != 'ignore' for m in mode)
        out.append({'opaque_id': r['opaque_id'], 'pool': r['pool'], 'image_path': r['image_path'],
                    'pixel_sha256': r['pixel_sha256'], 'context_string': ''.join(target), 'modes': mode,
                    'labelled_positions': n_lab, 'member_predictions': texts})
    used = [o for o in out if o['labelled_positions'] > 0]
    write(OUT / 'char_pseudo_labels.json', used)
    write(OUT / 'receipt.json', {
        'created_at': datetime.now(timezone.utc).isoformat(), 'source_predictions': str((SRC / 'all_predictions.json').relative_to(ROOT)),
        'source_predictions_sha256': sha(SRC / 'all_predictions.json'), 'members_epoch20': rc['members'],
        'rule': {'exact': 'all 5 identical, letter not case-ambiguous',
                 'case_insensitive': 'all 5 identical ignoring case, or identical but letter is case-ambiguous',
                 'ignore': 'members disagree beyond case; no loss, context token = majority'},
        'case_ambiguous_letters': ''.join(sorted(CASE_AMBIGUOUS)),
        'case_ambiguous_source': 'a priori typographic list (same glyph shape in both cases), not derived from labels',
        'images_in': 400, 'images_with_labels': len(used), 'positions': dict(modes), 'labelled_chars_by_pool': dict(by_pool),
        'per_image_labelled_positions': dict(Counter(o['labelled_positions'] for o in out)),
        'labels_read': False, 'char_pseudo_labels_sha256': sha(OUT / 'char_pseudo_labels.json'),
        'script_sha256': sha(Path(__file__)),
        'disclosure': 'Case-folding decision was informed by an earlier label diagnostic on the 58 unanimous images.'})
    print(json.dumps(json.loads((OUT / 'receipt.json').read_text())['positions']),
          len(used), dict(Counter(o['labelled_positions'] for o in out)), dict(by_pool))


if __name__ == '__main__':
    main()
