"""Hit-rate analysis from the frozen committee predictions (no new inference)."""
import json, sys
from collections import Counter
from pathlib import Path
from rapidfuzz.distance import Levenshtein
D = Path(__file__).resolve().parents[3] / 'model4/runs/cryptbb_committee300_epoch20_20261115'
R = json.loads((D / 'per_image_results.json').read_text())
N = ['M1A', 'M1B', 'M2A', 'M2B', 'M3A']
TH = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
out = {'n': len(R), 'per_model': {}, 'agreement': {}, 'agreement_case_insensitive_vote': {}}
for n in N:
    m = {'exact': sum(r[n]['text'] == r['label'] for r in R),
         'case_insensitive': sum(r[n]['text'].lower() == r['label'].lower() for r in R),
         'cer': sum(Levenshtein.distance(r[n]['text'], r['label']) for r in R) / sum(len(r['label']) for r in R),
         'by_confidence': []}
    for t in TH:
        s = [r for r in R if r[n]['conf'] >= t]
        c = sum(r[n]['text'] == r['label'] for r in s)
        m['by_confidence'].append({'min_conf': t, 'kept': len(s), 'correct': c, 'precision': c / len(s) if s else None})
    out['per_model'][n] = m
def rule(key, label_fn):
    res = {}
    for k in (2, 3, 4, 5):
        for t in (0.0, 0.9, 0.95):
            kept = cor = 0
            for r in R:
                text, cnt = Counter(key(r[n]['text']) for n in N).most_common(1)[0]
                agreeing = [r[n]['conf'] for n in N if key(r[n]['text']) == text]
                if cnt >= k and min(agreeing) >= t:
                    kept += 1; cor += text == label_fn(r['label'])
            res[f'>={k}/5, min_conf>={t}'] = {'kept': kept, 'correct': cor, 'precision': round(cor / kept, 3) if kept else None}
    return res
out['agreement'] = rule(lambda s: s, lambda s: s)
out['agreement_case_insensitive_vote'] = rule(lambda s: s.lower(), lambda s: s.lower())
(D / 'hit_rate_analysis.json').write_text(json.dumps(out, indent=2) + '\n')
for n, m in out['per_model'].items():
    print(n, 'exact', m['exact'], 'ci', m['case_insensitive'], 'cer %.3f' % m['cer'])
    print('   ', '  '.join(f"{b['min_conf']}:{b['kept']}/{b['correct']}" for b in m['by_confidence']))
for name in ('agreement', 'agreement_case_insensitive_vote'):
    print(name)
    for k, v in out[name].items(): print('   ', k, v)
