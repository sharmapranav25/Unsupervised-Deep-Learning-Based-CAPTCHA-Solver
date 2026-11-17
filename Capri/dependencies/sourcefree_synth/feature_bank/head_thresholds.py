"""Per-head visibility threshold tau (pixels) from synthetic statistics.

For every approved intensity recipe, render levels 1 and 2 ("first discernible trace", "sparse
trace") over N seeds and measure the head's visible label pixels with feature_bank.heads.
tau_h = max(8, round(0.5 * min over recipes of the 10th percentile at level 1)), so an effect at
the reviewed level-1 strength is almost always "present", and anything fainter is masked (null),
not called absent. Heads without an approved ladder keep the documented floor in heads.py.
Writes docs/feature_bank/HEAD_THRESHOLDS_01.json.

Usage: python -m feature_bank.head_thresholds --out runs/head_thresholds_NEW [--seeds 10]
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from feature_bank.bank02_smoke import ROOT, run_case


def approved():
    out = {}
    for p in sorted((ROOT / "docs/feature_bank").glob("INTENSITY_CONTROL_MAP_*.json")):
        for name, e in json.loads(p.read_text(encoding="utf-8")).items():
            if e.get("source", "php") == "php":
                out[name] = e
    return out


def job(args):
    name, entry, level, seed, out = args
    from feature_bank.heads import label_image
    d = Path(out) / name
    d.mkdir(parents=True, exist_ok=True)
    stem = f"L{level}_{seed}"
    r = run_case(entry["library"], seed, d, intercept=True, params=entry["params_by_level"][str(level)],
                 stem_name=stem, options={"font_split": "train"})
    if not r["ok"]:
        return name, level, None
    lab = label_image(d / stem)
    return name, level, lab["heads"][entry["feature"]]["px"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    recs = approved()
    jobs = [(n, e, lv, 31000 + s, str(out)) for n, e in recs.items() for lv in (1, 2) for s in range(a.seeds)]
    stats = {}
    with ProcessPoolExecutor(28) as ex:
        for name, level, px in ex.map(job, jobs):
            if px is not None:
                stats.setdefault(recs[name]["feature"], {}).setdefault(name, {}).setdefault(level, []).append(px)
    tau, detail = {}, {}
    for head, by_rec in stats.items():
        l1 = [float(np.percentile(v[1], 10)) for v in by_rec.values() if v.get(1)]
        if not l1 or any(px is None for v in by_rec.values() for px in v.get(1, [])):
            continue
        detail[head] = {r: {str(k): dict(n=len(x), min=int(min(x)), p10=float(np.percentile(x, 10)), median=float(np.median(x)))
                            for k, x in v.items()} for r, v in by_rec.items()}
        if max(l1) <= 0:
            continue  # px is not this head's measure (geometry/alpha heads)
        tau[head] = int(max(8, round(0.5 * min(x for x in l1 if x > 0))))
    res = dict(method=__doc__.strip().splitlines()[0], rule="tau = max(8, 0.5 * min_recipe p10(level-1 visible px))",
               run=str(out.relative_to(ROOT)).replace("\\", "/"), tau_px=tau, level_stats=detail)
    (ROOT / "docs/feature_bank/HEAD_THRESHOLDS_01.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out / "thresholds.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(tau))


if __name__ == "__main__":
    main()
