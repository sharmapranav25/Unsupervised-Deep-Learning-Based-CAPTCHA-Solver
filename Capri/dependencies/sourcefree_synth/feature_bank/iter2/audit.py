"""Audit an iteration-2 cache split: prevalence + level spread per head, text / font label spread, duplicates,
legibility-rejection bias per effect (from fresh draws), and a review sheet.

  python -m feature_bank.iter2.audit --cache runs/<name>/iter2_h96/<split> [--reject-draws 3000] [--sheet 48]
"""
import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from feature_bank.iter2 import generate as G


def load(cache):
    cache = Path(cache)
    meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
    idx = dict(np.load(cache / "index.npz"))
    cols = np.fromfile(cache / "images.u8", np.uint8).reshape(-1, G.H, 3)
    return meta, idx, cols


def image(idx, cols, i):
    o, w = int(idx["offsets"][i]), int(idx["widths"][i])
    return cols[o:o + w].transpose(1, 0, 2)


def draw_attempt(args):
    seed, p_on = args
    from captcha_effects.captcha import render_captcha
    rng = np.random.default_rng([7, seed])
    out = render_captcha(rng, G._bank(), G.canvas_size(rng), p_on=p_on)
    if out is None:
        return None
    req = [e["name"] for e in out["effects_spec"]["effects"] if e.get("level", 0) > 0]
    return req, bool(out["text"]["usable"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--reject-draws", type=int, default=3000)
    ap.add_argument("--sheet", type=int, default=48)
    ap.add_argument("--workers", type=int, default=28)
    a = ap.parse_args()
    meta, idx, cols = load(a.cache)
    n = meta["n"]
    rep = dict(n=n, usable_rate=meta["usable_rate"], seconds=meta["seconds"])
    rep["duplicates"] = n - len(set(meta["sha256"]))
    ep, el = idx["eff_present"], idx["eff_level"]
    rep["effects"] = {h: dict(prev=round(float(ep[:, i].mean()), 3),
                              levels=np.bincount(el[ep[:, i] == 1, i], minlength=11)[1:].tolist())
                      for i, h in enumerate(meta["effect_heads"])}
    rep["effects_per_image"] = dict(mean=round(float(ep.sum(1).mean()), 2),
                                    hist=np.bincount(ep.sum(1)).tolist())
    tp, tl = idx["txt_present"], idx["txt_level"]
    rep["text"] = {h: dict(prev=round(float(tp[:, i].mean()), 3),
                           levels=None if (tl[:, i] < 0).all() else np.bincount(tl[tp[:, i] == 1, i].clip(0), minlength=11)[1:].tolist())
                   for i, h in enumerate(meta["text_heads"])}
    rep["length"] = dict(Counter(idx["length"].tolist()))
    f = idx["font"]
    rep["font"] = {c: dict(Counter(f[:, j].tolist())) for j, c in enumerate(meta["font_cols"])}
    rep["font_share"] = {c: {int(k): round(v / n, 3) for k, v in sorted(Counter(f[:, j].tolist()).items())}
                         for j, c in enumerate(meta["font_cols"])}
    # target vs measured (dataset 1 scheme): how often the measured font level lands within +-1 of its target
    import gzip as _gz
    tw = []
    with _gz.open(Path(a.cache) / "labels.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            lab = json.loads(line)
            t = lab.get("text_spec", {}).get("font_targets")
            if t:
                st = lab["text"]["style"]
                tw.append((t["weight"], st.get("median_weight"), t["width"], st.get("median_width"), t["category"],
                           st.get("dominant_category")))
    if tw:
        rep["font_targets"] = dict(weight_pm1=round(float(np.mean([abs(x[0] - x[1]) <= 1 for x in tw])), 3),
                                   width_pm1=round(float(np.mean([abs(x[2] - x[3]) <= 1 for x in tw])), 3),
                                   category=round(float(np.mean([x[4] == x[5] for x in tw])), 3))
    rep["widths"] = dict(min=int(idx["widths"].min()), median=int(np.median(idx["widths"])), max=int(idx["widths"].max()))
    # rejection bias: share of draws with effect X requested that were usable, vs overall
    if a.reject_draws:
        from captcha_effects import render as R
        p_on = meta.get("p_on_effects") or {k: meta["p_on"] for k in R.SAMPLED}
        with ProcessPoolExecutor(a.workers) as ex:
            res = [r for r in ex.map(draw_attempt, [(s, p_on) for s in range(a.reject_draws)], chunksize=32) if r]
        base = np.mean([u for _, u in res])
        per = {}
        for name in R.SAMPLED:
            us = [u for req, u in res if name in req]
            if us:
                per[name] = dict(n=len(us), usable=round(float(np.mean(us)), 3), vs_base=round(float(np.mean(us) - base), 3))
        rep["rejection"] = dict(draws=len(res), base_usable=round(float(base), 3),
                                worst=sorted(per.items(), key=lambda kv: kv[1]["usable"])[:10], per_effect=per)
    out = Path(a.cache) / "audit.json"
    out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    # review sheet: image + its present effects (level) + text
    labs = []
    import gzip
    with gzip.open(Path(a.cache) / "labels.jsonl.gz", "rt", encoding="utf-8") as fh:
        for k, line in enumerate(fh):
            if k >= a.sheet:
                break
            labs.append(json.loads(line))
    tiles = []
    for i, lab in enumerate(labs):
        im = Image.fromarray(image(idx, cols, i)).resize((int(idx["widths"][i] * 0.75), 72))
        tile = Image.new("RGB", (480, 130), "white")
        tile.paste(im, (2, 2))
        eff = ", ".join(f"{h}:{e['level']}" for h, e in lab["effects"].items() if e["present"])
        st = lab["text"]["style"]
        txt = [f"'{lab['text']['text']}'  {st.get('dominant_category')} w{st.get('median_weight')} "
               f"wd{st.get('median_width')} {st.get('layout')}", eff[:85], eff[85:170]]
        d = ImageDraw.Draw(tile)
        for k, s in enumerate(txt):
            d.text((2, 78 + 16 * k), s, fill="black")
        tiles.append(tile)
    cols_n = 3
    sheet = Image.new("RGB", (480 * cols_n, 130 * ((len(tiles) + cols_n - 1) // cols_n)), "white")
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % cols_n) * 480, (k // cols_n) * 130))
    sheet.save(Path(a.cache) / "review.png")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("effects", "text")}, indent=0)[:3000])
    print("effects:", {h: v["prev"] for h, v in rep["effects"].items()})
    print("text:", {h: v["prev"] for h, v in rep["text"].items()})


if __name__ == "__main__":
    sys.exit(main())
