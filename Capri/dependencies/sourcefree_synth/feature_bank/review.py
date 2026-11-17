"""Overlay sheets for reviewing 46-head labels, text targets and clipping decisions.

Each row: original | layer heads coloured | answer-glyph boxes (green = kept, red = clipping reason)
with a caption listing present heads (levels in brackets) and masked (null) heads.

  python -m feature_bank.review stems <out.png> <stem> [<stem> ...]
  python -m feature_bank.review corpus <corpus_dir> [--per-head 8] [--per-source 12]
"""
import argparse
import colorsys
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from feature_bank.heads import LAYER_HEADS, HEADS, label_image, load

PALETTE = {h: tuple(int(255 * v) for v in colorsys.hsv_to_rgb(i / len(LAYER_HEADS), 0.85, 0.95))
           for i, h in enumerate(LAYER_HEADS)}


def row_images(stem, lab):
    meta, log, opaque, trans, cls_map, img = load(stem)
    calls = {c["id"]: c for c in log["calls"]}
    base = Image.fromarray(img.astype(np.uint8))
    grey = np.asarray(base.convert("L").convert("RGB"), np.float32) * 0.45 + 140
    out = grey.copy()
    # re-derive each call's head the same way the labeller did
    from feature_bank.heads import call_head, flat_params
    params = flat_params(meta)
    gl = set(lab.get("glyph_ids") or [])
    for layer, alpha in ((opaque, 0.85), (trans, 0.45)):
        for cid in np.unique(layer):
            if not cid:
                continue
            c = calls.get(int(cid))
            if c is None:
                continue
            if int(cid) in gl:
                col = (0, 160, 0)
            elif c.get("class") == "text":
                col = (90, 90, 90)
            else:
                h = c.get("head") if "head" in c else call_head(c, c.get("library") or meta.get("library", ""),
                                                                params if not c.get("library") or c.get("library") == meta.get("library") else {})
                col = PALETTE.get(h)
            if col is None:
                continue
            m = layer == cid
            out[m] = out[m] * (1 - alpha) + np.array(col, np.float32) * alpha
    ov = Image.fromarray(out.clip(0, 255).astype(np.uint8))
    bx = base.copy()
    d = ImageDraw.Draw(bx)
    bad = {int(r.split()[1]) for r in lab["clip"]["reasons"] if r.startswith("glyph ") and r.split()[1].isdigit()}
    for s in lab["text"]["slots"]:
        if s:
            x0, y0, x1, y1 = s["box_px"]
            d.rectangle([x0, y0, x1, y1], outline=(255, 0, 0) if s["answer_index"] in bad else (0, 200, 0))
    return [base, ov, bx]


def caption(stem, lab, extra=""):
    hs = lab["heads"]
    on = [f"{h}" + (f"[{v['level']}]" if v.get("level") not in (None, 0) else "") for h, v in hs.items() if v["present"] == 1]
    nul = [h for h, v in hs.items() if v["present"] is None]
    clip = "OK" if lab["clip"]["ok"] else "REJECT " + "; ".join(lab["clip"]["reasons"])[:90]
    return [f"{Path(stem).name} {extra} text={lab['text']['text']!r} clip={clip}",
            "ON: " + " ".join(on), "null: " + " ".join(nul)]


def sheet(items, out_path, max_w=1500):
    rows = []
    for stem, lab, extra in items:
        try:
            ims = row_images(stem, lab)
        except Exception as e:  # noqa: BLE001
            print("skip", stem, e)
            continue
        rows.append((ims, caption(stem, lab, extra)))
    if not rows:
        return None
    W = min(max_w, max(sum(i.width for i in r[0]) + 20 for r in rows))
    H = sum(max(i.height for i in r[0]) + 44 for r in rows)
    canvas = Image.new("RGB", (max(W, 900), H), "white")
    d = ImageDraw.Draw(canvas)
    y = 0
    for ims, cap in rows:
        for k, line in enumerate(cap):
            d.text((2, y + k * 12), line[:230], fill=(0, 0, 0) if k == 0 else (0, 0, 160) if k == 1 else (120, 120, 120))
        x = 0
        for im in ims:
            canvas.paste(im, (x, y + 38))
            x += im.width + 8
        y += max(i.height for i in ims) + 44
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def corpus_sheets(corpus, per_head=8, per_source=12, seed=0):
    """Per-head and per-source review sheets from a generated corpus manifest."""
    corpus = Path(corpus)
    rows = []
    for mf in sorted(corpus.glob("shards/*/manifest.jsonl")):
        rows += [json.loads(l) for l in mf.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = random.Random(seed)
    out_dir = corpus / "review"
    out_dir.mkdir(exist_ok=True)

    def items(sel):
        res = []
        for r in sel:
            stem = corpus / r["stem"]
            lab = dict(heads=r["heads"], text=r["text"], clip=r["clip"], glyph_ids=r.get("glyph_ids"))
            res.append((stem, lab, f"[{r['source']}|{r['split']}|{r.get('chain_desc', '')}]"))
        return res
    written = []
    by_src = {}
    for r in rows:
        by_src.setdefault(r["source"] + ("_chain" if r.get("chain_depth", 0) > 0 else ""), []).append(r)
    for src, rs in sorted(by_src.items()):
        p = sheet(items(rng.sample(rs, min(per_source, len(rs)))), out_dir / f"source_{src}.png")
        if p:
            written.append(p)
    for h in HEADS:
        pos = [r for r in rows if r["heads"][h]["present"] == 1]
        if pos:
            p = sheet(items(rng.sample(pos, min(per_head, len(pos)))), out_dir / f"head_{h}.png")
            if p:
                written.append(p)
    return written


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("stems")
    a.add_argument("out")
    a.add_argument("stems", nargs="+")
    b = sub.add_parser("corpus")
    b.add_argument("corpus")
    b.add_argument("--per-head", type=int, default=8)
    b.add_argument("--per-source", type=int, default=12)
    args = ap.parse_args()
    if args.cmd == "stems":
        print(sheet([(s, label_image(s), "") for s in args.stems], args.out))
    else:
        for p in corpus_sheets(args.corpus, args.per_head, args.per_source):
            print("wrote", p)


if __name__ == "__main__":
    main()
