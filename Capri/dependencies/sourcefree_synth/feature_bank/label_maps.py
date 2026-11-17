"""Decode interception label maps and build review overlays.

For a rendered case <stem>: <stem>.png (pixels), <stem>.labels.png (opaque call ids),
<stem>.labels_translucent.png, <stem>.calls.json (call log), <stem>.json (label text).

decode(stem) -> dict with id maps, call records, the answer glyph ids (text calls whose
characters spell the label, matched in drawing order, case-insensitive) and per-glyph boxes.
python -m feature_bank.label_maps <run_dir> [--libs ...] writes <run_dir>/<lib>/overlay_sheet.png:
per case: original | layer overlay | answer-glyph boxes.
"""
import argparse
import colorsys
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CLASS_COLOURS = {  # overlay colours per call class
    "line": (230, 40, 40), "rectangle": (230, 120, 0), "filled_rectangle": (255, 170, 0),
    "ellipse": (40, 90, 230), "filled_ellipse": (90, 140, 255), "arc": (0, 170, 220), "filled_arc": (0, 200, 200),
    "polygon": (150, 60, 200), "filled_polygon": (190, 110, 230), "pixel": (0, 180, 0), "fill": None,
    "loaded": (160, 120, 80), "text_decoy": (120, 120, 120),
}


def read_ids(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint32)
    return (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]


def answer_glyphs(calls, label, visible=None):
    """Text calls that spell the label in drawing order, choosing the subsequence with the most
    visible pixels (libraries often draw the word more than once: measuring passes on scratch
    canvases, shadows under the final text, discarded include-time renders).
    visible: {call id: visible pixel count}; without it the earliest match wins."""
    text = [c for c in calls if c.get("class") == "text" and not c.get("suppressed")]
    # explicit reading order when the renderer records one (shuffled SVG elements); else drawing order
    if any("order_key" in t for t in text):  # (decided before sorting: list.sort empties the list meanwhile)
        text.sort(key=lambda c: (c.get("order_key", float("inf")), c["id"]))
    want = list(str(label).lower())
    n, m = len(text), len(want)
    if m == 0 or n < m:
        return []
    vis = visible or {}
    NEG = float("-inf")
    # best[i][k]: best score using text[:i] to match want[:k]
    best = [[NEG] * (m + 1) for _ in range(n + 1)]
    take = [[False] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        best[i][0] = 0.0
    for i in range(1, n + 1):
        ch = str(text[i - 1].get("char", "")).lower()
        score = vis.get(text[i - 1]["id"], 0) if visible is not None else -i * 1e-6  # tie-break: earliest
        for k in range(1, m + 1):
            best[i][k] = best[i - 1][k]
            if ch == want[k - 1] and best[i - 1][k - 1] != NEG and best[i - 1][k - 1] + score >= best[i][k]:
                best[i][k] = best[i - 1][k - 1] + score
                take[i][k] = True
    if best[n][m] == NEG:
        return []
    out, i, k = [], n, m
    while k > 0:
        if take[i][k]:
            out.append(text[i - 1]["id"]); k -= 1
        i -= 1
    return out[::-1]


def decode(stem):
    stem = Path(stem)
    meta = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    log = json.loads(Path(str(stem) + ".calls.json").read_text(encoding="utf-8"))
    calls = log["calls"]
    by_id = {c["id"]: c for c in calls}
    opaque = read_ids(str(stem) + ".labels.png")
    trans = read_ids(str(stem) + ".labels_translucent.png")
    ids, counts = np.unique(np.concatenate([opaque.ravel(), trans.ravel()]), return_counts=True)
    visible = {int(i): int(c) for i, c in zip(ids, counts) if i}
    glyphs = answer_glyphs(calls, meta["label"], visible)
    boxes = []
    for i, gid in enumerate(glyphs):
        ys, xs = np.nonzero((opaque == gid) | (trans == gid))  # glyph may sit under a partial merge
        boxes.append(None if len(xs) == 0 else
                     dict(char=by_id[gid]["char"], index=i, x0=int(xs.min()), y0=int(ys.min()), x1=int(xs.max()), y1=int(ys.max()),
                          visible_pixels=int(len(xs))))
    return dict(meta=meta, log_meta=log["meta"], calls=by_id, opaque=opaque, translucent=trans, glyphs=glyphs, boxes=boxes)


def overlay(d, image):
    img = np.asarray(image.convert("L").convert("RGB"), dtype=np.float32) * 0.45 + 140
    out = img.copy()
    glyph_set = set(d["glyphs"])
    n = max(1, len(d["glyphs"]))
    def colour(cid):
        c = d["calls"].get(int(cid))
        if c is None:
            return None
        if cid in glyph_set:
            h = d["glyphs"].index(cid) / n
            return tuple(int(255 * v) for v in colorsys.hsv_to_rgb(h, 0.9, 0.95))
        if c.get("class") == "text":
            return CLASS_COLOURS["text_decoy"]
        if c["fn"].startswith("imagecreatefrom"):
            return CLASS_COLOURS["loaded"]
        # review-only background heuristic (semantic mapping is Step 4): gradient/background
        # routines and canvas-scale fills are left uncoloured
        site = c.get("site", "").lower()
        if any(k in site for k in ("gradient", "background", "image_create")) or c.get("canvas_fraction", 0) > 0.4:
            return None
        return CLASS_COLOURS.get(c.get("class"))
    for layer, alpha in ((d["opaque"], 0.85), (d["translucent"], 0.45)):
        for cid in np.unique(layer):
            if cid == 0:
                continue
            col = colour(int(cid))
            if col is None:
                continue
            m = layer == cid
            out[m] = out[m] * (1 - alpha) + np.array(col, np.float32) * alpha
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def boxes_image(d, image):
    im = image.convert("RGB").copy()
    dr = ImageDraw.Draw(im)
    for b in d["boxes"]:
        if b:
            dr.rectangle([b["x0"], b["y0"], b["x1"], b["y1"]], outline=(255, 0, 0))
    return im


def sheet(lib_dir, max_cases=6):
    stems = sorted(p.with_suffix("") for p in lib_dir.glob("*.labels.png"))
    stems = [Path(str(s)[: -len(".labels")]) for s in stems][:max_cases]
    rows = []
    for s in stems:
        d = decode(s)
        image = Image.open(s.with_suffix(".png")).convert("RGB")
        found = sum(b is not None for b in d["boxes"])
        caption = f"{d['meta']['label']}  glyphs {found}/{len(str(d['meta']['label']))}  canvas={d['log_meta']['label_canvas']}  copies={d['log_meta']['copy_writes']}"
        rows.append(([image, overlay(d, image), boxes_image(d, image)], caption))
    if not rows:
        return None
    W = max(sum(i.width for i in r[0]) + 20 for r in rows)
    H = sum(r[0][0].height + 18 for r in rows)
    out = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(out)
    y = 0
    for ims, cap in rows:
        dr.text((2, y), cap, fill="black")
        x = 0
        for im in ims:
            out.paste(im, (x, y + 14)); x += im.width + 10
        y += ims[0].height + 18
    out.save(lib_dir / "overlay_sheet.png")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--libs", nargs="*")
    a = ap.parse_args()
    run = Path(a.run)
    for lib_dir in sorted(p for p in run.iterdir() if p.is_dir()):
        if a.libs and lib_dir.name not in a.libs:
            continue
        if sheet(lib_dir) is not None:
            print("wrote", lib_dir / "overlay_sheet.png")


if __name__ == "__main__":
    main()
