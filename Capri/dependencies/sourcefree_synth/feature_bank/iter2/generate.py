"""Iteration 2 corpus: complete labelled CAPTCHAs from captcha-effects (effects + our own text) -> a height-96 cache.

Every image comes from `captcha_effects.render_captcha` and is redrawn (new seed) until it is usable (no clipping,
legible). Train / steering / selection / calibration / seen_test use train fonts; generalisation uses held-out fonts.

Cache (<out>/iter2_h96/):
  images.u8        RGB, height 96, native aspect (width capped), concatenated column blocks
  index.npz        offsets, widths, native sizes, labels:
    eff_present [N,37] 0/1, eff_level [N,37] 0..10                 (CNN heads; EFFECT_HEADS order)
    txt_present [N,T] 0/1, txt_level [N,T] 0..10 / -1 (presence-only)  (transformer heads; TEXT_HEADS order)
    chars [N,8] index into CHARS / -1, length [N]                  (transformer reading head)
    font_* image heads: cat, stroke (-1 unknown), weight, width, x_height (-1 unknown), slant, italic
    g_* per glyph [N,8]: cat, weight, width, x_height, slant, italic, box (x0,y0,x1,y1 relative)
  labels.jsonl.gz  full labels per image (effect styles, text style, slots, specs, seed)
  meta.json        splits, counts, heads, renderer version, font split, sha256 of every image

  python -m feature_bank.iter2.generate --out runs/<name> --split train --n 100000 --workers 28
"""
import argparse
import gzip
import hashlib
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

CE_ROOT = Path(__file__).resolve().parents[3] / "captcha-effects"
sys.path.insert(0, str(CE_ROOT))

H = 96
MAX_W = 480
CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_LEN = 8
CATEGORIES = ["Sans Serif", "Serif", "Display", "Handwriting", "Monospace"]
STROKES = ["Sans Serif", "Serif", "Slab Serif"]
TEXT_HEADS = ["per_char_rotation", "vertical_jitter", "size_variation", "text_on_curve", "slant",
              "overlap_tight_spacing", "wide_spacing", "condensed", "expanded", "font_mixture", "multicolor_text",
              "two_layer", "digits_only", "uppercase_only", "lowercase_only", "mixed_case_alnum"]
FONT_HEADS = ["weight", "width", "x_height", "slant_level"]
SPLIT_SALT = {"train": 11, "steering": 101, "selection": 102, "calibration": 103, "seen_test": 104,
              "generalisation": 105, "smoke": 9}
P_ON = 0.15  # per effect; exclusive groups apply (render.sample_spec)


def effect_heads():
    from captcha_effects import render as R
    from captcha_effects.effects import REGISTRY
    out = []
    for n in R.SAMPLED:
        h = REGISTRY[n]["head"]
        if h not in out:
            out.append(h)
    return out


def canvas_size(rng):
    """Canvas sizes covering common text CAPTCHAs: height 30-90 px, aspect 1.8-5."""
    h = int(rng.integers(30, 91))
    return int(round(h * rng.uniform(1.8, 5.0))), h


_BANK = None


def _bank():
    global _BANK
    if _BANK is None:
        from captcha_effects.text import FontBank
        _BANK = FontBank()
    return _BANK


K_EFFECTS = (3, 4, 5)  # dataset 1 scheme: exactly 3-5 visible effects per image (user, 2026-10-03)


def render_one(args):
    split, index, p_on, scheme, salt = args
    from captcha_effects.captcha import render_captcha
    from captcha_effects.render import sample_spec_k
    from captcha_effects.text import balanced_text_spec
    font_split = "heldout" if split == "generalisation" else "train"
    for attempt in range(80):
        seed = [SPLIT_SALT[split], index, attempt] if salt == 0 else [salt, SPLIT_SALT[split], index, attempt]  # salt 0 = dataset 0 seeds
        rng = np.random.default_rng(seed)
        size = canvas_size(rng)
        try:
            if scheme == "k35":
                es = sample_spec_k(rng, int(rng.choice(K_EFFECTS)), weights=p_on)
                ts = balanced_text_spec(rng, _bank(), size[1], split=font_split)
                out = render_captcha(rng, _bank(), size, effects_spec=es, text_spec=ts, font_split=font_split)
            else:
                out = render_captcha(rng, _bank(), size, font_split=font_split, p_on=p_on)
        except Exception as e:  # noqa: BLE001
            out = None
            err = repr(e)[:200]
        if out is None or not out["text"]["usable"]:
            continue
        if scheme == "k35" and not (K_EFFECTS[0] <= sum(e["present"] for e in out["effects"].values()) <= K_EFFECTS[-1]):
            continue  # an effect measured invisible (level 0) can drop the count below 3
        return pack(out, split, index, seed, size, attempt)
    return dict(split=split, index=index, failed=True, err=locals().get("err"))


def pack(out, split, index, seed, size, attempt):
    img = out["image"]
    w = max(8, min(MAX_W, int(round(size[0] * H / size[1]))))
    small = np.asarray(Image.fromarray(img).resize((w, H), Image.BILINEAR), dtype=np.uint8)
    t = out["text"]
    effects = {h: dict(present=e["present"], level=e["level"], style=e.get("style", {}), effect=e.get("effect"))
               for h, e in out["effects"].items()}
    lab = dict(id=f"{split}_{index:06d}", split=split, index=index, seed=seed, attempt=attempt, size=list(size),
               background=out["background"], effects=effects,
               text=dict(text=t["text"], length=t["length"], heads=t["heads"], style=t["style"], slots=t["slots"],
                         legibility=t["legibility"]),
               text_spec={k: v for k, v in out["text_spec"].items()},
               sha256=hashlib.sha256(img.tobytes()).hexdigest())
    return dict(split=split, index=index, image=small, label=lab, failed=False)


def label_rows(lab, eff_heads):
    """Label arrays for one image."""
    ep = np.zeros(len(eff_heads), np.int8)
    el = np.zeros(len(eff_heads), np.int8)
    for i, h in enumerate(eff_heads):
        e = lab["effects"].get(h)
        if e:
            ep[i], el[i] = e["present"], e["level"]
    t = lab["text"]
    tp = np.zeros(len(TEXT_HEADS), np.int8)
    tl = np.full(len(TEXT_HEADS), -1, np.int8)
    for i, h in enumerate(TEXT_HEADS):
        x = t["heads"][h]
        tp[i] = x["present"]
        if x.get("level") is not None:
            tl[i] = x["level"]
    chars = np.full(MAX_LEN, -1, np.int16)
    for i, c in enumerate(t["text"][:MAX_LEN]):
        chars[i] = CHARS.index(c)
    st = t["style"]
    font = np.array([CATEGORIES.index(st["dominant_category"]) if st.get("dominant_category") in CATEGORIES else -1,
                     STROKES.index(st["dominant_stroke_class"]) if st.get("dominant_stroke_class") in STROKES else -1,
                     st.get("median_weight", -1), st.get("median_width", -1),
                     -1 if st.get("median_x_height") is None else st["median_x_height"],
                     st.get("median_slant_level", -1), int(bool(st.get("italic_any")))], np.int8)
    g = np.full((MAX_LEN, 6), -1, np.int8)
    box = np.full((MAX_LEN, 4), -1.0, np.float16)
    W, Hh = lab["size"]
    for i, s in enumerate(t["slots"][:MAX_LEN]):
        g[i] = [CATEGORIES.index(s["category"]) if s["category"] in CATEGORIES else -1, s["weight"], s["width"],
                -1 if s["x_height"] is None else s["x_height"], s["slant_level"], int(s["italic"])]
        if s.get("box"):
            x0, y0, x1, y1 = s["box"]
            box[i] = [x0 / W, y0 / Hh, (x1 + 1) / W, (y1 + 1) / Hh]
    return ep, el, tp, tl, chars, len(t["text"]), font, g, box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", required=True, choices=list(SPLIT_SALT))
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--p-on", type=float, default=P_ON)
    ap.add_argument("--p-on-file", default=None, help="json with per-effect p_on (scheme p_on) or weights (scheme k35)")
    ap.add_argument("--scheme", default="p_on", choices=["p_on", "k35"])
    ap.add_argument("--salt", type=int, default=0, help="seed salt (dataset 0: 0; dataset 1: 1)")
    a = ap.parse_args()
    from captcha_effects import __version__ as ce_version
    from captcha_effects import render as R
    eff_heads = effect_heads()
    p_on = {n: a.p_on for n in R.SAMPLED}
    if a.p_on_file and a.scheme == "p_on":
        p_on.update(json.loads(Path(a.p_on_file).read_text(encoding="utf-8"))["p_on"])
    root = Path(a.out) / "iter2_h96" / (f"train_{a.start:06d}" if a.split == "train" else a.split)
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = [None] * a.n
    if a.scheme == "k35":
        p_on = {n: 1.0 for n in R.SAMPLED}
        if a.p_on_file:
            p_on.update(json.loads(Path(a.p_on_file).read_text(encoding="utf-8"))["weights"])
    jobs = [(a.split, a.start + i, p_on, a.scheme, a.salt) for i in range(a.n)]
    attempts = Counter()
    with ProcessPoolExecutor(a.workers) as ex:
        for k, r in enumerate(ex.map(render_one, jobs, chunksize=64)):
            results[k] = r
            if not r["failed"]:
                attempts[r["label"]["attempt"]] += 1
            if (k + 1) % 5000 == 0:
                print(f"{a.split}: {k + 1}/{a.n}  {time.time() - t0:.0f}s", flush=True)
    ok = [r for r in results if not r["failed"]]
    failed = [dict(index=r["index"], err=r.get("err")) for r in results if r["failed"]]
    widths = np.array([r["image"].shape[1] for r in ok], np.int32)
    offsets = np.concatenate([[0], np.cumsum(widths)[:-1]]).astype(np.int64)
    imgs = np.concatenate([r["image"] for r in ok], axis=1) if ok else np.zeros((H, 0, 3), np.uint8)
    np.ascontiguousarray(imgs.transpose(1, 0, 2)).tofile(root / "images.u8")  # column-major blocks: [sum(w), 96, 3]
    rows = [label_rows(r["label"], eff_heads) for r in ok]
    stack = lambda i, dt: np.stack([x[i] for x in rows]).astype(dt)  # noqa: E731
    np.savez_compressed(root / "index.npz", offsets=offsets, widths=widths,
                        native=np.array([r["label"]["size"] for r in ok], np.int16),
                        eff_present=stack(0, np.int8), eff_level=stack(1, np.int8),
                        txt_present=stack(2, np.int8), txt_level=stack(3, np.int8), chars=stack(4, np.int16),
                        length=np.array([x[5] for x in rows], np.int8), font=stack(6, np.int8),
                        glyph=stack(7, np.int8), glyph_box=stack(8, np.float16))
    with gzip.open(root / "labels.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in ok:
            f.write(json.dumps(r["label"], default=str) + "\n")
    ids = [r["label"]["id"] for r in ok]
    meta = dict(split=a.split, n=len(ok), requested=a.n, start=a.start, failed=failed,
                attempts_hist={int(k): v for k, v in sorted(attempts.items())},
                usable_rate=round(len(ok) / max(1, sum((k + 1) * v for k, v in attempts.items())), 3),
                seconds=round(time.time() - t0, 1), p_on=a.p_on, p_on_effects=p_on, scheme=a.scheme, salt=a.salt, h=H, max_w=MAX_W,
                layout="images.u8 = uint8 [sum(widths), 96, 3] (column blocks; transpose to HWC per image)",
                effect_heads=eff_heads, text_heads=TEXT_HEADS, chars=CHARS, max_len=MAX_LEN, categories=CATEGORIES,
                strokes=STROKES, font_cols=["cat", "stroke", "weight", "width", "x_height", "slant", "italic"],
                glyph_cols=["cat", "weight", "width", "x_height", "slant", "italic"],
                font_split="heldout" if a.split == "generalisation" else "train", captcha_effects=ce_version,
                font_meta_sha256=hashlib.sha256((CE_ROOT / "fonts/font_meta.json").read_bytes()).hexdigest(),
                ids=ids, sha256=[r["label"]["sha256"] for r in ok])
    (root / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    print(f"{a.split}: {len(ok)} ok, {len(failed)} failed, usable rate {meta['usable_rate']}, {meta['seconds']}s")


if __name__ == "__main__":
    main()
