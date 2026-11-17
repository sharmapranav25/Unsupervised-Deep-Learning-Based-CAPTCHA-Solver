"""Python renderers as an extra, flagged, capped source (handoff §7b).

Three sources, same output contract as the PHP adapters (<stem>.png, .json, .labels.png,
.labels_translucent.png, .labels_classes.png, .calls.json), so feature_bank.heads labels them
with the same code:
  python_shapes  shape layer: circles, ellipses, polygons, arcs, blobs, lines, dots, grids, blocks,
                 scribbles, gradients, textures, decoy characters. Knobs per layer:
                 kind, count, size, stroke, filled, colour, opacity, region, seed.
  python_text    font renderer with per-character font, size, weight (variable 'wght' axis or
                 FreeType-style emboldening), width, slant, rotation, vertical jitter, spacing /
                 overlap, outline / hollow / shadow, colour / multicolour, opacity, dotted strokes,
                 emboss, text-only warps (wave, morph, curved baseline) and blur. Exact per-glyph
                 masks, boxes and angles. Any TTF path works (fitted variable-font instances too).
  python_capri   wraps the CAPRI renderers (releases/CAPRI_phase123_v1): phase2/renderer.py
                 (operation log, gain knobs) and phase1/reproduction/corpus.py (15 switchable
                 effects), driven with generic random styles only - never the family profiles
                 measured from real targets.
Every source is seeded and replayable (one independent random stream per layer / glyph, so a
switched-off layer does not shift the others), accepts a background image (a previous chain
stage, whose label maps are carried over), and labels by drawing each operation with its own id.
"""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from feature_bank.bank02_smoke import ROOT

CLASS_BITS = {"line": 1, "rectangle": 2, "filled_rectangle": 4, "ellipse": 8, "filled_ellipse": 16, "arc": 32,
              "filled_arc": 64, "polygon": 128, "filled_polygon": 256, "pixel": 512, "fill": 1024, "text": 2048,
              "loaded": 4096}
SOURCES = ("python_shapes", "python_text", "python_capri")
TEXT_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz23456789"
SHAPE_HEAD = {"circle": "bg_circles", "ellipse": "bg_circles", "arc": "curves", "polygon": "scribbles",
              "scribble": "scribbles", "blob": "blob_noise", "dots": "dot_noise", "line": "straight_lines",
              "grid": "bg_grid", "blocks": "bg_blocks", "gradient": "bg_gradient", "texture": "bg_image_texture",
              "decoys": "bg_decoy_chars", "strike": "strike_through", "border": "border_frame"}


def stream(seed, *k):
    return np.random.default_rng(np.random.SeedSequence([int(seed) % (2 ** 32), *[int(x) for x in k]]))


# ------------------------------------------------------------------ labelled canvas

class Canvas:
    """RGB image + hidden id canvases. Each op is rasterised once as a coverage mask; the image is
    blended with it and the op's id is written where coverage >= 50% (translucent ops to the
    translucent map), class bits wherever coverage > 0."""

    def __init__(self, w, h, colour=(255, 255, 255), bg=None, first_id=1):
        if bg is not None:
            im = Image.open(bg["png"]).convert("RGB")
            self.img = np.asarray(im, np.float32).copy()
            from feature_bank.label_maps import read_ids
            self.opaque = read_ids(str(bg["stem"]) + ".labels.png").astype(np.uint32)
            self.trans = read_ids(str(bg["stem"]) + ".labels_translucent.png").astype(np.uint32)
            self.classes = read_ids(str(bg["stem"]) + ".labels_classes.png").astype(np.uint32)
            self.h, self.w = self.opaque.shape
            self.next = int(bg["next_id"])
            self.chain_seeded = 1
            self.base = tuple(int(v) for v in np.median(self.img.reshape(-1, 3), 0))
        else:
            self.w, self.h = int(w), int(h)
            self.img = np.zeros((self.h, self.w, 3), np.float32) + np.array(colour, np.float32)
            self.opaque = np.zeros((self.h, self.w), np.uint32)
            self.trans = np.zeros((self.h, self.w), np.uint32)
            self.classes = np.zeros((self.h, self.w), np.uint32)
            self.next = int(first_id)
            self.chain_seeded = 0
            self.base = tuple(int(c) for c in colour)
        self.calls = []
        self.disabled = set()
        self.head_cov = {}  # head -> bool mask of every pixel its ops touched (switch-test footprint)

    def new_id(self, **rec):
        cid = self.next
        self.next += 1
        rec["id"] = cid
        self.calls.append(rec)
        return cid

    def mask(self, draw_fn, pad=0):
        m = Image.new("L", (self.w + 2 * pad, self.h + 2 * pad), 0)
        draw_fn(ImageDraw.Draw(m), pad)
        return np.asarray(m, np.float32) / 255.0

    def apply(self, cov, colour, cls, site, head=None, opacity=1.0, **extra):
        """Blend colour through coverage cov (h x w, 0..1); label it. Returns the call id."""
        switched = head in self.disabled or cls in self.disabled or site in self.disabled
        cid = self.new_id(fn="py:" + site.split(":")[-1], site=site, **{"class": cls}, head=head,
                          rgba=[int(c) for c in colour] + [int(round(127 * (1 - opacity)))],
                          translucent=opacity < 0.5, **extra)
        if switched:
            self.calls[-1]["suppressed"] = True
            return cid
        a = (cov * opacity)[..., None]
        self.img = self.img * (1 - a) + np.array(colour, np.float32) * a
        key = head or ("text" if cls == "text" else cls)
        self.head_cov[key] = self.head_cov.get(key, np.zeros(cov.shape, bool)) | (cov > 0)
        sel = cov >= 0.5
        (self.trans if opacity < 0.5 else self.opaque)[sel] = cid
        self.classes[cov > 0] |= CLASS_BITS.get(cls, 0)
        return cid

    def field(self, rgb_field, cls, site, head, opacity=1.0, translucent_label=False):
        """Whole-canvas layer given as an H x W x 3 colour field (gradient, texture)."""
        cid = self.new_id(fn="py:" + site.split(":")[-1], site=site, **{"class": cls}, head=head, rgba=None,
                          translucent=translucent_label, canvas_fraction=1.0)
        if head in self.disabled or cls in self.disabled:
            self.calls[-1]["suppressed"] = True
            return cid
        self.img = self.img * (1 - opacity) + rgb_field * opacity
        self.head_cov[head] = np.ones(self.img.shape[:2], bool)
        (self.trans if translucent_label else self.opaque)[:] = cid
        self.classes |= CLASS_BITS.get(cls, 0)
        return cid

    def filter(self, name, filter_id, fn_img):
        self.calls.append(dict(id=self.next, fn="py:" + name, site="py:" + name, **{"class": "filter"},
                               filter=filter_id, rgba=None, translucent=False))
        self.next += 1
        if name not in self.disabled:
            self.img = np.asarray(fn_img(Image.fromarray(self.img.clip(0, 255).astype(np.uint8))), np.float32)

    def save(self, stem, meta):
        stem = Path(stem)
        Image.fromarray(self.img.clip(0, 255).round().astype(np.uint8)).save(stem.with_suffix(".png"))

        def ids_png(a, suffix):
            rgb = np.stack([(a >> 16) & 255, (a >> 8) & 255, a & 255], -1).astype(np.uint8)
            Image.fromarray(rgb).save(str(stem) + suffix)
        ids_png(self.opaque, ".labels.png")
        ids_png(self.trans, ".labels_translucent.png")
        ids_png(self.classes, ".labels_classes.png")
        log_meta = dict(label_canvas="python", copy_writes=0, unmirrored=[], palette_canvas=False, class_bits=CLASS_BITS,
                        disabled=sorted(self.disabled), calls=len(self.calls), chain_seeded=self.chain_seeded,
                        next_id=self.next)
        Path(str(stem) + ".calls.json").write_text(json.dumps(dict(meta=log_meta, calls=self.calls)), encoding="utf-8")
        if meta.get("params", {}).get("_debug_cov"):
            np.savez_compressed(str(stem) + ".head_cov.npz", **{k: v for k, v in self.head_cov.items()})
        meta = dict(meta, width=self.w, height=self.h, source_format="png", ok=True, labels=log_meta,
                    param_control={k: dict(value=v, overridden=False) for k, v in meta.get("params", {}).items()})
        stem.with_suffix(".json").write_text(json.dumps(meta, default=str), encoding="utf-8")


# ------------------------------------------------------------------ fonts

def bank_fonts(split="train", caseless=False):
    from feature_bank.font_split import load_bank
    return [r for r in load_bank() if r.get("split", "train") == split and (caseless or not r.get("caps_only"))]


def resolve_font(f):
    p = Path(f)
    if p.is_absolute() and p.exists():
        return p
    q = ROOT / f
    if q.exists():
        return q
    from feature_bank.font_split import by_key
    r = by_key().get(str(f))
    if r:
        return ROOT / r["path"]
    raise FileNotFoundError(f)


def has_glyph(font, ch):
    try:
        bb = font.getbbox(ch)
        return bb[2] - bb[0] > 0 and bb[3] - bb[1] > 0
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------------ python_shapes

def region_box(region, w, h, rng, text_box=None):
    if region == "text" and text_box is not None:
        return text_box
    if region == "top":
        return (0, 0, w, h // 2)
    if region == "bottom":
        return (0, h // 2, w, h)
    if isinstance(region, (list, tuple)) and len(region) == 4:
        return tuple(region)
    return (0, 0, w, h)


def colour_of(spec, rng, base):
    if spec in (None, "random"):
        return tuple(int(x) for x in rng.integers(0, 256, 3))
    if spec == "contrast":
        lum = 0.299 * base[0] + 0.587 * base[1] + 0.114 * base[2]
        lo, hi = (0, 110) if lum > 128 else (150, 256)
        return tuple(int(x) for x in rng.integers(lo, hi, 3))
    if spec == "similar":
        return tuple(int(np.clip(b + rng.integers(-60, 61), 0, 255)) for b in base)
    return tuple(int(x) for x in spec)


def draw_layer(cv, layer, li, seed, text_box=None, font_pool=None):
    """One shape layer: kind, count, size [lo,hi], stroke, filled, colour, opacity, region, seed."""
    kind = layer["kind"]
    head = SHAPE_HEAD[kind]
    rng = stream(layer.get("seed", seed), 100 + li)
    w, h = cv.w, cv.h
    x0, y0, x1, y1 = region_box(layer.get("region", "full"), w, h, rng, text_box)
    base = cv.base
    n = int(layer.get("count", 1))
    lo, hi = layer.get("size", [4, 20])
    stroke = int(layer.get("stroke", 1))
    op = float(layer.get("opacity", 1.0))
    site = f"py_shapes:{kind}"
    ids = []
    if kind == "gradient":
        c1 = np.array(colour_of(layer.get("colour"), rng, base), np.float32)
        c2 = np.array(colour_of("random", rng, base), np.float32)
        ang = rng.uniform(0, math.pi)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        t = (xx * math.cos(ang) + yy * math.sin(ang))
        t = (t - t.min()) / max(1e-6, t.max() - t.min())
        if layer.get("steep"):  # ramp across the middle of the canvas (where the text sits)
            t = np.clip((t - 0.2) / 0.6, 0, 1)
        ids.append(cv.field(c1 * (1 - t[..., None]) + c2 * t[..., None], "fill", site, head, opacity=op))
        return ids
    if kind == "texture":
        # luminance texture: blurred noise normalised to unit std, scaled to the requested std
        from scipy import ndimage
        sm = float(rng.uniform(0.8, 1.8))
        noise = ndimage.gaussian_filter(rng.normal(0, 1, (h, w)).astype(np.float32), sm)
        noise = noise / max(1e-6, float(noise.std())) * float(layer.get("amplitude", 30))
        ids.append(cv.field(np.clip(cv.img + noise[..., None], 0, 255), "fill", site, head, opacity=op, translucent_label=True))
        return ids
    if kind == "grid":
        step = int(rng.integers(max(4, lo), max(5, hi + 1)))
        col = colour_of(layer.get("colour", "similar"), rng, base)
        off = int(rng.integers(0, step))
        for gx in range(x0 + off, x1, step):
            ids.append(cv.apply(cv.mask(lambda d, p, gx=gx: d.line([(gx, y0), (gx, y1)], fill=255, width=stroke)),
                                col, "line", site, head, op))
        for gy in range(y0 + off, y1, step):
            ids.append(cv.apply(cv.mask(lambda d, p, gy=gy: d.line([(x0, gy), (x1, gy)], fill=255, width=stroke)),
                                col, "line", site, head, op))
        return ids
    if kind == "dots":  # one id per layer (like a pixel run), dots of radius lo..hi
        col = layer.get("colour", "random")
        cov = np.zeros((h, w), np.float32)
        for k in range(n):
            r = rng.integers(0, 10 ** 6)
            sub = stream(r, k)
            cx, cy = sub.integers(x0, max(x0 + 1, x1)), sub.integers(y0, max(y0 + 1, y1))
            rad = int(sub.integers(max(0, lo), max(1, hi) + 1))
            m = cv.mask(lambda d, p: d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=255) if rad > 0
                        else d.point([(cx, cy)], fill=255))
            cov = np.maximum(cov, m)
        ids.append(cv.apply(cov, colour_of(col, rng, base), "filled_ellipse" if hi > 0 else "pixel", site, head, op))
        return ids
    if kind == "decoys":
        fonts = font_pool or bank_fonts()
        for k in range(n):
            sub = stream(rng.integers(0, 10 ** 9), k)
            ch = TEXT_ALPHABET[int(sub.integers(len(TEXT_ALPHABET)))]
            f = ImageFont.truetype(str(ROOT / fonts[int(sub.integers(len(fonts)))]["path"]), int(sub.integers(lo, hi + 1)))
            px, py = int(sub.integers(x0, max(x0 + 1, x1 - 4))), int(sub.integers(y0, max(y0 + 1, y1 - 4)))
            ang = float(sub.uniform(-40, 40))
            g = Image.new("L", (hi * 3, hi * 3))
            ImageDraw.Draw(g).text((hi, hi // 2), ch, font=f, fill=255)
            g = g.rotate(ang, Image.Resampling.BILINEAR)
            m = Image.new("L", (w, h))
            m.paste(g, (px - hi, py - hi))
            ids.append(cv.apply(np.asarray(m, np.float32) / 255, colour_of(layer.get("colour", "similar"), sub, base),
                                "text", site, head, op, decoy_char=ch))
        return ids
    for k in range(n):
        sub = stream(rng.integers(0, 10 ** 9), k)
        col = colour_of(layer.get("colour", "random"), sub, base)
        cx, cy = float(sub.uniform(x0, x1)), float(sub.uniform(y0, y1))
        s1, s2 = float(sub.uniform(lo, hi)), float(sub.uniform(lo, hi))
        filled = bool(layer.get("filled", False))
        if kind in ("circle", "ellipse"):
            s2 = s1 if kind == "circle" else s2
            box = [cx - s1, cy - s2, cx + s1, cy + s2]
            m = cv.mask(lambda d, p: d.ellipse(box, fill=255) if filled else d.ellipse(box, outline=255, width=stroke))
            ids.append(cv.apply(m, col, "filled_ellipse" if filled else "ellipse", site, head, op))
        elif kind == "blob":
            pts = [(cx + s1 * (0.6 + 0.4 * sub.random()) * math.cos(t), cy + s1 * (0.6 + 0.4 * sub.random()) * math.sin(t))
                   for t in np.linspace(0, 2 * math.pi, 9)[:-1]]
            m = cv.mask(lambda d, p: d.polygon(pts, fill=255))
            ids.append(cv.apply(m, col, "filled_polygon", site, head, op))
        elif kind == "arc":
            start = float(sub.uniform(0, 360))
            box = [cx - s1 * 2, cy - s2, cx + s1 * 2, cy + s2]
            m = cv.mask(lambda d, p: d.arc(box, start, start + float(sub.uniform(90, 270)), fill=255, width=stroke))
            ids.append(cv.apply(m, col, "arc", site, head, op))
        elif kind in ("polygon", "scribble"):
            npts = int(sub.integers(3, 7)) if kind == "polygon" else int(layer.get("npts") or sub.integers(6, 14))
            pts = [(float(sub.uniform(x0, x1)), float(sub.uniform(y0, y1))) if kind == "scribble" else
                   (cx + s1 * math.cos(t + sub.uniform(-.3, .3)), cy + s2 * math.sin(t + sub.uniform(-.3, .3)))
                   for t in np.linspace(0, 2 * math.pi, npts + 1)[:-1]]
            if kind == "polygon" and filled:
                m = cv.mask(lambda d, p: d.polygon(pts, fill=255))
                ids.append(cv.apply(m, col, "filled_polygon", site, "bg_blocks", op))
            else:
                closed = pts + [pts[0]] if kind == "polygon" else pts
                m = cv.mask(lambda d, p: d.line(closed, fill=255, width=stroke, joint="curve"))
                ids.append(cv.apply(m, col, "polygon", site, head, op))
        elif kind == "line":
            ang = float(sub.uniform(0, math.pi))
            L = float(sub.uniform(lo, hi))
            pts = [(cx - L * math.cos(ang), cy - L * math.sin(ang)), (cx + L * math.cos(ang), cy + L * math.sin(ang))]
            m = cv.mask(lambda d, p: d.line(pts, fill=255, width=stroke))
            ids.append(cv.apply(m, col, "line", site, head, op))
        elif kind == "strike":
            tb = text_box or (0, h // 3, w, 2 * h // 3)
            ya, yb = float(sub.uniform(tb[1], tb[3])), float(sub.uniform(tb[1], tb[3]))
            pts = [(tb[0] - 4, ya), (tb[2] + 4, yb)]
            m = cv.mask(lambda d, p: d.line(pts, fill=255, width=stroke))
            ids.append(cv.apply(m, col, "line", site, head, op))
        elif kind == "blocks":
            bw, bh = s1 * 2, s2 * 2
            box = [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
            m = cv.mask(lambda d, p: d.rectangle(box, fill=255))
            ids.append(cv.apply(m, col, "filled_rectangle", site, head, op))
        elif kind == "border":
            m = cv.mask(lambda d, p: d.rectangle([0, 0, w - 1, h - 1], outline=255, width=stroke))
            ids.append(cv.apply(m, col, "rectangle", site, head, op))
            break
    return ids


# ------------------------------------------------------------------ python_text

def glyph_mask(ch, font_path, size, weight, width_scale, slant, angle, hollow, outline_px, dotted, seed):
    """Full glyph coverage (L image, 0..255) after per-glyph transforms; plus its offset."""
    font = ImageFont.truetype(str(font_path), int(size))
    axes_note = None
    if weight is not None:
        try:
            axes = font.get_variation_axes()
            names = [a.get("name", b"").decode() if isinstance(a.get("name"), bytes) else str(a.get("name", "")) for a in axes]
            vals = []
            for a, nm in zip(axes, names):
                if nm.lower().startswith("weight") or nm.lower() == "wght":
                    vals.append(float(np.clip(weight, a["minimum"], a["maximum"])))
                    axes_note = "wght"
                else:
                    vals.append(a.get("default", a["minimum"]))
            if axes_note:
                font.set_variation_by_axes(vals)
        except Exception:  # noqa: BLE001 - static font: embolden instead
            axes_note = None
    stroke = 0
    if weight is not None and axes_note is None:
        stroke = int(max(0, round((weight - 400) / 250)))  # FreeType-style emboldening via stroke
    S = int(size * 3 + 20)
    g = Image.new("L", (S, S))
    ImageDraw.Draw(g).text((S // 4, S // 6), ch, font=font, fill=255, stroke_width=stroke, stroke_fill=255)
    bb = g.getbbox()
    if not bb:
        return None, axes_note
    g = g.crop(bb)
    if width_scale and abs(width_scale - 1) > 1e-3:
        g = g.resize((max(1, int(round(g.width * width_scale))), g.height), Image.Resampling.BILINEAR)
    if slant:
        sh = math.tan(math.radians(slant))
        pad = int(abs(sh) * g.height) + 2
        g2 = Image.new("L", (g.width + pad, g.height))
        g2.paste(g, (pad if sh > 0 else 0, 0))
        g = g2.transform(g2.size, Image.Transform.AFFINE, (1, sh, -sh * g.height if sh > 0 else 0, 0, 1, 0),
                         Image.Resampling.BILINEAR)
    a = np.asarray(g, np.float32) / 255
    if hollow or outline_px:
        from scipy import ndimage
        t = max(1, int(outline_px or 2))
        solid = np.pad(a, t + 1)
        inner = ndimage.grey_erosion(solid, size=(2 * t + 1, 2 * t + 1))
        a = np.clip(solid - inner, 0, 1) if hollow else solid
        g = Image.fromarray(np.uint8(a * 255))
    if dotted:  # fragmentation level 1..10: dot spacing grows, dot radius shrinks relative to it
        a = np.asarray(g, np.float32) / 255
        yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
        lv = float(dotted)
        sp = 3.0 + 0.35 * lv
        rf = 0.64 - 0.036 * lv
        dots = (((xx % sp) - sp / 2) ** 2 + ((yy % sp) - sp / 2) ** 2) <= (sp * rf) ** 2
        g = Image.fromarray(np.uint8(a * dots * 255))
    if angle:
        g = g.rotate(angle, Image.Resampling.BILINEAR, expand=True)
    bb = g.getbbox()
    return (g.crop(bb) if bb else None), axes_note


def warp_field(kind, w, h, amp, period, seed):
    """Displacement (dx, dy) for text-only warps."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rng = stream(seed, 77)
    if kind == "wave":
        ph = float(rng.uniform(0, 2 * math.pi))
        return np.zeros_like(xx), amp * np.sin(2 * math.pi * xx / max(4.0, period) + ph)
    if kind == "morph":
        # coarse, smooth field (cells ~ a glyph wide): bends strokes without shredding them
        f = rng.normal(0, 1, (2, max(2, h // 28), max(2, w // 28))).astype(np.float32)
        out = []
        for c in f:
            im = Image.fromarray(c).resize((w, h), Image.Resampling.BICUBIC)
            out.append(np.asarray(im, np.float32) * amp)
        return out[0], out[1]
    return np.zeros_like(xx), np.zeros_like(xx)


def remap(a, dx, dy):
    import cv2
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return cv2.remap(a.astype(np.float32), xx + dx, yy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def draw_text(cv, text, tp, seed, font_split="train"):
    """Python font renderer. tp: dict of text knobs (scalars apply to every glyph; lists per slot)."""
    w, h = cv.w, cv.h
    n = len(text)
    rng = stream(seed, 500)

    def slot(key, i, default):
        v = tp.get(key, default)
        return v[i] if isinstance(v, (list, tuple)) and len(v) == n else v
    caseless = text == text.upper()
    pool = bank_fonts(font_split, caseless)
    fonts = tp.get("fonts")
    if not fonts:
        k = int(rng.integers(len(pool)))
        fonts = [pool[k]["path"]] * n
        if tp.get("font_mix"):
            fonts = [pool[int(stream(seed, 510, i).integers(len(pool)))]["path"] for i in range(n)]
    elif isinstance(fonts, str):
        fonts = [fonts] * n
    size = float(tp.get("size", h * 0.55))
    masks, metas = [], []
    for i, ch in enumerate(text):
        r = stream(seed, 520, i)
        fp = resolve_font(fonts[i])
        sz = size * (1 + float(slot("size_jitter", i, 0.0)) * float(r.uniform(-1, 1)))
        ang = float(slot("rotation", i, 0.0)) * float(r.uniform(-1, 1)) if not isinstance(tp.get("rotation"), list) else float(slot("rotation", i, 0))
        g, axis = glyph_mask(ch, fp, sz, slot("weight", i, None), float(slot("width", i, 1.0)), float(slot("slant", i, 0.0)),
                             ang, bool(slot("hollow", i, False)), int(slot("outline", i, 0)) if slot("outline", i, 0) and not slot("hollow", i, False) else 0,
                             int(slot("dotted", i, 0)), seed)
        if g is None:
            return None, f"font {Path(fp).parent.name} has no glyph for {ch!r}"
        solid = None
        if int(slot("dotted", i, 0)):  # the same glyph undotted: how much of the stroke the dots removed
            g0, _ = glyph_mask(ch, fp, sz, slot("weight", i, None), float(slot("width", i, 1.0)), float(slot("slant", i, 0.0)),
                               ang, bool(slot("hollow", i, False)), 0, 0, seed)
            solid = int((np.asarray(g0) >= 128).sum()) if g0 is not None else None
        masks.append(g)
        metas.append(dict(font_path=str(fp).replace("\\", "/"), size=round(sz, 2), angle=round(ang, 2), weight_axis=axis,
                          solid_px=solid, dotted_px=int((np.asarray(g) >= 128).sum()) if solid else None))
    spacing = tp.get("spacing", 0.08)
    total = sum(m.width for m in masks) + sum(float(slot("spacing", i, spacing)) * size for i in range(n - 1))
    fit = min(1.0, (w * 0.92) / max(1.0, total), (h * 0.9) / max(1, max(m.height for m in masks)))
    if fit < 1.0:
        masks = [m.resize((max(1, int(m.width * fit)), max(1, int(m.height * fit))), Image.Resampling.BILINEAR) for m in masks]
        size *= fit
        total *= fit
    x = (w - total) / 2 + float(rng.uniform(-1, 1)) * max(0.0, (w - total) / 2) * 0.5
    curve = tp.get("text_on_curve", 0.0)
    cph = float(rng.uniform(0, 2 * math.pi))
    P = int(max(w, h))  # padding: glyph raster outside the canvas still counts for clipping
    dx = dy = None
    warp = tp.get("warp")
    if warp in ("wave", "morph"):
        dx, dy = warp_field(warp, w + 2 * P, h + 2 * P, float(tp.get("warp_amp", 3.0)), float(tp.get("warp_period", 40.0)), seed)
    colours = tp.get("colours")
    base = cv.base
    main_col = colour_of(tp.get("colour", "contrast"), stream(seed, 530), base)
    opacity = float(tp.get("opacity", 1.0))
    shadow = tp.get("shadow")
    emboss = float(tp.get("emboss", 0.0))
    placed = []
    for i, (g, meta) in enumerate(zip(masks, metas)):
        r = stream(seed, 540, i)
        jy = float(slot("jitter", i, 0.0)) * h * float(r.uniform(-1, 1))
        cy = h / 2 + jy + (curve * h * math.sin(2 * math.pi * (x + g.width / 2) / max(1, w) + cph) if curve else 0)
        full = np.zeros((h + 2 * P, w + 2 * P), np.float32)
        gx, gy = int(round(x)) + P, int(round(cy - g.height / 2)) + P
        full[gy:gy + g.height, gx:gx + g.width] = np.asarray(g, np.float32) / 255
        if dx is not None:
            full = remap(full, dx, dy)
        raster = int((full >= 0.5).sum())
        cov = full[P:P + h, P:P + w]
        placed.append((cov, raster, int((cov >= 0.5).sum()), meta, i))
        x += g.width + float(slot("spacing", i, spacing)) * size
    # effects under the glyphs, then the glyphs
    for cov, raster, raster_in, meta, i in placed:
        r = stream(seed, 560, i)
        col = tuple(colours[i]) if colours else (colour_of("contrast", r, base) if tp.get("multicolour") else main_col)
        if shadow:
            from scipy import ndimage
            sdx, sdy = float(shadow.get("dx", 2)), float(shadow.get("dy", 2))
            sh = np.clip(ndimage.shift(cov, (sdy, sdx), order=1), 0, 1)
            cv.apply(sh, colour_of(shadow.get("colour", "similar"), r, base), "text", "py_text:shadow", "glyph_shadow",
                     float(shadow.get("opacity", 0.8)), shadow_of=i)
        if emboss:  # light/dark relief edges offset by a sub-pixel distance
            from scipy import ndimage
            k = float(emboss)
            hi_m = np.clip(ndimage.shift(cov, (-k, -k), order=1) - cov, 0, 1)
            lo_m = np.clip(ndimage.shift(cov, (k, k), order=1) - cov, 0, 1)
            cv.apply(hi_m, (250, 250, 250), "text_effect", "py_text:emboss", None, 0.9, emboss_of=i)
            cv.apply(lo_m, (20, 20, 20), "text_effect", "py_text:emboss", None, 0.9, emboss_of=i)
        if tp.get("outline_colour") and slot("outline", i, 0) and not slot("hollow", i, False):
            pass
        cv.apply(cov, col, "text", "py_text:glyph", None, opacity, char=text[i], char_index=i, primitive="py_glyph",
                 raster_px=raster, raster_px_in=raster_in, mask_px=raster, outline=bool(slot("hollow", i, False)), **meta)
    if tp.get("blur"):
        cv.filter("blur", 7, lambda im: im.filter(ImageFilter.GaussianBlur(float(tp["blur"]))))
    return True, None


def text_box_of(cv):
    ys, xs = np.nonzero(np.isin(cv.opaque, [c["id"] for c in cv.calls if c.get("site") == "py_text:glyph"]))
    if not len(xs):
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


# ------------------------------------------------------------------ python_capri

def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


CAPRI = ROOT / "releases" / "CAPRI_phase123_v1"


def generic_capri_style(rng, w, h):
    """A generic, random phase2 style. Built from scratch: no family profile is read."""
    bgc = [int(x) for x in rng.integers(150, 256, 3)] if rng.random() < 0.8 else [int(x) for x in rng.integers(0, 90, 3)]
    dark = sum(bgc) < 300
    fg = [[int(x) for x in (rng.integers(150, 256, 3) if dark else rng.integers(0, 110, 3))] for _ in range(3)]
    hist = rng.random(12)
    style = dict(size=[w, h], background_palette=[bgc, [int(np.clip(c + rng.integers(-25, 26), 0, 255)) for c in bgc]],
                 foreground_palette=fg, dark_background=dark, gradient_strength=float(rng.choice([0, 0, 40])),
                 gradient_left=[int(x) for x in rng.integers(0, 256, 3)], gradient_right=[int(x) for x in rng.integers(0, 256, 3)],
                 background_block_evidence=dict(state="supported" if rng.random() < 0.25 else "absent", count=int(rng.integers(2, 12)),
                                                candidates=[dict(width_fraction=float(rng.uniform(.05, .3)), height_fraction=float(rng.uniform(.1, .5)),
                                                                 color=[int(x) for x in rng.integers(0, 256, 3)]) for _ in range(4)]),
                 background_edge_fraction=float(rng.uniform(0, .25)), line_count_gain=float(rng.uniform(0, 3)),
                 line_angle_histogram=hist.tolist(), small_component_density=float(rng.uniform(0, .01)),
                 texture_fraction=float(rng.uniform(0, .02)), dot_count_gain=float(rng.uniform(0, 3)),
                 glyph_proposals=[], text_height=float(rng.uniform(.35, .65)), text_tilt_proxy=float(rng.uniform(0, 20)),
                 spacing_gain=float(rng.uniform(.8, 1.15)), multilabel_grid=bool(rng.random() < .15),
                 multilabel_grid_spacing=int(rng.integers(8, 18)), multilabel_blur=bool(rng.random() < .15),
                 classifier_features=dict(straight_interference="present" if rng.random() < .6 else "absent",
                                          curved_interference="present" if rng.random() < .5 else "absent",
                                          dot_scatter="present" if rng.random() < .6 else "absent"),
                 curve_count_gain=float(rng.uniform(.5, 4)))
    return style


def render_capri_phase2(cv, text, seed, params, font_split):
    """Wrap phase2/renderer.py: capture each glyph paste (exact masks) and re-draw each logged op."""
    rmod = _module(CAPRI / "phase2" / "renderer.py", "capri_phase2_renderer")
    rng = stream(seed, 900)
    style = params.get("style") or generic_capri_style(rng, cv.w, cv.h)
    style["size"] = [cv.w, cv.h]
    pool = bank_fonts(font_split, text == text.upper())
    fonts = [str(ROOT / pool[int(stream(seed, 910, i).integers(len(pool)))]["path"]) for i in range(len(text))] \
        if params.get("font_mix") else str(ROOT / pool[int(rng.integers(len(pool)))]["path"])
    pastes = []
    orig_paste = Image.Image.paste
    bg_img = Image.fromarray(cv.img.clip(0, 255).astype(np.uint8)) if cv.chain_seeded else None

    def spy(self, im, box=None, mask=None):
        if mask is not None and self.mode == "RGB" and self.size == (cv.w, cv.h):
            pastes.append(np.asarray(mask, np.float32) / 255)
        return orig_paste(self, im, box, mask)
    real_new = rmod.Image.new
    state = dict(first=True)

    class ImageProxy:  # the renderer's first canvas is the chain background when there is one
        def __getattr__(self, k):
            return getattr(Image, k)

        @staticmethod
        def new(mode, size, color=0):
            if bg_img is not None and state["first"] and mode == "RGB" and tuple(size) == (cv.w, cv.h):
                state["first"] = False
                return bg_img.copy()
            return real_new(mode, size, color)
    if bg_img is not None:
        style["gradient_strength"] = 0
    rmod.Image = ImageProxy()
    Image.Image.paste = spy
    try:
        image, textmask, rec = rmod.render(style, text, fonts, int(seed), int(params.get("variant", rng.integers(0, 4))))
    finally:
        Image.Image.paste = orig_paste
    final = np.asarray(image.convert("RGB"), np.float32)
    # labels: re-draw each logged operation with its own id, in drawing order
    if style.get("gradient_strength", 0) > 30 and bg_img is None:
        cv.field(cv.img, "fill", "py_capri:gradient", "bg_gradient")
    for op in rec["operations"]:
        k = op["kind"]
        if k == "block":
            b = op["box"]
            cv.apply(cv.mask(lambda d, p: d.rectangle((b[0], b[1], b[2] - 1, b[3] - 1), fill=255)), op["color"],
                     "filled_rectangle", "py_capri:block", "bg_blocks")
        elif k in ("line", "classifier_curve"):
            pts = op["points"]
            pts = [tuple(pts[i:i + 2]) for i in range(0, len(pts), 2)] if isinstance(pts[0], (int, float)) else [tuple(p) for p in pts]
            cv.apply(cv.mask(lambda d, p: d.line(pts, fill=255, width=op["width"])), op["color"],
                     "line" if k == "line" else "arc", f"py_capri:{k}", "straight_lines" if k == "line" else "curves")
        elif k == "point":
            pass  # grouped below
        elif k == "multilabel_grid":
            sp = op["spacing"]
            m = cv.mask(lambda d, p: ([d.line((gx, 0, gx, cv.h - 1), fill=255) for gx in range(0, cv.w, sp)],
                                      [d.line((0, gy, cv.w - 1, gy), fill=255) for gy in range(0, cv.h, sp)]))
            cv.apply(m, op["color"], "line", "py_capri:grid", "bg_grid")
    pts = [op["xy"] for op in rec["operations"] if op["kind"] == "point"]
    if pts:
        m = np.zeros((cv.h, cv.w), np.float32)
        for x, y in pts:
            m[y, x] = 1
        cv.apply(m, (0, 0, 0), "pixel", "py_capri:point", "dot_noise")
    for i, (g, cov) in enumerate(zip(rec["glyphs"], pastes[:len(rec["glyphs"])])):
        cv.apply(cov, g["color"], "text", "py_capri:glyph", None, 1.0, char=g["character"], char_index=i,
                 primitive="py_glyph", font_path=str(g["font"]).replace("\\", "/"), angle=round(float(g["angle"]), 2),
                 raster_px=int((cov >= 0.5).sum()), raster_px_in=int((cov >= 0.5).sum()),
                 mask_px=int((cov >= 0.5).sum()), outline=bool(g.get("hollow")))
    if any(op["kind"] == "multilabel_blur" for op in rec["operations"]):
        cv.calls.append(dict(id=cv.next, fn="py:blur", site="py_capri:blur", filter=7, rgba=None, translucent=False, **{"class": "filter"}))
        cv.next += 1
    cv.img = final  # pixels are the renderer's own; the label canvases were drawn above
    return dict(renderer="phase2", style={k: v for k, v in style.items() if k != "background_block_evidence"},
                glyph_effects=[g.get("font_effect") for g in rec["glyphs"]], _blur=int(style.get("multilabel_blur", False)))


def render_capri_phase1(cv, seed, params, font_split):
    """Wrap phase1/reproduction/corpus.py (15 switchable effects). Text is drawn by the renderer
    itself (length uniform 2-8); fonts come from our train/held-out split instead of its bank."""
    cmod = _module(CAPRI / "phase1" / "reproduction" / "corpus.py", "capri_phase1_corpus")
    pool = [dict(path=str(ROOT / r["path"])) for r in bank_fonts(font_split, False)[:60]]
    flags = np.array(params.get("flags") or (stream(seed, 950).random(15) < stream(seed, 951).uniform(.12, .5)), bool)
    if flags[8] and flags[9]:
        flags[9] = False
    if flags[13] and flags[11]:
        flags[11] = flags[11]
    draws, pastes = [], []
    orig_paste = Image.Image.paste
    names = ("line", "ellipse", "rectangle")
    origs = {n: getattr(ImageDraw.ImageDraw, n) for n in names}

    def spy_paste(self, im, box=None, mask=None):
        if mask is not None and self.mode == "RGB" and self.size == (192, 96):
            pastes.append((np.asarray(mask, np.float32) / 255, im))
        return orig_paste(self, im, box, mask)

    def make(nm):
        def f(self, *a, **k):
            if self.im.size == (192, 96) and self.mode == "RGB":
                draws.append((nm, a, k))
            return origs[nm](self, *a, **k)
        return f
    Image.Image.paste = spy_paste
    for nm in names:
        setattr(ImageDraw.ImageDraw, nm, make(nm))
    try:
        arr = cmod.render(int(seed), flags, pool, "train")
    finally:
        Image.Image.paste = orig_paste
        for nm in names:
            setattr(ImageDraw.ImageDraw, nm, origs[nm])
    # text drawn by the renderer: reproduce its deterministic draw to get the label string / font
    rr = np.random.default_rng(np.random.SeedSequence([int(seed), 46]))
    fidx = int(rr.integers(len(pool)))
    rr.integers(30, 48)
    text = "".join(rr.choice(list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"), int(rr.integers(2, 9))))
    # labels in drawing order: gradient, blocks, texture, grid, glyph pastes, interference, blur.
    # Every recorded draw on the canvas is classified by its own geometry.
    if flags[6]:
        cv.field(cv.img, "fill", "py_capri:gradient", "bg_gradient")
    def kind_of(nm, a, k):
        if nm == "rectangle":
            return "block"
        if nm == "ellipse":
            return "dot" if k.get("fill") is not None else "circle"
        pts = a[0]
        if len(pts) == 4 and not isinstance(pts[0], (tuple, list)):
            x0_, y0_, x1_, y1_ = pts
            if (x0_ == x1_ and y0_ == 0) or (y0_ == y1_ and x0_ == 0):
                return "grid"
        return "line" if len(pts) <= 2 else "curve"
    kinds = [kind_of(*d) for d in draws]
    pre = [d for d, kd in zip(draws, kinds) if kd in ("block",)]
    grid = [d for d, kd in zip(draws, kinds) if kd == "grid"]
    post = [(d, kd) for d, kd in zip(draws, kinds) if kd in ("line", "curve", "dot", "circle")]
    for nm, a, k in pre:
        cv.apply(cv.mask(lambda d, p, a=a: d.rectangle(a[0], fill=255)), tuple(k.get("fill") or (0, 0, 0)),
                 "filled_rectangle", "py_capri:block", "bg_blocks")
    if flags[5]:
        cv.field(cv.img, "fill", "py_capri:texture", "bg_image_texture", translucent_label=True)
    if grid:
        m = np.zeros((96, 192), np.float32)
        for nm, a, k in grid:
            m = np.maximum(m, cv.mask(lambda d, p, a=a: d.line(a[0], fill=255, width=1)))
        cv.apply(m, (0, 0, 0), "line", "py_capri:grid", "bg_grid")
    per_glyph = 1 + int(flags[10]) + int(flags[9])
    for j in range(len(text)):
        group = pastes[j * per_glyph:(j + 1) * per_glyph]
        if len(group) < per_glyph:
            break
        k = 0
        if flags[10]:
            cv.apply(group[k][0], (128, 128, 128), "text", "py_capri:shadow", "glyph_shadow", shadow_of=j)
            k += 1
        if flags[9]:
            cv.apply(group[k][0], (128, 128, 128), "text_effect", "py_capri:border", None, border_of=j)
            k += 1
        cov = group[k][0]
        col = group[k][1]
        col = tuple(int(x) for x in col) if isinstance(col, (tuple, list, np.ndarray)) else (0, 0, 0)
        # corpus.py rotates glyph j by rng(80+j).choice([-1,1]) * rng(90+j).uniform(12,35) when flag 13 is on
        ang = (float(np.random.default_rng(np.random.SeedSequence([int(seed), 81 + j])).choice([-1, 1])
                     * np.random.default_rng(np.random.SeedSequence([int(seed), 91 + j])).uniform(12, 35)) if flags[13] else 0.0)
        cv.apply(cov, col, "text", "py_capri:glyph", None, char=text[j], char_index=j, primitive="py_glyph", angle=round(ang, 2),
                 font_path=pool[fidx]["path"].replace("\\", "/"), raster_px=int((cov >= 0.5).sum()),
                 raster_px_in=int((cov >= 0.5).sum()), mask_px=int((cov >= 0.5).sum()), outline=bool(flags[8] or flags[9]))
    head_of = {"line": ("line", "straight_lines"), "curve": ("arc", "curves"), "dot": ("filled_ellipse", "dot_noise"),
               "circle": ("ellipse", "bg_circles")}
    for (nm, a, k2), kd in post:
        cls, head = head_of[kd]
        kw = dict(k2)
        if kw.get("fill") is not None:
            kw["fill"] = 255
        if kw.get("outline") is not None:
            kw["outline"] = 255
        col = k2.get("fill") if k2.get("fill") is not None else k2.get("outline")
        m2 = cv.mask(lambda d, p, nm=nm, a=a, kw=kw: getattr(ImageDraw.ImageDraw, nm)(d, *a, **kw))
        cv.apply(m2, tuple(int(x) for x in (col if col is not None else (0, 0, 0))), cls, f"py_capri:{head}", head)
    if flags[12]:
        cv.calls.append(dict(id=cv.next, fn="py:blur", site="py_capri:blur", filter=7, rgba=None, translucent=False, **{"class": "filter"}))
        cv.next += 1
    cv.img = arr.astype(np.float32)
    return text, dict(renderer="phase1", flags=flags.astype(int).tolist(), _blur=int(flags[12]), _wave_warp=int(flags[11]))


# ------------------------------------------------------------------ public API

def render(source, seed, params, stem, text_on=True, font_split="train", bg=None):
    """Render one Python stage to <stem>.*; returns {'ok', 'reason'}."""
    try:
        return _render(source, seed, params, Path(stem), text_on, font_split, bg)
    except Exception as e:  # noqa: BLE001
        import traceback
        return dict(ok=False, reason=f"{source}: {e!r} {traceback.format_exc()[-300:]}")


def _render(source, seed, params, stem, text_on, font_split, bg):
    w, h = int(params.get("width", 200)), int(params.get("height", 70))
    rng = stream(seed, 1)
    bgcol = tuple(params.get("background", [int(x) for x in rng.integers(170, 256, 3)]))
    if source == "python_capri" and params.get("renderer") == "phase1":
        cv = Canvas(192, 96, (0, 0, 0)) if bg is None else None
        if cv is None:
            return dict(ok=False, reason="phase1 CAPRI renderer draws its own fixed 192x96 canvas: no background input")
        cv.disabled = set(params.get("disable", []))
        text, extra = render_capri_phase1(cv, seed, params, font_split)
        meta = dict(library=source, seed=seed, label=text, params={**params, **extra})
        cv.save(stem, meta)
        return dict(ok=True)
    cv = Canvas(w, h, bgcol, bg=bg)
    cv.disabled = set(params.get("disable", []))
    text = params.get("text") or "".join(TEXT_ALPHABET[int(rng.integers(len(TEXT_ALPHABET)))] for _ in range(int(rng.integers(2, 9))))
    extra = {}
    if source == "python_capri":
        if not text_on:
            return dict(ok=False, reason="python_capri draws text and background together; use it as the text stage")
        extra = render_capri_phase2(cv, text, seed, params, font_split)
    else:
        layers = params.get("layers", [])
        under = [l for l in layers if not l.get("over_text")]
        over = [l for l in layers if l.get("over_text")]
        for li, layer in enumerate(under):
            draw_layer(cv, layer, li, seed)
        if source == "python_text" and text_on:
            ok, why = draw_text(cv, text, params.get("text_params", {}), seed, font_split)
            if not ok:
                return dict(ok=False, reason=why)
            tp = params.get("text_params", {})
            extra = dict(_wave_warp=int(tp.get("warp") == "wave"), _morph_distortion=int(tp.get("warp") == "morph"),
                         _text_on_curve=int(bool(tp.get("text_on_curve"))), _blur=int(bool(tp.get("blur"))),
                         _emboss_bevel=int(bool(tp.get("emboss"))))
        tb = text_box_of(cv)
        for li, layer in enumerate(over):
            draw_layer(cv, layer, 50 + li, seed, text_box=tb)
    label = text if (text_on and source != "python_shapes") else text
    meta = dict(library=source, seed=seed, label=label, params={**{k: v for k, v in params.items() if k != "layers"},
                                                               "layers": params.get("layers", []), **extra})
    cv.save(stem, meta)
    return dict(ok=True)


# ------------------------------------------------------------------ sampling (corpus generator)

SHAPE_KINDS = ["circle", "ellipse", "arc", "polygon", "scribble", "blob", "dots", "line", "grid", "blocks",
               "gradient", "texture", "decoys", "strike", "border"]


def random_layer(rng, kind=None):
    kind = kind or rng.choice(SHAPE_KINDS)
    L = dict(kind=kind, seed=rng.randint(1, 2 ** 31 - 2), opacity=rng.choice([1.0, 1.0, 0.8, 0.6, 0.4]),
             colour=rng.choice(["random", "contrast", "similar"]), region=rng.choice(["full", "full", "top", "bottom"]),
             stroke=rng.choice([1, 1, 2, 3]), filled=rng.random() < 0.4)
    L.update({"circle": dict(count=rng.randint(2, 25), size=[3, 18]), "ellipse": dict(count=rng.randint(2, 20), size=[3, 22]),
              "arc": dict(count=rng.randint(1, 10), size=[8, 40]), "polygon": dict(count=rng.randint(1, 8), size=[6, 25]),
              "scribble": dict(count=rng.randint(1, 6), size=[5, 30]), "blob": dict(count=rng.randint(1, 25), size=[3, 12]),
              "dots": dict(count=rng.randint(20, 900), size=[0, rng.choice([0, 1, 2])]), "line": dict(count=rng.randint(1, 15), size=[15, 120]),
              "grid": dict(count=1, size=[6, 22]), "blocks": dict(count=rng.randint(2, 14), size=[4, 30]),
              "gradient": dict(count=1), "texture": dict(count=1, amplitude=rng.choice([10, 20, 35, 50])),
              "decoys": dict(count=rng.randint(2, 18), size=[8, 18]), "strike": dict(count=rng.randint(1, 3), size=[1, 1], over_text=True),
              "border": dict(count=1, stroke=rng.choice([1, 2, 3]))}[kind])
    if kind in ("line", "dots", "arc", "scribble") and rng.random() < 0.35:
        L["over_text"] = True
    return L


def random_text_params(rng, h):
    tp = dict(size=h * rng.uniform(0.4, 0.75), size_jitter=rng.choice([0, 0, 0.1, 0.25]), rotation=rng.choice([0, 0, 8, 20, 35]),
              jitter=rng.choice([0, 0, 0.05, 0.12]), spacing=rng.choice([0.08, 0.0, -0.08, 0.25, 0.5]),
              slant=rng.choice([0, 0, 0, 12, -12]), width=rng.choice([1.0, 1.0, 0.8, 1.25]),
              weight=rng.choice([None, None, 300, 700, 900]), hollow=rng.random() < 0.1, outline=0,
              multicolour=rng.random() < 0.3, opacity=rng.choice([1.0, 1.0, 1.0, 0.8, 0.6]),
              font_mix=rng.random() < 0.3, dotted=rng.choice([0, 0, 0, 0, 2, 4]),
              warp=rng.choice([None, None, None, "wave", "morph"]), warp_amp=rng.choice([1.5, 2.5, 3.5]), warp_period=rng.choice([25.0, 40.0, 60.0]),
              text_on_curve=rng.choice([0, 0, 0, 0.12, 0.2]), blur=rng.choice([0, 0, 0, 0.7, 1.2]), emboss=rng.choice([0, 0, 0, 1.0, 1.5]))
    if rng.random() < 0.2:
        tp["shadow"] = dict(dx=rng.choice([-3, -2, 2, 3]), dy=rng.choice([2, 3]), colour="similar", opacity=0.8)
    return tp


def sample(rng, src, s, cfg, libs, chained):
    """Settings for one Python-source image (s is the generator's base settings dict)."""
    from feature_bank.generate_corpus import make_text, font_params, stage_seed, layer_libraries, MIDDLE_STAGE
    from feature_bank.render_chain import BG_LIBRARIES
    w, h = rng.randint(150, 300), rng.randint(50, 100)
    tops = sorted(set(BG_LIBRARIES) & set(libs))
    if src == "python_shapes":
        layers = [random_layer(rng) for _ in range(rng.randint(1, 3))]
        shapes = dict(source="python_shapes", seed=stage_seed(rng), params=dict(width=w, height=h, layers=layers))
        if chained or rng.random() < 0.7:  # shapes behind a library: its own processing applies to them
            top = rng.choice(tops)
            p = font_params(rng, top, cfg, s["font_split"])
            text, mode = make_text(rng, top, cfg, p)
            p["text"] = text
            s.update(stages=[shapes, dict(source="php", library=top, seed=stage_seed(rng), params=p)], text_mode=mode)
        else:
            text, mode = make_text(rng, "Gregwar_Captcha", cfg, {})
            tp = random_text_params(rng, h)
            s.update(stages=[dict(source="python_text", seed=stage_seed(rng),
                                  params=dict(width=w, height=h, layers=layers, text=text, text_params=tp))], text_mode=mode)
        return s
    if src == "python_text":
        text, mode = make_text(rng, "Gregwar_Captcha", cfg, {})
        tp = random_text_params(rng, h)
        layers = [random_layer(rng) for _ in range(rng.randint(0, 2))]
        top = dict(source="python_text", seed=stage_seed(rng), params=dict(width=w, height=h, layers=layers, text=text, text_params=tp))
        if chained:  # library background stage(s) under the Python font renderer
            first = dict(source="php", library=rng.choice(layer_libraries(libs)), seed=stage_seed(rng), params={})
            top["params"].pop("width"), top["params"].pop("height")
            s.update(stages=[first, top], text_mode=mode)
        else:
            s.update(stages=[top], text_mode=mode)
        return s
    # python_capri
    text, mode = make_text(rng, "Gregwar_Captcha", cfg, {})
    if rng.random() < 0.5:
        st = dict(source="python_capri", seed=stage_seed(rng), params=dict(renderer="phase1"))
        s.update(stages=[st], text_mode="random")
        return s
    st = dict(source="python_capri", seed=stage_seed(rng), params=dict(renderer="phase2", width=w, height=h, text=text,
                                                                      font_mix=rng.random() < 0.3))
    if chained:
        first = dict(source="php", library=rng.choice(layer_libraries(libs)), seed=stage_seed(rng), params={})
        st["params"].pop("width"), st["params"].pop("height")
        s.update(stages=[first, st], text_mode=mode)
    else:
        s.update(stages=[st], text_mode=mode)
    return s


def recipe_stage(name, recipe, level, rng, cfg):
    """Stage settings for a Python intensity recipe at a level (see python_recipes.py)."""
    from feature_bank import python_recipes
    return python_recipes.stage(name, recipe, level, rng, cfg)
