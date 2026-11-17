"""Step 4: per-image labels for all 46 heads, text targets and the clipping filter.

Inputs: one intercepted render (<stem>.png/.json/.calls.json/.labels*.png), single-library or
chained (calls then carry "stage"; stage metadata is passed in). Everything is measured from the
FINAL image's label maps; requested parameters are used only where a head has no pixel footprint
(global warps, emboss), and then only the library's own logged switch, never a level.

Per head the label is {"present": 1|0|None, "level": 0..10|None, "value": float|None, "px": int}:
  * layer heads: visible pixels of the calls the rule table maps to the head.
      px == 0 -> 0 (the complete draw-call log shows nothing visible), 0 < px < tau -> None,
      px >= tau -> 1. tau per head: docs/feature_bank/HEAD_THRESHOLDS_01.json.
  * glyph style / geometry heads: from the answer glyphs (calls, fonts, final-image pixels).
  * global heads: filter calls, the stage's logged parameter, a final-image measurement, or the
    taxonomy (library cannot draw the effect -> 0; draws it always -> 1; otherwise None).
  * ordinal level: 0 when absent; when present, only through a reviewed recipe curve
    (INTENSITY_CONTROL_MAP + calibration medians) applied to the value measured on this image,
    for the recipe's own library; otherwise None. Never copied from the request.
Unknown is always None (masked), never 0.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from feature_bank.label_maps import read_ids, answer_glyphs

ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = json.loads((ROOT / "docs/feature_bank/FEATURE_BUCKETS_02.json").read_text(encoding="utf-8"))
HEADS = [h["head"] for h in TAXONOMY["heads"]]
ORDINAL_HEADS = [
    "bg_gradient", "bg_image_texture", "bg_blocks", "bg_grid", "bg_circles", "bg_decoy_chars",
    "straight_lines", "curves", "strike_through", "dot_noise", "blob_noise", "scribbles", "noise_over_text",
    "pixelated_dotted_glyphs", "glyph_shadow", "translucent_text", "emboss_bevel",
    "per_char_rotation", "vertical_jitter", "overlap_tight_spacing", "wide_spacing", "size_variation",
    "text_on_curve", "edge_clipping", "wave_warp", "morph_distortion", "blur"]
LAYER_HEADS = ["bg_gradient", "bg_image_texture", "bg_blocks", "bg_grid", "bg_circles", "bg_decoy_chars",
               "border_frame", "straight_lines", "curves", "strike_through", "dot_noise", "blob_noise",
               "scribbles", "glyph_shadow", "fixed_caption"]
BACKGROUND = "_background"  # canvas fills / loaded plain canvases: the bg_solid layer
THRESHOLDS_PATH = ROOT / "docs/feature_bank/HEAD_THRESHOLDS_01.json"
DEFAULT_TAU = {h: 12 for h in LAYER_HEADS}
DEFAULT_TAU.update(bg_gradient=400, bg_image_texture=400, bg_blocks=150, bg_grid=60, border_frame=40, fixed_caption=30,
                   glyph_shadow=25, strike_through=25, scribbles=30, curves=25, bg_circles=25, blob_noise=25, straight_lines=20)
CLIP_MIN_VISIBLE = 0.8

GLYPH_FNS = ("imagettftext", "imagefttext")
BITMAP_FNS = ("imagestring", "imagestringup", "imagechar", "imagecharup")
BLUR_FILTERS = (7, 8)  # IMG_FILTER_GAUSSIAN_BLUR, IMG_FILTER_SELECTIVE_BLUR
PIXELATE_FILTER = 11


@lru_cache(maxsize=1)
def thresholds():
    if THRESHOLDS_PATH.exists():
        return {**DEFAULT_TAU, **json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))["tau_px"]}
    return dict(DEFAULT_TAU)


# ------------------------------------------------------------------ call-site rule table
# Per library: ordered (classes, site regex, head) - first match wins. The site string is
# "<file>:<line> <function>"; eval'd sources are matched by line number ("code:<line> ").
# Keyed on call site, not class alone (e.g. Kohana draws its gradient with imageline).
RULES = {
    "BSN4_P-Captcha": [({"line"}, r"imagegrid", "bg_grid"), ({"arc"}, r"imagecircles", "bg_circles"),
                       ({"rectangle"}, r"", "border_frame"), ({"line"}, r"gen_cpatcha_image", "straight_lines"),
                       ({"filled_arc"}, r"", "dot_noise")],
    "Dev-3bdulrahman_Laravel-Captcha": [({"filled_ellipse"}, r"", "dot_noise")],
    "Gregwar_Captcha": [({"line"}, r"drawLine", "straight_lines"), ({"pixel", "fill"}, r"distort", None)],
    "LordDashMe_php-simple-captcha": [({"line"}, r"drawTextCrossLine", "strike_through"),
                                      ({"text"}, r"drawShadow", "glyph_shadow"),
                                      ({"loaded"}, r"imageCanvas", "bg_image_texture")],
    "carlcraig_fuel-captcha": [({"loaded", "filled_rectangle", "filter"}, r"image_font", None)],
    "dapphp_securimage": [({"filled_rectangle"}, r"drawLines", "curves"), ({"filled_arc"}, r"drawNoise", "dot_noise"),
                          ({"filled_rectangle"}, r"setBackground", BACKGROUND)],
    "dubashi_simple-captcha-php": [({"filled_rectangle"}, r"_gradient", "bg_gradient"), ({"line"}, r"_noise", "dot_noise"),
                                   ({"line"}, r"_nightmare", "straight_lines")],
    "eusonlito_captcha": [({"filled_arc"}, r"noisePoints", "blob_noise"), ({"filled_rectangle"}, r"noiseLines", "straight_lines"),
                          ({"filled_rectangle"}, r"setBackground", BACKGROUND)],
    "igoshev_laravel-captcha": [({"line"}, r"drawScratch", "curves")],
    "innocead_CaptchaBundle": [({"arc"}, r"addNoiseToImage", "bg_circles"), ({"line"}, r"addNoiseToImage", "straight_lines"),
                               ({"pixel"}, r"addNoiseToImage", "dot_noise"), ({"rectangle"}, r"addBorder", "border_frame"),
                               ({"fill"}, r"setBrush", None)],
    "isszz_svg-captcha": [({"arc"}, r"", "curves")],
    "josecl_cool-php-captcha": [({"line"}, r"WriteLine", "strike_through")],
    "kolanos_kohana-captcha": [({"line"}, r"image_gradient", "bg_gradient"), ({"arc"}, r"Captcha_Black", "bg_circles"),
                               ({"filled_ellipse"}, r"Captcha_Alpha", "bg_circles"), ({"line"}, r"Captcha_Basic", "straight_lines")],
    "laminas_laminas-captcha": [({"filled_ellipse"}, r"", "dot_noise")],
    "lifei6671_php-captcha": [({"pixel"}, r"drawSineLine", "curves"), ({"line"}, r"drawLine", "straight_lines")],
    "sameer-shelavale_multi-captcha": [({"line"}, r"", "bg_grid")],
    "sibche2013_pure-captcha": [({"line"}, r"code:24[0-9] ", "bg_gradient"), ({"line"}, r"code:25[0-9] ", "bg_grid"),
                                ({"arc"}, r"", "curves"), ({"line"}, r"code:3[0-9][0-9] ", "straight_lines")],
    "top-think_think-captcha": [({"loaded"}, r"background", "bg_image_texture"), ({"pixel"}, r"writeCurve", "curves")],
    "yasirmturk_simple-php-captcha": [({"loaded"}, r"", BACKGROUND)],  # adapter's generated plain canvas
}
CLASS_DEFAULT = {"line": "straight_lines", "pixel": "dot_noise", "arc": "curves", "ellipse": "bg_circles",
                 "filled_ellipse": "dot_noise", "filled_arc": "dot_noise", "polygon": "scribbles",
                 "filled_polygon": None, "rectangle": None, "filled_rectangle": None, "fill": None, "loaded": None}


def site_matches(call, regex):
    return re.search(regex, call.get("site", "")) is not None


def freecap_head(call, params):
    """freeCap's background mode is an adapter-logged switch; its strokes are that mode's layer."""
    bg = params.get("bg_type")
    if call.get("class") in ("line", "polygon"):
        return {1: "bg_grid", 2: "scribbles"}.get(bg)
    if call.get("class") in ("filled_rectangle", "fill") and bg == 3:
        return "bg_blocks"
    return None


def call_head(call, library, params):
    """Head of a non-text call (None = no head, BACKGROUND = canvas)."""
    if "head" in call:  # Python sources and chain seeds carry their own head
        return call["head"]
    if call.get("chain_bg"):  # the previous chain stage loaded as background: its own calls carry the labels
        return None
    cls = call.get("class") or ("loaded" if str(call.get("fn", "")).startswith("imagecreatefrom") else None)
    if library == "sjbr_sr-freecap":
        h = freecap_head(call, params)
        if h:
            return h
    for classes, rx, head in RULES.get(library, []):
        if cls in classes and site_matches(call, rx):
            return head
    if cls in ("fill", "filled_rectangle", "loaded") and (call.get("canvas_fraction", 0) > 0.4 or cls == "loaded"):
        return BACKGROUND
    return CLASS_DEFAULT.get(cls)


# ------------------------------------------------------------------ helpers

def tax_mode(library, head):
    return (TAXONOMY["libraries"].get(library) or {}).get(head)


def taxonomy_presence(libraries, head):
    """Fallback for global heads: a library that cannot draw the effect gives 0, one that always
    draws it gives 1; param/random libraries without a rule give None."""
    modes = [tax_mode(lib, head) for lib in libraries if not lib.startswith("python_")]
    if any(m == "always" for m in modes):
        return 1
    if any(m in ("param", "random") for m in modes):
        return None
    return 0


def pv(meta, key, default=None):
    """Logged parameter value (param_control first, then the adapter's reported params)."""
    pc = meta.get("param_control")
    if isinstance(pc, dict) and key in pc:
        return pc[key].get("value")
    p = meta.get("params")
    if isinstance(p, dict) and key in p:
        return p[key]
    return default


def flat_params(meta):
    out = dict(meta.get("params") or {}) if isinstance(meta.get("params"), dict) else {}
    pc = meta.get("param_control")
    if isinstance(pc, dict):
        out.update({k: v.get("value") for k, v in pc.items()})
    return out


def tri(value, lo, hi):
    """1 above hi, 0 below lo, None between (ambiguous -> masked)."""
    if value is None:
        return None
    return 1 if value >= hi else 0 if value <= lo else None


# ------------------------------------------------------------------ recipe curves (ordinal levels)

@lru_cache(maxsize=1)
def recipe_curves():
    """Approved recipes: {name: dict(library, feature, metric, medians[11], site, call_class)}."""
    from feature_bank.intensity_corpus import RECIPES
    out = {}
    for path in sorted((ROOT / "docs/feature_bank").glob("INTENSITY_CONTROL_MAP_*.json")):
        for name, entry in json.loads(path.read_text(encoding="utf-8")).items():
            if not (isinstance(entry, dict) and "feature" in entry):
                continue
            cal = entry.get("calibration_medians")
            if cal is None:
                rep = ROOT / entry["calibration"]
                if not rep.exists():
                    continue
                data = json.loads(rep.read_text(encoding="utf-8"))
                cal = data["recipes"][name]["metric"]["median_target_value"]
            r = RECIPES.get(name)
            out[name] = dict(library=entry["library"], feature=entry["feature"], metric=entry["metric"],
                             medians=[float(x) for x in cal], site=r.site if r else entry.get("site"),
                             call_class=r.call_class if r else entry.get("call_class"), source=entry.get("source", "php"))
    return out


def level_from_curve(value, medians):
    """Nearest reviewed level for a measured value. Outside 0.6x..1.4x of the level-10 anchor,
    or present-but-below level 1, the level is unknown (None)."""
    if value is None:
        return None
    m = np.array(medians, float)
    if value <= 0:
        return 0
    if value > 1.4 * m[10]:
        return None
    lv = int(np.argmin(np.abs(m - value)))
    return max(1, lv)


# ------------------------------------------------------------------ main labelling

def load(stem):
    stem = Path(stem)
    meta = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    if Path(str(stem) + ".labels.npz").exists():  # compressed corpus storage
        import gzip
        log = json.loads(gzip.decompress(Path(str(stem) + ".calls.json.gz").read_bytes()).decode("utf-8"))
        z = np.load(str(stem) + ".labels.npz")
        opaque, trans, cls_map = z["opaque"], z["translucent"], z["classes"]
    else:
        log = json.loads(Path(str(stem) + ".calls.json").read_text(encoding="utf-8"))
        opaque = read_ids(str(stem) + ".labels.png")
        trans = read_ids(str(stem) + ".labels_translucent.png")
        cls_map = read_ids(str(stem) + ".labels_classes.png")
    from PIL import Image
    img = np.asarray(Image.open(stem.with_suffix(".png")).convert("RGB"), dtype=np.int16)
    return meta, log, opaque, trans, cls_map, img


def label_image(stem, stages=None, text_mode="random", font_lookup=None):
    """Labels for one rendered image.
    stages: list of dict(library=..., meta=<stage case json>) in chain order (last draws the text);
            default: the single case at <stem>.
    Returns dict(heads, text, clip, layers, notes)."""
    meta, log, opaque, trans, cls_map, img = load(stem)
    H, W = opaque.shape
    calls = {c["id"]: c for c in log["calls"]}
    if stages is None:
        stages = [dict(library=meta["library"], meta=meta)]
    for c in calls.values():  # single-library: every call belongs to stage 0
        c.setdefault("stage", len(stages) - 1)
    text_stage = len(stages) - 1
    text_lib = stages[text_stage]["library"]
    libraries = [s["library"] for s in stages]
    params = [flat_params(s["meta"]) for s in stages]
    label = str(meta["label"])
    text_bit = int(log["meta"]["class_bits"]["text"])

    ids, counts = np.unique(np.concatenate([opaque.ravel(), trans.ravel()]), return_counts=True)
    visible = {int(i): int(c) for i, c in zip(ids, counts) if i}
    op_ids, op_counts = np.unique(opaque.ravel(), return_counts=True)
    vis_opaque = {int(i): int(c) for i, c in zip(op_ids, op_counts) if i}

    # answer glyphs: text calls of the text stage only (earlier stages' text is off)
    text_calls = [c for c in calls.values() if c.get("class") == "text" and c.get("stage") == text_stage]
    word_path = text_lib == "NikolaiT_SVG-Captcha"  # whole answer is one path: no per-character glyphs
    glyph_ids = [] if word_path else answer_glyphs(text_calls, label, visible)
    glyph_set = set(glyph_ids) | ({c["id"] for c in text_calls} if word_path else set())

    # ---- layer pixels per head
    layer_px = {h: 0 for h in LAYER_HEADS}
    layer_ids = {h: [] for h in LAYER_HEADS}
    background_px, unmapped_px = 0, 0
    stage_px = {}
    drawn = {h: 0 for h in LAYER_HEADS}   # non-suppressed calls mapped to each head
    answer_by_char = {}
    for gid in glyph_ids:
        answer_by_char.setdefault(str(calls[gid].get("char", "")).lower(), []).append(gid)
    for cid, c in calls.items():
        px = visible.get(cid, 0)
        if c.get("class") == "filter" or cid in glyph_set:
            continue
        lib = libraries[c.get("stage", 0)] if c.get("stage", 0) < len(libraries) else text_lib
        if c.get("class") == "text":
            if "head" in c:
                head = c["head"]
            elif c.get("stage") != text_stage:
                head = "bg_decoy_chars"
            else:
                head = None
                for classes, rx, h in RULES.get(lib, []):
                    if "text" in classes and site_matches(c, rx):
                        head = h
                        break
                if head is None:
                    if len(str(c.get("text") or "")) > 10:  # a caption strip (freeCap, Fuel), not a decoy character
                        head = "fixed_caption"
                    else:
                        head = "glyph_shadow" if is_shadow(c, calls, answer_by_char, opaque, trans) else "bg_decoy_chars"
        else:
            head = call_head(c, lib, params[c.get("stage", 0)] if c.get("stage", 0) < len(params) else {})
        c["_head"] = head
        if head == BACKGROUND:
            background_px += px
        elif head in layer_px:
            layer_px[head] += px
            if not c.get("suppressed"):
                drawn[head] += 1
            sp = stage_px.setdefault(str(c.get("stage", 0)), {})
            sp[head] = sp.get(head, 0) + px
            layer_ids[head].append(cid)
        elif head is None and px:
            unmapped_px += px
    tau = thresholds()
    heads = {h: dict(present=None, level=None, value=None, px=None) for h in HEADS}
    for h in LAYER_HEADS:
        px = layer_px[h]
        # drawn but with no visible label pixels (e.g. labels lost through an unmirrored copy):
        # unknown, not absent
        heads[h].update(px=px, present=(None if drawn[h] else 0) if px == 0 else (1 if px >= tau[h] else None),
                        value=round(px / (H * W), 6))

    # ---- answer glyph geometry from the final maps
    boxes = []
    for i, gid in enumerate(glyph_ids):
        m = (opaque == gid) | (trans == gid)
        ys, xs = np.nonzero(m)
        c = calls[gid]
        if len(xs) == 0:
            boxes.append(None)
            continue
        colour = np.median(img[m], axis=0).astype(int).tolist()
        boxes.append(dict(gid=gid, char=c.get("char"), x0=int(xs.min()), y0=int(ys.min()), x1=int(xs.max()), y1=int(ys.max()),
                          px=int(len(xs)), colour=colour, angle=c.get("angle"), size=c.get("size"),
                          font_path=c.get("font_path"), fn=c.get("fn"), rgba=c.get("rgba"),
                          raster_px=c.get("raster_px"), raster_px_in=c.get("raster_px_in"),
                          glyph_scope=bool(c.get("glyph_scope")), primitives=c.get("primitives"),
                          primitive=c.get("primitive"), mask_px=c.get("mask_px"), outline=c.get("outline")))
    located = [b for b in boxes if b]

    # ---- clipping filter
    if word_path:
        wpx = sum(visible.get(c["id"], 0) for c in text_calls)
        clip = dict(ok=wpx > 0, reasons=[] if wpx > 0 else ["word path not visible"], glyphs=[], scale=1.0,
                    note="single word path: no per-character glyphs (holdout library)")
    else:
        clip = clipping(boxes, glyph_ids, label, calls, opaque, trans, cls_map, text_bit, log["meta"], W, H)

    # ---- glyph style heads
    fonts = [font_info(b, font_lookup) for b in located]
    ttf = [b for b in located if b["fn"] in GLYPH_FNS or b.get("primitive") == "py_glyph"]
    bitmap = [b for b in located if b["fn"] in BITMAP_FNS]
    svg = [b for b in located if str(b["fn"]).startswith("svg:")]
    scope = [b for b in located if b["glyph_scope"]]
    n = len(located)
    if n:
        styles = [f["style"] for f in fonts]
        heads["bitmap_font"]["present"] = int(len(bitmap) > 0)
        heads["sprite_glyphs"]["present"] = int(any("imagecopy" in (b["primitives"] or []) for b in scope))
        vec = any(set(b["primitives"] or []) & {"imageline", "imagepolygon", "imagefilledpolygon"} for b in scope) or \
            (len(svg) > 0) or text_lib == "NikolaiT_SVG-Captcha"
        heads["vector_stroke_glyphs"]["present"] = int(vec)
        if ttf:
            ttf_styles = [font_info(b, font_lookup)["style"] for b in ttf]
            if any(s is None for s in ttf_styles):
                heads["hollow_outline_glyphs"]["present"] = None
                heads["filled_glyphs"]["present"] = None
            else:
                hollow = any(s == "hollow" for s in ttf_styles) or any(b.get("outline") for b in ttf)
                heads["hollow_outline_glyphs"]["present"] = int(hollow)
                heads["filled_glyphs"]["present"] = int(any(s in ("filled", "distressed") for s in ttf_styles) and not
                                                        all(b.get("outline") for b in ttf))
                dotted = any(s == "dotted" for s in ttf_styles)
                distressed = any(s == "distressed" for s in ttf_styles)
                heads["pixelated_dotted_glyphs"]["present"] = 1 if dotted else (None if distressed else 0)
        else:
            heads["hollow_outline_glyphs"]["present"] = 1 if text_lib == "NikolaiT_SVG-Captcha" else (0 if (bitmap or scope) else None)
            heads["filled_glyphs"]["present"] = 0 if (bitmap or scope or svg) else None
            heads["pixelated_dotted_glyphs"]["present"] = 0 if (bitmap or scope) else None
        # P-Captcha pixelate filter breaks the glyph strokes into blocks
        if any(c.get("class") == "filter" and c.get("filter") == PIXELATE_FILTER and int((c.get("args") or [0])[0] or 0) > 1
               for c in calls.values()):
            heads["pixelated_dotted_glyphs"]["present"] = 1
        # font mixture: distinct fonts among answer glyphs
        keys = [f["key"] for f in fonts]
        if all(k is not None for k in keys):
            heads["font_mixture"]["present"] = int(len(set(keys)) > 1)
        elif bitmap and text_lib == "sjbr_sr-freecap":
            heads["font_mixture"]["present"] = None  # freeCap picks one of five GDF files per letter; not logged
        # multicolour: final-image glyph colours
        cols = np.array([b["colour"] for b in located])
        spread = float(np.max(np.abs(cols[:, None, :] - cols[None, :, :]))) if n > 1 else 0.0
        heads["multicolor_text"].update(present=tri(spread, 20, 48), value=spread)
        # translucent text: alpha of the answer glyph calls (GD alpha 0..127)
        alphas = [b["rgba"][3] for b in located if b.get("rgba")]
        if alphas and len(alphas) == n:
            a = float(np.mean(alphas)) / 127
            heads["translucent_text"].update(present=tri(a, 0.0, 0.08), value=round(a, 4))
        else:  # SVG opacity / glyph scopes: no per-glyph alpha logged
            heads["translucent_text"]["present"] = None
    # ---- geometry heads
    geometry(heads, located, calls, W, H, clip)
    # ---- noise over text: later-drawn interference topmost on text pixels
    first_glyph = min(glyph_ids) if glyph_ids else None
    noise_heads = {"straight_lines", "curves", "dot_noise", "blob_noise", "scribbles", "strike_through", "bg_circles", "bg_grid"}
    text_cov = (cls_map & text_bit) > 0
    if first_glyph is not None and text_cov.any():
        over_ids = [cid for cid, c in calls.items() if c.get("_head") in noise_heads and cid > first_glyph
                    and c.get("stage") == text_stage]
        over = np.isin(opaque, over_ids) & text_cov
        frac = float(over.sum() / text_cov.sum())
        heads["noise_over_text"].update(px=int(over.sum()), value=round(frac, 6),
                                        present=0 if over.sum() == 0 else (1 if over.sum() >= tau.get("noise_over_text", 12) and frac >= 0.01 else None))
    # ---- background / global heads
    globals_(heads, calls, stages, params, libraries, img, opaque, trans, glyph_ids, background_px, H, W)
    # ---- text content heads
    content(heads, label, text_mode)
    # ---- ordinal levels
    levels(heads, stages, calls, opaque, trans, located, glyph_ids, H, W, stage_px, stem)
    text = text_targets(label, boxes, fonts_for(boxes, font_lookup), W, H)
    return dict(heads=heads, text=text, clip=clip,
                layers=dict(px={h: layer_px[h] for h in LAYER_HEADS if layer_px[h]}, background_px=background_px,
                            unmapped_px=unmapped_px, by_stage=stage_px),
                glyph_ids=glyph_ids)


def is_shadow(c, calls, answer_by_char, opaque, trans):
    """A non-answer text call is a shadow when it repeats an answer character within a few pixels."""
    ch = str(c.get("char", "")).lower()
    for gid in answer_by_char.get(ch, []):
        g = calls[gid]
        if c.get("angle") == g.get("angle") and c.get("size") == g.get("size"):
            return True
    return False


@lru_cache(maxsize=1)
def bank_index():
    from feature_bank.font_split import load_bank, font_key
    rows = load_bank()
    by_key = {font_key(r): r for r in rows}
    return by_key


def font_key_of(path):
    """Bank key (directory name) for a bank font or a per-case copy named <key>.ttf; else 'lib:<file>'."""
    if not path:
        return None
    p = Path(str(path))
    idx = bank_index()
    if p.name == "font.ttf" and p.parent.name in idx:
        return p.parent.name
    if p.stem in idx:
        return p.stem
    return "lib:" + p.name


def font_info(b, font_lookup=None):
    from feature_bank.font_styles import style_of
    path = b.get("font_path")
    if not path:
        return dict(key=None, style=None, split=None)
    key = font_key_of(path)
    split = bank_index()[key].get("split", "train") if key in bank_index() else "library"
    real = Path(path)
    if not real.exists() and key in bank_index():
        real = ROOT / bank_index()[key]["path"]
    st = style_of(str(real)) if real.exists() else {}
    return dict(key=key, style=st.get("reviewed_style"), split=split)


def fonts_for(boxes, font_lookup):
    return [font_info(b, font_lookup) if b else None for b in boxes]


def clipping(boxes, glyph_ids, label, calls, opaque, trans, cls_map, text_bit, log_meta, W, H):
    """Reject when an answer glyph is missing or < 80% visible (canvas clipping or erasure).
    Visible = own label pixels + pixels of its box still carrying the text coverage bit under a
    later non-text call (noise drawn over the glyph is not clipping)."""
    reasons, per = [], []
    if len(glyph_ids) != len(label):
        reasons.append(f"answer_glyphs_matched {len(glyph_ids)}/{len(label)}")
    warped = bool(log_meta.get("copy_writes")) or bool(log_meta.get("unmirrored"))
    text_cov = (cls_map & text_bit) > 0
    answer = set(glyph_ids)
    raw = []
    from scipy import ndimage
    cov_comp = ndimage.label(text_cov, structure=np.ones((3, 3), int))[0]
    for i, b in enumerate(boxes):
        if b is not None:
            b["_i"] = i
        if b is None:
            reasons.append(f"glyph {i} missing")
            per.append(None)
            continue
        raster, raster_in = b.get("raster_px") or b.get("mask_px"), b.get("raster_px_in")
        # occluded part of the glyph: the text-coverage region connected to its visible pixels (coverage
        # bits survive occlusion) whose top owner is something drawn later than the glyph
        own = (opaque == b["gid"]) | (trans == b["gid"])
        region = cov_comp == -1
        hit = np.unique(cov_comp[own & text_cov])
        if len(hit):
            region = np.isin(cov_comp, hit[hit > 0])
        occl = int((region & (opaque > b["gid"])).sum())  # incl. later neighbours (overlap is not clipping)
        in_frac = (raster_in / raster) if raster and raster_in is not None else None
        raw.append((b["px"] + occl) / raster if raster and not warped else None)
        per.append(dict(in_canvas=None if in_frac is None else round(in_frac, 3), touches_edge=
                        b["x0"] <= 0 or b["y0"] <= 0 or b["x1"] >= W - 1 or b["y1"] >= H - 1, _b=b))
    # libraries that draw large and resample down shrink every glyph alike: normalise by the
    # image's median so only a glyph that lost more than its neighbours counts as clipped
    known = [r for r in raw if r is not None]
    scale = float(np.median(known)) if known else 1.0
    scale = scale if scale < 0.9 else 1.0
    for g, r in zip([g for g in per if g is not None], raw):
        b = g.pop("_b")
        in_frac = g["in_canvas"]
        vis_frac = None if r is None else min(1.0, r / scale)
        g["visible"] = None if vis_frac is None else round(vis_frac, 3)
        touches = g["touches_edge"]
        i = b["_i"]
        if in_frac is not None and in_frac < CLIP_MIN_VISIBLE:
            reasons.append(f"glyph {i} {b['char']!r} {in_frac:.2f} inside canvas")
        elif vis_frac is not None and vis_frac < CLIP_MIN_VISIBLE:
            reasons.append(f"glyph {i} {b['char']!r} {vis_frac:.2f} visible")
        elif in_frac is None and touches and b["px"] < 15:
            reasons.append(f"glyph {i} {b['char']!r} at edge with {b['px']} px")
    return dict(ok=not reasons, reasons=reasons, glyphs=per, scale=round(scale, 3))


def geometry(heads, located, calls, W, H, clip):
    n = len(located)
    if n == 0:
        return
    angles = [b["angle"] for b in located]
    if all(a is not None for a in angles):
        ang = [min(abs(float(a)) % 360, 360 - abs(float(a)) % 360) for a in angles]
        heads["per_char_rotation"].update(value=round(float(np.mean(ang)), 3), present=tri(float(np.mean(ang)), 0.5, 4.0))
    hts = np.array([b["y1"] - b["y0"] + 1 for b in located], float)
    cy = np.array([(b["y0"] + b["y1"]) / 2 for b in located], float)
    order = sorted(located, key=lambda b: (b["x0"] + b["x1"]) / 2)
    if n >= 2:
        jit = float(np.std(cy) / max(1.0, np.median(hts)))
        heads["vertical_jitter"].update(value=round(jit, 4), present=tri(jit, 0.04, 0.12))
        gaps, overlaps = [], []
        wid = np.array([b["x1"] - b["x0"] + 1 for b in order], float)
        for a, b in zip(order, order[1:]):
            g = b["x0"] - a["x1"] - 1
            gaps.append(g)
            overlaps.append(max(0, -g) / max(1, min(a["x1"] - a["x0"] + 1, b["x1"] - b["x0"] + 1)))
        mw = float(np.median(wid))
        ov = float(np.max(overlaps))
        heads["overlap_tight_spacing"].update(value=round(ov, 4), present=1 if ov >= 0.15 else (0 if min(gaps) >= 1 else None))
        gap = float(np.median(gaps)) / max(1.0, mw)
        heads["wide_spacing"].update(value=round(gap, 4), present=tri(gap, 0.3, 0.8))
        sizes = [b["size"] for b in located]
        if all(s is not None for s in sizes):
            sv = (max(sizes) - min(sizes)) / max(1e-6, min(sizes))
            heads["size_variation"].update(value=round(float(sv), 4), present=tri(float(sv), 0.0, 0.15))
    edge = any(g and g.get("in_canvas") is not None and g["in_canvas"] < 0.999 for g in clip["glyphs"])
    touch = any(g and g.get("touches_edge") for g in clip["glyphs"])
    known = all(g and g.get("in_canvas") is not None for g in clip["glyphs"])
    heads["edge_clipping"]["present"] = 1 if edge else (0 if known and not touch else (None if touch else (0 if known else None)))


def globals_(heads, calls, stages, params, libraries, img, opaque, trans, glyph_ids, background_px, H, W):
    def any_stage(fn):
        # Deformation / glyph-effect heads describe the answer strokes: only the text stage (last)
        # can set them. A background stage's own warp or blur moved only its own layers.
        vals = [fn(libraries[-1], params[-1], stages[-1]["meta"])]
        if any(v == 1 for v in vals):
            return 1
        if any(v is None for v in vals):
            return None
        return 0

    # wave warp: the library's own warp switch (pixels are moved by copy loops)
    def wave(lib, p, meta):
        if lib == "Gregwar_Captcha":
            return int(bool(p.get("distortion")))
        if lib == "dapphp_securimage":
            return int(float(p.get("perturbation") or 0) > 0)
        if lib == "carlcraig_fuel-captcha":
            return int(bool(p.get("distort")) and float(p.get("distort_amplitude") or 0) > 0)
        if lib == "josecl_cool-php-captcha":
            return int(float(p.get("x_amplitude") or 0) > 0 or float(p.get("y_amplitude") or 0) > 0)
        if lib.startswith("python_"):
            return p.get("_wave_warp", 0)
        return taxonomy_presence([lib], "wave_warp")
    heads["wave_warp"]["present"] = any_stage(wave)

    def morph(lib, p, meta):
        if lib == "dubashi_simple-captcha-php":
            return int(bool(p.get("randSym")) or bool(p.get("randPos")))
        if lib.startswith("python_"):
            return p.get("_morph_distortion", 0)
        return taxonomy_presence([lib], "morph_distortion")
    heads["morph_distortion"]["present"] = any_stage(morph)

    blur_filter = any(c.get("class") == "filter" and c.get("filter") in BLUR_FILTERS and c.get("stage", 0) == len(stages) - 1
                      for c in calls.values())

    def blur(lib, p, meta):
        if lib == "sjbr_sr-freecap":
            return int(bool(p.get("blur_bg")))
        if lib == "carlcraig_fuel-captcha":
            return int(bool(p.get("font_gaussian_blur")))
        if lib == "josecl_cool-php-captcha":
            return int(bool(p.get("blur")))
        if lib == "innocead_CaptchaBundle":
            return int(bool(p.get("effect_blur")))
        if lib.startswith("python_"):
            return p.get("_blur", 0)
        return 0 if tax_mode(lib, "blur") is None else None
    b = any_stage(blur)
    heads["blur"]["present"] = 1 if blur_filter else b

    def emboss(lib, p, meta):
        if lib == "dubashi_simple-captcha-php":
            return int(bool(p.get("emboss")))
        if lib.startswith("python_"):
            return p.get("_emboss_bevel", 0)
        return taxonomy_presence([lib], "emboss_bevel")
    heads["emboss_bevel"]["present"] = any_stage(emboss)

    def curve_text(lib, p, meta):
        if lib.startswith("python_"):
            return p.get("_text_on_curve", 0)
        return taxonomy_presence([lib], "text_on_curve")
    heads["text_on_curve"]["present"] = curve_text(libraries[-1], params[-1], stages[-1]["meta"])

    def caption(lib, p, meta):
        if lib == "carlcraig_fuel-captcha":
            return int(bool(p.get("message")))
        return taxonomy_presence([lib], "fixed_caption")
    cap = any_stage(caption)
    if heads["fixed_caption"]["present"] in (0, None):
        heads["fixed_caption"]["present"] = cap if cap is not None else heads["fixed_caption"]["present"]

    # greyscale: measured on the final image
    spread = int(np.max(np.max(img, axis=2) - np.min(img, axis=2)))
    heads["greyscale"].update(present=1 if spread <= 3 else 0, value=spread)

    # dark background: luminance of non-glyph pixels vs answer glyph pixels
    lum = img @ np.array([0.299, 0.587, 0.114])
    gmask = np.isin(opaque, glyph_ids) if glyph_ids else np.zeros_like(opaque, bool)
    bg_l = float(np.median(lum[~gmask])) if (~gmask).any() else None
    fg_l = float(np.median(lum[gmask])) if gmask.any() else None
    if bg_l is not None and fg_l is not None:
        heads["bg_dark"].update(value=round(bg_l, 1),
                                present=1 if (bg_l < 100 and fg_l > bg_l + 30) else (0 if bg_l > 130 or fg_l < bg_l else None))
    # border frame from a rule is a layer head; taxonomy fallback otherwise
    # solid background: canvas fill present and no other background layer
    others = [heads[h]["present"] for h in ("bg_gradient", "bg_image_texture", "bg_blocks")]
    if any(o == 1 for o in others):
        heads["bg_solid"]["present"] = 0
    elif background_px + int(((opaque == 0) & (trans == 0)).sum()) >= 0.3 * H * W and all(o == 0 for o in others):
        heads["bg_solid"]["present"] = 1
    else:
        heads["bg_solid"]["present"] = None


def content(heads, label, text_mode):
    s = str(label)
    d = any(ch.isdigit() for ch in s)
    u = any(ch.isupper() for ch in s)
    lo = any(ch.islower() for ch in s)
    heads["digits_only"]["present"] = int(d and not u and not lo)
    heads["uppercase_only"]["present"] = int(u and not lo)
    heads["lowercase_only"]["present"] = int(lo and not u)
    heads["mixed_case_alnum"]["present"] = int(u and lo)
    heads["pronounceable_word"]["present"] = int(text_mode == "pronounceable") if text_mode in ("random", "pronounceable") else None
    heads["length_short_le4"]["present"] = int(len(s) <= 4)
    heads["length_long_ge7"]["present"] = int(len(s) >= 7)


RECIPE_DECIDES = {"per_char_rotation", "vertical_jitter", "overlap_tight_spacing", "translucent_text", "size_variation",
                  "text_on_curve", "wave_warp", "morph_distortion", "blur", "pixelated_dotted_glyphs", "emboss_bevel",
                  "bg_gradient", "bg_image_texture"}


def levels(heads, stages, calls, opaque, trans, located, glyph_ids, H, W, stage_px=None, stem=None):
    """Ordinal levels: 0 if verified absent; a reviewed recipe curve for that recipe's own library;
    otherwise None."""
    curves = recipe_curves()
    libs = {s["library"]: i for i, s in enumerate(stages)}
    for h in ORDINAL_HEADS:
        st = heads[h]
        # glyph-geometry / alpha heads with a reviewed recipe metric: the recipe's own measurement
        # on this image decides presence and level for that library (the generic geometry
        # thresholds are only used where no reviewed metric applies)
        if h in RECIPE_DECIDES:
            hit = False
            for name, rc in curves.items():
                if rc["feature"] != h or libs.get(rc["library"]) != len(stages) - 1:
                    continue
                if not recipe_applies(name, rc, stages):
                    continue
                v = measure_any(name, rc, len(stages) - 1, calls, opaque, trans, located, glyph_ids, H, W, stages, stem)
                lv = level_from_curve(v, rc["medians"]) if v is not None else None
                if lv is not None:
                    st.update(present=int(lv >= 1), level=lv, level_source=name, level_value=round(v, 6))
                    hit = True
                    break
            if hit:
                continue
        if st["present"] == 0:
            st["level"] = 0
            continue
        if st["present"] != 1:
            continue
        for name, rc in curves.items():
            if rc["feature"] != h or rc["library"] not in libs:
                continue
            stage = libs[rc["library"]]
            if h in LAYER_HEADS and stage_px:
                tot = sum(d.get(h, 0) for d in stage_px.values())
                own = stage_px.get(str(stage), {}).get(h, 0)
                if tot and own < 0.95 * tot:
                    continue  # other stages draw the same effect: the recipe's scale no longer applies
            elif len(stages) > 1 and stage != len(stages) - 1:
                continue
            if not recipe_applies(name, rc, stages):
                continue
            v = measure_any(name, rc, stage, calls, opaque, trans, located, glyph_ids, H, W, stages, stem)
            if v is None:
                continue
            lv = level_from_curve(v, rc["medians"])
            if lv is not None:
                st.update(level=lv, level_source=name, level_value=round(v, 6))
                break


def recipe_applies(name, rc, stages):
    """PHP recipes apply to their library wherever it draws; Python recipes only to their own
    recipe-mode renders (their metrics assume the recipe's fixed settings)."""
    if rc.get("source", "php") == "php":
        return True
    p = stages[-1]["meta"].get("params") or {}
    return p.get("_recipe") == name


def measure_any(name, rc, stage, calls, opaque, trans, located, glyph_ids, H, W, stages, stem):
    if rc.get("source", "php") == "php":
        return measure_recipe(rc, stage, calls, opaque, trans, located, glyph_ids, H, W, stages)
    from feature_bank import python_recipes
    r = python_recipes.RECIPES[name]
    if r.metric.startswith("paired"):
        pv = stages[stage]["meta"].get("_paired_value")
        return pv.get(rc["feature"]) if isinstance(pv, dict) else None
    return python_recipes.measure(r, stem)


def measure_recipe(rc, stage, calls, opaque, trans, located, glyph_ids, H, W, stages):
    m = rc["metric"]
    tgt = [cid for cid, c in calls.items() if c.get("stage", stage) == stage and c.get("class") == rc["call_class"]
           and rc["site"] and rc["site"] in c.get("site", "")]
    if m == "pixels":
        if not tgt:
            return 0.0
        return float((np.isin(opaque, tgt) | np.isin(trans, tgt)).sum() / (H * W))
    if m == "angle":
        ang = [min(abs(float(calls[g].get("angle") or 0)) % 360, 360 - abs(float(calls[g].get("angle") or 0)) % 360)
               for g in glyph_ids]
        return float(np.mean(ang)) if ang else None
    if m == "alpha":
        al = [float((calls[g].get("rgba") or [0, 0, 0, 0])[3]) for g in glyph_ids]
        return float(np.mean(al) / 127) if al else None
    if m == "glyph_overlap":
        target = np.isin(opaque, tgt) | np.isin(trans, tgt)
        region = np.zeros_like(target)
        for b in located:
            region[b["y0"]:b["y1"] + 1, b["x0"]:b["x1"] + 1] = True
        return float((target & region).sum() / max(1, region.sum()))
    # paired metrics (glyph_displacement, xcompression, difference) need the level-0 twin;
    # the corpus generator supplies them as stage meta "_paired_value"
    return stages[stage]["meta"].get("_paired_value", {}).get(rc["feature"]) if isinstance(stages[stage]["meta"].get("_paired_value"), dict) else None


def text_targets(label, boxes, fonts, W, H):
    s = str(label)
    slots = []
    order = sorted([i for i, b in enumerate(boxes) if b], key=lambda i: (boxes[i]["x0"] + boxes[i]["x1"]) / 2)
    for k, i in enumerate(order[:8]):
        b, f = boxes[i], fonts[i] or {}
        style = f.get("style")
        slots.append(dict(slot=k, char=b["char"], answer_index=i,
                          box=[round(b["x0"] / W, 4), round(b["y0"] / H, 4), round((b["x1"] + 1) / W, 4), round((b["y1"] + 1) / H, 4)],
                          box_px=[b["x0"], b["y0"], b["x1"], b["y1"]],
                          angle=None if b["angle"] is None else float(b["angle"]),
                          colour=b["colour"],
                          outline=(True if (style == "hollow" or b.get("outline")) else False if style in ("filled", "distressed", "dotted") else None),
                          font=f.get("key"), font_split=f.get("split"), font_style=style, size=b.get("size")))
    while len(slots) < 8:
        slots.append(None)
    return dict(text=s, length=len(s),
                mix=dict(digits=sum(ch.isdigit() for ch in s), lower=sum(ch.islower() for ch in s), upper=sum(ch.isupper() for ch in s)),
                slots=slots, fonts_per_char=[(fonts[i] or {}).get("key") if boxes[i] else None for i in range(len(boxes))])
