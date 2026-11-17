"""Label maps for SVG-rendering libraries (the SVG counterpart of adapters/bank02/intercept.php).

The library's own SVG is never changed. For labelling, a copy is re-rendered with resvg where
each drawable element is painted in a unique id colour (fill and/or stroke as in the original),
opacity removed and antialiasing off (shape-rendering=crispEdges, text-rendering=optimizeSpeed).
Painter's order gives the opaque id map; per-class renders (other classes hidden) give the
coverage bits, so occluded parts are still covered. Outputs match the GD path:
<stem>.labels.png, <stem>.labels_translucent.png (zeros), <stem>.labels_classes.png, <stem>.calls.json.

Element roles are library-specific (rules below); glyph elements are assigned to label
characters in drawing order, or left-to-right when the library shuffles its elements (isszz).
"""
import copy
import json
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

from feature_bank.bank02_smoke import RESVG

SVG_NS = "http://www.w3.org/2000/svg"
CLASS_BITS = {"line": 1, "rectangle": 2, "filled_rectangle": 4, "ellipse": 8, "filled_ellipse": 16, "arc": 32,
              "filled_arc": 64, "polygon": 128, "filled_polygon": 256, "pixel": 512, "fill": 1024, "text": 2048, "loaded": 4096}
DRAWABLE = {"path", "rect", "line", "circle", "ellipse", "polyline", "polygon", "text"}


def _tag(el):
    return el.tag.split("}")[-1]


def _inline_style(el):
    """Return simple SVG presentation properties from an element's inline CSS."""
    props = {}
    for item in el.get("style", "").split(";"):
        if ":" in item:
            key, value = item.split(":", 1)
            props[key.strip()] = value.strip()
    return props


def _paint_property(el, name):
    # Inline CSS has precedence over the matching SVG presentation attribute.
    return _inline_style(el).get(name, el.get(name))


def role(library, el, index):
    """(class, is_glyph, primitive) for an element."""
    t = _tag(el)
    has_fill = _paint_property(el, "fill") not in (None, "none")
    has_stroke = _paint_property(el, "stroke") not in (None, "none")
    if t == "rect":
        return "fill", False, "rect"
    if t == "line":
        return "line", False, "line"
    if t in ("circle", "ellipse"):
        return "filled_ellipse" if has_fill else "ellipse", False, t
    if t == "text":
        return "text", True, "text"
    if t == "path":
        if library == "NikolaiT_SVG-Captcha":
            # The entire answer is a single stroked path, not one glyph per path.
            return "text", False, "word_path"
        if library == "isszz_svg-captcha":
            return ("text", True, "glyph_path") if has_fill and not has_stroke else ("arc", False, "bezier")
        if library == "splitbrain_dokuwiki-plugin-captcha":
            return "text", True, "glyph_path"
        return ("arc", False, "path") if has_stroke and not has_fill else ("filled_polygon", False, "path")
    return "polygon", False, t


def _hex(i):
    return "#%06x" % i


def _render(root, out_png, coverage=False):
    """Id map (alpha >= 128), or with coverage=True a mask of every pixel the paint touches."""
    with tempfile.TemporaryDirectory() as td:
        svg = Path(td) / "l.svg"
        ET.ElementTree(root).write(svg, encoding="utf-8", xml_declaration=True)
        p = subprocess.run([str(RESVG), str(svg), str(out_png)], capture_output=True, text=True, timeout=60)
        if p.returncode:
            raise RuntimeError(p.stderr[:300])
    a = np.asarray(Image.open(out_png).convert("RGBA"), dtype=np.uint32)
    if coverage:
        return a[..., 3] > 0
    ids = (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]
    ids[a[..., 3] < 128] = 0
    return ids


def label_svg(stem, library, label, char_order="drawing"):
    stem = Path(stem)
    ET.register_namespace("", SVG_NS)
    tree = ET.parse(stem.with_suffix(".svg"))
    root = tree.getroot()
    elements = [el for el in root.iter() if _tag(el) in DRAWABLE]
    calls = []
    for i, el in enumerate(elements, start=1):
        cls, glyph, prim = role(library, el, i)
        source_id = el.get("data-bank-source-id", str(i))
        calls.append(dict(id=i, fn="svg:" + _tag(el), site=f"{library}:svg#{source_id}", class_=cls, glyph=glyph, primitive=prim,
                          rgba=None, translucent=False))

    def painted(keep=None, crisp=True):
        r = copy.deepcopy(root)
        if crisp:
            r.set("shape-rendering", "crispEdges")
            r.set("text-rendering", "optimizeSpeed")
        els = [e for e in r.iter() if _tag(e) in DRAWABLE]
        for c, e in zip(calls, els):
            # Keep the effective paint and stroke geometry before dropping inline CSS.
            # In SVG-Captcha these exist only in style, including fill:none.
            for k, v in _inline_style(e).items():
                e.set(k, v)
            e.attrib.pop("style", None)
            for k in ("opacity", "fill-opacity", "stroke-opacity", "filter"):
                e.attrib.pop(k, None)
            if keep is not None and c["id"] not in keep:
                e.set("display", "none")
                continue
            col = _hex(c["id"])
            # SVG fills shapes black when no fill attribute is given, so only an explicit "none"
            # leaves the interior unpainted (lines/polylines have no interior to fill)
            if e.get("fill") != "none" and _tag(e) not in ("line",):
                e.set("fill", col)
            if e.get("stroke") not in (None, "none"):
                e.set("stroke", col)
        return r

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        opaque = _render(painted(), td / "all.png")
        cov = np.zeros_like(opaque)
        for cls, bit in CLASS_BITS.items():
            keep = {c["id"] for c in calls if c["class_"] == cls}
            if keep:
                # antialiased render: the coverage bit marks every pixel the class touches at all
                cov |= np.where(_render(painted(keep, crisp=False), td / f"{cls}.png", coverage=True), bit, 0).astype(np.uint32)
        glyph_ids = [c["id"] for c in calls if c["glyph"]]
        if char_order == "left_to_right":
            lefts = {}
            for gid in glyph_ids:
                m = _render(painted({gid}), td / f"g{gid}.png") > 0
                xs = np.nonzero(m.any(axis=0))[0]
                lefts[gid] = xs.min() if len(xs) else 1e9
            glyph_ids.sort(key=lambda g: lefts[g])
    chars = list(str(label))
    for k, gid in enumerate(glyph_ids):
        c = calls[gid - 1]
        c["char"] = chars[k] if k < len(chars) else "?"
        c["char_index"] = k
        c["order_key"] = k  # reading order (elements may be shuffled in the document)
    for c in calls:
        c["class"] = c.pop("class_")

    def save(a, name):
        rgb = np.stack([(a >> 16) & 255, (a >> 8) & 255, a & 255], axis=-1).astype(np.uint8)
        Image.fromarray(rgb).save(str(stem) + name)
    save(opaque, ".labels.png")
    save(np.zeros_like(opaque), ".labels_translucent.png")
    save(cov, ".labels_classes.png")
    meta = dict(label_canvas="svg", copy_writes=0, unmirrored=[], palette_canvas=False, class_bits=CLASS_BITS,
                disabled=[], calls=len(calls), glyph_count_matches_label=len(glyph_ids) == len(chars))
    Path(str(stem) + ".calls.json").write_text(json.dumps(dict(meta=meta, calls=calls)), encoding="utf-8")
    return meta


CHAR_ORDER = {"isszz_svg-captcha": "left_to_right"}


def apply_disable(svg_path, library, classes):
    """Per-class switch for SVG renderers: remove elements whose role is in `classes` from the
    library's SVG before rasterising (the untouched original is kept as <stem>.orig.svg).
    Painting order of the remaining elements is unchanged, so the result is exactly the image
    without those elements. Returns the number of removed elements."""
    svg_path = Path(svg_path)
    ET.register_namespace("", SVG_NS)
    tree = ET.parse(svg_path)
    root = tree.getroot()
    parents = {c: p for p in root.iter() for c in p}
    removed = 0
    for i, el in enumerate([e for e in root.iter() if _tag(e) in DRAWABLE], start=1):
        cls, _, _ = role(library, el, i)
        if cls in classes:
            parents[el].remove(el)
            removed += 1
        else:
            el.set("data-bank-source-id", str(i))
    svg_path.with_suffix(".orig.svg").write_bytes(svg_path.read_bytes())
    tree.write(svg_path, encoding="utf-8", xml_declaration=True)
    return removed
