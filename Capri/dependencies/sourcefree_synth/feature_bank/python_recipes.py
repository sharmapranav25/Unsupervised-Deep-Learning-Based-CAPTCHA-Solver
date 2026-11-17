"""0-10 intensity ladders driven through the Python sources (python_shapes / python_text).

Same method and gates as feature_bank.intensity_corpus (PHP recipes):
  calibrate: render levels 0..10 on paired seeds (everything except the knob fixed per seed),
             measure the visible effect on the FINAL image's label maps (or against the image's own
             level-0 twin for geometric/global effects), write contact sheets and an automatic report
             (zero clear, glyphs present, per-seed Spearman >= 0.9, >= 10 distinct medians);
  review:    a human look at every sheet is required before a recipe may enter a corpus;
  export:    docs/feature_bank/INTENSITY_CONTROL_MAP_02.json (python recipes, calibration medians
             embedded, source tag) - read by feature_bank.heads for measured levels.

  python -m feature_bank.python_recipes calibrate --out runs/py_intensity_cal_NEW [--recipes ...] [--seeds 6]
  python -m feature_bank.python_recipes review --calibration <report.json> --recipes a b --note "..."
  python -m feature_bank.python_recipes revoke --calibration <report.json> --recipes a --note "..."
  python -m feature_bank.python_recipes export --calibration <report.json> [...]
"""
import argparse
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from feature_bank.bank02_smoke import ROOT

PY_TEXT_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZabdefghjkmnqrtuyz2345678"


@dataclass(frozen=True)
class PRecipe:
    name: str
    source: str            # python_shapes | python_text
    feature: str
    kind: str              # shape kind or text knob
    metric: str            # pixels | alpha | angle | sizes | dotted | paired_iou | paired_disp | paired_xcomp | paired_sharp | bg_range | bg_texture
    note: str = ""
    call_class: str = ""
    site: str = ""


def lin(level, lo, hi):
    return lo + (hi - lo) * level / 10.0


# ---- knob ladders (level -> stage params). Everything not named here is fixed per seed.

def shape_layer(kind, level, base, **kw):
    if level == 0:
        return []
    return [dict(kind=kind, seed=base["layer_seed"], **kw)]


def ladder_params(r, level, base):
    """Stage params for recipe r at level, from per-seed base settings."""
    tp = dict(base["text_params"])
    layers = []
    k = r.kind
    if k == "gradient":
        layers = shape_layer("gradient", level, base, colour=base["text_colour"], opacity=lin(level, 0, 0.92), steep=True)
    elif k == "texture":
        layers = shape_layer("texture", level, base, amplitude=lin(level, 0, 75))
    elif k == "blocks":
        layers = shape_layer("blocks", level, base, count=int(round(lin(level, 0, 24))), size=[5, 18], colour=base["text_colour"],
                             region="text", opacity=0.85)
    elif k == "grid":
        layers = [] if level == 0 else [dict(kind="grid", seed=base["layer_seed"], size=[max(4, int(round(lin(level, 22, 4)))),
                                                                                       max(5, int(round(lin(level, 22, 4))))],
                                             stroke=1 if level < 7 else 2, colour=base["text_colour"], opacity=0.9)]
    elif k == "decoys":
        layers = shape_layer("decoys", level, base, count=int(round(lin(level, 0, 70))), size=[12, 28], colour=base["text_colour"], opacity=0.9)
    elif k == "scribble":
        layers = shape_layer("scribble", level, base, count=int(round(lin(level, 0, 12))), size=[5, 30], stroke=1, npts=4, colour=base["text_colour"],
                             region="text", over_text=True)
    elif k in ("line", "dots", "blob", "arc", "circle", "strike"):
        spec = {"line": dict(count=int(round(lin(level, 0, 30))), size=[20, 90], stroke=2, over_text=True, colour=base["text_colour"]),
                "dots": dict(count=int(round(lin(level, 0, 1600))), size=[0, 1], colour=base["text_colour"], over_text=True),
                "blob": dict(count=int(round(lin(level, 0, 60))), size=[3, 8], colour=base["text_colour"], over_text=True, filled=True),
                "arc": dict(count=int(round(lin(level, 0, 22))), size=[10, 45], stroke=2, colour=base["text_colour"], over_text=True),
                "circle": dict(count=int(round(lin(level, 0, 40))), size=[4, 16], stroke=2, colour=base["text_colour"]),
                "strike": dict(count=1, size=[1, 1], stroke=max(1, int(round(lin(level, 0, 9)))), colour=base["text_colour"], over_text=True)}[k]
        layers = shape_layer(k, level, base, **spec)
    elif k == "over_text_dots":
        layers = shape_layer("dots", level, base, count=int(round(lin(level, 0, 900))), size=[0, 1], colour=base["text_colour"],
                             over_text=True, region="text")
    elif k == "shadow":
        if level:
            o = lin(level, 0.8, 9.0)  # sub-pixel offsets; level 10: an overlapping second copy
            tp["shadow"] = dict(dx=o, dy=0.6 * o, colour=base["text_colour"], opacity=lin(level, 0.55, 1.0))
    elif k == "emboss":
        tp["emboss"] = 0 if level == 0 else lin(level, 0.5, 6.0)
    elif k == "dotted":
        tp["dotted"] = level  # fragmentation 1..10 (dot radius shrinks relative to spacing)
    elif k == "size":
        tp["size_jitter"] = lin(level, 0, 0.72)
    elif k == "curve":
        tp["text_on_curve"] = lin(level, 0, 0.32)
    elif k == "wave":
        tp.update(warp="wave" if level else None, warp_amp=lin(level, 0, 8.0), warp_period=base["period"])
    elif k == "morph":
        tp.update(warp="morph" if level else None, warp_amp=lin(level, 0, 9.0))
    elif k == "blur":
        tp["blur"] = lin(level, 0, 4.0)
    elif k == "opacity":
        tp["opacity"] = lin(level, 1.0, 0.13)
    elif k == "rotation":
        tp["rotation"] = lin(level, 0, 75)
    elif k == "jitter":
        tp["jitter"] = lin(level, 0, 0.4)
    elif k == "tight":
        tp["spacing"] = lin(level, 0.12, -0.26)
    return dict(width=base["width"], height=base["height"], background=base["background"], text=base["text"],
                text_params=tp, layers=layers)


RECIPES = {r.name: r for r in (
    # 16 heads without an approved library ladder
    PRecipe("py_gradient", "python_shapes", "bg_gradient", "gradient", "bg_range"),
    PRecipe("py_texture", "python_shapes", "bg_image_texture", "texture", "bg_texture"),
    PRecipe("py_blocks", "python_shapes", "bg_blocks", "blocks", "pixels", call_class="filled_rectangle", site="py_shapes:blocks"),
    PRecipe("py_grid", "python_shapes", "bg_grid", "grid", "pixels", call_class="line", site="py_shapes:grid"),
    PRecipe("py_decoys", "python_shapes", "bg_decoy_chars", "decoys", "pixels", call_class="text", site="py_shapes:decoys"),
    PRecipe("py_scribbles", "python_shapes", "scribbles", "scribble", "pixels", call_class="polygon", site="py_shapes:scribble"),
    PRecipe("py_dotted", "python_text", "pixelated_dotted_glyphs", "dotted", "dotted"),
    PRecipe("py_shadow", "python_text", "glyph_shadow", "shadow", "paired_diff"),
    PRecipe("py_emboss", "python_text", "emboss_bevel", "emboss", "pixels", call_class="text_effect", site="py_text:emboss"),
    PRecipe("py_size", "python_text", "size_variation", "size", "sizes"),
    PRecipe("py_curve", "python_text", "text_on_curve", "curve", "paired_disp"),
    PRecipe("py_wave", "python_text", "wave_warp", "wave", "paired_iou"),
    PRecipe("py_morph", "python_text", "morph_distortion", "morph", "paired_iou"),
    PRecipe("py_blur", "python_text", "blur", "blur", "paired_sharp"),
    # second sources for the 11 approved heads
    PRecipe("py_lines", "python_shapes", "straight_lines", "line", "pixels", call_class="line", site="py_shapes:line"),
    PRecipe("py_dots", "python_shapes", "dot_noise", "dots", "pixels", call_class="filled_ellipse", site="py_shapes:dots"),
    PRecipe("py_blobs", "python_shapes", "blob_noise", "blob", "pixels", call_class="filled_polygon", site="py_shapes:blob"),
    PRecipe("py_curves", "python_shapes", "curves", "arc", "pixels", call_class="arc", site="py_shapes:arc"),
    PRecipe("py_circles", "python_shapes", "bg_circles", "circle", "pixels", call_class="ellipse", site="py_shapes:circle"),
    PRecipe("py_over_text", "python_shapes", "noise_over_text", "over_text_dots", "glyph_overlap", call_class="filled_ellipse", site="py_shapes:dots"),
    PRecipe("py_strike", "python_shapes", "strike_through", "strike", "pixels", call_class="line", site="py_shapes:strike"),
    PRecipe("py_translucent", "python_text", "translucent_text", "opacity", "alpha"),
    PRecipe("py_rotation", "python_text", "per_char_rotation", "rotation", "angle"),
    PRecipe("py_jitter", "python_text", "vertical_jitter", "jitter", "paired_jitter"),
    PRecipe("py_tight", "python_text", "overlap_tight_spacing", "tight", "paired_xcomp"),
)}
MASKED = {
    "text_on_curve": "A curved baseline (Python ladder py_curve, ordered and measured) never makes the answer hard to "
                     "read: its level-10 endpoint fails on every reviewed seed. RealCaptcha draws curved text as a fixed "
                     "style. Presence stays labelled; the level is masked.",
    "wide_spacing": "Gaps between complete glyphs never make the answer hard to read: wide spacing has no level-10 "
                    "endpoint under the agreed scale (library ladder Innocead char_px_spacing failed for the same reason; "
                    "a Python spacing knob cannot change that). Presence stays labelled; the 0-10 level is masked.",
    "emboss_bevel": "Light/dark relief edges (Python ladder py_emboss, offsets up to 6 px) never make the answer hard "
                    "to read, and the measured relief stops rising at large offsets; Dubashi exposes only an on/off "
                    "emboss. Presence stays labelled; the level is masked.",
    "edge_clipping": "A level-10 'hard to read' clipping would cut glyphs below the 0.8 visible-fraction rule that the "
                     "clipping filter enforces; any ladder within the filter stays easy to read. Presence stays labelled; "
                     "the level is masked.",
}


def base_settings(seed):
    """Per-seed fixed settings (text, fonts, canvas, colours) shared by every level of a ladder."""
    rng = random.Random(f"pyrecipe|{seed}")
    from feature_bank.font_styles import style_of
    from feature_bank.py_sources import bank_fonts
    fonts = [f for f in bank_fonts("train", False) if style_of(str(ROOT / f["path"])).get("reviewed_style") == "filled"]
    n = rng.randint(4, 7)
    text = "".join(rng.choice(PY_TEXT_ALPHABET) for _ in range(n))
    w, h = rng.randint(190, 260), rng.randint(64, 84)
    bg = [rng.randint(225, 255) for _ in range(3)]
    tc = [rng.randint(20, 90) for _ in range(3)]
    return dict(text=text, width=w, height=h, background=bg, text_colour=tc, layer_seed=rng.randint(1, 2 ** 31 - 2),
                period=float(rng.choice([30, 40, 50])), seed=seed,
                text_params=dict(size=h * 0.62, fonts=fonts[rng.randrange(len(fonts))]["path"], colour=tc, spacing=0.12))


# ---- measurement on the final image

def glyph_masks(lab_stem):
    from feature_bank.heads import load
    meta, log, opaque, trans, cls_map, img = load(lab_stem)
    calls = [c for c in log["calls"] if c.get("site") == "py_text:glyph"]
    ids = [c["id"] for c in calls]
    return meta, log, opaque, trans, img, calls, ids


def measure(r, stem, twin=None):
    meta, log, opaque, trans, img, gcalls, gids = glyph_masks(stem)
    H, W = opaque.shape
    m = r.metric
    if m == "pixels":
        ids = [c["id"] for c in log["calls"] if c.get("class") == r.call_class and c.get("site") == r.site and not c.get("suppressed")]
        return float((np.isin(opaque, ids) | np.isin(trans, ids)).sum() / (H * W))
    if m == "glyph_overlap":
        ids = [c["id"] for c in log["calls"] if c.get("site") == r.site and not c.get("suppressed")]
        target = np.isin(opaque, ids) | np.isin(trans, ids)
        region = np.zeros_like(target)
        for gid in gids:
            ys, xs = np.nonzero((opaque == gid) | (trans == gid))
            if len(xs):
                region[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
        return float((target & region).sum() / max(1, region.sum()))
    if m == "alpha":
        return float(np.mean([c["rgba"][3] for c in gcalls]) / 127) if gcalls else None
    if m == "angle":
        return float(np.mean([abs(c.get("angle") or 0) for c in gcalls])) if gcalls else None
    if m == "sizes":
        s = [c["size"] for c in gcalls]
        return float((max(s) - min(s)) / max(1e-6, min(s))) if s else None
    if m == "dotted":
        return float(np.mean([1 - c["mask_px"] / max(1, c.get("solid_px") or c["mask_px"]) for c in gcalls])) if gcalls else None
    lum = img @ np.array([0.299, 0.587, 0.114])
    gm = np.isin(opaque, gids) | np.isin(trans, gids)
    from scipy import ndimage as _nd
    band = _nd.binary_dilation(gm, iterations=3)
    if m == "bg_range":
        bgl = lum[~band]
        return float((np.percentile(bgl, 97) - np.percentile(bgl, 3)) / 255)
    if m == "bg_texture":
        if not any(c.get("site") == "py_shapes:texture" and not c.get("suppressed") for c in log["calls"]):
            return 0.0
        from scipy import ndimage
        hp = lum - ndimage.uniform_filter(lum, 5)
        return float(np.std(hp[~_nd.binary_dilation(band, iterations=3)]) / 255)
    if twin is None:
        return None
    t_meta, t_log, t_op, t_tr, t_img, t_calls, t_ids = glyph_masks(twin)
    tm = np.isin(t_op, t_ids) | np.isin(t_tr, t_ids)
    if m == "paired_diff":  # mean image change against the level-0 twin
        return float(np.mean(np.abs(img.astype(np.float32) - t_img.astype(np.float32))) / 255)
    if m == "paired_iou":
        inter, union = (gm & tm).sum(), (gm | tm).sum()
        return float(1 - inter / max(1, union))
    if m in ("paired_disp", "paired_jitter"):
        def centres(op, tr, ids):
            out = []
            for i in ids:
                ys, xs = np.nonzero((op == i) | (tr == i))
                out.append(ys.mean() if len(ys) else np.nan)
            return np.array(out)
        a, b = centres(opaque, trans, gids), centres(t_op, t_tr, t_ids)
        if len(a) != len(b) or np.isnan(a).any() or np.isnan(b).any():
            return None
        return float(np.mean(np.abs(a - b)) / H)
    if m == "paired_xcomp":
        def span(op, tr, ids):
            cs = []
            for i in ids:
                ys, xs = np.nonzero((op == i) | (tr == i))
                if len(xs):
                    cs.append(xs.mean())
            return (max(cs) - min(cs)) if cs else 0
        s0, s1 = span(t_op, t_tr, t_ids), span(opaque, trans, gids)
        return float(max(0.0, (s0 - s1) / max(1.0, s0)))
    if m == "paired_sharp":
        from scipy import ndimage
        t_lum = t_img @ np.array([0.299, 0.587, 0.114])
        band = ndimage.binary_dilation(tm, iterations=2)
        g0 = np.hypot(ndimage.sobel(t_lum, 0), ndimage.sobel(t_lum, 1))[band].mean()
        g1 = np.hypot(ndimage.sobel(lum, 0), ndimage.sobel(lum, 1))[band].mean()
        return float(max(0.0, 1 - g1 / max(1e-6, g0)))
    return None


# ---- rendering a ladder

def render_level(r, seed, level, out_dir):
    from feature_bank.render_chain import render
    base = base_settings(seed)
    st = dict(source="python_text", seed=seed, params=ladder_params(r, level, base))
    stem = out_dir / f"seed_{seed}" / f"level_{level:02d}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    res = render(dict(stages=[st], font_split="train"), stem)
    return stem, res


def stage(name, recipe_entry, level, rng, cfg):
    """Corpus generator: a python recipe stage at a level (base drawn from the image's rng)."""
    r = RECIPES[name]
    seed = rng.randint(1, 2 ** 31 - 2)
    return dict(source="python_text", seed=seed, params=dict(ladder_params(r, level, base_settings(seed)), _recipe=name))


def twin_value(r, stem, stage_settings, work):
    """Paired metrics: render the level-0 twin of a stage (same seed) and measure against it."""
    if not r.metric.startswith("paired"):
        return measure(r, stem)
    from feature_bank.render_chain import render
    base = base_settings(stage_settings["seed"])
    t = dict(source="python_text", seed=stage_settings["seed"], params=ladder_params(r, 0, base))
    tstem = Path(work) / "pytwin0"
    res = render(dict(stages=[t], font_split="train"), tstem)
    if not res["ok"]:
        return None
    return measure(r, stem, tstem)


def calibrate(args):
    from feature_bank.intensity_corpus import monotonic_report
    from feature_bank.heads import label_image
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    names = args.recipes or list(RECIPES)
    report = dict(method="python_recipes calibrate", recipes={})
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(n, args.first_seed + s, lv, str(out)) for n in names for s in range(args.seeds) for lv in range(11)]
    rows = {}
    with ProcessPoolExecutor(28) as ex:
        for n, row in ex.map(_job, jobs):
            rows.setdefault(n, []).append(row)
    for n in names:
        rr = sorted(rows.get(n, []), key=lambda x: (x["seed"], x["level"]))
        # ladders with a failed level are dropped whole (a seed either has all 11 levels or none)
        bad = {x["seed"] for x in rr if not x["ok"]}
        rr_ok = [x for x in rr if x["seed"] not in bad]
        metric = monotonic_report(rr_ok) if rr_ok else dict(passes=False, reason="no complete ladder")
        metric["complete_ladders"] = len({x["seed"] for x in rr_ok})
        metric["failed_ladders"] = sorted(bad)
        report["recipes"][n] = dict(source=RECIPES[n].source, feature=RECIPES[n].feature, metric_name=RECIPES[n].metric,
                                    metric=metric, rows=rr)
        sheet(n, rr_ok, out)
        print(n, {k: metric.get(k) for k in ("passes", "spearman", "min_seed_spearman", "distinct_levels", "zero_clear", "glyphs_ok")}, flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")


def _job(args):
    name, seed, level, out = args
    from feature_bank.heads import label_image
    r = RECIPES[name]
    out = Path(out) / name
    stem, res = render_level(r, seed, level, out)
    if not res["ok"]:
        return name, dict(ok=False, seed=seed, level=level, reason=res["reason"][:200])
    twin = Path(str(stem.parent / "level_00")) if r.metric.startswith("paired") else None
    if twin is not None and not Path(str(twin) + ".png").exists():
        render_level(r, seed, 0, out)
    v = measure(r, stem, twin)
    lab = res["labels"]
    return name, dict(ok=v is not None, seed=seed, level=level, target_value=v, glyph_ok=lab["clip"]["ok"],
                      image=str(stem.with_suffix(".png")), present=lab["heads"][r.feature]["present"])


def sheet(name, rows, out):
    by = {}
    for r in rows:
        by.setdefault(r["seed"], []).append(r)
    tw, th = 220, 110
    canvas = Image.new("RGB", (tw * 11, th * max(1, len(by))), "white")
    d = ImageDraw.Draw(canvas)
    for y, (seed, rs) in enumerate(sorted(by.items())):
        for r in sorted(rs, key=lambda r: r["level"]):
            with Image.open(r["image"]) as im:
                im = im.convert("RGB")
                im.thumbnail((tw - 6, th - 22))
                canvas.paste(im, (r["level"] * tw + 3, y * th + 18))
            d.text((r["level"] * tw + 3, y * th + 2), f"L{r['level']} {r['target_value']:.3f} g={int(r['glyph_ok'])}", fill="black")
    canvas.save(out / f"{name}_sheet.png")


def review(args, value=True):
    p = Path(args.calibration)
    data = json.loads(p.read_text(encoding="utf-8"))
    for n in args.recipes:
        m = data["recipes"][n]["metric"]
        if value and not m.get("passes"):
            raise SystemExit(f"automatic gate failed: {n}")
        m.update(reviewed=value, review_note=args.note)
    p.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    print("reviewed" if value else "revoked", args.recipes)


def export(args):
    out = {}
    for path in args.calibration:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for n, e in data["recipes"].items():
            m = e["metric"]
            if m.get("passes") and m.get("reviewed"):
                r = RECIPES[n]
                out[n] = dict(library="python_text", source=r.source, feature=r.feature, control=r.kind, metric=r.metric,
                              call_class=r.call_class, site=r.site, calibration=str(path),
                              calibration_medians=m["median_target_value"], calibration_endpoint=m["median_target_value"][10],
                              review_note=m.get("review_note"),
                              params_by_level={str(k): f"ladder_params({r.kind!r}, level={k}, base_settings(seed))" for k in range(11)})
    dst = ROOT / "docs/feature_bank/INTENSITY_CONTROL_MAP_02.json"
    dst.write_text(json.dumps(dict(sorted(out.items())), indent=1), encoding="utf-8")
    print("exported", len(out), "python recipes over", len({v["feature"] for v in out.values()}), "heads ->", dst)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--out", required=True)
    c.add_argument("--recipes", nargs="*")
    c.add_argument("--seeds", type=int, default=6)
    c.add_argument("--first-seed", type=int, default=41000)
    for nm in ("review", "revoke"):
        r = sub.add_parser(nm)
        r.add_argument("--calibration", required=True)
        r.add_argument("--recipes", nargs="+", required=True)
        r.add_argument("--note", required=True)
    e = sub.add_parser("export")
    e.add_argument("--calibration", nargs="+", required=True)
    a = ap.parse_args()
    if a.cmd == "calibrate":
        calibrate(a)
    elif a.cmd == "review":
        review(a, True)
    elif a.cmd == "revoke":
        review(a, False)
    else:
        export(a)


if __name__ == "__main__":
    main()
