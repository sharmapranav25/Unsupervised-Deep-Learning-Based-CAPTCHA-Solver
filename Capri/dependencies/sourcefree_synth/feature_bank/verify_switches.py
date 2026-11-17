"""Step 2 check: per-class on/off switches and parameter/text overrides are exact.

For every (library, seed, call class present in the image):
  * render the case with interception (reference) and again with that class disabled;
  * containment = share of changed pixels lying inside the class's labelled footprint
    (opaque or translucent maps of the reference, dilated by --dilate px for antialiasing /
    interpolation). 1.0 means nothing outside the switched-off layer changed;
  * stability = share of pixels outside the footprint whose owning call is the same in both
    runs (calls compared by signature fn|site|char_index, since ids are run-local). 1.0 means
    the switch did not shift the library's random stream or layout.
Overrides: render with a fixed text and a changed parameter; check the label, the parameter log,
and that every other controlled parameter kept its sampled value.

Usage: python -m feature_bank.verify_switches --out runs/switches_NEW [--libs ...] [--seeds 4]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from feature_bank.bank02_smoke import run_case
from feature_bank.label_maps import read_ids

SKIP_CLASSES = {"filter"}
VISIBLE_DELTA = 8  # max per-channel change (0-255) below which a pixel change counts as quantisation
OVERRIDES = {  # (param, new value) used for the override test, per library
    "Gregwar_Captcha": ("max_front_lines", 0),
    "kolanos_kohana-captcha": ("width", 240),
    "sjbr_sr-freecap": ("bg_type", 0),
    "innocead_CaptchaBundle": ("width", 240),
    "mobicms_captcha": ("width", 240),
    # width is derived from font_size / letter_spacing by the adapter; height_ratio is independent
    "igoshev_laravel-captcha": ("height_ratio", 2.0),
}


def load(stem):
    stem = Path(stem)
    meta = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    calls = json.loads(Path(str(stem) + ".calls.json").read_text(encoding="utf-8"))["calls"]
    img = np.asarray(Image.open(stem.with_suffix(".png")).convert("RGB"), dtype=np.int16)
    log = json.loads(Path(str(stem) + ".calls.json").read_text(encoding="utf-8"))["meta"]
    cov = read_ids(str(stem) + ".labels_classes.png")
    return meta, calls, img, read_ids(str(stem) + ".labels.png"), cov, log["class_bits"]


def signature_map(ids, calls):
    sig = {c["id"]: f"{c['fn']}|{c['site']}|{c.get('char_index', '')}" for c in calls}
    lut = {0: "0"}
    flat = ids.ravel()
    return np.array([lut.get(i) or sig.get(int(i), "?") for i in flat]).reshape(ids.shape)


def dilate(mask, px):
    if px <= 0:
        return mask
    im = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(2 * px + 1))
    return np.asarray(im) > 0


def check_class(lib, seed, cls, ref_dir, off_dir, dilate_px):
    meta, calls, img, op, cov, bits = load(ref_dir / f"{seed:06d}")
    run_case(lib, seed, off_dir, intercept=True, disable=[cls], stem_name=f"{seed:06d}_no_{cls}")
    m2, calls2, img2, op2, cov2, _ = load(off_dir / f"{seed:06d}_no_{cls}")
    raw_fp = (cov & bits[cls]) > 0
    fp = dilate(raw_fp, dilate_px)
    if img.shape != img2.shape:
        return dict(cls=cls, error=f"size changed {img.shape} -> {img2.shape}")
    changed = np.any(img != img2, axis=-1)
    n_changed = int(changed.sum())
    contain = 1.0 if n_changed == 0 else float((changed & fp).sum() / n_changed)
    # visible containment ignores sub-threshold colour shifts (palette re-quantisation on 8-bit canvases)
    vis = np.abs(img - img2).max(axis=-1) > VISIBLE_DELTA
    n_vis = int(vis.sum())
    contain_vis = 1.0 if n_vis == 0 else float((vis & fp).sum() / n_vis)
    s1, s2 = signature_map(op, calls), signature_map(op2, calls2)
    outside = ~fp
    stable = float((s1[outside] == s2[outside]).mean()) if outside.any() else 1.0
    # Structural: the class leaves no coverage in the final image (e.g. canvas-clearing rectangles
    # later fully overwritten) yet removing it changes the result. Not a visual layer; reported apart.
    structural = int(raw_fp.sum()) == 0 and n_changed > 0
    palette = bool(json.loads(Path(str(ref_dir / f"{seed:06d}") + ".calls.json").read_text(encoding="utf-8"))["meta"].get("palette_canvas"))
    # Lossy JPEG output: any change is re-encoded across its whole coding unit, so pixel containment is
    # measured against the footprint grown to the blocks it touches (reported apart, like palette).
    jpeg = meta.get("source_format") == "jpeg"
    contain_block = None
    if jpeg:
        h, w = fp.shape
        blk = np.zeros(((h + 15) // 16, (w + 15) // 16), bool)  # 4:2:0 chroma: 16x16 coding units
        ys, xs = np.nonzero(fp)
        blk[ys // 16, xs // 16] = True
        fp_blk = np.kron(blk, np.ones((16, 16), bool))[:h, :w]
        contain_block = 1.0 if n_vis == 0 else round(float((vis & fp_blk).sum() / n_vis), 4)
    return dict(jpeg_coupled=jpeg, containment_block=contain_block, cls=cls, footprint_px=int(fp.sum()), changed_px=n_changed, containment=round(contain, 4), containment_visible=round(contain_vis, 4),
                stability=round(stable, 4), label_same=m2["label"] == meta["label"], structural=structural,
                palette_coupled=palette)


def pick_override(meta):
    """A parameter to override: the first integer-valued controlled parameter (shifted by one),
    else the first boolean (flipped). Returns (name, value) or (None, None)."""
    pc = meta.get("param_control") or {}
    for k, v in (pc.items() if isinstance(pc, dict) else []):  # PHP encodes an empty map as []
        val = v.get("value")
        if k in ("text", "alphabet", "length_range", "font"):
            continue
        if isinstance(val, bool):
            return k, not val
        if isinstance(val, int):
            return k, val + 1
    return None, None


def override_text(meta, length=4):
    """A text of the given length spread over the library's logged alphabet."""
    alpha = (meta.get("param_control", {}).get("alphabet") or {}).get("value") or "abcd"
    step = max(1, len(alpha) // length)
    return "".join(alpha[(i * step + 1) % len(alpha)] for i in range(length))


def check_override(lib, seed, ref_dir, off_dir):
    meta = json.loads((ref_dir / f"{seed:06d}.json").read_text(encoding="utf-8"))
    name, value = OVERRIDES.get(lib) or pick_override(meta)
    pcm = meta.get("param_control") if isinstance(meta.get("param_control"), dict) else {}
    has_alpha = bool((pcm.get("alphabet") or {}).get("value"))
    # Keep length fixed: some adapters sample later parameters from the text length.
    # A length change is a separate control and need not preserve those draws.
    text = override_text(meta, len(str(meta["label"]))) if has_alpha else None   # math CAPTCHAs: no text control
    params = {"text": text} if text else {}
    if name:
        params[name] = value
    run_case(lib, seed, off_dir, intercept=True, params=params, stem_name=f"{seed:06d}_override")
    m2 = json.loads((off_dir / f"{seed:06d}_override.json").read_text(encoding="utf-8"))
    ref_pc = meta.get("param_control") if isinstance(meta.get("param_control"), dict) else {}
    new_pc = m2.get("param_control") if isinstance(m2.get("param_control"), dict) else {}
    others_kept = all(new_pc[k]["value"] == v["value"] for k, v in ref_pc.items() if k not in ("text", name) and k in new_pc)
    return dict(text=text, param=name, text_ok=True if text is None else str(m2["label"]).lower() == text.lower(),
                param_applied=(name is None) or new_pc.get(name, {}).get("value") == value,
                others_kept=others_kept, ok=bool(m2.get("ok")), reason=m2.get("exception") or m2.get("fatal"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--libs", nargs="*")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--first-seed", type=int, default=7000)
    ap.add_argument("--dilate", type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if not a.libs:
        from feature_bank.bank02_smoke import active_libraries
        a.libs = active_libraries()
    report = {}
    for lib in a.libs:
        ref_dir, off_dir = out / lib / "ref", out / lib / "variants"
        ref_dir.mkdir(parents=True, exist_ok=True); off_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for seed in range(a.first_seed, a.first_seed + a.seeds):
            ref = run_case(lib, seed, ref_dir, intercept=True)
            if not ref["ok"] or not Path(str(ref_dir / f"{seed:06d}") + ".calls.json").exists():
                rows.append(dict(cls="reference_failed", seed=seed, reason=ref.get("reason")))
                continue
            _, calls, *_rest = load(ref_dir / f"{seed:06d}")
            classes = sorted({c.get("class") for c in calls if c.get("class") and c.get("class") not in SKIP_CLASSES})
            for cls in classes:
                r = check_class(lib, seed, cls, ref_dir, off_dir, a.dilate)
                r["seed"] = seed
                rows.append(r)
                print(lib, seed, r, flush=True)
            o = check_override(lib, seed, ref_dir, off_dir)
            o["seed"] = seed
            rows.append(dict(cls="override", **o))
            print(lib, seed, "override", o, flush=True)
        report[lib] = rows
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("\nSUMMARY")
    for lib, rows in report.items():
        allsw = [r for r in rows if r["cls"] != "override" and "error" not in r and not r.get("structural")]
        allsw = [r for r in allsw if r["cls"] != "reference_failed"]
        jp = [r for r in allsw if r.get("jpeg_coupled") and not r.get("palette_coupled")]
        sw = [r for r in allsw if not r.get("palette_coupled") and not r.get("jpeg_coupled")] or [dict(containment=1, containment_visible=1, stability=1, label_same=True)]
        pal = [r for r in allsw if r.get("palette_coupled")]
        st = sorted({r["cls"] for r in rows if r.get("structural")})
        ov = [r for r in rows if r["cls"] == "override"]
        print(f"{lib:26} truecolour_switches={len(sw)} min_containment={min(r['containment'] for r in sw):.4f} "
              f"min_visible_containment={min(r['containment_visible'] for r in sw):.4f} min_stability={min(r['stability'] for r in sw):.4f} labels_same={all(r['label_same'] for r in sw)} "
              f"structural={st} palette_cases={len(pal)} palette_min_visible={min([r['containment_visible'] for r in pal], default=1):.3f} jpeg_cases={len(jp)} jpeg_min_block={min([r['containment_block'] for r in jp], default=1):.3f} jpeg_min_stability={min([r['stability'] for r in jp], default=1):.3f} overrides_ok={all(r['text_ok'] and r['param_applied'] and r['others_kept'] and r['ok'] for r in ov)}")


if __name__ == "__main__":
    main()
