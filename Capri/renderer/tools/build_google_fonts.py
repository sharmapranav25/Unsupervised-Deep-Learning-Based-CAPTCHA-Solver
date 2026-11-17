"""Integrate the Google Fonts collection into the font table (fonts/font_meta.json).

Input: a checkout of github.com/google/fonts (ofl/, apache/, ufl/; commit recorded in font_data/google_fonts_commit.txt)
plus the public family metadata (fonts.google.com/metadata/fonts: category, stroke, classifications, subsets).
Per family (folder with METADATA.pb):
  * files: the roman variable font if there is one, else the static Regular; the italic (variable or static) is added as
    a separate entry flagged italic=True (true italic letterforms);
  * kept only if: the latin subset is listed, the cmap has all 62 ASCII alphanumerics, it is not a symbol family, and it
    renders (non-empty glyphs);
  * caps_only: lowercase glyphs identical to the uppercase ones (same glyph name or near-identical rasters);
  * measured: stroke_rel (mean stroke width / cap height on 'H', 'n', 'o'), x_height (height of 'x' / height of 'H'),
    width_rel ('n' advance / cap height);
  * split: the 175 existing bank fonts keep their train/heldout assignment; new families are ranked by
    sha256("captcha_effects_gf_split|" + family) inside their category and the first 15% are held out.
Licence file path and sha256 recorded per family.

  python tools/build_google_fonts.py --gf-root <font_data/google-fonts> --gf-metadata gf_metadata.json \
      --existing fonts/font_meta.json --out fonts/font_meta.json
"""
import argparse
import hashlib
import json
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

ALNUM = string.ascii_letters + string.digits
SALT = "captcha_effects_gf_split|"
HELD_OUT = 0.15


def parse_metadata_pb(text):
    """Minimal parser for METADATA.pb (protobuf text format): name, category, license, fonts[], subsets[], axes[]."""
    out = dict(name=None, category=None, license=None, fonts=[], subsets=[], axes=[])
    m = re.search(r'^name:\s*"([^"]+)"', text, re.M)
    out["name"] = m.group(1) if m else None
    out["category"] = (re.findall(r'^category:\s*"([^"]+)"', text, re.M) or [None])[0]
    out["license"] = (re.findall(r'^license:\s*"([^"]+)"', text, re.M) or [None])[0]
    out["subsets"] = re.findall(r'^subsets:\s*"([^"]+)"', text, re.M)
    for blk in re.findall(r"fonts\s*\{(.*?)\n\}", text, re.S):
        g = lambda k: (re.findall(rf'{k}:\s*"?([^"\n]+)"?', blk) or [None])[0]  # noqa: E731
        out["fonts"].append(dict(filename=g("filename"), style=g("style"), weight=g("weight")))
    for blk in re.findall(r"axes\s*\{(.*?)\n\}", text, re.S):
        tag = re.findall(r'tag:\s*"([^"]+)"', blk)
        lo = re.findall(r"min_value:\s*([\d.\-]+)", blk)
        hi = re.findall(r"max_value:\s*([\d.\-]+)", blk)
        if tag:
            out["axes"].append(dict(tag=tag[0], min=float(lo[0]) if lo else None, max=float(hi[0]) if hi else None))
    return out


def raster(font, ch, size=96):
    l, t, r, b = font.getbbox(ch)
    if r <= l or b <= t:
        return None
    im = Image.new("L", (r - l + 8, b - t + 8), 0)
    ImageDraw.Draw(im).text((4 - l, 4 - t), ch, font=font, fill=255)
    a = np.asarray(im) > 127
    return a if a.sum() > 10 else None


def stroke_width(a):
    pad = np.pad(a, 1)
    inner = pad[1:-1, 1:-1] & pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    per = (a & ~inner).sum()
    return 2 * a.sum() / max(1, per)


def measure(path):
    f = ImageFont.truetype(str(path), 96)
    H, x, n, o, A, aa = (raster(f, c) for c in "HxnoAa")
    if H is None or x is None or n is None:
        return None
    cap = H.shape[0] - 8
    sw = np.mean([stroke_width(g) for g in (H, n, o) if g is not None])
    caps = A is not None and aa is not None and A.shape == aa.shape and (A == aa).mean() > 0.97
    adv = f.getlength("n")
    return dict(stroke_rel=round(float(sw / cap), 4), x_height=round(float((x.shape[0] - 8) / cap), 4),
                width_rel=round(float(adv / cap), 4), caps_only_raster=bool(caps))


def pick_files(fam_dir, md):
    files = [f for f in md["fonts"] if f["filename"]]
    out = []
    for italic in (False, True):
        cand = [f for f in files if (f["style"] == "italic") == italic]
        var = [f for f in cand if "[" in f["filename"]]
        if var:
            out.append((var[0]["filename"], italic, True))
        else:
            reg = sorted(cand, key=lambda f: abs(int(f["weight"] or 400) - 400))
            if reg:
                out.append((reg[0]["filename"], italic, False))
    return [(fam_dir / fn, it, isvar) for fn, it, isvar in out if (fam_dir / fn).exists()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gf-root", required=True)
    ap.add_argument("--gf-metadata", required=True)
    ap.add_argument("--existing", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.gf_root)
    commit = (root.parent / "google_fonts_commit.txt").read_text().strip() if (root.parent / "google_fonts_commit.txt").exists() else None
    gf = {f["family"]: f for f in json.loads(Path(a.gf_metadata).read_text(encoding="utf-8"))["familyMetadataList"]}
    existing = json.loads(Path(a.existing).read_text(encoding="utf-8"))
    for e in existing:
        e.setdefault("source", "font_bank_02")
        e.setdefault("italic", False)
    have = {e["family"].lower().replace(" ", "") for e in existing}
    new, skipped = [], Counter()
    for md_path in sorted(root.glob("*/*/METADATA.pb")):
        fam_dir = md_path.parent
        md = parse_metadata_pb(md_path.read_text(encoding="utf-8", errors="ignore"))
        name = md["name"]
        if not name:
            skipped["no name"] += 1
            continue
        if name.lower().replace(" ", "") in have:
            skipped["already in bank"] += 1
            continue
        meta = gf.get(name, {})
        if "latin" not in md["subsets"]:
            skipped["no latin subset"] += 1
            continue
        if "Symbols" in (meta.get("classifications") or []) or md["category"] in ("SYMBOLS",):
            skipped["symbols"] += 1
            continue
        lic = next((p for p in fam_dir.iterdir() if p.name.upper().startswith(("OFL", "LICENSE", "UFL"))), None)
        for path, italic, isvar in pick_files(fam_dir, md):
            try:
                tt = TTFont(str(path), lazy=True)
                cmap = tt.getBestCmap() or {}
                chars = "".join(c for c in ALNUM if ord(c) in cmap)
                if len(chars) < 62:
                    skipped["missing alnum"] += 1
                    continue
                axes = [dict(tag=x.axisTag, min=x.minValue, default=x.defaultValue, max=x.maxValue) for x in tt["fvar"].axes] \
                    if "fvar" in tt else []
                same = sum(cmap.get(ord(c)) == cmap.get(ord(c.upper())) for c in string.ascii_lowercase)
                m = measure(path)
                if m is None:
                    skipped["does not render"] += 1
                    continue
            except Exception:  # noqa: BLE001
                skipped["unreadable"] += 1
                continue
            key = fam_dir.name + ("_italic" if italic else "")
            new.append(dict(key=key, family=name, path=str(path.resolve()).replace("\\", "/"), split=None,
                            caps_only=bool(same >= 20 or m["caps_only_raster"]),
                            category=meta.get("category", (md["category"] or "unknown").title().replace("_", " ")),
                            stroke=meta.get("stroke") or "unknown", classifications=meta.get("classifications", []),
                            axes=axes, variable=isvar, italic=italic, chars=chars,
                            licence=str(lic).replace("\\", "/") if lic else None,
                            licence_sha256=hashlib.sha256(lic.read_bytes()).hexdigest() if lic else None,
                            licence_dir=fam_dir.parent.name, source=f"google/fonts@{commit}", popularity=meta.get("popularity"),
                            **{k: v for k, v in m.items() if k != "caps_only_raster"}))
    # split new families: sha256 rank inside each category, first 15% held out (family-level, italics follow roman)
    fams = {}
    for f in new:
        fams.setdefault(f["category"], set()).add(f["family"])
    held = set()
    for cat, fs in fams.items():
        ranked = sorted(fs, key=lambda x: hashlib.sha256((SALT + x).encode()).hexdigest())
        held |= set(ranked[:round(len(ranked) * HELD_OUT)])
    for f in new:
        f["split"] = "heldout" if f["family"] in held else "train"
    # measure the existing bank too (same features), keep its split
    for e in existing:
        if "stroke_rel" not in e:
            m = measure(e["path"])
            if m:
                e.update({k: v for k, v in m.items() if k != "caps_only_raster"})
    allf = existing + new
    Path(a.out).write_text(json.dumps(allf, indent=1), encoding="utf-8")
    print("existing", len(existing), "new entries", len(new), "new families", len({f["family"] for f in new}),
          "skipped", dict(skipped))
    print("split", Counter(f["split"] for f in allf), "category", Counter(f["category"] for f in allf))
    print("variable", sum(bool(f["axes"]) for f in allf), "italic", sum(f.get("italic", False) for f in allf),
          "caps_only", sum(f["caps_only"] for f in allf))


if __name__ == "__main__":
    main()
