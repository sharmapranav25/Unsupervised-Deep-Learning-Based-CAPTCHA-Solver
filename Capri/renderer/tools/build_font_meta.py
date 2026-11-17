"""Font table for the text renderer -> fonts/font_meta.json.

Per font of the bank (sourcefree_synth resources/font_bank_02/fonts.json): path, family, key, train/heldout split,
caps_only, Google Fonts category (Sans Serif / Serif / Display / Handwriting / Monospace) and stroke class
(Sans Serif / Serif / Slab Serif) from the public Google Fonts family metadata, variable axes (fontTools fvar), and the
ASCII alphanumerics the font really contains (cmap), so the renderer never asks a font for a glyph it lacks.

  python tools/build_font_meta.py --bank <sourcefree_synth>/resources/font_bank_02/fonts.json \
      --gf-metadata gf_metadata.json --out fonts/font_meta.json
"""
import argparse
import json
import re
import string
from pathlib import Path

from fontTools.ttLib import TTFont

ALNUM = string.ascii_letters + string.digits
WEIGHT_SUFFIX = r"(thin|extralight|ultralight|light|medium|semibold|bold|extrabold|black|regular)$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--gf-metadata", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bank_path = Path(a.bank)
    root = bank_path.parents[2]  # sourcefree_synth/
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    gf = json.loads(Path(a.gf_metadata).read_text(encoding="utf-8"))["familyMetadataList"]
    by = {f["family"].lower().replace(" ", ""): f for f in gf}
    out = []
    for r in bank:
        key = Path(r["path"]).parent.name
        fam = r["family"].lower().replace(" ", "")
        g = by.get(fam) or by.get(re.sub(WEIGHT_SUFFIX, "", fam))
        path = (root / r["path"]).resolve()
        tt = TTFont(str(path), lazy=True)
        cmap = tt.getBestCmap() or {}
        chars = "".join(c for c in ALNUM if ord(c) in cmap)
        axes = [dict(tag=x.axisTag, min=x.minValue, default=x.defaultValue, max=x.maxValue) for x in tt["fvar"].axes] \
            if "fvar" in tt else []
        out.append(dict(key=key, family=r["family"], path=str(path).replace("\\", "/"), split=r.get("split", "train"),
                        caps_only=bool(r.get("caps_only")), category=(g or {}).get("category", "unknown"),
                        stroke=(g or {}).get("stroke") or "unknown", classifications=(g or {}).get("classifications", []),
                        axes=axes, chars=chars, licence=r.get("license_path")))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    from collections import Counter
    print(len(out), "fonts;", Counter(f["category"] for f in out), Counter(f["split"] for f in out),
          "variable", sum(bool(f["axes"]) for f in out), "full alnum", sum(len(f["chars"]) == 62 for f in out))


if __name__ == "__main__":
    main()
