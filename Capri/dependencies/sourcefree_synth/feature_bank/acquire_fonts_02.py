"""Font bank 02: expand the 35-font Phase 2 bank 5x (to 175) with OFL fonts from google/fonts
at the same pinned commit as the existing bank.

Selection is deterministic: a forced list of CAPTCHA-relevant styles the current bank lacks
(outline/hollow, inline, shadow, stencil, pixel/dotted, heavy, typewriter, marker), then a
category-stratified fill ordered by sha256(SEED + directory). The existing 35 fonts are kept
unchanged and referenced from the Phase 2 manifest.

Every accepted font must: carry an OFL.txt containing the SIL OFL text; map all 62 ASCII
alphanumerics in its cmap; render each of them with ink and not as the .notdef box (PIL).
Fonts whose lowercase renders identical to uppercase are kept but flagged caps_only, so a
renderer can restrict them to uppercase/digit text instead of producing wrong labels.

Inputs: a sparse partial clone of google/fonts (METADATA.pb + OFL.txt) at COMMIT; TTF bytes
are fetched from raw.githubusercontent.com at the same commit.
Usage: python -m feature_bank.acquire_fonts_02 --gfonts <clone dir>
Output: resources/font_bank_02/{fonts/<dir>/font.ttf, fonts/<dir>/OFL.txt, fonts.json, acquisition.json}
"""
import argparse
import hashlib
import io
import json
import re
import string
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "resources" / "font_bank_02"
PHASE2 = ROOT / "releases" / "CAPRI_phase123_v1" / "phase2"
COMMIT = "23e54b51ddffbc7713c583748e3bd86f62b1fa4a"
SEED = "font-bank-02"
TARGET_NEW = 140
QUOTA = {"SANS_SERIF": 35, "SERIF": 30, "DISPLAY": 40, "HANDWRITING": 20, "MONOSPACE": 15}
FORCED = [  # CAPTCHA-relevant appearance families absent from the 35-font bank
    "bungeeoutline", "bungeeinline", "bungeeshade", "bungeehairline", "monoton", "londrinaoutline",
    "londrinashadow", "londrinasketch", "trainone", "fascinateinline", "kumarone", "codystar",
    "ralewaydots", "monofett", "stardosstencil", "allertastencil", "emblemaone", "pressstart2p",
    "vt323", "silkscreen", "dotgothic16", "pixelifysans", "blackopsone", "specialelite",
    "permanentmarker", "indieflower", "comicneue", "creepster", "frijole", "ewert", "sixcaps",
    "rubikmonoone", "bungee", "rampartone", "shojumaru", "nosifer", "faster one", "plaster",
]
ALNUM = string.ascii_letters + string.digits


def parse_metadata(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    fonts = re.findall(r'fonts \{\s*name: "([^"]*)"\s*style: "(\w+)"\s*weight: (\d+)\s*filename: "([^"]+)"', text)
    return dict(
        dir=path.parent.name,
        name=(re.search(r'^name: "([^"]+)"', text, re.M) or [None, path.parent.name])[1],
        category=(re.search(r'^category: "(\w+)"', text, re.M) or [None, "UNKNOWN"])[1],
        classifications=re.findall(r'^classifications: "(\w+)"', text, re.M),
        subsets=re.findall(r'^subsets: "([\w-]+)"', text, re.M),
        fonts=[dict(style=s, weight=int(w), filename=f) for _, s, w, f in fonts],
    )


def choose_file(meta):
    files = [f for f in meta["fonts"] if f["style"] == "normal"] or meta["fonts"]
    return min(files, key=lambda f: (abs(f["weight"] - 400), "[" in f["filename"], f["filename"]))["filename"]


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "CAPRI-font-bank-02"})
    return urllib.request.urlopen(req, timeout=60).read()


def render_check(data):
    font = ImageFont.truetype(io.BytesIO(data), 48)
    def bitmap(ch):
        im = Image.new("L", (80, 80), 0)
        ImageDraw.Draw(im).text((10, 5), ch, font=font, fill=255)
        return im.tobytes()
    notdef = bitmap("\ue000")
    bmps = {c: bitmap(c) for c in ALNUM}
    empty = [c for c, b in bmps.items() if not any(b)]
    as_notdef = [c for c, b in bmps.items() if b == notdef and any(notdef)]
    caps_only = sum(bmps[c] == bmps[c.upper()] for c in string.ascii_lowercase) >= 20
    distinct = len(set(bmps.values()))
    return dict(empty=empty, as_notdef=as_notdef, caps_only=caps_only, caps_like=caps_like(font),
                distinct_glyphs=distinct)


def caps_like(font, threshold=0.6, min_letters=16):
    """Small-caps / unicase detection: lowercase glyphs that are scaled capitals.
    Each glyph is cropped to its ink box and resized to 32x32; a letter counts when the
    lowercase/uppercase IoU >= threshold. Ambiguous pairs (c/C, o/O, s/S, v/V, w/W, x/X, z/Z)
    are excluded because they legitimately look alike in ordinary fonts."""
    def shape(ch):
        im = Image.new("L", (96, 96), 0)
        ImageDraw.Draw(im).text((12, 8), ch, font=font, fill=255)
        box = im.getbbox()
        return im.crop(box).resize((32, 32)).point(lambda v: 255 if v > 96 else 0) if box else None
    same = 0
    for c in "abdefghijklmnpqrtuy":
        a, b = shape(c), shape(c.upper())
        if a is None or b is None:
            continue
        pa, pb = a.tobytes(), b.tobytes()
        inter = sum(1 for x, y in zip(pa, pb) if x and y)
        union = sum(1 for x, y in zip(pa, pb) if x or y)
        same += union and inter / union >= threshold
    return same >= min_letters


def acquire(meta, gfonts):
    d = meta["dir"]
    try:
        if "latin" not in meta["subsets"]:
            return dict(rejected=d, reason="no latin subset")
        lic_path = gfonts / "ofl" / d / "OFL.txt"
        notice = lic_path.read_bytes()
        if b"SIL OPEN FONT LICENSE" not in notice.upper():
            return dict(rejected=d, reason="OFL text missing")
        fname = choose_file(meta)
        url = f"https://raw.githubusercontent.com/google/fonts/{COMMIT}/ofl/{d}/{fname}"
        data = fetch(url)
        with TTFont(io.BytesIO(data)) as tt:
            cmap = tt.getBestCmap() or {}
            missing = [c for c in ALNUM if ord(c) not in cmap]
            family = tt["name"].getDebugName(1)
        if missing:
            return dict(rejected=d, reason=f"cmap missing {''.join(missing)[:20]}")
        chk = render_check(data)
        if chk["empty"] or chk["as_notdef"]:
            return dict(rejected=d, reason=f"renders empty/notdef for {''.join(chk['empty'] + chk['as_notdef'])[:20]}")
        folder = OUT / "fonts" / d
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "font.ttf").write_bytes(data)
        (folder / "OFL.txt").write_bytes(notice)
        return dict(family=family or meta["name"], dir=d, path=f"fonts/{d}/font.ttf",
                    sha256=hashlib.sha256(data).hexdigest(), license_path=f"fonts/{d}/OFL.txt",
                    license_sha256=hashlib.sha256(notice).hexdigest(), source=url, commit=COMMIT,
                    source_file=fname, variable_font="[" in fname, category=meta["category"],
                    classifications=meta["classifications"], caps_only=chk["caps_only"] or chk["caps_like"], caps_like=chk["caps_like"],
                    distinct_glyphs=chk["distinct_glyphs"], ascii_alphanumeric_checked=True)
    except Exception as e:  # noqa: BLE001 - record and continue
        return dict(rejected=d, reason=repr(e)[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gfonts", required=True)
    args = ap.parse_args()
    gfonts = Path(args.gfonts)
    existing = json.loads((PHASE2 / "fonts.json").read_text(encoding="utf-8"))
    existing_hashes = {r["sha256"] for r in existing}
    existing_dirs = {Path(r["path"].replace("\\", "/")).parent.name for r in existing}
    existing_names = {re.sub(r"[^a-z0-9]", "", r["family"].lower()) for r in existing}
    metas = []
    for p in sorted((gfonts / "ofl").glob("*/METADATA.pb")):
        m = parse_metadata(p)
        if m["dir"] in existing_dirs or re.sub(r"[^a-z0-9]", "", m["name"].lower()) in existing_names:
            continue
        if "SYMBOLS" in m["classifications"] or not m["fonts"]:
            continue
        metas.append(m)
    by_dir = {m["dir"]: m for m in metas}
    order = sorted(metas, key=lambda m: hashlib.sha256((SEED + m["dir"]).encode()).hexdigest())

    accepted, rejected, seen = [], [], set(existing_hashes)
    def take(cands, limit):
        got = 0
        with ThreadPoolExecutor(max_workers=6) as pool:
            for i in range(0, len(cands), 12):
                if got >= limit:
                    break
                for r in pool.map(lambda m: acquire(m, gfonts), cands[i:i + 12]):
                    if "rejected" in r:
                        rejected.append(r)
                    elif r["sha256"] in seen or got >= limit:
                        if got >= limit:  # fetched beyond quota: remove the unused files
                            for f in ("font.ttf", "OFL.txt"):
                                (OUT / r["path"]).parent.joinpath(f).unlink(missing_ok=True)
                            (OUT / r["path"]).parent.rmdir()
                        else:
                            rejected.append(dict(rejected=r["dir"], reason="duplicate bytes"))
                    else:
                        seen.add(r["sha256"])
                        accepted.append(r)
                        got += 1
        return got

    forced = [by_dir[d] for d in FORCED if d in by_dir]
    missing_forced = [d for d in FORCED if d not in by_dir]
    take(forced, len(forced))
    for r in accepted:
        r["selection"] = "forced_style"
    counts = {c: sum(r["category"] == c for r in accepted) for c in QUOTA}
    for cat, quota in QUOTA.items():
        need = max(0, quota - counts[cat])
        pool = [m for m in order if m["category"] == cat and m["dir"] not in {r["dir"] for r in accepted}]
        before = len(accepted)
        take(pool, need)
        for r in accepted[before:]:
            r["selection"] = f"stratified_{cat.lower()}"
    if len(accepted) < TARGET_NEW:  # top up from any category, same deterministic order
        pool = [m for m in order if m["dir"] not in {r["dir"] for r in accepted}]
        before = len(accepted)
        take(pool, TARGET_NEW - len(accepted))
        for r in accepted[before:]:
            r["selection"] = "topup_any"
    accepted = accepted[:TARGET_NEW]

    phase2_rows = [dict(r, path=str((PHASE2 / r["path"].replace("\\", "/")).relative_to(ROOT)).replace("\\", "/"),
                        license_path=str((PHASE2 / r["license_path"].replace("\\", "/")).relative_to(ROOT)).replace("\\", "/"),
                        bank="phase2_existing_35") for r in existing]
    new_rows = [dict(r, path=f"resources/font_bank_02/{r['path']}", license_path=f"resources/font_bank_02/{r['license_path']}",
                     bank="font_bank_02_new") for r in accepted]
    manifest = phase2_rows + new_rows
    (OUT / "fonts.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (OUT / "acquisition.json").write_text(json.dumps(dict(
        commit=COMMIT, seed=SEED, quota=QUOTA, forced_requested=FORCED, forced_missing=missing_forced,
        existing=len(existing), new_accepted=len(accepted), total=len(manifest), rejected=rejected,
        paths_relative_to="E:/pranav/Final sem/sourcefree_synth"), indent=1), encoding="utf-8")
    from collections import Counter
    print("new", len(accepted), "total", len(manifest), Counter(r["category"] for r in accepted),
          "caps_only", sum(r["caps_only"] for r in accepted), "rejected", len(rejected), "forced missing", missing_forced)


if __name__ == "__main__":
    main()
