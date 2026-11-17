"""Smoke-render feature-bank-02 libraries through their adapters.

Each case runs in a fresh PHP process (adapters/bank02/harness.php) with its own
seed and temporary session directory. Outputs per library:
  <out>/<library>/<seed>.{png,json[,svg,gif,jpeg]}, contact_sheet.png, summary.json
A run-level summary.json reports per-library success counts and failure reasons.

Usage:
  python -m feature_bank.bank02_smoke --out runs/bank02_smoke_01 [--libs a b] [--seeds 8]
"""
import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageStat

ROOT = Path(__file__).resolve().parents[1]
PHP = ROOT / "resources" / "feature_bank_01" / "php"
ADAPTERS = ROOT / "adapters" / "bank02"
EXCLUDED_LIBRARIES = frozenset({"magkopian_php-math-captcha", "mulertech_captcha"})


def active_libraries():
    """Text CAPTCHA adapters selected for hybrid v3 runs."""
    return sorted(p.stem for p in ADAPTERS.glob("*.php")
                  if p.stem not in {"harness", "intercept", *EXCLUDED_LIBRARIES}
                  and not p.stem.startswith("_"))
STAGING = ROOT / "resources" / "feature_bank_02" / "staging"
BANK01 = ROOT / "resources" / "feature_bank_01"


def source_dir(library):
    return STAGING / library if (STAGING / library).exists() else BANK01 / library
RESVG = ROOT / "resources" / "tools" / "resvg" / "resvg.exe"


def php_cmd(case_path):
    return [
        str(PHP / "php.exe"), "-n",
        "-d", f"extension_dir={PHP / 'ext'}",
        "-d", "extension=gd", "-d", "extension=mbstring", "-d", "extension=fileinfo",
        "-d", "extension=intl", "-d", "extension=openssl", "-d", "memory_limit=512M",
        str(ADAPTERS / "harness.php"), str(case_path),
    ]


def rasterise_svg(svg_path, png_path):
    if not RESVG.exists():
        return "resvg missing"
    proc = subprocess.run([str(RESVG), "--background", "white", str(svg_path), str(png_path)],
                          capture_output=True, text=True, timeout=60)
    return None if proc.returncode == 0 else proc.stderr.strip()[:300]


def run_case(library, seed, out_dir, options=None, intercept=False, disable=(), params=None, stem_name=None):
    stem = out_dir / (stem_name or f"{seed:06d}")
    with tempfile.TemporaryDirectory() as tmp:
        case = {
            "library": library,
            "adapter": str(ADAPTERS / f"{library}.php"),
            "source": str(source_dir(library)),
            "intercept": intercept,
            "disable": list(disable),
            "params": dict(params or {}),
            "seed": seed,
            "output_stem": str(stem),
            "tmp": tmp,
            "options": options or {},
        }
        case_path = Path(tmp) / "case.json"
        case_path.write_text(json.dumps(case), encoding="utf-8")
        proc = subprocess.run(php_cmd(case_path), capture_output=True, text=True,
                              timeout=120, cwd=tmp)
    meta_path = stem.with_suffix(".json")
    if not meta_path.exists():
        return {"seed": seed, "ok": False, "reason": "no metadata",
                "stderr": (proc.stderr or proc.stdout)[-600:]}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("source_format") == "svg":
        if disable:
            from feature_bank.svg_labels import apply_disable
            meta["svg_disabled_elements"] = apply_disable(stem.with_suffix(".svg"), library, set(disable))
        err = rasterise_svg(stem.with_suffix(".svg"), stem.with_suffix(".png"))
        if err:
            meta["ok"] = False
            meta["raster_error"] = err
        else:
            with Image.open(stem.with_suffix(".png")) as im:
                meta["width"], meta["height"] = im.size
            if intercept and meta.get("label"):
                from feature_bank.svg_labels import CHAR_ORDER, label_svg
                try:
                    meta["labels"] = label_svg(stem, library, meta["label"], CHAR_ORDER.get(library, "drawing"))
                except Exception as e:  # noqa: BLE001 - recorded, image still usable without labels
                    meta["labels_error"] = repr(e)[:300]
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    if meta.get("ok") and not stem.with_suffix(".png").exists():
        meta["ok"] = False
        meta["reason"] = "image did not decode"
    if meta.get("ok"):
        with Image.open(stem.with_suffix(".png")) as im:
            spread = ImageStat.Stat(im.convert("L")).stddev[0]
        meta["luminance_std"] = round(spread, 2)
        if spread < 4:  # effectively uniform canvas: nothing drawn
            meta["ok"] = False
            meta["reason"] = "blank image"
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    reason = meta.get("exception") or meta.get("fatal") or meta.get("raster_error") or meta.get("reason")
    if not meta.get("ok") and not reason:
        reason = "no image" if not meta.get("source_format") else "no label"
    return {"seed": seed, "ok": bool(meta.get("ok")), "reason": reason,
            "label": meta.get("label"), "shown": meta.get("shown_text"),
            "size": [meta.get("width"), meta.get("height")], "stderr": proc.stderr[-300:]}


def contact_sheet(library, out_dir, results):
    tiles = []
    for r in results:
        p = out_dir / f"{r['seed']:06d}.png"
        if r["ok"] and p.exists():
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((240, 90))
                tiles.append((im.copy(), str(r["shown"])))
    if not tiles:
        return
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 250, rows * 110 + 20), "white")
    d = ImageDraw.Draw(sheet)
    d.text((4, 2), library, fill="black")
    for i, (im, label) in enumerate(tiles):
        x, y = (i % cols) * 250 + 5, (i // cols) * 110 + 20
        sheet.paste(im, (x, y))
        d.text((x, y + 92), label[:40], fill="blue")
    sheet.save(out_dir / "contact_sheet.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--libs", nargs="*")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--first-seed", type=int, default=1000)
    ap.add_argument("--intercept", action="store_true", help="write label maps + call logs")
    ap.add_argument("--disable", nargs="*", default=[], help="call classes to suppress (needs --intercept)")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    libs = args.libs or active_libraries()
    summary = {}
    for lib in libs:
        lib_dir = out / lib
        if lib_dir.exists():
            shutil.rmtree(lib_dir)
        lib_dir.mkdir()
        results = [run_case(lib, args.first_seed + i, lib_dir, intercept=args.intercept, disable=args.disable)
                   for i in range(args.seeds)]
        contact_sheet(lib, lib_dir, results)
        (lib_dir / "summary.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
        ok = sum(r["ok"] for r in results)
        reasons = sorted({str(r["reason"]) for r in results if not r["ok"]})
        summary[lib] = {"ok": ok, "total": len(results), "reasons": reasons}
        print(f"{lib:40} {ok}/{len(results)} {'; '.join(reasons)[:160]}", flush=True)
    merged = out / "summary.json"
    old = json.loads(merged.read_text()) if merged.exists() else {}
    old.update(summary)
    merged.write_text(json.dumps(old, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
