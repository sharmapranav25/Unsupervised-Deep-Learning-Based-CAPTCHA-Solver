"""Step 3 rollout check: per library, does interception keep output byte-identical and
locate every answer glyph?

For each library and seed: render plain and intercepted (same seed), compare the library's
output bytes, decode the label maps, count answer glyphs located (text calls matched to the
label in order, with pixels in the final maps), and write an overlay sheet.
Results: <out>/rollout.json and a printed table.

Usage: python -m feature_bank.rollout_check --out runs/rollout_NEW [--libs ...] [--seeds 4]
"""
import argparse
import json
from pathlib import Path

from feature_bank.bank02_smoke import active_libraries, run_case
from feature_bank.label_maps import decode, sheet


def check(lib, seeds, out):
    plain, inter = out / lib / "plain", out / lib / "intercept"
    plain.mkdir(parents=True, exist_ok=True); inter.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        a = run_case(lib, seed, plain)
        b = run_case(lib, seed, inter, intercept=True)
        row = dict(seed=seed, ok_plain=a["ok"], ok_intercept=b["ok"], reason=a.get("reason") or b.get("reason"))
        meta_p = plain / f"{seed:06d}.json"
        if a["ok"] and b["ok"]:
            m = json.loads(meta_p.read_text(encoding="utf-8"))
            fmt = m["source_format"]
            row["identical"] = (plain / f"{seed:06d}.{fmt}").read_bytes() == (inter / f"{seed:06d}.{fmt}").read_bytes()
            row["format"] = fmt
            stem = inter / f"{seed:06d}"
            if Path(str(stem) + ".labels.png").exists():
                d = decode(stem)
                row["label_len"] = len(str(d["meta"]["label"]))
                row["glyphs_matched"] = len(d["glyphs"])
                row["glyphs_located"] = sum(b is not None for b in d["boxes"])
                row["label_canvas"] = d["log_meta"]["label_canvas"]
                row["palette"] = d["log_meta"].get("palette_canvas")
                row["text_calls"] = sum(1 for c in d["calls"].values() if c.get("class") == "text")
                row["unmirrored"] = len(d["log_meta"]["unmirrored"])
            else:
                row["labels"] = "none (svg or no canvas)"
        rows.append(row)
    try:
        sheet(inter)
    except Exception as e:  # noqa: BLE001
        rows.append(dict(sheet_error=repr(e)))
    return rows


def summarise(lib, rows):
    rs = [r for r in rows if "seed" in r]
    both = [r for r in rs if r["ok_plain"] and r["ok_intercept"]]
    ident = sum(r.get("identical", False) for r in both)
    lab = [r for r in both if "glyphs_located" in r]
    loc = sum(r["glyphs_located"] for r in lab)
    want = sum(r["label_len"] for r in lab)
    full = sum(r["glyphs_located"] == r["label_len"] for r in lab)
    reasons = sorted({str(r["reason"])[:70] for r in rs if r["reason"]})
    return dict(lib=lib, rendered=f"{len(both)}/{len(rs)}", identical=f"{ident}/{len(both)}",
                glyphs=f"{loc}/{want}", cases_all_glyphs=f"{full}/{len(lab)}",
                canvas=sorted({r.get("label_canvas") for r in lab} - {None}), svg=len(both) - len(lab),
                reasons=reasons)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--libs", nargs="*")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--first-seed", type=int, default=9000)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    libs = a.libs or active_libraries()
    seeds = list(range(a.first_seed, a.first_seed + a.seeds))
    summary = []
    for lib in libs:
        rows = check(lib, seeds, out)
        s = summarise(lib, rows)
        summary.append(s)
        (out / lib / "rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(f"{lib:40} rendered {s['rendered']:5} identical {s['identical']:5} glyphs {s['glyphs']:7} "
              f"all-glyph cases {s['cases_all_glyphs']:5} svg {s['svg']} {s['canvas']} {'; '.join(s['reasons'])[:90]}", flush=True)
    merged = out / "rollout.json"
    old = {r["lib"]: r for r in (json.loads(merged.read_text()) if merged.exists() else [])}
    old.update({s["lib"]: s for s in summary})
    merged.write_text(json.dumps(list(old.values()), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
