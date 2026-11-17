"""Deterministic train / held-out split of font bank 02 (hybrid v3).

Version 1 (2026-10-01): families ranked by sha256("hybrid_v3_font_split|" + family) inside two strata
(caps-only and mixed-case); the first round(15%) of each stratum held out -> 149 train / 26 held out.

Version 2 (2026-10-02, user decision): the 26 v1 held-out families are split 13 / 13 by the same salted
sha256 ranking, again inside the caps-only / mixed-case strata so both sides keep caps-only faces.
The first half of each ranked stratum stays held out, the rest moves to train -> 162 train / 13 held out.

The split is written into resources/font_bank_02/fonts.json (field "split") and pinned by
resources/font_bank_02/font_split.json (version 2; the v1 record is kept under "v1").

bank_font() in adapters/bank02/harness.php and the Python sources only draw from the split given
by the case option "font_split" (default "train").

Usage: python -m feature_bank.font_split [--check]
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FONTS = ROOT / "resources" / "font_bank_02" / "fonts.json"
SPLIT = ROOT / "resources" / "font_bank_02" / "font_split.json"
HELD_OUT_FRACTION = 0.15
SALT = "hybrid_v3_font_split|"
VERSION = 2
V2_HELDOUT = 13


def rank(fams):
    return sorted(fams, key=lambda f: hashlib.sha256((SALT + f).encode()).hexdigest())


def caps_of(rows):
    return {r["family"]: bool(r.get("caps_only")) for r in rows}


def assign_v1(rows):
    out = {}
    for caps in (True, False):
        fams = rank({r["family"] for r in rows if bool(r.get("caps_only")) == caps})
        k = round(len(fams) * HELD_OUT_FRACTION)
        for i, f in enumerate(fams):
            out[f] = "heldout" if i < k else "train"
    return out


def assign(rows):
    """Version 2: split the v1 held-out families into V2_HELDOUT held out / the rest train."""
    v1 = assign_v1(rows)
    caps = caps_of(rows)
    held = [f for f, s in v1.items() if s == "heldout"]
    n_caps = sum(caps[f] for f in held)
    k_caps = max(1, min(n_caps - 1, round(V2_HELDOUT * n_caps / len(held))))  # caps-only on both sides
    k = {True: k_caps, False: V2_HELDOUT - k_caps}
    out = dict(v1)
    for c in (True, False):
        for i, f in enumerate(rank(f for f in held if caps[f] == c)):
            out[f] = "heldout" if i < k[c] else "train"
    return out, v1


def font_key(row):
    """Stable short id of a bank font: its directory name (unique in the bank)."""
    return Path(row["path"]).parent.name


def load_bank():
    return json.loads(FONTS.read_text(encoding="utf-8"))


def by_key():
    return {font_key(r): r for r in load_bank()}


def counts_of(rows, split):
    return {s: {"families": sum(1 for v in split.values() if v == s),
                "caps_only": sum(1 for r in rows if split[r["family"]] == s and r.get("caps_only"))} for s in ("train", "heldout")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify fonts.json matches the pinned split")
    a = ap.parse_args()
    rows = load_bank()
    split, v1 = assign(rows)
    if a.check:
        bad = [r["family"] for r in rows if r.get("split") != split[r["family"]]]
        pinned = json.loads(SPLIT.read_text(encoding="utf-8"))
        assert not bad, f"split mismatch: {bad[:5]}"
        assert pinned.get("version") == VERSION, f"pinned split is version {pinned.get('version')}, expected {VERSION}"
        assert pinned["heldout_families"] == sorted(f for f, s in split.items() if s == "heldout")
        assert pinned["fonts_json_sha256"] == hashlib.sha256(FONTS.read_bytes()).hexdigest(), "fonts.json changed since pinning"
        print("font split OK (v%d):" % VERSION, pinned["counts"])
        return
    old = json.loads(SPLIT.read_text(encoding="utf-8")) if SPLIT.exists() else {}
    v1_record = old.get("v1") or {k: v for k, v in old.items() if k != "version"}
    assert v1_record.get("heldout_families") == sorted(f for f, s in v1.items() if s == "heldout"), "v1 record does not match the v1 rule"
    for r in rows:
        r["split"] = split[r["family"]]
    FONTS.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    counts = counts_of(rows, split)
    SPLIT.write_text(json.dumps(dict(
        version=VERSION,
        method=("v2: the 26 v1 held-out families ranked by sha256(salt+family) within caps-only / mixed-case strata; "
                f"first {V2_HELDOUT} (caps-only kept on both sides) stay held out, the rest join train"),
        salt=SALT, counts=counts,
        heldout_families=sorted(f for f, s in split.items() if s == "heldout"),
        heldout_keys=sorted(font_key(r) for r in rows if r["split"] == "heldout"),
        moved_to_train_v2=sorted(f for f in split if v1[f] == "heldout" and split[f] == "train"),
        fonts_json_sha256=hashlib.sha256(FONTS.read_bytes()).hexdigest(),
        v1=v1_record), indent=1), encoding="utf-8")
    print(counts)


if __name__ == "__main__":
    main()
