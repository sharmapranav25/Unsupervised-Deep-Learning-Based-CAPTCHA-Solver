# Font bank 02 — 175 OFL fonts (35 existing + 140 new)

Built 2026-10-01 by `feature_bank/acquire_fonts_02.py` from google/fonts at the **same pinned commit** as the Phase 2 bank (`23e54b51ddffbc7713c583748e3bd86f62b1fa4a`). The original 35 fonts are unchanged and referenced in place.

| File | Content |
|---|---|
| `fonts.json` | 175-row manifest: path, SHA-256, licence path/hash, source URL, category, selection reason, `caps_only`, `caps_like` |
| `fonts/<family>/font.ttf`, `OFL.txt` | The 140 new fonts and their licence texts |
| `acquisition.json` | Commit, seed, quotas, forced list, rejections with reasons |
| `specimen_new140.png` | `Ab3xQ7kz` in each new font; red label = caps-only |

## Selection

1. **Forced styles (33 accepted).** CAPTCHA-relevant appearances the 35-font bank lacked: outline/hollow (Bungee Outline, Londrina Outline, Monoton), inline/shadow (Bungee Inline/Shade, Londrina Shadow, Fascinate Inline), stencil (Stardos, Allerta), pixel/dotted (Press Start 2P, VT323, Silkscreen, DotGothic16, Codystar, Raleway Dots), heavy/distressed display (Black Ops One, Frijole, Creepster, Nosifer, Ewert) and marker/comic hands. Not taken: Pixelify Sans is already in the bank; Special Elite and Permanent Marker are Apache-licensed, not OFL; one name was misspelled.
2. **Category-stratified fill** to fixed quotas, ordered by `sha256("font-bank-02" + directory)`: SANS_SERIF 35, SERIF 30, DISPLAY 40, HANDWRITING 20, MONOSPACE 15. Symbol fonts and families without a latin subset are skipped.

## Checks per font

- OFL.txt present and containing the SIL OFL text.
- cmap covers all 62 ASCII alphanumerics.
- Each of the 62 renders with ink (PIL) and not as the `.notdef` box.
- Duplicate bytes rejected.

9 rejected: 8 without a latin subset, and Kumar One, which crashed FreeType.

## Caps-only fonts (22) — use for uppercase/digit text only

Lowercase renders as capitals, small capitals or unicase forms, so a lowercase label would be wrong:
- **Exact-bitmap check:** uppercase-only display faces.
- **Shape check `caps_like`:** each lowercase/uppercase pair is cropped to its ink box at 32×32, and the font is flagged when at least 16 of 19 unambiguous letters reach IoU ≥ 0.6. Calibration: small-caps fonts score 18–19; ordinary fonts score ≤10.
- **Manual flag:** Major Mono Display (unicase, recorded with a reason).

A few heavy display faces may be conservative false positives. That only costs lowercase variety, not label correctness.

**The renderer must honour `caps_only`.** Otherwise mixed-case labels drawn in these fonts are wrong.
