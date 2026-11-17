# Feature bank 02 — 30 additional CAPTCHA renderers

Built 2026-10-01. Extends feature bank 01 (Gregwar, Securimage, MobiCMS, simple-php-captcha; Laminas withheld) to **35 renderers**. Downloaded sources are never edited.

## What is here

| File | Content |
|---|---|
| `candidates.txt` | 64 candidate repositories, shortlisted from GitHub search (~660 hits) and Packagist (`captcha`, service clients filtered out) |
| `staging/<owner_repo>/` | Each candidate extracted at a pinned commit (`zips/` holds the archives) |
| `fetch.log` | Name, pinned commit, archive URL, archive SHA-256, size |
| `analysis.py` → `analysis.json` | Static triage: licence evidence, own renderer or not, wrapper dependencies, bundled fonts, effect keyword hits (leads, not verified effects) |
| `similarity.py` → `similarity.json` | Pairwise 5-token-shingle containment, used to detect forks/ports (also checked against bank 01) |
| `acquisition.json` | **Decision receipt**: every candidate with commit, hash, licence, status, reason, 20-seed smoke result |
| `source_files.json` | SHA-256 of all 1,549 extracted files of the 30 accepted libraries |
| `gallery_30_libraries.png` | Six smoke renders per accepted library |

## Decisions (64 candidates)

| Status | Count | Meaning |
|---|---:|---|
| `accepted` | 30 | Licensed, own renderer, not a fork; adapter written; renders labelled images |
| `reserve_not_adapted` | 10 | Licensed own renderer, adapter not written yet (mews/captcha, jasonencode, cocoq need Intervention Image; hyperf-ext needs Imagick; lubosdz, dereuromark, hamid-hpd, solutionforest, pavelmaca, zablose need more framework stand-ins) |
| `excluded_fork` | 7 | Ports/forks of Gregwar (webman, belomaxorka, simonetoo), think-captcha (thinkcmf, lzy309, Tinywan) or Laminas (zend-captcha ZF1) |
| `excluded_wrapper` | 4 | Depend on another CAPTCHA package for drawing |
| `excluded_no_renderer` | 6 | Validation/Turnstile integrations with no drawing code |
| `excluded_no_licence` | 6 | No licence file, so no permission to use (incl. OWASP PureCaptcha, PHPGangsta) |
| `excluded_not_static` | 1 | visavi: animated sliding-window reveal; no single frame shows the whole code |

Licences of the 30 accepted: MIT (most), BSD-3, Apache-2.0, GPL-2/3. **Bundled fonts/backgrounds carry their own, mostly unverified, terms** (RealCaptcha says so explicitly). Fine for local research training; check before redistributing any generated corpus.

## Running them

`adapters/bank02/harness.php` + one adapter per library; `feature_bank/bank02_smoke.py` runs each case in a fresh PHP 8.5 process (portable runtime from bank 01, GD enabled) and writes PNG + JSON metadata (label, shown text, sampled parameters, randomness notes, warnings, compatibility patches).

```powershell
$py = 'E:\pranav\model 1\.venv\Scripts\python.exe'
& $py -m feature_bank.bank02_smoke --out runs/bank02_smoke_NEW --seeds 20 [--libs name ...]
```

Harness behaviour:
- `mt_rand`/`rand` are seeded per case. Adapter parameter sampling uses a separate seeded stream, so it doesn't perturb library randomness. Libraries that use `random_int`/OpenSSL are marked not replayable.
- Image bytes are taken from the library's return value or captured stdout. PNG/JPEG/GIF are normalised to PNG; SVG is rasterised with `resources/tools/resvg` (v0.47.0, the latest release with a Windows build).
- Framework touch points (Laravel, ThinkPHP, Kohana, Fuel, Yii, Symfony, WordPress, DokuWiki) are replaced by minimal stand-ins (`_laravel_stubs.php` and per-adapter shims). Drawing code is always upstream.
- PHP 8 incompatibilities in unmaintained code (brace string offsets, nested ternaries, old `implode` order, float-to-int under strict types) are patched **in memory** via `bank_require_patched`. Each case's metadata records file hash and exact replacements.
- A case fails if there is no image, no label, an exception, or a blank (uniform) image.
- `label_case: insensitive` marks libraries whose drawn case differs from the stored label (lifei6671, RealCaptcha caps-only fonts, isszz, Kohana, innocead, DokuWiki SVG fonts).
- `text_kind: math` marks arithmetic CAPTCHAs (magkopian, mulertech); the label is the drawn expression.

20-seed smoke (`runs/bank02_smoke_02`): **593/600 renders succeeded**. The 7 failures are upstream colour-range bugs in P-Captcha (3), lifei6671 (2) and multi-captcha (2); handle them by redrawing.

## Known limitations

- Clipping is not yet filtered: P-Captcha, Tinywan-style think-captcha layouts and hnhx sometimes cut glyphs at the edge. A corpus build needs a per-sample glyph-visibility/clipping check before labels are trusted.
- Effect labels per image are not instrumented yet. The feature taxonomy (`docs/feature_bank/FEATURE_BUCKETS_02.md`) records what each library **can** draw, not what each image **shows**.
- freeCap draws a fixed caption ("freeCap v1.41 - puremango.co.uk"), and Fuel optionally draws a footer strip. These are upstream style, recorded as the `fixed_caption` head.
