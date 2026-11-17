"""Controller v2 PREVIEW: fixes the gaps found in the v1 custom/DINO/mixed controllers and adds a
SynthTIGER-rendered fourth arm. New, separately named outputs only; v1 code, entries and corpora are untouched.

DEFAULT MODE: 'free' (option A), set by the user 2026-11-14 after comparing A / B-strict / B+text / textonly previews.
History the same day: 'bounded' (B) was briefly the default, then widened to give text the full search (current
'bounded' = CAPRI background kept, strength +/-2, no replacement; text fully searched). The earlier strict B is
'bounded_strict'. 'textonly' keeps CAPRI's background verbatim including its 'random' colour. All remain via --mode.
Note: weight_absolute/bold_absolute were added to NUMERIC after the shown A previews (controller_v2b_diverse4_*), so a
new 'free' run searches two extra text controls compared with those images.

What changes relative to v1 (pipeline.py):
 1. Search no longer stops at a fixed grid edge: numeric settings (size, width, spacing, rotation,
    jitter, size jitter, slant) are refined by repeated coordinate steps around each arm's leader,
    so a winner at an edge keeps moving until the score stops improving or a renderer limit is hit.
 2. Rotation/spacing/etc. are searched AROUND CAPRI's own per-image value instead of being replaced
    by one menu constant.
 3. Font candidates: CAPRI's next-ranked fonts in its predicted category plus the best-ranked font of
    every other category (the font was never searched in v1).
 4. Background and effect candidates: background textures/noise/post effects, and +/- strength for the
    effects CAPRI predicted (v1 only had 'all effects off').
 5. Less noisy selection: 3 render seeds per candidate (v1: 2) and 4 fresh validation seeds (v1: 3).
 6. Arms that would pick an identical recipe take their next validated alternative when it is within 5%.
 7. Fourth arm 'synthtiger': the same search, but every candidate is drawn with SynthTIGER glyph
    rendering (synthtiger_recipe_adapter.glyph_mask), scored with the mixed objective.
Scoring objective, calibration, reference pixels and CAPRI starting recipes are reused unchanged from
the existing controller runs. No reader, no real labels, no VLM. Visual preview only: no corpus.
"""
import argparse, copy, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCES = dict(
    darkbay=ROOT / 'Capri/controllers/darkbay',
    cryptbb=ROOT / 'Capri/runs/cryptbb_model1_preparation_20261113/generator')
CAPRI_ARMS = ['custom', 'dino', 'mixed']
ARMS = CAPRI_ARMS + ['synthtiger']
METRICS = dict(custom='custom_objective', dino='dino_objective', mixed='objective', synthtiger='objective')
FIT_SEEDS, VAL_SEEDS = 3, 4
# (control, step, low, high, recipe field used as the starting centre)
NUMERIC = [('size_absolute', .08, .30, .95, 'size'), ('width_absolute', .12, .40, 1.90, 'x_scale'),
           ('spacing_absolute', .08, -.35, .60, 'spacing'), ('rotation_absolute', 5., 0., 40., 'rotation'),
           ('jitter_absolute', .04, 0., .30, 'jitter'), ('size_jitter_absolute', .05, 0., .30, 'size_jitter'),
           ('slant_absolute', .08, -.30, .50, 'slant'), ('weight_absolute', .15, 0., 1., 'weight'), ('bold_absolute', 1, 0, 2, 'bold_stroke')]
BACKGROUND_FX = ['gradient', 'texture', 'grid', 'dots', 'lines', 'curves', 'blobs', 'circles', 'pattern']
SURFACE_FX = ['grain', 'blur', 'noise_over', 'jpeg']
VERSION = 'v2b'  # v2 + background replacement, per-letter colours, text-contrast rejection
MIN_TEXT_CONTRAST = 40.  # RGB distance: median text colour vs median colour just around the text

# ---------------------------------------------------------------- worker side
_ST = None
def _install():
    from .rendering import renderer as R
    R.install()

def text_contrast(row):
    """Median text colour vs median colour of a thin ring around the text (rendered mask; no target data)."""
    from scipy import ndimage as ndi
    im = row['image'].astype(np.float32); m = row['mask'] > 127
    ring = ndi.binary_dilation(m, iterations=3) & ~m
    if m.sum() < 10 or ring.sum() < 10:return 0.
    return float(np.linalg.norm(np.median(im[m], 0) - np.median(im[ring], 0)))

def check_contrast(row):
    if row.get('image') is not None and row.get('mask') is not None:
        row['text_contrast'] = text_contrast(row)
        if row['text_contrast'] < MIN_TEXT_CONTRAST:row['usable'] = False; row['low_contrast'] = True
    return row

def render_job(item):
    """(recipe, controls, engine) -> renderer row. engine 'synthtiger' swaps in SynthTIGER glyphs."""
    global _ST
    from .rendering import renderer as R
    recipe, controls, engine = item
    if engine != 'synthtiger':return check_contrast(R.safe_render((recipe, controls)))
    if _ST is None:
        if not hasattr(np, 'sctypes'):
            # imgaug (imported by synthtiger) reads np.sctypes at import; NumPy 2 removed it. Same type lists as NumPy 1.x.
            np.sctypes = dict(int=[np.int8, np.int16, np.int32, np.int64], uint=[np.uint8, np.uint16, np.uint32, np.uint64],
                              float=[np.float16, np.float32, np.float64], complex=[np.complex64, np.complex128], others=[bool, object, bytes, str, np.void])
        from . import synthtiger_recipe_adapter as ST
        _ST = ST
    before = R.ORIGINAL
    np.random.seed(int(recipe['seed']) % 2**32)
    try:
        R.ORIGINAL = _ST.glyph_mask
        row = R.safe_render((recipe, controls))
    finally:R.ORIGINAL = before
    row['engine'] = 'synthtiger'
    return check_contrast(row)

# ---------------------------------------------------------------- main side
def font_candidates(description, current, k=5):
    from feature_bank.iter2.controller import bank
    from feature_bank.iter2.generate import CATEGORIES
    from captcha_effects import text as T
    b = bank(); font = description['font']; caseless = description['content'] in ('digits', 'upper')
    rank = lambda f: (int(bool(f.get('italic')) != bool(font['italic'])), int(f['stroke'] != font['stroke']),
                      abs((T.x_height_level(f) if T.x_height_level(f) is not None else 5) - font['x_height']))
    out = []
    for f in sorted(b.pool('train', category=font['category'], caseless=caseless), key=rank):
        if f['key'] not in current and f['key'] not in out:out.append(f['key'])
        if len(out) >= k:break
    for cat in CATEGORIES:
        if cat == font['category']:continue
        pool = sorted(b.pool('train', category=cat, caseless=caseless), key=rank)
        if pool and pool[0]['key'] not in out:out.append(pool[0]['key'])
    return out

def palette(im, mask):
    from PIL import Image
    im = np.asarray(Image.fromarray(im).resize((256, 64)))
    def reduce(where):
        x = im[where]
        if len(x) < 16:x = im.reshape(-1, 3)
        q = Image.fromarray(x.reshape(1, -1, 3)).quantize(colors=5); p = q.getpalette()
        return [p[3*i:3*i+3] for _, i in sorted(q.getcolors(), reverse=True)]
    return reduce(mask > .5), reduce(mask < .2)

def letter_colors(im, mask, bg, k=8):
    """One median colour per letter-sized ink blob of the real image (pixel arithmetic only)."""
    from PIL import Image
    from scipy import ndimage as ndi
    im = np.asarray(Image.fromarray(im).resize((256, 64))).astype(np.float32)
    labels, n = ndi.label(mask > .6); out = []
    for i in range(1, n + 1):
        sel = labels == i
        if sel.sum() < 15:continue
        core = ndi.binary_erosion(sel)
        c = np.median(im[core if core.sum() >= 5 else sel], 0)
        if np.linalg.norm(c - np.array(bg)) < 60:continue
        if all(np.linalg.norm(c - np.array(o)) >= 35 for o in out):out.append([int(v) for v in c])
    return out[:k]

def draw_recipe(base, seed):
    from .rendering import renderer as R
    r = copy.deepcopy(base); r['seed'] = int(seed)
    r['text_spec']['text'] = R.C.R.T.make_text(np.random.default_rng(seed + 101), int(r['description']['length']), r['description']['content'])
    return r

def usable(row, length):return bool(row.get('complete') and row.get('usable') and len(row.get('text', '')) == length)
def key(p):return json.dumps(p, sort_keys=True)
def slice_features(features, idx):return tuple({k: v[idx] for k, v in part.items()} for part in features)

def pick_diverse(whole, n):
    """Most typical reference first, then farthest-point picks on standardized whole-image descriptors."""
    x = (whole - whole.mean(0)) / np.maximum(whole.std(0), 1e-6)
    picks = [int(np.argmin(np.linalg.norm(x, axis=1)))]
    while len(picks) < n:
        d = np.min(np.stack([np.linalg.norm(x - x[i], axis=1) for i in picks]), axis=0)
        picks.append(int(np.argmax(d)))
    return picks

def run(dataset, inputs, out, workers, diverse=False, mode='free'):
    """mode 'free' (option A): controllers may override CAPRI. mode 'bounded' (option B): stay within one step of
    CAPRI's own values, CAPRI's top-2 fonts in its predicted category, CAPRI's effect set (strength +/-2 only), no
    background replacement. Pixel colours are allowed in both, because CAPRI predicts no colour."""
    bounded = mode in ('bounded', 'bounded_strict')   # background rules of option B
    text_bounded = mode == 'bounded_strict'            # text also within one step of CAPRI (earlier B)
    # mode 'textonly': background stays exactly as CAPRI predicts (its background colour setting and every effect);
    # controllers adjust only font, text style (slant/rotation/jitter/width/spacing/outline), text colour and size.
    textonly = mode == 'textonly'
    src = SOURCES[dataset]
    if out.exists():raise FileExistsError('Preserve evidence; choose a new output: ' + str(out))
    out.mkdir(parents=True)
    refs = json.loads((src / 'inputs.json').read_text(encoding='utf-8-sig'))
    local_refs = src / 'reference_images'
    for r in refs:  # prefer the in-workspace copy when present; hash is re-verified on load
        p = local_refs / Path(r['source']).name
        if p.exists():r['source'] = str(p)
    (out / 'targets.json').write_text(json.dumps(dict(fit=refs), indent=1))
    os.environ['CAPRI_RUN_DIR'] = str(out); os.environ['CAPRI_TARGET_MANIFEST'] = str(out / 'targets.json')
    from . import runtime as U
    from .controllers import Mixed, Objective
    from .rendering import renderer as R
    from .artifacts import load_images
    from concurrent.futures import ProcessPoolExecutor
    from PIL import Image
    started = time.time()
    recipes = U.read(src / 'refined_recipes.json'); old_entries = U.read(src / 'entries.json')
    length = int(U.read(src / 'length_decision.json')['default_length'])
    real = load_images(refs); model = Mixed(); target = model.extract(real)
    with np.load(src / 'calibration_features.npz', allow_pickle=False) as z:
        calibration = tuple({k.split('__', 1)[1]: z[k] for k in z.files if k.startswith(str(i) + '__')} for i in range(2))
    obj = Objective(target, calibration)
    chosen_inputs = pick_diverse(np.asarray(target[0]['whole']), inputs) if diverse else list(range(inputs))
    U.dump(out / 'protocol.json', dict(dataset=dataset, source_run=str(src), preview_inputs=chosen_inputs, input_selection='most typical + farthest-point on custom whole-image descriptors (no labels, no visual cherry-picking)' if diverse else 'first N', arms=ARMS,
        reused_unchanged=['reference pixels', 'CAPRI refined recipes', 'calibration features', 'objective', 'model-derived length'],
        fit_seeds=FIT_SEEDS, validation_seeds=VAL_SEEDS, numeric_search=NUMERIC, mode=mode, version=VERSION, min_text_contrast=MIN_TEXT_CONTRAST, background='replacement candidates over every under-text effect', surface_fx=SURFACE_FX,
        target_labels=False, reader_used=False, vlm=False, corpus=False, code_sha256=U.sha(Path(__file__))))
    entries = []; histories = []; samples = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_install) as pool:
        def evaluate(records, seen, candidates, engine, phase, fit_recipes):
            fresh = []
            for p in candidates:
                p = R.expand(p)
                if (engine, key(p)) not in seen:seen.add((engine, key(p))); fresh.append(p)
            if not fresh:return
            rows = list(pool.map(render_job, [(r, p, engine) for p in fresh for r in fit_recipes], chunksize=4))
            imgs = [r.get('image', np.full((64, 256, 3), 255, np.uint8)) for r in rows]
            f = model.extract(imgs); n = len(fit_recipes)
            for j, p in enumerate(fresh):
                rr = rows[j*n:(j+1)*n]; ff = slice_features(f, slice(j*n, (j+1)*n))
                g = obj.score(ff, target, rr); l = obj.score(ff, local, rr)
                arms = CAPRI_ARMS if engine == 'capri' else ['synthtiger']
                records.append(dict(index=len(records), engine=engine, phase=phase, controls=p, usable=sum(usable(r, length) for r in rr),
                    scores={a: .5*(g[METRICS[a]] + l[METRICS[a]]) for a in arms}))
        for index in chosen_inputs:
            base = recipes[index]; ts = base['text_spec']; d = base['description']
            local = slice_features(target, slice(index, index + 1))
            colors, bgs = palette(real[index], target[0]['mask'][index]); bg = bgs[0]
            ink = [c for c in colors if np.linalg.norm(np.array(c) - np.array(bg)) >= 60] or [max(colors, key=lambda c: np.linalg.norm(np.array(c) - np.array(bg)))]
            start = dict(length=length, background=bg, text_palette=ink, effect_palette=ink, color_mode='alternating' if len(ink) > 1 else 'single')
            if textonly:start = dict(length=length, text_palette=ink, color_mode='alternating' if len(ink) > 1 else 'single')  # no background / effect colours
            neutral = dict(length=length)
            present = {e['name']: e['level'] for e in base['effects_spec']['effects']}
            UNDER = list(R.C.R.R.UNDER); keep = [n for n in present if n not in UNDER]
            letters = letter_colors(real[index], target[0]['mask'][index], bg)
            colour = [{**start, 'color_mode': 'random'}]
            if len(letters) > 1:
                colour += [{**start, 'text_palette': letters, 'color_mode': m} for m in ('alternating', 'random')]
                if not textonly:colour += [{**start, 'text_palette': letters, 'effect_palette': letters, 'color_mode': 'random'}]
            background = [{**start, 'background_pool': bgs, 'background': None}, {**start, 'effects': {n: 0 for n in R.C.R.R.SAMPLED}},
                          {**start, 'keep_only': keep, 'effects': {}}]
            # REPLACE CAPRI's background: drop its under-text effects, then try each background type alone,
            # with ink-coloured effects and with the renderer's own effect colours.
            for n in UNDER:
                background += [{**start, 'keep_only': keep + [n], 'effects': {n: 3}}, {**start, 'keep_only': keep + [n], 'effects': {n: 3}, 'effect_palette': None}]
            if bounded:background = [{**start, 'background_pool': bgs, 'background': None}]
            if textonly:background = []
            groups = dict(
                base=[neutral, start, {**start, 'glyph_outline': 1}],
                geometry=([{**start, 'size_absolute': v} for v in [.54, .70]] + [{**start, 'width_absolute': round(float(np.clip(ts['x_scale'] + s, .4, 1.9)), 4)} for s in (-.12, .12)]) if text_bounded else
                         [{**start, 'size_absolute': v} for v in [.5, .65, .8, .9]] + [{**start, 'width_absolute': v} for v in [.75, 1.1, 1.4]],
                font=[{**start, 'font': f} for f in (font_candidates(d, ts['fonts'], k=2)[:2] if text_bounded else font_candidates(d, ts['fonts']))],
                colour=colour, background=background,
                surface=[] if textonly else ([] if bounded else [{**start, 'effects': {n: 2}} for n in SURFACE_FX]) + [{**start, 'effects': {n: int(np.clip(lv + s, 1, 10))}} for n, lv in present.items() for s in (-2, 2)])
            old = {e['controller']: R.expand(e['controls']) for e in old_entries if e['input_index'] == index}
            fit_recipes = [draw_recipe(base, 1200003 + index*1009 + j*104729) for j in range(FIT_SEEDS)]
            records = []; seen = set()
            for engine in ['capri', 'synthtiger']:
                for g, cands in groups.items():evaluate(records, seen, cands, engine, 'initial_' + g, fit_recipes)
                if not (bounded or textonly):evaluate(records, seen, list(old.values()), engine, 'v1_choice', fit_recipes)
                arms = CAPRI_ARMS if engine == 'capri' else ['synthtiger']
                # merge: overall leader + best font / colour / background choice, and pairs of the best replacement backgrounds
                OWN = dict(font=['font'], colour=['text_palette', 'effect_palette', 'color_mode'],
                           background=['keep_only', 'effects', 'background', 'background_pool', 'effect_palette'])
                merged = []
                for arm in arms:
                    mine = [r for r in records if r['engine'] == engine]; lead = min(mine, key=lambda r: r['scores'][arm])['controls']
                    combo = dict(lead)
                    for g, names in OWN.items():
                        cands = [r for r in mine if r['phase'] == 'initial_' + g]
                        if not cands:continue
                        c = min(cands, key=lambda r: r['scores'][arm])['controls']
                        merged.append({**lead, **{k2: c[k2] for k2 in names}}); combo.update({k2: c[k2] for k2 in names})
                    merged.append(combo)
                    top = [] if bounded else sorted((r for r in mine if r['phase'] == 'initial_background' and r['controls'].get('keep_only') is not None and r['controls']['effects']),
                                 key=lambda r: r['scores'][arm])[:3]
                    for i1 in range(len(top)):
                        for i2 in range(i1 + 1, len(top)):
                            a1, a2 = top[i1]['controls'], top[i2]['controls']
                            merged.append({**combo, 'keep_only': sorted(set(a1['keep_only']) | set(a2['keep_only'])),
                                           'effects': {**a1['effects'], **a2['effects']}, 'effect_palette': a1['effect_palette']})
                evaluate(records, seen, merged, engine, 'merge', fit_recipes)
                for round_ in range(3):
                    proposals = []
                    for arm in arms:
                        lead = min((r for r in records if r['engine'] == engine), key=lambda r: r['scores'][arm])['controls']
                        for name, step, lo, hi, field in NUMERIC:
                            centre = lead.get(name)
                            if centre is None:centre = .5 if field == 'weight' and ts.get('weight') is None else float(ts.get(field) or 0.)
                            centre = float(centre)
                            if text_bounded:  # never more than one step from CAPRI's own value
                                capri = float(ts.get(field) or 0.); lo, hi = max(lo, capri - step), min(hi, capri + step)
                            for s in (-step, step):proposals.append({**lead, name: int(np.clip(centre + s, lo, hi)) if name == 'bold_absolute' else round(float(np.clip(centre + s, lo, hi)), 4)})
                        # effect-strength refinement around the leader
                        if lead.get('effects'):
                            for n, lv in lead['effects'].items():
                                if lv and bounded and n in present:proposals += [{**lead, 'effects': {**lead['effects'], n: int(np.clip(lv + s, max(1, present[n] - 2), min(10, present[n] + 2)))}} for s in (-1, 1)]
                                elif lv and not bounded and not textonly:proposals += [{**lead, 'effects': {**lead['effects'], n: int(np.clip(lv + s, 1, 10))}} for s in (-1, 1)]
                    evaluate(records, seen, proposals, engine, f'refine_{round_ + 1}', fit_recipes)
            # validation on fresh seeds
            vr = [draw_recipe(base, 1300003 + index*1009 + j*104729) for j in range(VAL_SEEDS)]
            validated = {}
            for arm in ARMS:
                engine = 'synthtiger' if arm == 'synthtiger' else 'capri'
                pool_rec = sorted((r for r in records if r['engine'] == engine), key=lambda r: r['scores'][arm])[:4]
                for rec in pool_rec:
                    k = (engine, key(rec['controls']))
                    if k in validated:continue
                    rows = list(pool.map(render_job, [(r, rec['controls'], engine) for r in vr]))
                    if not all(usable(r, length) for r in rows):validated[k] = None; continue
                    ff = model.extract([r['image'] for r in rows]); g = obj.score(ff, target, rows); l = obj.score(ff, local, rows)
                    a = CAPRI_ARMS if engine == 'capri' else ['synthtiger']
                    validated[k] = dict(**rec, validation={x: .5*(g[METRICS[x]] + l[METRICS[x]]) for x in a})
            chosen = {}
            for arm in ARMS:
                engine = 'synthtiger' if arm == 'synthtiger' else 'capri'
                cand = sorted((v for (e, _), v in validated.items() if v and e == engine), key=lambda v: v['validation'][arm])
                if not cand:
                    # no candidate usable on all 4 validation seeds (e.g. CAPRI's random background colour can clash with the text):
                    # fall back to the best fit-phase record with the most usable renders, and mark it.
                    mine = [r for r in records if r['engine'] == engine and r['usable'] > 0]
                    if not mine:raise RuntimeError(f'No usable recipe for {dataset} input {index} arm {arm}; retained for inspection')
                    best = max(mine, key=lambda r: (r['usable'], -r['scores'][arm]))
                    cand = [dict(**best, validation={x: best['scores'][x] for x in best['scores']}, unvalidated=True)]
                pick = cand[0]; taken = {key(c['controls']) for a, c in chosen.items() if a != 'synthtiger'}
                if arm != 'synthtiger' and key(pick['controls']) in taken:
                    alt = [c for c in cand[1:] if key(c['controls']) not in taken and c['validation'][arm] <= 1.05*pick['validation'][arm]]
                    if alt:pick = dict(**alt[0], deduplicated=True)
                chosen[arm] = pick
            v1_val = {}
            for arm in CAPRI_ARMS:  # v1 choice re-scored with the same 4 validation seeds for a like-for-like comparison
                rows = list(pool.map(render_job, [(r, old[arm], 'capri') for r in vr]))
                ff = model.extract([r.get('image', np.full((64, 256, 3), 255, np.uint8)) for r in rows]); g = obj.score(ff, target, rows); l = obj.score(ff, local, rows)
                v1_val[arm] = .5*(g[METRICS[arm]] + l[METRICS[arm]])
            for arm, pick in chosen.items():
                engine = 'synthtiger' if arm == 'synthtiger' else 'capri'
                show = [draw_recipe(base, 1400003 + index*1009 + j*10000019) for j in range(4)]
                rows = list(pool.map(render_job, [(r, pick['controls'], engine) for r in show]))
                samples[(index, arm)] = [r for r in rows if usable(r, length)]
                entries.append(dict(entry_id=f'{arm}_{index:03d}', controller=arm, engine=engine, input_index=index, controls=pick['controls'],
                    selected_phase=pick['phase'], validation_score=pick['validation'][arm], v1_validation_score=v1_val.get(arm),
                    deduplicated=bool(pick.get('deduplicated')), unvalidated=bool(pick.get('unvalidated')), candidates_tried=sum(r['engine'] == engine for r in records)))
            histories.append(dict(input_index=index, records=records))
            U.dump(out / 'entries.json', entries)
            U.dump(out / 'status.json', dict(processed=len(histories), total=len(chosen_inputs), seconds=time.time() - started))
            print(dataset, 'input', len(histories), '/', len(chosen_inputs), f'{time.time() - started:.0f}s', flush=True)
    U.dump(out / 'controller_history.json', histories)
    folder = out / 'samples'; folder.mkdir()
    for (i, arm), rows in samples.items():
        for j, r in enumerate(rows):Image.fromarray(r['image']).save(folder / f'{i:03d}_{arm}_{j}.png')
    sheet(dataset, src, out, refs, real, chosen_inputs, samples)
    U.dump(out / 'status.json', dict(state='complete', inputs=chosen_inputs, seconds=time.time() - started))

def sheet(dataset, src, out, refs, real, chosen, samples):
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 18); small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14)
    cols = ['Real', 'v1 custom', 'v1 DINO', 'v1 mixed', 'v2 custom', 'v2 DINO', 'v2 mixed', 'v2 SynthTIGER']
    W, H = 210, 80
    im = Image.new('RGB', (W*len(cols) + 20, 50 + len(chosen)*(H + 14)), '#f4f6f8'); d = ImageDraw.Draw(im)
    for j, c in enumerate(cols):d.text((10 + j*W, 14), c, font=font, fill='#182033')
    for row, i in enumerate(chosen):
        tiles = [Image.fromarray(real[i])] + [Image.open(src / a / f'{i:04d}.png').convert('RGB') for a in CAPRI_ARMS]
        tiles += [Image.fromarray(samples[(i, a)][0]['image']) if samples.get((i, a)) else Image.new('RGB', (200, 60), '#ffdddd') for a in ARMS]
        for j, t in enumerate(tiles):
            t = ImageOps.contain(t, (W - 12, H), Image.Resampling.NEAREST); im.paste(t, (10 + j*W, 46 + row*(H + 14)))
    im.save(out / f'{dataset}_v1_vs_v2.png')
    # variety sheet: 4 fresh samples per v2 arm for the first 4 inputs
    n = min(4, len(chosen)); im = Image.new('RGB', (20 + 5*W, 40 + n*len(ARMS)*(H + 10)), '#f4f6f8'); d = ImageDraw.Draw(im); y = 10
    for i in chosen[:n]:
        for a in ARMS:
            d.text((10, y + 28), f'{i}:{a}', font=small, fill='#182033')
            t = ImageOps.contain(Image.fromarray(real[i]), (W - 12, H), Image.Resampling.NEAREST); im.paste(t, (110, y))
            for j, r in enumerate(samples.get((i, a), [])[:3]):
                t = ImageOps.contain(Image.fromarray(r['image']), (W - 12, H), Image.Resampling.NEAREST); im.paste(t, (110 + (j + 1)*W, y))
            y += H + 10
    im.save(out / f'{dataset}_v2_samples.png')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dataset', choices=list(SOURCES), required=True)
    ap.add_argument('--inputs', type=int, default=12); ap.add_argument('--workers', type=int, default=8); ap.add_argument('--out', required=True); ap.add_argument('--diverse', action='store_true'); ap.add_argument('--mode', choices=['free', 'bounded', 'bounded_strict', 'textonly'], default='free')
    a = ap.parse_args(); out = Path(a.out).resolve(); assert out.drive.upper() == 'E:'
    run(a.dataset, a.inputs, out, a.workers, a.diverse, a.mode)

if __name__ == '__main__':main()
