"""Controller v3 PREVIEW: one job per controller ("split"), plus a parallel feature extractor.

Requested by the user 2026-11-14 after v2 arms produced near-identical images. Job assignment follows the judge
sensitivity test in Capri/runs/controller_split_sensitivity_20261114 (change one group at a time, measure which
judge's score moves):
  custom      fits thickness (weight/bold) + text colour        judged by custom   (custom sees thickness/font/colour best)
  dino        fits background/effects + size/width/spacing      judged by DINO     (DINO sees background and size best)
  synthtiger  fits font, drawn with SynthTIGER glyphs           judged by custom   (custom sees font best)
  mixed       combines the three winners, then +/-1-step refine  judged by mixed; contrast/usability fallback to
              next-best specialist choices
Slant, rotation and wobble (jitter/size jitter) stay CAPRI's for every arm: neither judge reacted to them above noise.
Everything not owned by an arm stays at the shared start (CAPRI recipe + reference pixel colours, as in v2).

Speed: FastMixed computes exactly the features of controllers.Mixed (verified bitwise by --verify), but the per-image
NumPy/SciPy work runs in the process pool instead of one parent core, and the discarded 'old' custom feature pass is
skipped (only its encoder embedding 'whole' was used). GPU passes keep the same batch partitioning.
Reuses v2 rendering, seeds, objective, calibration and reference pixels unchanged. No reader, labels or VLM.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
from . import controller_v2 as V

VERSION = 'v3split'
ARMS = ['custom', 'dino', 'synthtiger', 'mixed']
ENGINE = dict(custom='capri', dino='capri', synthtiger='synthtiger', mixed='capri')
JUDGE = dict(custom='custom_objective', dino='dino_objective', synthtiger='custom_objective', mixed='objective')
OWNS = dict(custom=['weight_absolute', 'bold_absolute', 'text_palette', 'effect_palette', 'color_mode'],
            dino=['keep_only', 'effects', 'background', 'background_pool', 'size_absolute', 'width_absolute', 'spacing_absolute'],
            synthtiger=['font'])
STEPS = dict(weight_absolute=(.15, 0., 1.), bold_absolute=(1, 0, 2), size_absolute=(.08, .30, .95),
             width_absolute=(.12, .40, 1.90), spacing_absolute=(.08, -.35, .60))

# ---------------------------------------------------------------- parallel, exact feature extraction
def _features_job(item):
    """Per-image CPU part of Mixed.extract (same functions, same inputs)."""
    from .features.base import descriptors
    from .features.regions import extra
    from .features.custom import ink_shape
    from .features.dino import pixel_features
    px, mask, z, dpx, focus = item
    new = {k: v[0] for k, v in descriptors(px[None], mask[None], z[None]).items()}
    glyph, layout, bg, diag = extra(px, mask)
    new.update(glyph_shape=glyph, layout_v2=layout, background_structure=bg, mask_diagnostic=diag, ink_shape=ink_shape(px, mask))
    return new, pixel_features((dpx, focus))

class FastMixed:
    def __init__(self, pool):
        from .controllers import Mixed
        self.m = Mixed(); self.pool = pool
    def extract(self, images):
        import torch
        from PIL import Image
        c, d = self.m.custom, self.m.dino
        with torch.inference_mode():
            pixels = np.stack([np.asarray(Image.fromarray(im).resize((256, 64), Image.Resampling.BILINEAR)) for im in images])
            zs = []; masks = []
            for s in range(0, len(pixels), 32):  # encoder embedding = the only part of the old pass that Mixed keeps
                x = torch.tensor(pixels[s:s+32].transpose(0, 3, 1, 2).copy(), device=c.device).float()/255
                zs.extend(c.encoder(x)['z'].cpu().numpy())
            for s in range(0, len(pixels), 32):
                x = torch.tensor(pixels[s:s+32].transpose(0, 3, 1, 2).copy(), device=c.device).float()/255
                masks.extend(c.segmenter(x).sigmoid()[:, 0].cpu().numpy())
            fd = super(type(d), d).extract(images)  # DINO GPU part (DinoComparator.extract)
        masks = np.asarray(masks); zs = np.asarray(zs)
        dpx = [np.asarray(Image.fromarray(im).convert('RGB').resize((256, 64), Image.Resampling.BILINEAR)) for im in images]
        out = list(self.pool.map(_features_job, [(pixels[i], masks[i], zs[i], dpx[i], fd['focus'][i]) for i in range(len(images))], chunksize=2))
        new = {k: np.asarray([o[0][k] for o in out]) for k in out[0][0]}; new['mask'] = masks
        for k in out[0][1]:fd[k] = np.asarray([o[1][k] for o in out])
        return new, fd

def verify(dataset, n=84):
    """Bitwise comparison FastMixed vs controllers.Mixed on rendered images."""
    from concurrent.futures import ProcessPoolExecutor
    from .rendering import renderer as R
    from . import runtime as U
    src = V.SOURCES[dataset]; recipes = U.read(src / 'refined_recipes.json')
    with ProcessPoolExecutor(max_workers=12, initializer=V._install) as pool:
        rows = list(pool.map(V.render_job, [(V.draw_recipe(recipes[i % 7], 900001 + i), R.expand(dict(length=6)), 'capri') for i in range(n)]))
        imgs = [r['image'] for r in rows if r.get('image') is not None]
        fast = FastMixed(pool); fast.extract(imgs[:24]); t = time.time(); a = fast.extract(imgs); tf = time.time() - t
        t = time.time(); b = fast.m.extract(imgs); ts = time.time() - t
    worst = 0.
    for i in range(2):
        for k in b[i]:
            x, y = np.asarray(a[i][k], float), np.asarray(b[i][k], float)
            assert x.shape == y.shape, (i, k, x.shape, y.shape); worst = max(worst, float(np.abs(x - y).max()) if x.size else 0.)
        assert set(a[i]) == set(b[i]), (i, set(a[i]) ^ set(b[i]))
    print(f'verify {dataset}: {len(imgs)} imgs, max abs diff {worst}, fast {tf:.2f}s vs serial {ts:.2f}s'); return worst

# ---------------------------------------------------------------- split controllers
def run(dataset, inputs, out, workers, input_list=None):
    from concurrent.futures import ProcessPoolExecutor
    src = V.SOURCES[dataset]
    if out.exists():raise FileExistsError('Preserve evidence; choose a new output: ' + str(out))
    out.mkdir(parents=True)
    refs = json.loads((src / 'inputs.json').read_text(encoding='utf-8-sig'))
    for r in refs:
        p = src / 'reference_images' / Path(r['source']).name
        if p.exists():r['source'] = str(p)
    (out / 'targets.json').write_text(json.dumps(dict(fit=refs), indent=1))
    os.environ['CAPRI_RUN_DIR'] = str(out); os.environ['CAPRI_TARGET_MANIFEST'] = str(out / 'targets.json')
    from . import runtime as U
    from .controllers import Objective
    from .rendering import renderer as R
    from .artifacts import load_images
    started = time.time()
    recipes = U.read(src / 'refined_recipes.json'); old_entries = U.read(src / 'entries.json')
    length = int(U.read(src / 'length_decision.json')['default_length'])
    chosen = input_list if input_list is not None else list(range(inputs))
    U.dump(out / 'protocol.json', dict(dataset=dataset, version=VERSION, inputs=chosen, arms=ARMS, engine=ENGINE, judge=JUDGE, owns=OWNS,
        fixed_at_capri=['slant', 'rotation', 'jitter', 'size_jitter'], fit_seeds=V.FIT_SEEDS, validation_seeds=V.VAL_SEEDS,
        min_text_contrast=V.MIN_TEXT_CONTRAST, split_evidence='Capri/runs/controller_split_sensitivity_20261114',
        target_labels=False, reader_used=False, vlm=False, corpus=False, code_sha256=U.sha(Path(__file__))))
    entries = []; samples = {}; histories = []
    with ProcessPoolExecutor(max_workers=workers, initializer=V._install) as pool:
        real = load_images(refs); model = FastMixed(pool); target = model.extract(real)
        with np.load(src / 'calibration_features.npz', allow_pickle=False) as z:
            cal = tuple({k.split('__', 1)[1]: z[k] for k in z.files if k.startswith(str(i) + '__')} for i in range(2))
        obj = Objective(target, cal)
        for index in chosen:
            base = recipes[index]; ts = base['text_spec']; d = base['description']
            local = V.slice_features(target, slice(index, index + 1))
            colors, bgs = V.palette(real[index], target[0]['mask'][index]); bg = bgs[0]
            ink = [c for c in colors if np.linalg.norm(np.array(c) - np.array(bg)) >= 60] or [max(colors, key=lambda c: np.linalg.norm(np.array(c) - np.array(bg)))]
            start = dict(length=length, background=bg, text_palette=ink, effect_palette=ink, color_mode='alternating' if len(ink) > 1 else 'single')
            present = {e['name']: e['level'] for e in base['effects_spec']['effects']}
            UNDER = list(R.C.R.R.UNDER); keep = [n for n in present if n not in UNDER]
            letters = V.letter_colors(real[index], target[0]['mask'][index], bg)
            fit = [V.draw_recipe(base, 1200003 + index*1009 + j*104729) for j in range(V.FIT_SEEDS)]
            cache = {}   # (engine, controls) -> scores for every judge, fit seeds
            def evaluate(cands, engine):
                fresh = []
                for p in cands:
                    p = R.expand(p); k = (engine, V.key(p))
                    if k not in cache and k not in [(engine, V.key(f)) for f in fresh]:fresh.append(p)
                if fresh:
                    rows = list(pool.map(V.render_job, [(r, p, engine) for p in fresh for r in fit], chunksize=4))
                    f = model.extract([r.get('image', np.full((64, 256, 3), 255, np.uint8)) for r in rows]); n = len(fit)
                    for j, p in enumerate(fresh):
                        rr = rows[j*n:(j+1)*n]; ff = V.slice_features(f, slice(j*n, (j+1)*n))
                        g = obj.score(ff, target, rr); l = obj.score(ff, local, rr)
                        cache[(engine, V.key(p))] = dict(controls=p, engine=engine, usable=sum(V.usable(r, length) for r in rr),
                            scores={m: .5*(g[m] + l[m]) for m in set(JUDGE.values())})
                return [cache[(engine, V.key(R.expand(p)))] for p in cands]
            def best(recs, arm, k=1):return sorted(recs, key=lambda r: r['scores'][JUDGE[arm]])[:k]
            def refine(arm, recs, names, rounds, engine, effects=False):
                for _ in range(rounds):
                    lead = best(recs, arm)[0]['controls']; props = []
                    for nm in names:
                        step, lo, hi = STEPS[nm]; c = lead.get(nm)
                        if c is None:c = .5 if nm == 'weight_absolute' and ts.get('weight') is None else float(ts.get(dict(weight_absolute='weight', bold_absolute='bold_stroke', size_absolute='size', width_absolute='x_scale', spacing_absolute='spacing')[nm]) or 0.)
                        for s in (-step, step):props.append({**lead, nm: int(np.clip(c + s, lo, hi)) if nm == 'bold_absolute' else round(float(np.clip(float(c) + s, lo, hi)), 4)})
                    if effects and lead.get('effects'):
                        for n, lv in lead['effects'].items():
                            if lv:props += [{**lead, 'effects': {**lead['effects'], n: int(np.clip(lv + s, 1, 10))}} for s in (-1, 1)]
                    before = best(recs, arm)[0]['scores'][JUDGE[arm]]
                    recs = recs + evaluate(props, engine)
                    if best(recs, arm)[0]['scores'][JUDGE[arm]] >= before:break  # no improvement: stop
                return recs
            recs = {}
            # custom: thickness + text colour
            colour = [start, {**start, 'color_mode': 'random'}]
            if len(letters) > 1:colour += [{**start, 'text_palette': letters, 'color_mode': m} for m in ('alternating', 'random')] + [{**start, 'text_palette': letters, 'effect_palette': letters, 'color_mode': 'random'}]
            thick = [{**start, 'weight_absolute': w} for w in (.2, .35, .5, .65, .8)] + [{**start, 'bold_absolute': b} for b in (0, 1, 2)]
            r_col = evaluate(colour, 'capri'); r_th = evaluate(thick, 'capri')
            bc = best(r_col, 'custom')[0]['controls']; bt = best(r_th, 'custom')[0]['controls']
            combo = {**bt, **{k: bc[k] for k in ('text_palette', 'effect_palette', 'color_mode')}}
            recs['custom'] = refine('custom', r_col + r_th + evaluate([combo], 'capri'), ['weight_absolute', 'bold_absolute'], 3, 'capri')
            # dino: background/effects + size/width/spacing
            bgc = [start, {**start, 'background_pool': bgs, 'background': None}, {**start, 'effects': {n: 0 for n in R.C.R.R.SAMPLED}}, {**start, 'keep_only': keep, 'effects': {}}]
            for n in UNDER:bgc += [{**start, 'keep_only': keep + [n], 'effects': {n: 3}}, {**start, 'keep_only': keep + [n], 'effects': {n: 3}, 'effect_palette': None}]
            bgc += [{**start, 'effects': {n: 2}} for n in V.SURFACE_FX] + [{**start, 'effects': {n: int(np.clip(lv + s, 1, 10))}} for n, lv in present.items() for s in (-2, 2)]
            geo = [{**start, 'size_absolute': v} for v in (.5, .65, .8, .9)] + [{**start, 'width_absolute': v} for v in (.75, 1.1, 1.4)] + [{**start, 'spacing_absolute': v} for v in (-.15, .15, .35)]
            r_bg = evaluate(bgc, 'capri'); r_geo = evaluate(geo, 'capri')
            bb = best(r_bg, 'dino')[0]['controls']; bg_ = best(r_geo, 'dino')[0]['controls']
            G = ('size_absolute', 'width_absolute', 'spacing_absolute'); B = ('keep_only', 'effects', 'background', 'background_pool', 'effect_palette')
            merged = [{**bb, **{k: bg_[k] for k in G if k in bg_}}]
            top = [r['controls'] for r in best([r for r in r_bg if r['controls'].get('keep_only') is not None and r['controls']['effects']], 'dino', 3)]
            for i1 in range(len(top)):
                for i2 in range(i1 + 1, len(top)):
                    a1, a2 = top[i1], top[i2]
                    merged.append({**merged[0], 'keep_only': sorted(set(a1['keep_only']) | set(a2['keep_only'])), 'effects': {**a1['effects'], **a2['effects']}, 'effect_palette': a1['effect_palette']})
            recs['dino'] = refine('dino', r_bg + r_geo + evaluate(merged, 'capri'), list(G), 3, 'capri', effects=True)
            # synthtiger: font, SynthTIGER glyphs
            fonts = [start] + [{**start, 'font': f} for f in V.font_candidates(d, ts['fonts'])]
            recs['synthtiger'] = evaluate(fonts, 'synthtiger')
            # mixed: combine winners (top-2 of each specialist for fallback), then +/-1-step refine with the mixed judge
            def take(arm, rec, names):return {k: rec['controls'][k] for k in names if k in rec['controls']}
            picks = {a: [r for r in best(recs[a], a, 6) if r['usable'] == len(fit)][:2] or best(recs[a], a, 1) for a in ('custom', 'dino', 'synthtiger')}
            combos = []
            for ci, c_ in enumerate(picks['custom']):
                for di, d_ in enumerate(picks['dino']):
                    for si, s_ in enumerate(picks['synthtiger']):
                        # text colour from custom, background + effect colours from DINO, font from SynthTIGER
                        combos.append({**start, **take('dino', d_, OWNS['dino'] + ['effect_palette']), **take('custom', c_, ['weight_absolute', 'bold_absolute', 'text_palette', 'color_mode']), **take('synthtiger', s_, ['font'])})
            r_mix = evaluate(combos, 'capri')
            first = [r for r in r_mix if r['usable'] == len(fit)]
            base_mix = first if first else r_mix
            recs['mixed'] = refine('mixed', base_mix, list(STEPS), 2, 'capri', effects=True)
            # validation on 4 fresh seeds: top-4 usable per arm, best validated wins
            vr = [V.draw_recipe(base, 1300003 + index*1009 + j*104729) for j in range(V.VAL_SEEDS)]
            vcache = {}
            for arm in ARMS:
                eng = ENGINE[arm]; pick = None
                for rec in best(recs[arm], arm, 4):
                    k = (eng, V.key(rec['controls']))
                    if k not in vcache:
                        rows = list(pool.map(V.render_job, [(r, rec['controls'], eng) for r in vr]))
                        if not all(V.usable(r, length) for r in rows):vcache[k] = None
                        else:
                            ff = model.extract([r['image'] for r in rows]); g = obj.score(ff, target, rows); l = obj.score(ff, local, rows)
                            vcache[k] = {m: .5*(g[m] + l[m]) for m in set(JUDGE.values())}
                    if vcache[k] and (pick is None or vcache[k][JUDGE[arm]] < pick[1][JUDGE[arm]]):pick = (rec, vcache[k])
                unvalidated = pick is None
                if unvalidated:
                    rec = max(recs[arm], key=lambda r: (r['usable'], -r['scores'][JUDGE[arm]])); pick = (rec, rec['scores'])
                rec, val = pick
                show = [V.draw_recipe(base, 1400003 + index*1009 + j*10000019) for j in range(4)]
                rows = list(pool.map(V.render_job, [(r, rec['controls'], eng) for r in show]))
                samples[(index, arm)] = [r for r in rows if V.usable(r, length)]
                entries.append(dict(entry_id=f'{arm}_{index:03d}', controller=arm, engine=eng, judge=JUDGE[arm], input_index=index, controls=rec['controls'],
                    validation_score=val[JUDGE[arm]], validation_all_judges=val, unvalidated=unvalidated, candidates_tried=len(recs[arm])))
            histories.append(dict(input_index=index, records=[dict(engine=r['engine'], controls=r['controls'], usable=r['usable'], scores=r['scores']) for r in cache.values()],
                                  arm_candidates={a: len(recs[a]) for a in ARMS}))
            U.dump(out / 'entries.json', entries); U.dump(out / 'controller_history.json', histories)
            U.dump(out / 'status.json', dict(processed=len({e['input_index'] for e in entries}), total=len(chosen), seconds=time.time() - started))
            print(dataset, 'input', index, f'{time.time() - started:.0f}s', flush=True)
    folder = out / 'samples'; folder.mkdir()
    from PIL import Image
    for (i, arm), rows in samples.items():
        for j, r in enumerate(rows):Image.fromarray(r['image']).save(folder / f'{i:03d}_{arm}_{j}.png')
    U.dump(out / 'status.json', dict(state='complete', inputs=chosen, seconds=time.time() - started))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=list(V.SOURCES), required=True)
    ap.add_argument('--inputs', type=int, default=4); ap.add_argument('--input-list', type=str, default=None)
    ap.add_argument('--workers', type=int, default=14); ap.add_argument('--out', type=Path)
    ap.add_argument('--verify', action='store_true')
    a = ap.parse_args()
    if a.verify:
        os.environ.setdefault('CAPRI_RUN_DIR', str(V.ROOT / 'runtime/temporary/v3_verify'))
        os.environ.setdefault('CAPRI_TARGET_MANIFEST', str(V.ROOT / f'Capri/runs/controller_v2c_Afree_diverse4_{a.dataset}_20261114/targets.json')); verify(a.dataset); sys.exit()
    run(a.dataset, a.inputs, a.out, a.workers, [int(x) for x in a.input_list.split(',')] if a.input_list else None)
