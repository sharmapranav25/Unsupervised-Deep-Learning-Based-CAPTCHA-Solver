"""Which judge reacts to which factor? Change one group at a time from the controller's starting recipe and
measure how much each judge's appearance score moves (custom vs DINO), vs. a seed-noise floor. Read-only."""
import json, os, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(r'E:\pranav\Thesis-Pranav-Final')
def main():
    d = sys.argv[1]
    PREV = ROOT / f'Capri/runs/controller_v2c_Afree_diverse4_{d}_20261114'
    OUT = ROOT / 'Capri/runs/controller_split_sensitivity_20261114'; OUT.mkdir(exist_ok=True)
    os.environ['CAPRI_RUN_DIR'] = str(OUT / f'tmp_{d}'); os.environ['CAPRI_TARGET_MANIFEST'] = str(PREV / 'targets.json')
    from capri_controller import controller_v2 as V
    from capri_controller import runtime as U
    from capri_controller.controllers import Mixed, Objective
    from capri_controller.rendering import renderer as R
    from capri_controller.artifacts import load_images
    from concurrent.futures import ProcessPoolExecutor
    src = V.SOURCES[d]; refs = json.loads((PREV / 'targets.json').read_text())['fit']
    recipes = U.read(src / 'refined_recipes.json'); length = int(U.read(src / 'length_decision.json')['default_length'])
    inputs = json.loads((PREV / 'protocol.json').read_text())['preview_inputs']
    R.install(); real = load_images(refs); model = Mixed(); target = model.extract(real)
    with np.load(src / 'calibration_features.npz', allow_pickle=False) as z:
        cal = tuple({k.split('__', 1)[1]: z[k] for k in z.files if k.startswith(str(i) + '__')} for i in range(2))
    obj = Objective(target, cal)
    results = []
    with ProcessPoolExecutor(max_workers=12, initializer=V._install) as pool:
        for idx in inputs:
            base = recipes[idx]; ts = base['text_spec']; desc = base['description']; local = V.slice_features(target, slice(idx, idx + 1))
            colors, bgs = V.palette(real[idx], target[0]['mask'][idx]); bg = bgs[0]
            ink = [c for c in colors if np.linalg.norm(np.array(c) - np.array(bg)) >= 60] or [max(colors, key=lambda c: np.linalg.norm(np.array(c) - np.array(bg)))]
            start = dict(length=length, background=bg, text_palette=ink, effect_palette=ink, color_mode='alternating' if len(ink) > 1 else 'single')
            keep = [n for n in (e['name'] for e in base['effects_spec']['effects']) if n not in R.C.R.R.UNDER]
            groups = {
              'thickness': [{**start, 'weight_absolute': w} for w in (.2, .8)] + [{**start, 'bold_absolute': b} for b in (0, 2)],
              'size/width/spacing': [{**start, 'size_absolute': v} for v in (.5, .85)] + [{**start, 'width_absolute': v} for v in (.75, 1.4)] + [{**start, 'spacing_absolute': v} for v in (-.2, .4)],
              'font': [{**start, 'font': f} for f in V.font_candidates(desc, ts['fonts'])[:4]],
              'slant/rotation/wobble': [{**start, 'slant_absolute': v} for v in (-.2, .35)] + [{**start, 'rotation_absolute': v} for v in (0., 25.)] + [{**start, 'jitter_absolute': .2}, {**start, 'size_jitter_absolute': .25}],
              'text colour': [{**start, 'text_palette': p, 'effect_palette': p, 'color_mode': 'single'} for p in ([[200, 30, 30]], [[30, 60, 200]], [[20, 140, 40]])] + [{**start, 'color_mode': 'random'}],
              'background/effects': [{**start, 'keep_only': keep + [n], 'effects': {n: 3}} for n in ('grid', 'dots', 'blobs', 'lines')] + [{**start, 'keep_only': keep, 'effects': {}}],
            }
            def score(ctrl, seeds):
                rr = [V.draw_recipe(base, s) for s in seeds]
                rows = list(pool.map(V.render_job, [(r, R.expand(ctrl), 'capri') for r in rr]))
                if not all(r.get('complete') and r.get('image') is not None for r in rows):return None
                f = model.extract([r['image'] for r in rows]); g = obj.score(f, target, rows); l = obj.score(f, local, rows)
                return {j: .5*(g[j + '_appearance'] + l[j + '_appearance']) for j in ('custom', 'dino')}
            fit = [1200003 + idx*1009 + j*104729 for j in range(3)]; other = [1500007 + idx*1009 + j*104729 for j in range(3)]
            b0 = score(start, fit); noise = score(start, other)
            row = dict(input=idx, base=b0, noise={j: abs(noise[j] - b0[j]) for j in b0}, groups={})
            for gname, cands in groups.items():
                ds = [s for s in (score(c, fit) for c in cands) if s]
                row['groups'][gname] = {j: float(np.mean([abs(s[j] - b0[j]) for s in ds])) for j in b0} if ds else None
            results.append(row); print(d, idx, 'done', flush=True)
    json.dump(results, open(OUT / f'{d}_sensitivity.json', 'w'), indent=1)
    G = list(results[0]['groups']); print(f'\n{d}: mean |score change| per judge (noise = same recipe, new seeds)')
    print(f"{'group':24s} {'custom':>8s} {'dino':>8s}  custom/dino")
    for gname in ['noise'] + G:
        vals = {j: np.mean([(r['noise'] if gname == 'noise' else r['groups'][gname])[j] for r in results if (gname == 'noise' or r['groups'][gname])]) for j in ('custom', 'dino')}
        print(f"{gname:24s} {vals['custom']:8.3f} {vals['dino']:8.3f}  {vals['custom']/max(vals['dino'],1e-9):6.2f}")

if __name__ == '__main__':main()
