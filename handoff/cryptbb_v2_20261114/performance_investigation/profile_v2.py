"""Isolated profile of one controller_v2 evaluate() step (CryptBB input 0, 'start' controls). Read-only on study files."""
import cProfile, pstats, io, json, os, time
from pathlib import Path
import numpy as np
ROOT = Path(r'E:\pranav\Thesis-Pranav-Final'); RUN = ROOT / 'Capri/runs/cryptbb_controller_v2_free_full_20261114'
os.environ['CAPRI_RUN_DIR'] = str(ROOT / 'runtime/temporary/profile_v2_20261114'); os.environ['CAPRI_TARGET_MANIFEST'] = str(RUN / 'targets.json')
from capri_controller import controller_v2 as V
from capri_controller import runtime as U
from capri_controller.controllers import Mixed, Objective
from capri_controller.rendering import renderer as R
from capri_controller.artifacts import load_images
R.install()
src = V.SOURCES['cryptbb']; refs = json.loads((RUN / 'targets.json').read_text())['fit']
recipes = U.read(src / 'refined_recipes.json'); length = int(U.read(src / 'length_decision.json')['default_length'])
t = time.time(); real = load_images(refs); model = Mixed(); print('load+init', round(time.time() - t, 2))
t = time.time(); target = model.extract(real); print('extract 100 refs', round(time.time() - t, 2))
with np.load(src / 'calibration_features.npz', allow_pickle=False) as z:
    calibration = tuple({k.split('__', 1)[1]: z[k] for k in z.files if k.startswith(str(i) + '__')} for i in range(2))
obj = Objective(target, calibration)
idx = 0; base = recipes[idx]; local = V.slice_features(target, slice(idx, idx + 1))
fit = [V.draw_recipe(base, 1200003 + idx*1009 + j*104729) for j in range(3)]
ctrl = R.expand(dict(length=length))
res = {}
for engine in ['capri', 'synthtiger']:
    ts = []; rows = []
    for k in range(14):
        for r in fit:
            t = time.time(); rows.append(V.render_job((r, ctrl, engine))); ts.append(time.time() - t)
    res[engine] = dict(first=round(ts[0], 3), median=round(float(np.median(ts[1:])), 4), p90=round(float(np.quantile(ts[1:], .9)), 4), n=len(ts))
    print(engine, 'serial render s/img', res[engine])
imgs = [r.get('image', np.full((64, 256, 3), 255, np.uint8)) for r in rows]  # 42 = 14 candidates x 3 seeds
model.extract(imgs[:6])  # warm
for name, fn in [('custom.extract_pair', lambda: model.custom.extract_pair(imgs)), ('dino.extract', lambda: model.dino.extract(imgs))]:
    t = time.time(); fn(); res[name] = round(time.time() - t, 3); print(name, '42 imgs', res[name])
f = model.extract(imgs)
t = time.time()
for j in range(14):
    ff = V.slice_features(f, slice(3*j, 3*j + 3)); rr = rows[3*j:3*j + 3]; obj.score(ff, target, rr); obj.score(ff, local, rr)
res['score 14 candidates'] = round(time.time() - t, 3); print('score x14', res['score 14 candidates'])
pr = cProfile.Profile(); pr.enable(); model.extract(imgs); pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(30); print(s.getvalue()[:6000])
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('tottime').print_stats(15); print(s.getvalue()[:4000])
json.dump(res, open(ROOT / 'runtime/temporary/profile_v2_20261114/profile_result.json', 'w'), indent=1)
