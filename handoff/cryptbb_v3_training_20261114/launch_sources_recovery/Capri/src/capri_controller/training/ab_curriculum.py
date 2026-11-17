"""DarkBay A/B curricula: full-pool training, cross teachers, fixed final selection.

This module deliberately does not import the historical run orchestrators.
"""
import argparse
import contextlib
import copy
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
STUDY = ROOT / 'handoff/darkbay_curriculum_20261113'
N = 100000
BATCH = 128
STEPS = math.ceil(N / BATCH)
DRAWS = STEPS * BATCH
EPOCHS = 20
SEED = 48


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.pending')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def bootstrap():
    for key, sub in {'TEMP':'temporary', 'TMP':'temporary', 'TORCH_HOME':'cache/torch',
                     'HF_HOME':'cache/huggingface', 'XDG_CACHE_HOME':'cache',
                     'MPLCONFIGDIR':'cache/matplotlib', 'PIP_CACHE_DIR':'cache/pip'}.items():
        p = ROOT / 'runtime' / sub
        p.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(p)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', NO_ALBUMENTATIONS_UPDATE='1',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    sys.path[:0] = [str(ROOT/'model1/src/parseq_packages'), str(ROOT/'model1/src/OpenOCR')]
    import torch
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), 'CUDA required'
    torch.backends.cudnn.benchmark = False


@contextlib.contextmanager
def gpu_slot():
    """Windows releases this byte lock on crashes; never change batch/BatchNorm.

    Independent readers prepare CPU inputs concurrently. Full GPU steps are
    serialized, and unused allocator blocks are released before handing over.
    """
    import msvcrt
    import torch
    STUDY.mkdir(parents=True, exist_ok=True)
    with (STUDY/'gpu.lock').open('a+b') as f:
        f.seek(0, 2)
        if f.tell() == 0:
            f.write(b'0'); f.flush()
        while True:
            try:
                f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                time.sleep(.05)
        try:
            yield
        finally:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def config(tag, out):
    import yaml
    cfg = yaml.safe_load((ROOT/f'model1/configs/darkbay/{tag}/relocated.yml').read_text())
    cfg['Global'].update(output_dir=str(out), save_res_path=str(out/'unused.txt'))
    return cfg


def make_model(cfg):
    import numpy as np
    import torch
    from openrec.modeling import build_model
    from openrec.postprocess import build_post_process
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    decoder = build_post_process(cfg['PostProcess'], cfg['Global'])
    cfg['Architecture']['Decoder']['out_channels'] = decoder.get_character_num()
    model = build_model(copy.deepcopy(cfg['Architecture']))
    for i, module in enumerate(model.modules()):
        if isinstance(getattr(module, 'rng', None), np.random.Generator):
            module.rng = np.random.default_rng(SEED + i)
    return model.cuda(), decoder


def dataset(tag):
    import lmdb
    import numpy as np
    from .ds2_data import ReaderDataset
    env = lmdb.open(str(ROOT/f'datasets/darkbay/{tag}/train_lmdb'), readonly=True,
                    lock=False, readahead=False)
    with env.begin() as tx:
        assert int(tx.get(b'num-samples')) == N
    return ReaderDataset(env, np.arange(1, N+1), ROOT/f'datasets/darkbay/{tag}/charset.txt')


def batch_at(ds, indices):
    from torch.utils.data._utils.collate import default_collate
    return default_collate([ds[int(i)] for i in indices])


def rng_state(model):
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all(),
                modules={n:copy.deepcopy(m.rng.bit_generator.state) for n,m in model.named_modules()
                         if isinstance(getattr(m, 'rng', None), np.random.Generator)})


def restore_rng(model, state):
    import numpy as np
    import torch
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
    for n,m in model.named_modules():
        if n in state['modules']:
            m.rng.bit_generator.state = copy.deepcopy(state['modules'][n])


def model_hash(model):
    h = hashlib.sha256()
    for n,v in model.state_dict().items():
        h.update(n.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def score_batch(model, decoder, batch, chars):
    import torch
    # Retry at the architecture horizon if early EOS prevents synthetic NLL.
    # No ground-truth tokens or lengths are passed to the decoder.
    with torch.inference_mode():
        length = int(batch[2].max())
        memory = model.encoder(batch[0].cuda())
        pred = model.decoder.forward_test(memory)
        if pred.shape[1] < length:
            pred = model.decoder.forward_test(memory, max_length=25)
        char_ids = [decoder.character.index(c) for c in chars]
        q = pred[:, :length, char_ids].double()
        q = q / q.sum(-1, keepdim=True).clamp_min(1e-300)
        truth = batch[1][:, 1:length+1]
        mask = torch.arange(length)[None, :] < batch[2][:, None]
        indices = torch.zeros_like(truth)
        for i,k in enumerate(char_ids):
            indices[truth == k] = i
        assert all(int(k) in char_ids for k in truth[mask].unique())
        p = q.gather(-1, indices.cuda().unsqueeze(-1)).squeeze(-1)
        values = ((-torch.log(p+1e-9)).clamp_min(0)*mask.cuda()).sum(1).cpu().numpy()
        assert torch.isfinite(pred).all()
        predictions = [v[0] for v in decoder(pred)]
    return values, predictions


def score_pool(model, decoder, ds, dest, limit=None):
    import numpy as np
    count = len(ds) if limit is None else limit
    chars = list(ds.encode.dict)
    chars = [c for c in chars if c not in ('<s>', '</s>', '<pad>')]
    model.eval(); values = []; predictions = []
    start = time.monotonic()
    for i in range(0, count, 64):
        batch = batch_at(ds, range(i, min(i+64, count)))
        with gpu_slot():
            v,p = score_batch(model, decoder, batch, chars)
        values.extend(v.tolist()); predictions.extend(p)
        if i % 6400 == 0:
            write(dest/'progress.json', dict(processed=len(values), total=count, seconds=time.monotonic()-start))
            print(f'score {len(values)}/{count}', flush=True)
    raw = np.asarray(values, dtype=np.float64)
    assert raw.shape == (count,) and np.isfinite(raw).all() and (raw >= 0).all()
    write(dest/'predictions.json', [dict(index=i+1, prediction=p) for i,p in enumerate(predictions)])
    return raw, time.monotonic()-start


def rank_dir(tag):
    return ROOT/f'model2/rankings/darkbay/{tag}'


def run_dir(model, tag):
    return ROOT/f'model{model}/runs/darkbay/{tag}'


def code_files():
    paths = [Path(__file__), Path(__file__).with_name('ds2_data.py'), Path(__file__).with_name('ds3_data.py')]
    paths.extend(sorted((ROOT/'model1/src/OpenOCR').rglob('*.py')))
    return paths


def setup():
    manifest = read(ROOT/'datasets/darkbay/manifests/datasets.json')
    inputs = {}
    for tag in ('A', 'B'):
        p = ROOT/manifest[tag]['lmdb']/'data.mdb'
        assert sha(p) == manifest[tag]['lmdb_sha256']
        ck = ROOT/f'model1/models/parseq/darkbay/{tag}/epoch_20.pth'
        expected = read(ck.parent/'manifest.json')['sha256']
        assert sha(ck) == expected
        inputs[tag] = dict(data_sha256=manifest[tag]['lmdb_sha256'], teacher_checkpoint_sha256=expected)
    sources = {str(p.relative_to(ROOT)):sha(p) for p in code_files()}
    protocol = dict(version=1, family='DarkBay', architecture='Model1 SVTRv2LNConvTwo33 + PARSeq',
        seed=SEED, epochs=EPOCHS, pool_size=N, batch=BATCH, draws_per_epoch=DRAWS,
        steps_per_epoch=STEPS, total_steps=STEPS*EPOCHS, replacement=True,
        budget_note='Model1 RatioSampler pads its final batch: 100096 draws, 782 full batches. Matched here.',
        preprocessing='Model1 RGB PIL bicubic resize32x128, ToTensor, Normalize(.5,.5); no added augmentation',
        initialization='from scratch each run, seed48 incl module NumPy generators; no Model1/Model2 weight initialization',
        optimizer='Model1 AdamW lr.00035 weight_decay.05 filtered bias/norm, OneCycle warmup2/20, BF16 AMP, clip20',
        selection='fixed final epoch20; no validation or real-label selection',
        teachers={'A':'Model1-B epoch20','B':'Model1-A epoch20'},
        ranking='FP32 free autoregressive decoder with one refinement and normal early stopping; no label tokens/length passed. Character-only renormalization, summed -log(p_true+1e-9), clamp0; ordinal NLL tertiles with image-ID tie-break.',
        scoring_revision='If output horizon is too short to score synthetic answers, retry free AR at fixed architecture max_length25 instead of failing the historical assertion. Applies to initial and adaptive scores; no target labels or manually chosen family length.',
        static_schedule={'1':[1,0,0],'2':[.5,.5,0],'3-20':[.15,.25,.60]},
        adaptive='Historical ds3_data.AdaptiveSampler.update unchanged: .5 rank smoothing, equal thirds, risk1-exp(-min(NLL,50)), .2-.5 masses, max .1 change, first uniform reference; rescore entire pool after every epoch including20.',
        parallel='Two independent processes active; exclusive GPU batch lock preserves batch128/BatchNorm within16GB. CPU preparation concurrent; no 2x speed claim.',
        checkpoint='Every completed epoch: weights, optimizer, scheduler, AMP, RNG incl module generators, sampler/adaptive state; resume only committed epoch boundaries',
        evaluation='All four final predictions frozen before opening real scoring labels; same900 historical DarkBay images; no threshold sweeps',
        length_rule='Unique leading model count >=50/100 votes; ties rejected. Not40% length accuracy. Existing DarkBay89/100 votes for6; no regeneration.',
        family_stop='Stop after DarkBay report; CryptBB and City await ready inputs; no other STR models',
        inputs=inputs, sources=sources)
    if (STUDY/'protocol.json').exists():
        assert read(STUDY/'protocol.json') == protocol, 'Frozen protocol/source changed'
    else:
        write(STUDY/'protocol.json', protocol)
        for p in code_files():
            dst = STUDY/'source'/p.relative_to(ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p,dst)
    print('Protocol and inputs frozen', flush=True)


def check_sources():
    protocol = read(STUDY/'protocol.json')
    for p,digest in protocol['sources'].items():
        assert sha(ROOT/p) == digest, 'Source changed since freeze: '+p
    return sha(STUDY/'protocol.json')


def rank(tag):
    import numpy as np
    import torch
    digest = check_sources()
    out = rank_dir(tag)
    if (out/'completion.json').exists():
        done = read(out/'completion.json')
        assert done['protocol_sha256'] == digest and sha(out/'ranking.npz') == done['sha256']
        return
    assert not out.exists() or not any(out.iterdir()), 'Inspect partial ranking before retry'
    out.mkdir(parents=True, exist_ok=True)
    opposite = 'B' if tag == 'A' else 'A'
    ckpath = ROOT/f'model1/models/parseq/darkbay/{opposite}/epoch_20.pth'
    cfg = config(tag, out)
    with gpu_slot():
        model, decoder = make_model(cfg)
        ck = torch.load(ckpath, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['state_dict'], strict=True); del ck
    ds = dataset(tag)
    raw, seconds = score_pool(model, decoder, ds, out)
    ids = np.arange(1,N+1)
    buckets = np.empty(N, dtype=np.int64)
    for b,group in enumerate(np.array_split(np.lexsort((ids, raw)),3)):
        buckets[group] = b
    np.savez_compressed(out/'ranking.npz', ids=ids, nll=raw, buckets=buckets)
    write(out/'completion.json', dict(dataset=tag, teacher=opposite, images=N, seconds=seconds,
          protocol_sha256=digest, teacher_sha256=sha(ckpath), sha256=sha(out/'ranking.npz'),
          predictions_sha256=sha(out/'predictions.json'), bucket_counts=np.bincount(buckets).tolist()))
    print('Ranking complete '+tag, flush=True)


def sampler_for(tag):
    import numpy as np
    from .ds3_data import AdaptiveSampler, NAMES
    ranking = rank_dir(tag)/'ranking.npz'
    assert sha(ranking) == read(rank_dir(tag)/'completion.json')['sha256']
    with np.load(ranking) as z:
        rows = [dict(index=int(i), nll=float(n), bucket=NAMES[int(b)])
                for i,n,b in zip(z['ids'],z['nll'],z['buckets'])]
    return AdaptiveSampler(rows, SEED)


def draws_for(sampler, model, epoch, count=DRAWS):
    import numpy as np
    assert sampler.epoch == epoch
    if model == 2:
        sampler.mass = np.array([1.,0.,0.] if epoch == 1 else [.5,.5,0.] if epoch == 2 else [.15,.25,.6])
    counts = np.rint(count * sampler.mass).astype(int)
    counts[np.argmax(sampler.mass)] += count-counts.sum()
    rng = np.random.default_rng(SEED+epoch*100003)
    draws = np.concatenate([rng.choice(np.flatnonzero(sampler.buckets == b), n, replace=True)
                            for b,n in enumerate(counts) if n])
    rng.shuffle(draws)
    assert len(draws) == count
    receipt = dict(epoch=epoch, mass=sampler.mass.tolist(), draws=counts.tolist(),
                   total=count, dropped=0, sample_ids_sha256=hashlib.sha256(sampler.ids[draws].tobytes()).hexdigest())
    sampler.epoch += 1
    return draws, receipt


def optimizer_for(model, cfg):
    import torch
    from openrec.optimizer import build_optimizer
    opt, sched = build_optimizer(cfg['Optimizer'], cfg['LRScheduler'], EPOCHS, STEPS, model)
    return opt, sched, torch.amp.GradScaler()


def step(model, opt, sched, scaler, batch):
    import torch
    model.train()
    with gpu_slot():
        batch = [v.cuda() for v in batch]
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, _ = model(batch[0], data=batch[1:])
        assert torch.isfinite(loss), 'Nonfinite training loss'
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 20, error_if_nonfinite=True)
        scaler.step(opt); scaler.update(); sched.step()
        value = float(loss.detach()); gradient = float(norm)
        opt.zero_grad(set_to_none=True)
        del loss, batch, norm, _
    return value, gradient


def save_checkpoint(path, model, opt, sched, scaler, sampler, cfg, epoch, history, digest):
    import torch
    state = dict(state_dict=model.state_dict(), optimizer=opt.state_dict(),
        lr_scheduler=sched.state_dict(), scaler=scaler.state_dict(), rng=rng_state(model),
        adaptive=sampler.state_dict(), config=cfg, epoch=epoch, global_step=epoch*STEPS,
        history=history, protocol_sha256=digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pending')
    torch.save(state, tmp); os.replace(tmp, path)


def load_checkpoint(path, model, opt, sched, scaler, sampler, digest):
    import torch
    ck = torch.load(path, map_location='cpu', weights_only=False)
    assert ck['protocol_sha256'] == digest
    model.load_state_dict(ck['state_dict'], strict=True)
    opt.load_state_dict(ck['optimizer']); sched.load_state_dict(ck['lr_scheduler'])
    scaler.load_state_dict(ck['scaler'])
    # Static schedules may violate the adaptive sampler's bounds by design.
    for k,v in ck['adaptive'].items():
        setattr(sampler,k,copy.deepcopy(v))
    restore_rng(model, ck['rng'])
    return ck


def train(model_number, tag, resume=False, pilot=False):
    import numpy as np
    import torch
    import yaml
    digest = check_sources()
    if not pilot:
        assert read(STUDY/'readiness.json')['passed']
    out = STUDY/f'pair_pilot/{tag}' if pilot else run_dir(model_number, tag)
    if (out/'completion.json').exists():
        done = read(out/'completion.json')
        assert done['protocol_sha256'] == digest
        assert sha(out/done['checkpoint']) == done['sha256']
        return
    assert resume or not out.exists(), 'Existing run requires --resume'
    out.mkdir(parents=True, exist_ok=True)
    cfg = config(tag, out)
    (out/'config.yml').write_text(yaml.safe_dump(cfg), encoding='utf8')
    sampler = sampler_for(tag)
    with gpu_slot():
        model, decoder = make_model(cfg)
        opt, sched, scaler = optimizer_for(model, cfg)
    ds = dataset(tag)
    history = []; first = 1
    if resume:
        ck = load_checkpoint(out/'latest.pth',model,opt,sched,scaler,sampler,digest)
        first = ck['epoch']+1; history = ck['history']; del ck
    else:
        write(out/'initialization.json', dict(model_tensor_sha256=model_hash(model), pretrained=False, seed=SEED))
    for epoch in range(first, 2 if pilot else EPOCHS+1):
        if (STUDY/'STOP').exists():
            raise RuntimeError('STOP requested at epoch boundary; resume from latest committed checkpoint')
        started = time.monotonic(); losses = []; active_seconds = 0
        draws, receipt = draws_for(sampler, model_number, epoch, BATCH*4 if pilot else DRAWS)
        np.save(out/f'draws_epoch_{epoch:02d}.npy', sampler.ids[draws])
        write(out/f'sampling_epoch_{epoch:02d}.json', receipt)
        for batch_index, offset in enumerate(range(0,len(draws),BATCH),1):
            batch = batch_at(ds, draws[offset:offset+BATCH])
            t = time.monotonic(); value, norm = step(model,opt,sched,scaler,batch)
            active_seconds += time.monotonic()-t; losses.append(value)
            if batch_index == 1 or batch_index % 25 == 0 or pilot:
                progress = dict(model=model_number,dataset=tag,state='training',epoch=epoch,
                    completed_epoch=epoch-1,batch=batch_index,batches=len(draws)//BATCH,
                    global_step=(epoch-1)*STEPS+batch_index,loss=value,gradient_norm=norm,
                    lr=opt.param_groups[0]['lr'],epoch_seconds=time.monotonic()-started,updated_at=now(),
                    peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
                write(out/'status.json',progress)
                print(json.dumps(progress),flush=True)
        train_seconds = time.monotonic()-started
        rescore_seconds = 0; update = None
        if model_number == 3 and not pilot:
            saved_rng = rng_state(model); before = model_hash(model)
            write(out/'status.json',dict(state='rescoring',epoch=epoch,completed_epoch=epoch-1,updated_at=now()))
            rescore = out/f'adaptive/epoch_{epoch:02d}'
            raw, rescore_seconds = score_pool(model,decoder,ds,rescore)
            assert before == model_hash(model), 'Scoring changed model state'
            restore_rng(model,saved_rng)
            update = sampler.update(raw)
            np.savez_compressed(rescore/'scores.npz',ids=sampler.ids,nll=raw,
                                percentile=sampler.normalized,smoothed=sampler.smooth,buckets=sampler.buckets)
            write(rescore/'update.json',update)
        record = dict(epoch=epoch,mean_loss=float(np.mean(losses)),training_wall_seconds=train_seconds,
                      gpu_step_including_wait_seconds=active_seconds,rescore_wall_seconds=rescore_seconds,update=update)
        history.append(record)
        if pilot:
            write(out/'completion.json',dict(pilot=True,steps=4,record=record,passed=True,
                  peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,protocol_sha256=digest))
            return
        latest = out/'latest.pth'
        save_checkpoint(latest,model,opt,sched,scaler,sampler,cfg,epoch,history,digest)
        archive = out/f'checkpoints/epoch_{epoch:02d}.pth'
        archive.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(latest,archive)
        write(out/'history.json',history)
        write(out/'status.json',dict(state='epoch_complete',completed_epoch=epoch,global_step=epoch*STEPS,updated_at=now()))
        if epoch == EPOCHS:
            write(out/'completion.json',dict(model=model_number,dataset=tag,epoch=epoch,global_step=epoch*STEPS,
                  checkpoint=str(archive.relative_to(out)),sha256=sha(archive),protocol_sha256=digest,
                  training_wall_seconds=sum(r['training_wall_seconds'] for r in history),
                  rescore_wall_seconds=sum(r['rescore_wall_seconds'] for r in history),selection='fixed final epoch20'))
            plot_history(out,history)


def plot_history(out, history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(figsize=(7,4))
    ax.plot([h['epoch'] for h in history],[h['mean_loss'] for h in history],marker='o')
    ax.set(xlabel='Epoch',ylabel='Synthetic training loss',title='DarkBay PARSeq curriculum')
    fig.tight_layout(); fig.savefig(out/'training_loss.png',dpi=160); plt.close(fig)


def launch_child(args, log):
    log.parent.mkdir(parents=True,exist_ok=True)
    stream = log.open('a',encoding='utf8')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT/'Capri/src')+os.pathsep+env.get('PYTHONPATH','')
    p = subprocess.Popen([sys.executable,'-B','-u','-m',__package__+'.ab_curriculum',*args],
                         cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,
                         creationflags=subprocess.CREATE_NO_WINDOW)
    stream.close()
    return p


def pair(model_number, pilot=False):
    children = []
    for tag in ('A','B'):
        args = ['train','--model',str(model_number),'--dataset',tag]
        if pilot:
            args.append('--pilot')
            log = STUDY/f'pair_pilot/{tag}.console.log'
        else:
            out = run_dir(model_number,tag)
            if (out/'latest.pth').exists(): args.append('--resume')
            log = out.parent/f'{tag}.console.log'
        children.append((tag,launch_child(args,log)))
    write(STUDY/'active_children.json',dict(model=model_number,pilot=pilot,pids={t:p.pid for t,p in children},updated_at=now()))
    # Let a healthy peer finish its epoch/run if the other fails; never silently restart.
    codes = {t:p.wait() for t,p in children}
    assert not any(codes.values()), str(codes)


def readiness():
    """Exercise budgets, preprocessing, adaptive state, real save/load and paired memory."""
    import numpy as np
    import torch
    from PIL import Image
    from .ds3_data import AdaptiveSampler, percentile
    from tools.data.ratio_dataset_tvresize import RatioDataSetTVResize
    from torchvision import transforms as T
    digest = check_sources()
    if (STUDY/'readiness.json').exists():
        assert read(STUDY/'readiness.json')['protocol_sha256'] == digest
        assert read(STUDY/'readiness.json')['passed']
        return
    out = STUDY/'checks'
    out.mkdir(parents=True,exist_ok=True)
    ds = dataset('A')
    # Compare the replacement dataset transform to the actual baseline method.
    baseline = RatioDataSetTVResize.__new__(RatioDataSetTVResize)
    baseline.base_shape = [[64,64],[96,48],[112,40],[128,32]]
    baseline.base_h = 32; baseline.padding_rand = False; baseline.padding_doub = False
    baseline.interpolation = T.InterpolationMode.BICUBIC
    baseline.transforms = T.Compose([T.ToTensor(),T.Normalize(.5,.5)])
    baseline.error = 0
    for index in (0,49999,99999):
        with ds.env.begin() as tx:
            im = Image.open(io.BytesIO(tx.get(f'image-{index+1:09d}'.encode()))).convert('RGB')
        reference = baseline.resize_norm_img({'image':im},4,padding=False)['image']
        assert torch.equal(reference,ds[index][0]), 'Baseline preprocessing differs'
    # Model1 uses padded full batches, not a 32-image final batch.
    sampler = sampler_for('A')
    counts = []
    for epoch in range(1,21):
        draws, receipt = draws_for(sampler,2,epoch)
        assert len(draws)==DRAWS and len(draws)//128==782 and draws.min()>=0 and draws.max()<N
        assert np.bincount(sampler.buckets[draws],minlength=3).tolist()==receipt['draws']
        counts.append(receipt['draws'])
    adaptive = sampler_for('A')
    draw1,_ = draws_for(adaptive,3,1)
    assert np.all(adaptive.buckets[draw1]==0)
    for epoch in range(1,21):
        # Exercise changes, ties, boundary masses and deterministic continuation.
        raw = np.random.default_rng(epoch).uniform(0,30,N) if epoch%2 else np.zeros(N)
        info = adaptive.update(raw)
        assert info['max_mass_change'] <= .10000000001
        assert np.min(adaptive.mass)>=.2-1e-12 and np.max(adaptive.mass)<=.5+1e-12
        assert sorted(np.bincount(adaptive.buckets))==[33333,33333,33334]
        other = sampler_for('A'); other.load_state_dict(adaptive.state_dict())
        d1,r1=draws_for(adaptive,3,epoch+1); d2,r2=draws_for(other,3,epoch+1)
        assert np.array_equal(d1,d2) and r1==r2
    assert np.all(np.diff(percentile(np.zeros(N),np.arange(N)))>0)
    # Use the same training/save/load implementation as production.
    cfg = config('A',out)
    with gpu_slot():
        model,decoder = make_model(cfg)
        opt,sched,scaler = optimizer_for(model,cfg)
    initial = model_hash(model)
    sm = sampler_for('A')
    draws_for(sm,3,1)
    small = batch_at(ds,range(8))
    losses=[]
    for _ in range(3):
        losses.append(step(model,opt,sched,scaler,small)[0])
    saved_rng = rng_state(model); before = model_hash(model)
    model.eval()
    chars = (ROOT/'datasets/darkbay/A/charset.txt').read_text().splitlines()
    with gpu_slot():
        raw,pred = score_batch(model,decoder,small,chars)
        raw2,pred2 = score_batch(model,decoder,small,chars)
    assert np.array_equal(raw,raw2) and pred==pred2
    assert before==model_hash(model)
    restore_rng(model,saved_rng)
    sm.update(np.resize(raw,N))
    saved = out/'resume_smoke.pth'
    save_checkpoint(saved,model,opt,sched,scaler,sm,cfg,1,[],digest)
    loss1,_ = step(model,opt,sched,scaler,small)
    expected = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    expected_rng = rng_state(model)
    expected_lr = sched.state_dict()
    expected_opt = copy.deepcopy(opt.state_dict())
    # Convert optimizer comparison to CPU before releasing the first instance.
    for state in expected_opt['state'].values():
        for k,v in state.items():
            if torch.is_tensor(v): state[k]=v.cpu().clone()
    del model,opt,sched,scaler
    gc.collect(); torch.cuda.empty_cache()
    with gpu_slot():
        model,decoder = make_model(cfg)
        assert initial==model_hash(model)
        opt,sched,scaler = optimizer_for(model,cfg)
    sm2=sampler_for('A')
    ck=load_checkpoint(saved,model,opt,sched,scaler,sm2,digest)
    assert np.array_equal(sm.smooth,sm2.smooth) and np.array_equal(sm.mass,sm2.mass)
    del ck
    loss2,_=step(model,opt,sched,scaler,small)
    error=max(float((expected[k]-v.detach().cpu()).abs().max()) for k,v in model.state_dict().items())
    assert error <= 1e-6 and abs(loss1-loss2)<=1e-6, (error,loss1,loss2)
    assert expected_lr==sched.state_dict()
    for key,state in opt.state_dict()['state'].items():
        for k,v in state.items():
            if torch.is_tensor(v):
                assert torch.allclose(v.cpu(),expected_opt['state'][key][k],atol=1e-6,rtol=0)
    actual_rng=rng_state(model)
    assert torch.equal(expected_rng['torch'],actual_rng['torch'])
    assert all(torch.equal(a,b) for a,b in zip(expected_rng['cuda'],actual_rng['cuda']))
    assert expected_rng['modules']==actual_rng['modules']
    del model,opt,sched,scaler,expected_opt,expected
    ds.env.close(); gc.collect(); torch.cuda.empty_cache()
    pair(2,pilot=True)
    pilots={tag:read(STUDY/f'pair_pilot/{tag}/completion.json') for tag in ('A','B')}
    assert all(r['passed'] for r in pilots.values())
    write(STUDY/'readiness.json',dict(passed=True,protocol_sha256=digest,created_at=now(),
          preprocessing_exact_matches=3,pool_size=N,draws_per_epoch=DRAWS,steps_per_epoch=STEPS,
          static_schedule_counts=counts,adaptive_twenty_updates_and_replay=True,
          checkpoint_next_step_max_tensor_error=error,resume_losses=[loss1,loss2],
          optimizer_scheduler_rng_resume_verified=True,deterministic_scoring=True,
          tiny_training_losses=losses,paired_full_batch_pilots=pilots,
          real_labels_used_for_readiness=False))
    print('Readiness PASSED',flush=True)


def evaluate_all():
    import numpy as np
    import torch
    from PIL import Image
    from .ds2_data import transform
    digest=check_sources()
    manifest=read(ROOT/'evaluation_data/darkbay/split_manifests/image_manifest.json')
    assert len(manifest)==900
    refs=read(ROOT/'Capri/configs/targets/darkbay_final.json')['fit']
    assert not {r['pixel_sha256'] for r in refs}&{r['pixel_sha256'] for r in manifest}
    for row in manifest:
        pixels=np.asarray(Image.open(ROOT/row['image_path']).convert('RGB'))
        assert hashlib.sha256(pixels.tobytes()).hexdigest()==row['pixel_sha256']
    frozen=[]
    tr=transform()
    for number in (2,3):
        for tag in ('A','B'):
            run=run_dir(number,tag); done=read(run/'completion.json')
            assert done['epoch']==20 and done['global_step']==15640 and done['protocol_sha256']==digest
            ckpath=run/done['checkpoint']; assert sha(ckpath)==done['sha256']
            dest=ROOT/f'model{number}/results/darkbay/{tag}/evaluation_epoch20'
            dest.mkdir(parents=True,exist_ok=True)
            if (dest/'prediction_freeze.json').exists():
                fr=read(dest/'prediction_freeze.json')
                assert sha(dest/'predictions.json')==fr['predictions_sha256']
                assert fr['checkpoint_sha256']==done['sha256']
            else:
                cfg=config(tag,run)
                with gpu_slot():
                    model,decoder=make_model(cfg)
                    ck=torch.load(ckpath,map_location='cpu',weights_only=False)
                    model.load_state_dict(ck['state_dict'],strict=True); del ck
                model.eval(); rows=[]
                for start in range(0,900,64):
                    chunk=manifest[start:start+64]
                    x=torch.stack([tr(Image.open(ROOT/r['image_path']).convert('RGB')) for r in chunk])
                    with gpu_slot(),torch.inference_mode():
                        predictions=decoder(model(x.cuda()))
                    rows.extend(dict(opaque_id=r['opaque_id'],prediction=p[0]) for r,p in zip(chunk,predictions))
                write(dest/'predictions.json',rows)
                write(dest/'prediction_freeze.json',dict(created_at=now(),checkpoint_sha256=done['sha256'],
                      predictions_sha256=sha(dest/'predictions.json'),images=900,protocol_sha256=digest))
                del model; gc.collect(); torch.cuda.empty_cache()
            frozen.append((number,tag,dest))
    # The only scoring-label open in this runner occurs AFTER all four freezes.
    labels=read(ROOT/'evaluation_data/darkbay/scoring_labels/scoring_labels.json')
    summaries=[]
    for number,tag,dest in frozen:
        rows=read(dest/'predictions.json')
        assert len(rows)==900 and all(r['opaque_id'] in labels for r in rows)
        correct=sum(r['prediction']==labels[r['opaque_id']] for r in rows)
        baseline=615 if tag=='A' else 573
        result=dict(model=number,dataset=tag,epoch=20,correct=correct,total=900,accuracy=correct/900,
                    model1_correct=baseline,difference_correct=correct-baseline,difference_percentage_points=(correct-baseline)/9,
                    selection='fixed final epoch20',historical_repeatedly_inspected_test=True)
        write(dest/'summary.json',result); summaries.append(result)
    write(STUDY/'family_report.json',dict(state='DarkBay complete; waiting for CryptBB readiness',
          created_at=now(),model1={'A':615,'B':573},results=summaries,
          length_threshold=dict(forty_percent_supported=False,actual_rule='unique leading count >=50/100 votes',
                                DarkBay_votes=89,predicted_count=6,accuracy_threshold_claim='Not established'),
          output_paths={f'Model{n}-{t}':dict(run=str(run_dir(n,t).relative_to(ROOT)),
              results=str(d.relative_to(ROOT)),ranking=str(rank_dir(t).relative_to(ROOT))) for n,t,d in frozen},
          limits=['Single run per dataset; no repeated seeds','Historical real test repeatedly inspected',
                  'GPU steps serialized to preserve batch128','Ranking retries at architecture horizon25 if early stopping leaves insufficient positions for synthetic NLL']))
    lines=['# DarkBay PARSeq final results','', '| Reader | Correct / 900 | Accuracy |','|---|---:|---:|',
           '| Model1-A | 615 | 68.33% |','| Model1-B | 573 | 63.67% |']
    lines.extend(f"| Model{r['model']}-{r['dataset']} | {r['correct']} | {100*r['accuracy']:.2f}% |" for r in summaries)
    lines+=['','The 40% length-accuracy threshold is not supported. The generator uses a unique leading count with at least50/100 votes; DarkBay had89 votes for6.',
            '', 'All run paths, timings and limitations: family_report.json and each run completion.json. Stop here until CryptBB inputs are ready.']
    (STUDY/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


def coordinate():
    import msvcrt
    STUDY.mkdir(parents=True,exist_ok=True)
    with (STUDY/'coordinator.lock').open('a+b') as f:
        f.seek(0,2)
        if f.tell()==0: f.write(b'0'); f.flush()
        f.seek(0); msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try:
            check_sources()
            for tag in ('A','B'):
                write(STUDY/'status.json',dict(state='ranking',dataset=tag,pid=os.getpid(),updated_at=now()))
                p=launch_child(['rank','--dataset',tag],STUDY/f'rank_{tag}.console.log')
                assert p.wait()==0, 'Ranking failed '+tag
            write(STUDY/'status.json',dict(state='readiness',pid=os.getpid(),updated_at=now()))
            p=launch_child(['check'],STUDY/'readiness.console.log')
            assert p.wait()==0,'Readiness failed; production not launched'
            for number in (2,3):
                write(STUDY/'status.json',dict(state='training_pair',model=number,pid=os.getpid(),updated_at=now()))
                pair(number)
            write(STUDY/'status.json',dict(state='final_evaluation',pid=os.getpid(),updated_at=now()))
            p=launch_child(['evaluate'],STUDY/'evaluation.console.log')
            assert p.wait()==0,'Final evaluation failed'
            write(STUDY/'status.json',dict(state='complete_darkbay_waiting_for_cryptbb',pid=os.getpid(),updated_at=now()))
        except Exception:
            write(STUDY/'status.json',dict(state='failed',pid=os.getpid(),error=traceback.format_exc(),updated_at=now()))
            raise
        finally:
            f.seek(0); msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=['setup','rank','check','train','coordinate','evaluate'])
    parser.add_argument('--dataset',choices=['A','B'])
    parser.add_argument('--model',type=int,choices=[2,3],default=2)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--pilot',action='store_true')
    args = parser.parse_args(); bootstrap()
    try:
        if args.command == 'setup': setup()
        elif args.command == 'rank': rank(args.dataset)
        elif args.command == 'check': readiness()
        elif args.command == 'train': train(args.model,args.dataset,args.resume,args.pilot)
        elif args.command == 'coordinate': coordinate()
        elif args.command == 'evaluate': evaluate_all()
    except Exception:
        print(traceback.format_exc(),flush=True)
        raise


if __name__ == '__main__':
    main()
