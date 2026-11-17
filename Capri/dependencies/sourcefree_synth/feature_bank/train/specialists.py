"""Hybrid v3 specialists v1 (docs/SPECIALISTS_V1.md): shared start with TAG affinity -> learned head groups ->
group specialists -> distilled merge -> intensity -> final selection/calibration/report. No curriculum, no freezing.

  python -m feature_bank.train.specialists <phase> --out runs/spec_v1_20261003 [--scale 0.05] [...]
    phases: A | groups | pools | B --group <name> | teachers | C | D | final

Every phase writes into <out>/<phase dir>/ and is resumable at epoch granularity.
"""
import argparse
import copy
import json
import math
import queue
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from feature_bank.train import data as D
from feature_bank.train import losses as LS
from feature_bank.train import metrics as M
from feature_bank.train.model import build, param_counts
from feature_bank.train.run import predict, subset, lr_at, param_groups, jdump

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
FIXED_DIR = ROOT / "runs/fixed_v3_20261002"
CFG = ROOT / "configs/train_hybrid_v3.yaml"
SPEC_CFG = ROOT / "configs/specialists_v1.yaml"
SHARED_DIRS = [ROOT / f"runs/r1_main_20261002/data/epoch_{k:02d}" for k in (1, 2, 3, 4)] + [ROOT / "runs/spec_data_20261003/base_extra"]
POOL_ROOT = ROOT / "runs/spec_data_20261003"
FIXED = ["steering", "selection", "calibration", "seen_test", "generalisation"]
PHASES = dict(A=dict(epochs=3, lr=5e-4, warmup=300), B=dict(epochs=4, lr=3e-4, warmup=100),
              C=dict(epochs=4, lr=5e-4, warmup=300), D=dict(epochs=2, lr=2e-4, warmup=100),
              E=dict(epochs=8, lr=3e-4, warmup=100))  # E = "C2": merge extension (supervisor, 03:37)
SAMPLES = 20000
TAG_EPS = 0.5          # normalised lookahead step length in shared-parameter space (~ one Adam step at lr 5e-4)
TAG_PER_EPOCH = 4


def log(out, msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(Path(out) / "specialists.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ------------------------------------------------------------------ setup

class Ctx:
    def __init__(self, out, scale):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.scale = scale
        self.cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
        self.tc, self.gates = self.cfg["train"], self.cfg["gates"]
        self.dev = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Windows/WDDM pages GPU memory into system RAM when VRAM is full (silent 5x slowdowns seen in R1 and the smoke
        # test at 16.07/16.38 GB): cap the caching allocator so it frees cache and retries instead of spilling.
        torch.cuda.set_per_process_memory_fraction(0.80)
        self.heads, self.ordinal = D.heads_lists()
        self.oi = [self.heads.index(h) for h in self.ordinal]
        self.fcs = D.CacheSet(D.build_cache(FIXED_DIR, workers=16))
        self.fixed_idx = {}
        for sp in FIXED:
            idx = self.fcs.select(lambda r, sp=sp: r["split"] == sp)
            if scale < 1:
                by = {}
                for i in idx:
                    by.setdefault(self.fcs.rows[i]["stratum"], []).append(i)
                idx = sorted(j for v in by.values() for j in v[:max(1, int(round(len(v) * scale)))])
            self.fixed_idx[sp] = idx
        self.fixed_ids = {r["id"] for r in self.fcs.rows}
        self.fixed_hashes = {r["sha256"] for r in self.fcs.rows}
        self.samples = max(256, int(round(SAMPLES * scale)))

    def pool(self, dirs):
        items = []
        for d in dirs:
            cs = D.CacheSet(D.build_cache(d, workers=16))
            for i, r in enumerate(cs.rows):
                if r["id"] not in self.fixed_ids and r["sha256"] not in self.fixed_hashes and r["split"] not in FIXED:
                    items.append((cs, i))
        return items

    def shared(self):
        return self.pool([d for d in SHARED_DIRS if (d / "audit.json").exists()])

    def new_model(self):
        torch.manual_seed(int(self.cfg["seed"]))
        return build(self.cfg["model"], self.oi).to(self.dev)


def assert_no_leak(ctx, items):
    for cs, i in items:
        r = cs.rows[i]
        assert r["id"] not in ctx.fixed_ids and r["sha256"] not in ctx.fixed_hashes and r["split"] not in FIXED, r["id"]


class Prefetch:
    def __init__(self, batches, depth=4):
        self.q = queue.Queue(depth)
        threading.Thread(target=self.run, args=(batches,), daemon=True).start()

    def run(self, batches):
        for b in batches:
            self.q.put((b, D.collate(b)))
        self.q.put(None)

    def __iter__(self):
        while True:
            x = self.q.get()
            if x is None:
                return
            yield x


# ------------------------------------------------------------------ evaluation

def evaluate(ctx, model, ep_dir, k, tag, extra=None):
    """Steering (states not used for decisions here), selection, generalisation (watch-only)."""
    t0 = time.time()
    rep = {}
    # Phases A-C do not train intensity: ordinal heads are scored on presence there (use_ordinal only in D).
    # mean_presence (use_ordinal=False) is the common yardstick for checkpoint choice across phases.
    uo = tag.startswith("D")
    for sp in ("steering", "selection"):
        P, L = predict(model, [(ctx.fcs, i) for i in ctx.fixed_idx[sp]], ctx.dev)
        sc, text = M.head_scores(P, L, ctx.heads, ctx.ordinal, ctx.gates, use_ordinal=uo)
        scp, _ = M.head_scores(P, L, ctx.heads, ctx.ordinal, ctx.gates, use_ordinal=False)
        rep[sp] = dict(mean_score=M.mean_score(sc), mean_presence=M.mean_score(scp),
                       passed=sum(1 for v in sc.values() if v.get("passed")), scores=sc, text=text)
    g_idx = ctx.fixed_idx["generalisation"]
    gP, gL = predict(model, [(ctx.fcs, i) for i in g_idx], ctx.dev)
    strata = np.array([ctx.fcs.rows[i]["stratum"] for i in g_idx])
    libs = np.array([ctx.fcs.rows[i]["text_lib"] for i in g_idx])
    held = {}
    for name, msk in [("all", np.ones(len(g_idx), bool)), ("holdout_library", strata == "holdout_library"),
                      ("heldout_font", strata == "heldout_font")] + \
                     [(f"library:{l}", libs == l) for l in sorted(set(libs[strata == "holdout_library"]))]:
        if msk.sum():
            s_, t_ = M.head_scores(*subset(gP, gL, msk), ctx.heads, ctx.ordinal, ctx.gates, use_ordinal=uo)
            held[name] = dict(n=int(msk.sum()), mean_score=M.mean_score(s_), passed=sum(1 for v in s_.values() if v.get("passed")),
                              scores=s_, text=t_)
    rep["generalisation"] = dict(watch_only=True, strata=held)
    rep.update(epoch=k, phase=tag, eval_s=round(time.time() - t0, 1), **(extra or {}))
    jdump(ep_dir / "report.json", rep)
    return rep


# ------------------------------------------------------------------ TAG affinity (Fifty et al., NeurIPS 2021)

SHARED_PREFIX = ("cnn.", "proj.", "width_emb.", "blocks.", "norm.", "pool.")


def task_losses(out, b, heads):
    """Per-task losses for TAG: 46 presence BCE (masked mean; NaN when the head has no label in the batch) + text."""
    y = b["presence"].long()
    v = (y >= 0).float()
    bce = F.binary_cross_entropy_with_logits(out["presence"].float(), y.clamp(min=0).float(), reduction="none")
    n = v.sum(0)
    pres = (bce * v).sum(0) / n.clamp(min=1.0)
    pres = torch.where(n > 0, pres, torch.full_like(pres, float("nan")))
    g, _ = LS.compute(out, b, torch.zeros(len(heads), dtype=torch.bool, device=out["presence"].device))
    return torch.cat([pres, g["text"].view(1)])


def tag_measure(model, batch, heads, eps=TAG_EPS):
    """One TAG measurement: Z[i, j] = 1 - L_j(theta - eps g_i/|g_i|) / L_j(theta); cosine of g_i, g_j."""
    model.eval()
    shared = [(n, p) for n, p in model.named_parameters() if n.startswith(SHARED_PREFIX)]
    params = [p for _, p in shared]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(batch["image"], batch["cols"])
    L = task_losses(out, batch, heads)
    T = L.shape[0]
    ok = [i for i in range(T) if torch.isfinite(L[i]) and L[i].item() > 0]
    G = torch.empty(len(ok), sum(p.numel() for p in params), device=L.device)  # unit gradients, one copy only
    for a, i in enumerate(ok):
        g = torch.autograd.grad(L[i], params, retain_graph=a < len(ok) - 1, allow_unused=True)
        off = 0
        for x, p in zip(g, params):
            n_ = p.numel()
            if x is None:
                G[a, off:off + n_] = 0
            else:
                G[a, off:off + n_] = x.flatten().float()
            off += n_
        del g
        G[a] /= G[a].norm().clamp(min=1e-12)
    del out
    L0 = L.detach()
    Z = torch.full((T, T), float("nan"))
    C = torch.full((T, T), float("nan"))
    C_ok = (G @ G.T).cpu()
    for a, i in enumerate(ok):
        for b_, j in enumerate(ok):
            C[i, j] = C_ok[a, b_]
    flat0 = torch.cat([p.detach().flatten().float() for p in params])
    with torch.no_grad():
        for a, i in enumerate(ok):
            step = eps * G[a]
            off = 0
            for p in params:
                n_ = p.numel()
                p.sub_(step[off:off + n_].view_as(p).to(p.dtype))
                off += n_
            with torch.autocast("cuda", dtype=torch.bfloat16):
                Lp = task_losses(model(batch["image"], batch["cols"]), batch, heads)
            Z[i] = (1 - Lp / L0).cpu()
            off = 0
            for p in params:  # restore exactly
                n_ = p.numel()
                p.copy_(flat0[off:off + n_].view_as(p).to(p.dtype))
                off += n_
    del G, flat0
    torch.cuda.empty_cache()
    model.train()
    return Z.numpy(), C.numpy(), L0.cpu().numpy()


# ------------------------------------------------------------------ one training phase

def train_phase(ctx, phase_dir, tag, model, sampler, head_mask, ord_on, text_on, init_from=None, teachers=None,
                tag_measure_on=False, epochs=None):
    """Generic epoch loop. sampler(k, rng) -> list of (CacheSet, i) of length ctx.samples."""
    phase_dir = Path(phase_dir)
    phase_dir.mkdir(parents=True, exist_ok=True)
    P = dict(PHASES[tag[0]])
    E = epochs or P["epochs"]
    dev, tc = ctx.dev, ctx.tc
    hm = torch.tensor([h in head_mask for h in ctx.heads], device=dev)
    om = torch.tensor([h in head_mask for h in ctx.ordinal], device=dev)
    active = {"presence", "coverage"} | ({"ordinal"} if ord_on else set()) | ({"text"} if text_on else set())
    uw = LS.UncertaintyWeights().to(dev)
    if init_from:
        ck = torch.load(init_from, map_location=dev, weights_only=False)
        # raw weights: phase A's EMA (fixed decay 0.999 over ~480 steps) was mostly the random init (supervisor, 02:28)
        model.load_state_dict(ck["model"])
    dec, nodec = param_groups(model, uw)
    opt = torch.optim.AdamW([dict(params=dec, weight_decay=tc["weight_decay"]), dict(params=nodec, weight_decay=0.0)], lr=P["lr"])
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    steps_per = math.ceil(ctx.samples / tc["batch"])
    total = E * steps_per
    sched = dict(lr=P["lr"], min_lr=tc["min_lr"], warmup_steps=max(1, int(P["warmup"] * min(1.0, ctx.scale * 10))))
    step, start = 0, 1
    st_path = phase_dir / "state.json"
    if st_path.exists():
        st = json.loads(st_path.read_text(encoding="utf-8"))
        ck = torch.load(phase_dir / f"epoch_{st['epoch']:02d}" / "model.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"]); uw.load_state_dict(ck["uw"])
        step, start = ck["step"], st["epoch"] + 1
        log(ctx.out, f"{tag}: resumed after epoch {st['epoch']}")
    jdump(phase_dir / "phase_config.json", dict(tag=tag, epochs=E, schedule=sched, samples=ctx.samples, heads=sorted(head_mask),
                                                ordinal=ord_on, text=text_on, init_from=str(init_from) if init_from else None,
                                                distill=bool(teachers), tag_measure=tag_measure_on, params=param_counts(model)))
    tag_Z, tag_C, tag_n = None, None, None
    if tag_measure_on and (phase_dir / "tag_partial.npz").exists():
        z = np.load(phase_dir / "tag_partial.npz")
        tag_Z, tag_C, tag_n = z["Z"], z["C"], z["n"]
    for k in range(start, E + 1):
        ep = phase_dir / f"epoch_{k:02d}"
        ep.mkdir(exist_ok=True)
        rng = random.Random(f"{ctx.cfg['seed']}|{tag}|{k}")
        items = sampler(k, rng)
        assert_no_leak(ctx, items)
        pres = np.stack([cs.ix["presence"][i] for cs, i in items])
        prev = np.array([(pres[:, j] == 1).sum() / max(1, (pres[:, j] >= 0).sum()) for j in range(len(ctx.heads))])
        focal = torch.tensor(prev < float(tc["focal_prevalence"]), device=dev)
        widths = [int(cs.ix["widths"][i]) for cs, i in items]
        batches = D.width_batches(items, widths, int(tc["batch"]), rng)
        tag_at = set(np.linspace(0, len(batches) - 1, TAG_PER_EPOCH + 2).astype(int)[1:-1]) if tag_measure_on else set()
        sums = dict(n=0, groups={g: 0.0 for g in LS.GROUPS}, pres=np.zeros(len(ctx.heads)), pres_n=np.zeros(len(ctx.heads)),
                    ordl=np.zeros(len(ctx.ordinal)), ordl_n=np.zeros(len(ctx.ordinal)), text={}, distill=0.0)
        t0 = time.time()
        model.train()
        for bi, (its, nb) in enumerate(Prefetch(batches)):
            b = D.to_torch(nb, dev)
            if bi in tag_at:
                Z, C, _ = tag_measure(model, b, ctx.heads)
                tag_Z = np.zeros_like(Z) if tag_Z is None else tag_Z
                tag_C = np.zeros_like(C) if tag_C is None else tag_C
                tag_n = np.zeros_like(Z) if tag_n is None else tag_n
                fz, fc = np.isfinite(Z), np.isfinite(C)
                tag_Z[fz] += Z[fz]; tag_C[fc] += C[fc]; tag_n[fz] += 1
                np.savez(phase_dir / "tag_partial.npz", Z=tag_Z, C=tag_C, n=tag_n)
            lr = lr_at(step, total, sched)
            for g in opt.param_groups:
                g["lr"] = lr
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(tc["bf16"])):
                o = model(b["image"], b["cols"])
            g, per = LS.compute(o, b, focal, float(tc["char_weight"]), float(tc["focal_gamma"]), head_mask=hm, ord_mask=om)
            loss = uw(g, active)
            if teachers is not None:
                tp = torch.from_numpy(np.stack([teachers[id(cs)]["presence"][i] for cs, i in its]).astype(np.float32)).to(dev)
                dl = F.binary_cross_entropy_with_logits(o["presence"].float(), torch.sigmoid(tp))
                if "length" in teachers[id(its[0][0])]:
                    tl = torch.from_numpy(np.stack([teachers[id(cs)]["length"][i] for cs, i in its]).astype(np.float32)).to(dev)
                    tcmp = torch.from_numpy(np.stack([teachers[id(cs)]["comp"][i] for cs, i in its]).astype(np.float32)).to(dev)
                    dl = dl + F.kl_div(F.log_softmax(o["length"].float(), -1), F.softmax(tl, -1), reduction="batchmean") \
                        + F.kl_div(F.log_softmax(o["comp"].float(), -1), F.softmax(tcmp, -1), reduction="batchmean")
                loss = loss + dl
                sums["distill"] += float(dl)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tc["grad_clip"]))
            opt.step()
            with torch.no_grad():
                # warm-up EMA: decay min(0.999, (1+t)/(10+t)) tracks the raw weights early and smooths later. A fixed 0.999
                # over a 500-650-step phase left the EMA dominated by its initial weights (scores far below the raw model).
                d_ = min(float(tc["ema"]), (1 + step) / (10 + step))
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(d_).add_(pm.detach(), alpha=1 - d_)
            step += 1
            sums["n"] += 1
            for kk in LS.GROUPS:
                sums["groups"][kk] += float(g[kk])
            pv, pn = per["presence"]
            sums["pres"] += (pv * pn).detach().cpu().numpy(); sums["pres_n"] += pn.cpu().numpy()
            ov, on_ = per["ordinal"]
            sums["ordl"] += (ov * on_).detach().cpu().numpy(); sums["ordl_n"] += on_.cpu().numpy()
            for kk, v in per["text"].items():
                sums["text"][kk] = sums["text"].get(kk, 0.0) + float(v)
        torch.cuda.synchronize()
        train_s = time.time() - t0
        n = max(1, sums["n"])
        lv = uw.log_var.detach().cpu().numpy()
        groups = {kk: round(v / n, 5) for kk, v in sums["groups"].items()}
        glob = sum(math.exp(-lv[i]) * groups[gname] + lv[i] for i, gname in enumerate(LS.GROUPS) if gname in active)
        losses = dict(global_loss=round(float(glob), 5), groups=groups, active_groups=sorted(active),
                      uncertainty_log_var=dict(zip(LS.GROUPS, lv.round(4).tolist())),
                      presence={h: round(float(sums["pres"][j] / max(1, sums["pres_n"][j])), 5) for j, h in enumerate(ctx.heads)},
                      ordinal={h: round(float(sums["ordl"][j] / max(1, sums["ordl_n"][j])), 5) for j, h in enumerate(ctx.ordinal)},
                      text={kk: round(v / n, 5) for kk, v in sums["text"].items()},
                      distill=round(sums["distill"] / n, 5) if teachers is not None else None,
                      focal_heads=[h for j, h in enumerate(ctx.heads) if bool(focal[j])], trained_heads=sorted(head_mask),
                      lr_end=lr, step=step, steps=sums["n"], train_s=round(train_s, 1))
        jdump(ep / "train_losses.json", losses)
        torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), uw=uw.state_dict(), step=step, epoch=k),
                   ep / "model.pt")
        rep = evaluate(ctx, ema, ep, k, tag, extra=dict(train_s=round(train_s, 1)))
        jdump(st_path, dict(epoch=k, step=step))
        sel = rep["selection"]["mean_score"]
        log(ctx.out, f"{tag} epoch {k}/{E}: global {losses['global_loss']} presence {groups['presence']} text {groups['text']} "
                     f"ordinal {groups['ordinal'] if ord_on else '-'} | steering {rep['steering']['mean_score']} passed "
                     f"{rep['steering']['passed']} | selection {sel} | generalisation {rep['generalisation']['strata']['all']['mean_score']} "
                     f"| train {train_s:.0f}s eval {rep['eval_s']}s")
    if tag_measure_on and tag_Z is not None:
        np.savez(phase_dir / "tag_final.npz", Z=tag_Z / np.maximum(tag_n, 1), C=tag_C / np.maximum(tag_n, 1), n=tag_n)
    return phase_dir / f"epoch_{E:02d}" / "model.pt"


# ------------------------------------------------------------------ phases

def phase_A(ctx, a):
    shared = ctx.shared()
    log(ctx.out, f"A: shared pool {len(shared)} images")

    def sampler(k, rng):
        return [shared[rng.randrange(len(shared))] for _ in range(ctx.samples)]
    model = ctx.new_model()
    train_phase(ctx, ctx.out / "phaseA", "A", model, sampler, set(ctx.heads), ord_on=False, text_on=True, tag_measure_on=True)


def cluster_groups(S, names, kmin=4, kmax=7, smin=4, smax=14):
    """Spectral clustering on W = exp((S - max S) / std(S)) (non-negative similarity; balanced partitions), sizes then
    repaired to smin..smax. Chooses k maximising mean within-group minus mean between-group affinity."""
    from sklearn.cluster import SpectralClustering
    off = S[~np.eye(len(S), dtype=bool)]
    tau = float(np.nanstd(off)) or 1.0
    W = np.exp((np.nan_to_num(S, nan=np.nanmin(off)) - np.nanmax(off)) / tau)
    W = (W + W.T) / 2
    np.fill_diagonal(W, 1.0)
    best, tried = None, []
    for k in range(kmin, kmax + 1):
        lab = SpectralClustering(n_clusters=k, affinity="precomputed", random_state=0, assign_labels="discretize").fit_predict(W)
        lab = rebalance(lab, S, smin, smax)
        within, between = [], []
        for i in range(len(names)):
            for j in range(len(names)):
                if i != j and np.isfinite(S[i, j]):
                    (within if lab[i] == lab[j] else between).append(S[i, j])
        score = (np.mean(within) if within else 0) - (np.mean(between) if between else 0)
        tried.append(dict(k=k, score=round(float(score), 6), sizes=sorted(np.bincount(lab).tolist())))
        if best is None or score > best[0]:
            best = (score, k, lab)
    return best, tried


def rebalance(lab, S, smin, smax):
    """Repair group sizes to smin..smax: members of too-small groups move to the group with the highest mean affinity that
    has room; oversized groups shed their lowest-affinity member to the best group with room."""
    lab = lab.copy()
    for _ in range(500):
        cnt = np.bincount(lab, minlength=lab.max() + 1)
        live = [g for g in range(len(cnt)) if cnt[g] > 0]
        small = [g for g in live if cnt[g] < smin]
        big = [g for g in live if cnt[g] > smax]
        if not small and not big:
            break
        g = (small or big)[0]
        members = np.nonzero(lab == g)[0]
        if small:
            aff = [np.nanmean(S[i, lab != g]) for i in members]
            i = members[int(np.nanargmax(aff))]
        else:
            aff = [np.nanmean(S[i, members[members != i]]) for i in members]
            i = members[int(np.nanargmin(aff))]
        others = [h for h in live if h != g and cnt[h] < smax] or [h for h in live if h != g]
        if not others:
            break
        lab[i] = max(others, key=lambda h: np.nanmean(S[i, lab == h]))
    _, lab = np.unique(lab, return_inverse=True)
    return lab


def phase_tag_remeasure(ctx, a, eps_list=(0.1, 0.03, 0.01), batches_per_ckpt=4):
    """TAG re-measured post hoc on phase A's epoch checkpoints (raw weights) with a step-size sweep.
    Chooses the largest eps at which self-affinity Z(i->i) is positive for >= 90% of tasks (lookahead in its valid,
    near-linear range). Reason: the in-training measurement used eps=0.5, where even self-affinity was negative
    (median -0.006): the step overshot and the affinities were mostly noise (supervisor, 2026-10-03)."""
    gd = ctx.out / "groups"
    gd.mkdir(exist_ok=True)
    shared = ctx.shared()
    rng = random.Random(f"{ctx.cfg['seed']}|tag_remeasure")
    widths = None
    res = {}
    model = ctx.new_model()
    for k in range(1, PHASES["A"]["epochs"] + 1):
        ck = torch.load(ctx.out / "phaseA" / f"epoch_{k:02d}" / "model.pt", map_location=ctx.dev, weights_only=False)
        model.load_state_dict(ck["model"])
        items = [shared[rng.randrange(len(shared))] for _ in range(128 * batches_per_ckpt * 2)]
        widths = [int(cs.ix["widths"][i]) for cs, i in items]
        bl = D.width_batches(items, widths, 128, rng)[:batches_per_ckpt]
        for bt in bl:
            b_ = D.to_torch(D.collate(bt), ctx.dev)
            for eps in eps_list:
                Z, C, _ = tag_measure(model, b_, ctx.heads, eps=eps)
                acc = res.setdefault(eps, dict(Z=np.zeros_like(Z), C=np.zeros_like(C), n=np.zeros_like(Z)))
                f = np.isfinite(Z)
                acc["Z"][f] += Z[f]; acc["C"][np.isfinite(C)] += C[np.isfinite(C)]; acc["n"][f] += 1
    summary = {}
    for eps, acc in res.items():
        Zm = acc["Z"] / np.maximum(acc["n"], 1)
        d = np.diag(Zm)
        summary[eps] = dict(self_positive_frac=round(float((d > 0).mean()), 3), self_median=round(float(np.median(d)), 6),
                            offdiag_median=round(float(np.median(Zm[~np.eye(len(Zm), dtype=bool)])), 6))
        np.savez(gd / f"tag_eps{eps}.npz", Z=Zm, C=acc["C"] / np.maximum(acc["n"], 1), n=acc["n"])
    ok = [e for e in sorted(res, reverse=True) if summary[e]["self_positive_frac"] >= 0.9]
    chosen = ok[0] if ok else min(res)
    jdump(gd / "tag_remeasure.json", dict(eps_sweep=summary, chosen_eps=chosen, checkpoints=PHASES["A"]["epochs"],
                                          batches_per_checkpoint=batches_per_ckpt))
    import shutil
    shutil.copyfile(gd / f"tag_eps{chosen}.npz", gd / "tag_used.npz")
    log(ctx.out, f"tag re-measure: sweep {summary} -> chosen eps {chosen}")


def phase_groups(ctx, a):
    gd = ctx.out / "groups"
    gd.mkdir(exist_ok=True)
    src = gd / "tag_used.npz" if (gd / "tag_used.npz").exists() else ctx.out / "phaseA" / "tag_final.npz"
    z = np.load(src)
    Z, C, n = z["Z"], z["C"], z["n"]
    names = ctx.heads + ["text"]
    S = (Z + Z.T) / 2
    np.save(gd / "affinity_Z.npy", Z); np.save(gd / "cosine.npy", C); np.save(gd / "measurements.npy", n)
    (score, k, lab), tried = cluster_groups(S, names)
    groups = {}
    for gi in range(lab.max() + 1):
        mem = [names[i] for i in np.nonzero(lab == gi)[0]]
        groups[f"g{gi + 1}"] = mem
    # importance-overlap check (CNN channels, gradient x activation on steering) from the phase-A model
    from feature_bank.train.freezing import Freezer
    from feature_bank.train.run import cnn_stats
    model = ctx.new_model()
    ck = torch.load(ctx.out / "phaseA" / f"epoch_{PHASES['A']['epochs']:02d}" / "model.pt", map_location=ctx.dev, weights_only=False)
    model.load_state_dict(ck["ema"])
    fr = Freezer(model, ctx.cfg["freezing"])
    stats = cnn_stats(model, ctx.fcs, ctx.fixed_idx["steering"][:256], ctx.dev, ctx.heads, ctx.ordinal, fr,
                      torch.zeros(len(ctx.heads), dtype=torch.bool, device=ctx.dev), bs=32)
    top = {}
    for h, per_unit in stats["importance"].items():
        v = np.concatenate([per_unit[u] for u, _, _ in fr.units])
        if v.max() > 0:
            top[h] = set(np.nonzero(v >= np.quantile(v, 0.8))[0].tolist())
    jac_w, jac_b = [], []
    names_t = [h for h in names if h in top]
    J = np.full((len(names), len(names)), np.nan)
    for i, hi in enumerate(names):
        for j, hj in enumerate(names):
            if i != j and hi in top and hj in top:
                J[i, j] = len(top[hi] & top[hj]) / max(1, len(top[hi] | top[hj]))
                (jac_w if lab[i] == lab[j] else jac_b).append(J[i, j])
    np.save(gd / "importance_overlap.npy", J)
    per_head = {}
    for i, h in enumerate(names):
        row = [(names[j], float(S[i, j])) for j in range(len(names)) if j != i and np.isfinite(S[i, j])]
        row.sort(key=lambda x: -x[1])
        per_head[h] = dict(group=[g for g, m in groups.items() if h in m][0], top_affinity=[(x, round(v, 5)) for x, v in row[:5]],
                           measurements=int(n[i].max()))
    rep = dict(method="TAG lookahead affinity (Fifty et al., NeurIPS 2021, arXiv:2109.04617), re-measured on phase-A checkpoints "
                      "with a step-size sweep (groups/tag_remeasure.json); S=(Z+Z^T)/2; spectral clustering on exp((S-max)/std); "
                      "k in 4..7, sizes 4..14; best within-between", source=str(src),
               chosen_k=k, score=round(float(score), 6), tried=tried, groups=groups,
               importance_overlap=dict(within_mean=round(float(np.mean(jac_w)), 4) if jac_w else None,
                                       between_mean=round(float(np.mean(jac_b)), 4) if jac_b else None,
                                       heads_with_importance=len(names_t)),
               cosine_within_minus_between=None, per_head=per_head)
    cw, cb = [], []
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j and np.isfinite(C[i, j]):
                (cw if lab[i] == lab[j] else cb).append(C[i, j])
    rep["cosine_within_minus_between"] = round(float(np.mean(cw) - np.mean(cb)), 5) if cw and cb else None
    jdump(gd / "groups_report.json", rep)
    jdump(gd / "groups.json", groups)
    log(ctx.out, f"groups: k={k} score {score:.5f} sizes {[len(v) for v in groups.values()]} importance overlap within "
                 f"{rep['importance_overlap']['within_mean']} vs between {rep['importance_overlap']['between_mean']}; "
                 f"cosine within-between {rep['cosine_within_minus_between']}")
    for g, m in groups.items():
        log(ctx.out, f"  {g}: {', '.join(m)}")


def group_heads(ctx):
    G = json.loads((ctx.out / "groups" / "groups.json").read_text(encoding="utf-8"))
    return {g: [h for h in m if h != "text"] for g, m in G.items()}, {g: "text" in m for g, m in G.items()}


def pool_dir(g):
    return POOL_ROOT / f"group_{g}"


def phase_pools(ctx, a):
    """Write per-group targets and render each group's pool (sequential; each 10k, scaled)."""
    G, _ = group_heads(ctx)
    n = max(200, int(round(10000 * ctx.scale)))
    for gi, (g, hs) in enumerate(G.items()):
        if not hs:
            continue
        pf = {h: (0.4 if h in ("bg_solid", "filled_glyphs") else 0.6) for h in hs}
        tp = POOL_ROOT / f"targets_{ctx.out.name}_{g}.json"
        jdump(tp, dict(group=g, heads={h: 1 / len(hs) for h in hs}, pos_fraction=pf))
        cmd = [PY, "-m", "feature_bank.generate_corpus", "--config", str(SPEC_CFG), "--out", str(pool_dir(f"{ctx.out.name}_{g}")),
               "--epoch", str(70 + gi), "--n", str(n), "--targets", str(tp), "--workers", str(a.workers)]
        t0 = time.time()
        with open(POOL_ROOT / f"pool_{ctx.out.name}_{g}.log", "a", encoding="utf-8") as f:
            rc = subprocess.call(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
        log(ctx.out, f"pool {g}: rendered {n} in {time.time() - t0:.0f}s (rc {rc})")
        if rc:
            raise SystemExit(f"pool {g} failed")


def phase_B(ctx, a):
    G, text_of = group_heads(ctx)
    g = a.group
    pdir = pool_dir(f"{ctx.out.name}_{g}")
    n = max(200, int(round(10000 * ctx.scale)))
    while True:  # wait for this group's pool
        au = pdir / "audit.json"
        if au.exists() and json.loads(au.read_text(encoding="utf-8")).get("images", 0) >= n - 5:
            break
        time.sleep(20)
    shared = ctx.shared()
    gpool = ctx.pool([pdir]) if G[g] else []
    log(ctx.out, f"B {g}: heads {G[g]} text {text_of[g]} | group pool {len(gpool)} shared {len(shared)}")

    def sampler(k, rng):
        half = ctx.samples // 2 if gpool else 0
        return [gpool[rng.randrange(len(gpool))] for _ in range(half)] + \
               [shared[rng.randrange(len(shared))] for _ in range(ctx.samples - half)]
    model = ctx.new_model()
    initA = ctx.out / "phaseA" / f"epoch_{PHASES['A']['epochs']:02d}" / "model.pt"
    train_phase(ctx, ctx.out / f"phaseB_{g}", "B_" + g, model, sampler, set(G[g]), ord_on=False, text_on=text_of[g], init_from=initA)


def all_pool_items(ctx):
    G, _ = group_heads(ctx)
    dirs = [d for d in SHARED_DIRS if (d / "audit.json").exists()] + [pool_dir(f"{ctx.out.name}_{g}") for g in G if G[g]]
    return dirs


@torch.no_grad()
def phase_teachers(ctx, a):
    """Precompute each specialist's logits (its heads; text specialist: length, composition) on every training image."""
    G, text_of = group_heads(ctx)
    td = ctx.out / "teachers"
    td.mkdir(exist_ok=True)
    models = {}
    for g in G:
        m = ctx.new_model()
        ck = torch.load(ctx.out / f"phaseB_{g}" / f"epoch_{PHASES['B']['epochs']:02d}" / "model.pt", map_location=ctx.dev, weights_only=False)
        m.load_state_dict(ck["ema"])
        models[g] = m.eval()
    hidx = {g: [ctx.heads.index(h) for h in hs] for g, hs in G.items()}
    tg = [g for g in G if text_of[g]]
    for d in all_pool_items(ctx):
        f = td / (Path(d).name + "__" + Path(d).parent.name + ".npz")
        if f.exists():
            continue
        cs = D.CacheSet(D.build_cache(d, workers=16))
        N = len(cs)
        pres = np.zeros((N, len(ctx.heads)), np.float16)
        ln = np.zeros((N, 7), np.float16)
        cp = np.zeros((N, 3), np.float16)
        order = np.argsort(cs.ix["widths"])
        t0 = time.time()
        for s in range(0, N, 256):
            idx = order[s:s + 256]
            b = D.to_torch(D.collate([(cs, int(i)) for i in idx]), ctx.dev)
            for g, m in models.items():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = m(b["image"], b["cols"])
                pres[np.ix_(idx, hidx[g])] = o["presence"][:, hidx[g]].float().cpu().numpy()
                if g in tg:
                    ln[idx] = o["length"].float().cpu().numpy()
                    cp[idx] = o["comp"].float().cpu().numpy()
        out = dict(presence=pres)
        if tg:
            out.update(length=ln, comp=cp)
        np.savez(f, ids=np.array([r["id"] for r in cs.rows]), **out)
        log(ctx.out, f"teachers: {Path(d).parent.name}/{Path(d).name} {N} images in {time.time() - t0:.0f}s")


def load_teachers(ctx, items):
    td = ctx.out / "teachers"
    T = {}
    for cs, _ in items:
        if id(cs) in T:
            continue
        d = cs.cdir.parent
        z = np.load(td / (d.name + "__" + d.parent.name + ".npz"))
        assert list(z["ids"]) == [r["id"] for r in cs.rows], f"teacher rows out of sync for {d}"
        T[id(cs)] = {k: z[k] for k in z.files if k != "ids"}
    return T


def phase_C(ctx, a, tag="C", ord_on=False, init=None):
    dirs = all_pool_items(ctx)
    items = ctx.pool(dirs)
    teachers = load_teachers(ctx, items)
    log(ctx.out, f"{tag}: {len(items)} training images over {len(dirs)} pools; distillation from specialists")

    def sampler(k, rng):
        return [items[rng.randrange(len(items))] for _ in range(ctx.samples)]
    model = ctx.new_model()
    init = init or ctx.out / "phaseA" / f"epoch_{PHASES['A']['epochs']:02d}" / "model.pt"
    train_phase(ctx, ctx.out / f"phase{tag}", tag, model, sampler, set(ctx.heads), ord_on=ord_on, text_on=True,
                init_from=init, teachers=teachers)


def phase_C2(ctx, a):
    """Merge extension: phase C was still rising (steering 0.24 -> 0.29 -> 0.37 -> 0.39) when its cosine schedule ended.
    8 more distillation epochs from C's last raw weights with a fresh schedule (peak 3e-4, warmup 100)."""
    initC = ctx.out / "phaseC" / f"epoch_{PHASES['C']['epochs']:02d}" / "model.pt"
    phase_C(ctx, a, tag="E", ord_on=False, init=initC)


def phase_D(ctx, a):
    c2 = ctx.out / "phaseE" / f"epoch_{PHASES['E']['epochs']:02d}" / "model.pt"
    initC = c2 if c2.exists() else ctx.out / "phaseC" / f"epoch_{PHASES['C']['epochs']:02d}" / "model.pt"
    phase_C(ctx, a, tag="D", ord_on=True, init=initC)


# ------------------------------------------------------------------ final: selection, temperature, test

def fit_temperature(logit, y):
    """Per-head temperature on calibration (presence), by minimising BCE over a grid then refining."""
    v = y >= 0
    if v.sum() < 20 or len(set(y[v].tolist())) < 2:
        return 1.0
    lz, yt = torch.tensor(logit[v], dtype=torch.float64), torch.tensor(y[v], dtype=torch.float64)
    best = min(np.exp(np.linspace(np.log(0.2), np.log(5), 60)), key=lambda T: float(F.binary_cross_entropy_with_logits(lz / T, yt)))
    return float(best)


def ordinal_summary(sc, ordinal):
    """Intensity (0-10) metrics of the ordinal heads that have level labels in the scored set."""
    rows = {h: {k: sc[h].get(k) for k in ("n_levels", "mae", "within1", "within2", "exact")} for h in ordinal
            if sc.get(h, {}).get("n_levels", 0) >= M.MIN_LEVELS}
    w1 = [v["within1"] for v in rows.values() if v["within1"] is not None]
    return dict(heads_with_levels=len(rows), mean_within1=round(float(np.mean(w1)), 4) if w1 else None, per_head=rows)


def score_model(ctx, path, label):
    """Calibrate (per-head temperature on calibration) and score once on seen_test and generalisation.
    Presence is scored on its own (use_ordinal=False); intensity is reported separately."""
    model = ctx.new_model()
    model.load_state_dict(torch.load(path, map_location=ctx.dev, weights_only=False)["ema"])
    Pc, Lc = predict(model, [(ctx.fcs, i) for i in ctx.fixed_idx["calibration"]], ctx.dev)
    T = np.array([fit_temperature(Pc["presence"][:, j], Lc["presence"][:, j]) for j in range(len(ctx.heads))])
    res = dict(model=label, path=str(path), temperature=dict(zip(ctx.heads, T.round(3).tolist())))
    for sp in ("seen_test", "generalisation"):
        idx = ctx.fixed_idx[sp]
        P, L = predict(model, [(ctx.fcs, i) for i in idx], ctx.dev)
        P["presence"] = P["presence"] / T[None, :]
        strata = np.array([ctx.fcs.rows[i]["stratum"] for i in idx])
        libs = np.array([ctx.fcs.rows[i]["text_lib"] for i in idx])
        parts = [("all", np.ones(len(idx), bool))]
        if sp == "generalisation":
            parts += [("holdout_library", strata == "holdout_library"), ("heldout_font", strata == "heldout_font")] +                      [(f"library:{l}", libs == l) for l in sorted(set(libs[strata == "holdout_library"]))]
        res[sp] = {}
        for name, msk in parts:
            s_, t_ = M.head_scores(*subset(P, L, msk), ctx.heads, ctx.ordinal, ctx.gates, use_ordinal=False)
            feats = {h: v for h, v in s_.items() if h in ctx.heads and v.get("score") is not None}
            res[sp][name] = dict(n=int(msk.sum()), mean_presence=M.mean_score(s_),
                                 heads_f1_ge_0_7=sum(1 for v in feats.values() if v["f1"] >= 0.7), heads_scored=len(feats),
                                 passed=sum(1 for v in s_.values() if v.get("passed")), intensity=ordinal_summary(s_, ctx.ordinal),
                                 scores=s_, text=t_)
    return res


def phase_final(ctx, a):
    """Checkpoint choice on selection (presence-only mean, once, over C / C2 / D), then calibration and a single scoring
    of the chosen model; the intensity model (D, last epoch) is scored the same way for its 0-10 ratings."""
    fd = ctx.out / "final"
    fd.mkdir(exist_ok=True)
    cands = []
    for tag in ("C", "E", "D"):
        for ep in sorted((ctx.out / f"phase{tag}").glob("epoch_*")):
            r = json.loads((ep / "report.json").read_text(encoding="utf-8"))
            cands.append(dict(phase=tag, epoch=int(ep.name[6:]), selection=r["selection"].get("mean_presence", r["selection"]["mean_score"]),
                              path=str(ep / "model.pt")))
    best = max(cands, key=lambda c: c["selection"])
    res = dict(selection_rule="max selection mean_presence over C/C2(E)/D epochs, once", chosen=best, candidates=cands)
    res["chosen_model"] = score_model(ctx, best["path"], f"{best['phase']} epoch {best['epoch']}")
    dlast = sorted((ctx.out / "phaseD").glob("epoch_*"))[-1] / "model.pt"
    if str(dlast) != best["path"]:
        res["intensity_model"] = score_model(ctx, dlast, "D last epoch (intensity trained)")
    jdump(fd / "final_report.json", res)
    for k in ("chosen_model", "intensity_model"):
        if k in res:
            m = res[k]
            st, g = m["seen_test"]["all"], m["generalisation"]
            log(ctx.out, f"final {k} ({m['model']}): seen test presence mean {st['mean_presence']} F1>=0.7 {st['heads_f1_ge_0_7']}/"
                         f"{st['heads_scored']} passed {st['passed']} length {st['text']['length_acc']} intensity within1 "
                         f"{st['intensity']['mean_within1']} | generalisation presence all {g['all']['mean_presence']} "
                         f"libs {g['holdout_library']['mean_presence']} fonts {g['heldout_font']['mean_presence']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["A", "tag", "groups", "pools", "B", "teachers", "C", "C2", "D", "final"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--group")
    ap.add_argument("--workers", type=int, default=28)
    a = ap.parse_args()
    ctx = Ctx(a.out, a.scale)
    dict(A=phase_A, tag=phase_tag_remeasure, groups=phase_groups, pools=phase_pools, B=phase_B, teachers=phase_teachers, C=phase_C, C2=phase_C2, D=phase_D,
         final=phase_final)[a.phase](ctx, a)


if __name__ == "__main__":
    main()
