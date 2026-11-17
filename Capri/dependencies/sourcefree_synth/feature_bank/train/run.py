"""Hybrid v3 curriculum training loop (docs/HYBRID_V3_TRAINING_HANDOFF.md §3, §4 step 9).

  python -m feature_bank.train.run --config configs/train_hybrid_v3.yaml --fixed runs/fixed_v3_20261002 --out runs/<name>
      [--epoch1-dir runs/<shared_epoch1>] [--no-freeze] [--static runs/<static_corpus>] [--scale 0.1 --epochs 2]
      [--gen-workers 14]

Per epoch k: (base half of batch k+1 renders in the background) -> train 20k samples (batch k once + uniform replay;
epoch 1: batch x3) -> save epoch_k/model.pt -> score steering (report.json), selection (selection.json), generalisation
(heldout_report.json, watch-only) -> targets from steering only (targets.json) -> targeted half of batch k+1 ->
freeze-map update (freeze_map.json). Resumable from the last saved epoch (state.json). Asserts that no training batch
holds a fixed-set or generalisation image (ids and hashes).
"""
import argparse
import copy
import json
import math
import os
import queue
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from feature_bank.train import data as D
from feature_bank.train import losses as LS
from feature_bank.train import metrics as M
from feature_bank.train.freezing import Freezer
from feature_bank.train.model import build, GrowBranch, param_counts

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
FIXED = ["steering", "selection", "calibration", "seen_test", "generalisation"]


def log(out, msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(out / "train.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def jdump(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")


# ------------------------------------------------------------------ data generation (subprocess)

def gen_cmd(cfg_path, out_dir, k, n, part, workers, targets=None):
    c = [PY, "-m", "feature_bank.generate_corpus", "--config", str(cfg_path), "--out", str(out_dir), "--epoch", str(k),
         "--n", str(n), "--part", part, "--workers", str(workers)]
    if targets:
        c += ["--targets", str(targets)]
    return c


def run_gen(cmd, logf, wait=True):
    f = open(logf, "a", encoding="utf-8")
    p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONPATH=str(ROOT)))
    if wait:
        rc = p.wait()
        f.close()
        if rc != 0:
            raise RuntimeError(f"generator failed ({rc}): {' '.join(cmd)}; see {logf}")
        return None
    return p


def batch_complete(d, n):
    a = Path(d) / "audit.json"
    if not a.exists():
        return False
    au = json.loads(a.read_text(encoding="utf-8"))
    return au.get("images") == n and au.get("complete")


# ------------------------------------------------------------------ evaluation

@torch.no_grad()
def predict(model, sets_items, dev, batch=256):
    """sets_items: [(CacheSet, i)] -> (P dict of numpy outputs, L dict of labels, order)."""
    model.eval()
    widths = [int(s.ix["widths"][i]) for s, i in sets_items]
    order = np.argsort(widths, kind="stable")
    P, L = {}, {}
    keys_p = ["presence", "ordinal", "length", "comp", "slot_present", "slot_box", "slot_char"]
    # pixel-budgeted chunks (no grad: ~2x the training budget). A fixed 256 images of the widest test images ran out of
    # memory after training had fragmented the cache (specialists v1, g5, 03:08).
    torch.cuda.empty_cache()
    chunks, cur = [], []
    for j in order:
        w = widths[j] + D.MARGIN
        if cur and (len(cur) >= batch or (len(cur) + 1) * w > 2 * D.PIXEL_BUDGET):
            chunks.append(cur)
            cur = []
        cur.append(j)
    if cur:
        chunks.append(cur)
    for ch in chunks:
        its = [sets_items[j] for j in ch]
        b = D.to_torch(D.collate(its), dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(b["image"], b["cols"])
        for k in keys_p:
            P.setdefault(k, []).append(o[k].float().cpu().numpy())
        for k in ["presence", "level", "length", "comp", "slot_mask", "slot_present", "slot_box", "slot_char"]:
            L.setdefault(k, []).append(b[k].cpu().numpy())
    inv = np.argsort(order)
    P = {k: np.concatenate(v)[inv] for k, v in P.items()}
    L = {k: np.concatenate(v)[inv] for k, v in L.items()}
    return P, L


def subset(P, L, mask):
    return {k: v[mask] for k, v in P.items()}, {k: v[mask] for k, v in L.items()}


def score_set(model, cs, idx, dev, heads, ordinal, gates):
    P, L = predict(model, [(cs, i) for i in idx], dev)
    sc, text = M.head_scores(P, L, heads, ordinal, gates)
    return sc, text, P, L


# ------------------------------------------------------------------ freezing statistics on steering

def cnn_stats(model, cs, idx, dev, heads, ordinal, freezer, focal_mask, bs=128):
    """Per CNN channel: weights, mean |activation|, importance (|grad x act|) per head, on a steering subset."""
    model.eval()
    acts = {}
    hooks = []
    for n, conv, gn in freezer.units:
        mod = dict(model.cnn.named_modules())[n]

        def hook(m, i, o, n=n):
            acts[n] = o
        hooks.append(mod.register_forward_hook(hook))
    imp = {}
    nch = {n: conv.out_channels for n, conv, _ in freezer.units}
    act_sum = {n: np.zeros(c) for n, c in nch.items()}
    nb = 0
    oi = {h: j for j, h in enumerate(ordinal)}
    for a in range(0, len(idx), bs):
        b = D.to_torch(D.collate([(cs, i) for i in idx[a:a + bs]]), dev)
        with torch.enable_grad():
            out = model(b["image"], b["cols"])
            names = list(acts)
            tensors = [acts[n] for n in names]
            for n, t in zip(names, tensors):
                act_sum[n] = act_sum[n] + t.detach().abs().mean(dim=(0, 2, 3)).float().cpu().numpy()
            nb += 1
            y = b["presence"].long()
            for j, h in enumerate(heads):
                v = y[:, j] >= 0
                if v.sum() == 0 or (y[v, j] == 1).sum() == 0:
                    continue
                l = torch.nn.functional.binary_cross_entropy_with_logits(out["presence"][v, j].float(), y[v, j].float())
                if h in oi:
                    lv = b["level"][:, oi[h]].long()
                    vv = lv >= 0
                    if vv.any():
                        thr = torch.arange(1, 11, device=dev).view(1, 10)
                        l = l + torch.nn.functional.binary_cross_entropy_with_logits(
                            out["ordinal"][vv, oi[h]].float(), (lv[vv].unsqueeze(-1) >= thr).float())
                gr = torch.autograd.grad(l, tensors, retain_graph=True, allow_unused=True)
                acc = imp.setdefault(h, {n: np.zeros(nch[n]) for n in names})
                for n, t, g in zip(names, tensors, gr):
                    if g is not None:
                        acc[n] = acc[n] + (g * t).detach().abs().sum(dim=(0, 2, 3)).float().cpu().numpy()
            for tn, fn in (("length", lambda o: torch.nn.functional.cross_entropy(o["length"].float(), b["length"].long())),
                           ("composition", lambda o: (torch.softmax(o["comp"].float(), -1) - b["comp"]).abs().sum(-1).mean())):
                gr = torch.autograd.grad(fn(out), tensors, retain_graph=True, allow_unused=True)
                acc = imp.setdefault(tn, {n: np.zeros(nch[n]) for n in names})
                for n, t, g in zip(names, tensors, gr):
                    if g is not None:
                        acc[n] = acc[n] + (g * t).detach().abs().sum(dim=(0, 2, 3)).float().cpu().numpy()
        acts.clear()
    for h in hooks:
        h.remove()
    model.train()
    return dict(weights=freezer.weights(), act={n: v / max(1, nb) for n, v in act_sum.items()}, importance=imp)


# ------------------------------------------------------------------ targets (steering only)

SUPERVISOR_DEFAULTS = dict(
    policy="v2",
    focus_k=12,            # targeted half goes to at most this many heads (the furthest below gate)
    min_share=0.05,        # per selected head, of the targeted half
    max_share=0.20,
    min_steering_pos=5,    # a head needs this many positives on steering to be scored reliably
    pos_frac_miss=0.8,     # head misses the effect (recall < 0.5, FPR ok): mostly hard positives
    pos_frac_overfire=0.2, # head fires on look-alikes (FPR > gate, recall ok): mostly hard negatives
    pos_frac_both=0.5,     # both failing
    pos_frac_levels=0.6,   # presence fine, intensity (within +-1) below gate: low-level positives
    force=[],              # heads always included (supervisor decision)
    exclude=[],            # heads never targeted (supervisor decision)
)


def supervisor_policy(out):
    """Targeting policy: defaults, overridden by <run>/supervisor.json (edited between epochs by the supervisor)."""
    p = dict(SUPERVISOR_DEFAULTS)
    f = out / "supervisor.json"
    if f.exists():
        p.update(json.loads(f.read_text(encoding="utf-8")).get("policy", {}))
    return p


def decide_targets(scores, states, heads, k, cur, policy=None):
    """Targets for batch k+1 from steering only.

    policy v1 (handoff §3): every improving/stalled head, shares proportional to gap in [2%, 20%], 50/50 pos/neg.
    policy v2 (supervisor, 2026-10-03): the `focus_k` heads furthest below gate get the targeted half (shares by gap in
    [min_share, max_share]); each head's positive fraction follows its failure: misses -> mostly positives, fires on
    look-alikes -> mostly negatives. Reason: under v1, 45 heads x ~75 images (half negatives) left rare heads' prevalence
    unchanged, so "always absent" stayed the easy answer.
    """
    from feature_bank.targeting import normalise_shares
    p = policy or dict(SUPERVISOR_DEFAULTS, policy="v1")
    gates_fpr = 0.10
    cand = {h: scores[h]["gap"] for h in heads
            if states.get(h) in ("improving", "stalled") and scores[h].get("gap") and scores[h]["gap"] > 0
            and h not in p.get("exclude", [])}
    if p["policy"] == "v1":
        shares = normalise_shares(cand, float(cur["head_min_share"]), float(cur["head_max_share"])) if cand else {}
        pos = {h: float(cur["hard_positive_fraction"]) for h in shares}
        why = {}
    else:
        cand = {h: g for h, g in cand.items() if (scores[h].get("n_pos") or 0) >= int(p["min_steering_pos"])}
        ranked = sorted(cand, key=lambda h: (-cand[h], scores[h].get("recall") or 0, h))
        sel = list(dict.fromkeys([h for h in p.get("force", []) if h in heads] + ranked))[:int(p["focus_k"])]
        w = {h: max(cand.get(h, 0.0), 1e-3) for h in sel}
        shares = normalise_shares(w, float(p["min_share"]), float(p["max_share"])) if w else {}
        pos, why = {}, {}
        for h in shares:
            s = scores[h]
            rec, fpr = s.get("recall") or 0.0, s.get("fpr") or 0.0
            if rec < 0.5 and fpr <= gates_fpr:
                pos[h], why[h] = float(p["pos_frac_miss"]), f"misses the effect (recall {rec:.2f})"
            elif fpr > gates_fpr and rec >= 0.5:
                pos[h], why[h] = float(p["pos_frac_overfire"]), f"fires on look-alikes (FPR {fpr:.2f})"
            elif fpr > gates_fpr:
                pos[h], why[h] = float(p["pos_frac_both"]), f"misses and over-fires (recall {rec:.2f}, FPR {fpr:.2f})"
            else:
                pos[h], why[h] = float(p["pos_frac_levels"]), f"presence ok, below gate on F1/levels (score {s.get('score')})"
    return dict(epoch=k + 1, decided_after_epoch=k, source="steering", policy=p, heads=shares, pos_fraction=pos,
                reasons=why, gaps=cand, not_selected=sorted(set(cand) - set(shares)),
                text_heads_below_gate={h: scores[h]["gap"] for h in ("length", "composition", "slots")
                                       if scores.get(h, {}).get("gap")},
                note="text heads are scored and logged but have no generator lever beyond the base mix")


# ------------------------------------------------------------------ main loop

class Prefetch:
    def __init__(self, batches, dev, depth=4):
        self.q = queue.Queue(depth)
        self.t = threading.Thread(target=self.run, args=(batches, dev), daemon=True)
        self.t.start()

    def run(self, batches, dev):
        for b in batches:
            self.q.put(D.collate(b))
        self.q.put(None)

    def __iter__(self):
        while True:
            b = self.q.get()
            if b is None:
                return
            yield b


def lr_at(step, total, tc):
    if step < tc["warmup_steps"]:
        return tc["lr"] * (step + 1) / tc["warmup_steps"]
    t = (step - tc["warmup_steps"]) / max(1, total - tc["warmup_steps"])
    return tc["min_lr"] + 0.5 * (tc["lr"] - tc["min_lr"]) * (1 + math.cos(math.pi * min(1.0, t)))


def param_groups(model, uw):
    dec, nodec = [], []
    for n, p in model.named_parameters():
        if p.ndim <= 1 or "emb" in n or n.endswith((".bias", ".b1", ".b2", ".q")) or "attn.bias" in n or "norm" in n:
            nodec.append(p)
        elif "rel" in n or n.endswith("attn.bias"):
            nodec.append(p)
        else:
            dec.append(p)
    # RelAttention.bias is a 3-d parameter named "...attn.bias": no decay (relative position bias)
    return dec, nodec + list(uw.parameters())


def grow_model(model, ema, opt, dev):
    for m in (model, ema):
        m.cnn.r2.grow.append(GrowBranch(96, 16, 96).to(dev))
    opt.add_param_group(dict(params=list(model.cnn.r2.grow[-1].parameters()), weight_decay=0.05))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/train_hybrid_v3.yaml"))
    ap.add_argument("--fixed", required=True, help="frozen fixed-set dir (runs/fixed_v3_*)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epoch1-dir", help="shared epoch-1 batch dir (R1/R3)")
    ap.add_argument("--static", help="static control: pre-generated corpus dir with split train (R2)")
    ap.add_argument("--no-freeze", action="store_true")
    ap.add_argument("--scale", type=float, default=1.0, help="smoke runs: batch, samples and fixed-set sizes x scale")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--gen-workers", type=int)
    ap.add_argument("--stop-after", type=int, help="stop after this epoch (resume test)")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    cur, tc, gates = cfg["curriculum"], cfg["train"], cfg["gates"]
    E = a.epochs or int(cur["epochs"])
    N = max(64, int(round(cur["new_per_epoch"] * a.scale)))
    S = max(128, int(round(cur["samples_per_epoch"] * a.scale)))
    freeze_on = cfg["freezing"]["enabled"] and not a.no_freeze
    gw = a.gen_workers or int(cfg["workers"])
    run_cfg = dict(config=str(a.config), fixed=str(a.fixed), epoch1_dir=a.epoch1_dir, static=a.static, freeze=freeze_on,
                   scale=a.scale, epochs=E, new_per_epoch=N, samples_per_epoch=S, gen_workers=gw, argv=sys.argv)
    st_path = out / "state.json"
    state = json.loads(st_path.read_text(encoding="utf-8")) if st_path.exists() else None
    if state is None:
        jdump(out / "run_config.json", run_cfg)
        (out / "config.yaml").write_text(Path(a.config).read_text(encoding="utf-8"), encoding="utf-8")
    else:
        old = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
        for k in ("fixed", "epoch1_dir", "static", "freeze", "scale", "epochs", "new_per_epoch", "samples_per_epoch"):
            if old.get(k) != run_cfg.get(k):
                raise SystemExit(f"resume with a different {k}: {old.get(k)} vs {run_cfg.get(k)}")
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    heads, ordinal = D.heads_lists()
    oi = [heads.index(h) for h in ordinal]

    # ---------------- fixed sets
    fcs = D.CacheSet(D.build_cache(a.fixed, workers=16))
    fixed_idx = {}
    for sp in FIXED:
        idx = fcs.select(lambda r, sp=sp: r["split"] == sp)
        if a.scale < 1:  # 1/10 of every split and stratum
            by = {}
            for i in idx:
                by.setdefault(fcs.rows[i]["stratum"], []).append(i)
            idx = sorted(j for v in by.values() for j in v[:max(1, int(round(len(v) * a.scale)))])
        fixed_idx[sp] = idx
    fixed_ids = {r["id"] for r in fcs.rows}
    fixed_hashes = {r["sha256"] for r in fcs.rows}
    log(out, f"fixed sets: { {k: len(v) for k, v in fixed_idx.items()} }  N={N} S={S} epochs={E} freeze={freeze_on} "
             f"static={bool(a.static)} gen_workers={gw}")

    # ---------------- model
    torch.manual_seed(int(cfg["seed"]))
    model = build(cfg["model"], oi).to(dev)
    uw = LS.UncertaintyWeights().to(dev)
    dec, nodec = param_groups(model, uw)
    opt = torch.optim.AdamW([dict(params=dec, weight_decay=tc["weight_decay"]), dict(params=nodec, weight_decay=0.0)],
                            lr=tc["lr"], betas=(0.9, 0.999))
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    freezer = Freezer(model, cfg["freezing"])
    steps_per_epoch = math.ceil(S / tc["batch"])
    total_steps = E * steps_per_epoch
    tcs = dict(tc)
    if a.scale < 1:
        tcs["warmup_steps"] = max(1, int(tc["warmup_steps"] * a.scale))
    sup_train = (json.loads((out / "supervisor.json").read_text(encoding="utf-8")).get("train", {})
                 if (out / "supervisor.json").exists() else {})
    tcs.update(sup_train)  # supervisor overrides of the schedule (logged in supervisor.json)
    history, best, stats_hist, timings = {}, {}, [], []
    start = 1
    step = 0
    if state:
        k0 = state["epoch"]
        ck = torch.load(out / f"epoch_{k0:02d}" / "model.pt", map_location=dev, weights_only=False)
        for _ in range(len(ck["freezer"].get("grown", []))):
            grow_model(model, ema, opt, dev)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        uw.load_state_dict(ck["uw"])
        opt.load_state_dict(ck["opt"])
        freezer.load(ck["freezer"], model)
        history, best, step, timings = state["history"], state["best"], state["step"], state.get("timings", [])
        stats_hist = [torch.load(out / f"epoch_{j:02d}" / "cnn_stats.pt", weights_only=False)
                      for j in range(max(1, k0 - 1), k0 + 1) if (out / f"epoch_{j:02d}" / "cnn_stats.pt").exists()]
        torch.set_rng_state(ck["torch_rng"].cpu())
        start = k0 + 1
        log(out, f"resumed after epoch {k0} (step {step})")
    jdump(out / "params.json", param_counts(model))

    # ---------------- data batches
    def batch_dir(k):
        if a.static:
            return Path(a.static)
        if k == 1 and a.epoch1_dir:
            return Path(a.epoch1_dir)
        return out / "data" / f"epoch_{k:02d}"

    static_items = None
    if a.static:
        scs = D.CacheSet(D.build_cache(a.static, workers=16))
        static_items = [(scs, i) for i in scs.select(lambda r: r["split"] == "train")]
        assert not any(scs.rows[i]["id"] in fixed_ids or scs.rows[i]["sha256"] in fixed_hashes for _, i in static_items), \
            "static corpus overlaps the fixed sets"
        log(out, f"static corpus: {len(static_items)} images")

    def ensure_batch(k, targets_path=None):
        d = batch_dir(k)
        if batch_complete(d, N):
            return d
        t = time.time()
        run_gen(gen_cmd(a.config, d, k, N, "all", gw, targets_path), out / "generation.log")
        log(out, f"batch {k}: generated/finished in {time.time() - t:.0f}s")
        return d

    caches = {}

    def batch_items(k):
        if k not in caches:
            d = batch_dir(k)
            cs = D.CacheSet(D.build_cache(d, workers=16))
            idx = [i for i, r in enumerate(cs.rows) if r["id"] not in fixed_ids and r["sha256"] not in fixed_hashes]
            dropped = len(cs.rows) - len(idx)
            if dropped:
                log(out, f"batch {k}: dropped {dropped} images whose id/hash is in the fixed sets")
            caches[k] = [(cs, i) for i in idx]
        return caches[k]

    base_proc = None
    for k in range(start, E + 1):
        ep = out / f"epoch_{k:02d}"
        ep.mkdir(exist_ok=True)
        T = dict(epoch=k)
        t0 = time.time()
        if not a.static:
            tp = out / f"epoch_{k - 1:02d}" / "targets.json" if k > 1 else None
            ensure_batch(k, tp if tp and tp.exists() else None)
        T["wait_batch_s"] = round(time.time() - t0, 1)
        t0 = time.time()
        new = static_items if a.static else batch_items(k)
        T["cache_s"] = round(time.time() - t0, 1)
        # base half of batch k+1 renders while this epoch trains
        if not a.static and k < E and not batch_complete(batch_dir(k + 1), N):
            base_proc = run_gen(gen_cmd(a.config, batch_dir(k + 1), k + 1, N, "base", gw), out / "generation.log", wait=False)
        # ---- assemble this epoch's samples
        rng = random.Random(f"{cfg['seed']}|samples|{k}")
        if a.static:
            samples = rng.sample(static_items, min(S, len(static_items)))
        else:
            reps = math.ceil(S / len(new)) if k == 1 else 1
            samples = (new * reps)[:S] if k == 1 else list(new)
            if k > 1:
                pool = [it for j in range(1, k) for it in batch_items(j)]
                samples += [pool[rng.randrange(len(pool))] for _ in range(S - len(samples))]
        # leak assertion (ids and hashes) on every training item
        for cs, i in samples:
            r = cs.rows[i]
            assert r["id"] not in fixed_ids and r["sha256"] not in fixed_hashes, f"fixed-set image in training: {r['id']}"
            assert r["split"] not in FIXED, f"fixed split in training: {r['id']}"
        # focal heads: positive prevalence < 5% in the current pool
        pool_items = static_items if a.static else [it for j in range(1, k + 1) for it in batch_items(j)]
        pres = np.stack([cs.ix["presence"][i] for cs, i in pool_items])
        prev = np.array([(pres[:, j] == 1).sum() / max(1, (pres[:, j] >= 0).sum()) for j in range(len(heads))])
        focal_mask = torch.tensor(prev < float(tc["focal_prevalence"]), device=dev)
        # ---- grow channels decided at the previous epoch
        while freeze_on and len(model.cnn.r2.grow) < len(freezer.grown):
            grow_model(model, ema, opt, dev)
            log(out, f"grew r2 by 16 channels for {freezer.grown[len(model.cnn.r2.grow) - 1]}")
        # ---- train
        t0 = time.time()
        model.train()
        widths = [int(cs.ix["widths"][i]) for cs, i in samples]
        batches = D.width_batches(samples, widths, int(tc["batch"]), rng)
        sums = dict(n=0, groups={g: 0.0 for g in LS.GROUPS}, presence=np.zeros(len(heads)), presence_n=np.zeros(len(heads)),
                    ordinal=np.zeros(len(ordinal)), ordinal_n=np.zeros(len(ordinal)), text={})
        for nb in Prefetch(batches, dev):
            b = D.to_torch(nb, dev)
            lr = lr_at(step, total_steps, tcs)
            for g in opt.param_groups:
                g["lr"] = lr
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(tc["bf16"])):
                o = model(b["image"], b["cols"])
            g, per = LS.compute(o, b, focal_mask, float(tc["char_weight"]), float(tc["focal_gamma"]))
            loss = uw(g)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if freeze_on:
                freezer.mask_grads()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tc["grad_clip"]))
            opt.step()
            if freeze_on:
                freezer.restore()
            with torch.no_grad():
                d_ = float(tc["ema"])
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(d_).add_(pm.detach(), alpha=1 - d_)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            step += 1
            sums["n"] += 1
            for kk in LS.GROUPS:
                sums["groups"][kk] += float(g[kk])
            pv, pn = per["presence"]
            sums["presence"] += (pv * pn).detach().cpu().numpy()
            sums["presence_n"] += pn.cpu().numpy()
            ov, on_ = per["ordinal"]
            sums["ordinal"] += (ov * on_).detach().cpu().numpy()
            sums["ordinal_n"] += on_.cpu().numpy()
            for kk, v in per["text"].items():
                sums["text"][kk] = sums["text"].get(kk, 0.0) + float(v)
        torch.cuda.synchronize()
        T["train_s"] = round(time.time() - t0, 1)
        T["steps"] = sums["n"]
        train_losses = dict(
            groups={kk: round(v / max(1, sums["n"]), 5) for kk, v in sums["groups"].items()},
            uncertainty_log_var=dict(zip(LS.GROUPS, uw.log_var.detach().cpu().numpy().round(4).tolist())),
            presence={h: round(float(sums["presence"][j] / max(1, sums["presence_n"][j])), 5) for j, h in enumerate(heads)},
            ordinal={h: round(float(sums["ordinal"][j] / max(1, sums["ordinal_n"][j])), 5) for j, h in enumerate(ordinal)},
            text={kk: round(v / max(1, sums["n"]), 5) for kk, v in sums["text"].items()},
            focal_heads=[h for j, h in enumerate(heads) if bool(focal_mask[j])], lr_end=lr, step=step,
            samples=len(samples), new=len(new) if not a.static else None)
        jdump(ep / "train_losses.json", train_losses)
        # ---- save
        torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), uw=uw.state_dict(), opt=opt.state_dict(),
                        freezer=freezer.state(), step=step, epoch=k,
                        torch_rng=torch.get_rng_state()), ep / "model.pt")
        # ---- score (EMA weights)
        t0 = time.time()
        sc, text, P, L = score_set(ema, fcs, fixed_idx["steering"], dev, heads, ordinal, gates)
        nfb = int(supervisor_policy(out).get("no_flag_before_epoch", 0))
        states = M.update_states(history, sc, k, dict(cur, flag_after=10 ** 6) if k < nfb else cur)
        for h, s in sc.items():
            if s.get("score") is not None:
                best[h] = max(best.get(h, -1), s["score"])
        jdump(ep / "report.json", dict(epoch=k, set="steering", n=len(fixed_idx["steering"]), mean_score=M.mean_score(sc),
                                       passed=sum(1 for v in states.values() if v == "passed"),
                                       states=states, scores=sc, text=text, train_losses=train_losses))
        sel, sel_text, _, _ = score_set(ema, fcs, fixed_idx["selection"], dev, heads, ordinal, gates)
        jdump(ep / "selection.json", dict(epoch=k, set="selection", mean_score=M.mean_score(sel), scores=sel, text=sel_text))
        g_idx = fixed_idx["generalisation"]
        gP, gL = predict(ema, [(fcs, i) for i in g_idx], dev)
        strata = np.array([fcs.rows[i]["stratum"] for i in g_idx])
        libs = np.array([fcs.rows[i]["text_lib"] for i in g_idx])
        held = {}
        for name, msk in [("all", np.ones(len(g_idx), bool)), ("holdout_library", strata == "holdout_library"),
                          ("heldout_font", strata == "heldout_font")] + \
                         [(f"library:{l}", libs == l) for l in sorted(set(libs[strata == "holdout_library"]))]:
            if msk.sum():
                s_, t_ = M.head_scores(*subset(gP, gL, msk), heads, ordinal, gates)
                held[name] = dict(n=int(msk.sum()), mean_score=M.mean_score(s_),
                                  passed=sum(1 for v in s_.values() if v.get("passed")), scores=s_, text=t_)
        jdump(ep / "heldout_report.json", dict(epoch=k, set="generalisation", watch_only=True, strata=held))
        T["eval_s"] = round(time.time() - t0, 1)
        # ---- targets from steering only
        if not a.static:
            tg = decide_targets(sc, states, heads, k, cur, supervisor_policy(out))
            jdump(ep / "targets.json", tg)
        # ---- batch k+1: wait for its base half, then render the targeted half while the freeze statistics run
        t0 = time.time()
        if base_proc is not None:
            base_proc.wait()
            base_proc = None
        T["wait_base_s"] = round(time.time() - t0, 1)
        tgt_proc, t_gen = None, time.time()
        if not a.static and k < E and not batch_complete(batch_dir(k + 1), N):
            tgt_proc = run_gen(gen_cmd(a.config, batch_dir(k + 1), k + 1, N, "all", gw, ep / "targets.json"),
                               out / "generation.log", wait=False)
        # ---- freeze map
        t0 = time.time()
        if freeze_on:
            st_idx = fixed_idx["steering"][:512]
            stats = cnn_stats(model, fcs, st_idx, dev, heads, ordinal, freezer, focal_mask)
            torch.save(stats, ep / "cnn_stats.pt")
            stats_hist.append(stats)
            prev1 = stats_hist[-2] if len(stats_hist) >= 2 else None
            prev2 = stats_hist[-3] if len(stats_hist) >= 3 else None
            ev = freezer.update(k, stats, prev1, prev2, states, {h: s.get("score") for h, s in sc.items()}, best,
                                lambda h: M.flagged_streak(history, h))
            jdump(ep / "freeze_map.json", dict(epoch=k, events=ev, **freezer.freeze_map()))
            stats_hist = stats_hist[-3:]
        else:
            jdump(ep / "freeze_map.json", dict(epoch=k, enabled=False, total_frozen=0))
        T["freeze_s"] = round(time.time() - t0, 1)
        if tgt_proc is not None:
            t0 = time.time()
            if tgt_proc.wait() != 0:
                raise RuntimeError(f"generator failed for batch {k + 1}; see {out / 'generation.log'}")
            T["wait_targeted_s"] = round(time.time() - t0, 1)
            T["gen_targeted_s"] = round(time.time() - t_gen, 1)
        if not a.static and k < E:
            ensure_batch(k + 1, ep / "targets.json")  # no-op when complete; retries any unfilled slot
        timings.append(T)
        state = dict(epoch=k, step=step, history=history, best=best, timings=timings)
        jdump(st_path, state)
        log(out, f"epoch {k}: steering mean {M.mean_score(sc)} passed {sum(1 for v in states.values() if v == 'passed')}"
                 f"/{len(states)}  selection mean {M.mean_score(sel)}  generalisation mean {held['all']['mean_score']}  "
                 f"loss {train_losses['groups']}  frozen {freezer.n_frozen()}  times {T}")
        if a.stop_after and k >= a.stop_after:
            log(out, f"stopping after epoch {k} (--stop-after)")
            return
    log(out, "done")


if __name__ == "__main__":
    main()
