"""Live status page for a hybrid v3 training run (read-only; never touches the run).

  python -m feature_bank.train.monitor --run runs/r1_main_20261002 [--port 8765]   -> open http://localhost:8765
  python -m feature_bank.train.monitor --run runs/r1_main_20261002 --once          -> text status in the terminal

Reads state.json, train.log, generation.log and epoch_k/*.json; the page refreshes itself every 15 s.
"""
import argparse
import html
import json
import re
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def jload(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - file being written / missing
        return None


def tail(p, n):
    try:
        return Path(p).read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except Exception:  # noqa: BLE001
        return []


def gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip().split(", ")
        return dict(util=int(out[0]), mem=int(out[1]), total=int(out[2]))
    except Exception:  # noqa: BLE001
        return None


def collect(run):
    run = Path(run)
    rc = jload(run / "run_config.json") or {}
    st = jload(run / "state.json") or {}
    E = rc.get("epochs", 15)
    done = st.get("epoch", 0)
    log = tail(run / "train.log", 400)
    last = log[-1] if log else ""
    age = time.time() - (run / "train.log").stat().st_mtime if (run / "train.log").exists() else None
    out_log = tail(Path(str(run) + ".log"), 40)
    crashed = any("Traceback" in l for l in out_log)
    finished = last.endswith("done")
    # current phase of epoch done+1
    k = done + 1
    ep = run / f"epoch_{k:02d}"
    gen = [l for l in tail(run / "generation.log", 60) if re.match(r"^\d+/\d+ ok", l)]
    if finished:
        phase = "finished"
    elif crashed:
        phase = "CRASHED (see " + run.name + ".log)"
    elif (ep / "targets.json").exists():
        phase = f"epoch {k}: freeze statistics / rendering batch {k + 1}"
    elif (ep / "report.json").exists():
        phase = f"epoch {k}: scoring selection / generalisation"
    elif (ep / "model.pt").exists():
        phase = f"epoch {k}: scoring steering"
    elif ep.exists():
        phase = f"epoch {k}: training"
    else:
        phase = f"epoch {k}: waiting for batch {k}"
    epochs = []
    for j in range(1, E + 1):
        d = run / f"epoch_{j:02d}"
        r, s, h, f, tl = (jload(d / x) for x in ("report.json", "selection.json", "heldout_report.json", "freeze_map.json", "train_losses.json"))
        if not r:
            continue
        T = next((t for t in st.get("timings", []) if t.get("epoch") == j), {})
        hs = (h or {}).get("strata", {})
        epochs.append(dict(epoch=j, steer=r.get("mean_score"), passed=r.get("passed"), n_heads=len(r.get("states", {})),
                           sel=(s or {}).get("mean_score"), gen=hs.get("all", {}).get("mean_score"),
                           gen_lib=hs.get("holdout_library", {}).get("mean_score"), gen_font=hs.get("heldout_font", {}).get("mean_score"),
                           loss=(tl or {}).get("groups", {}), frozen=(f or {}).get("total_frozen"),
                           minutes=round(sum(v for kk, v in T.items() if kk.endswith("_s") and kk != "gen_targeted_s") / 60, 1) if T else None))
    eta = None
    mins = [e["minutes"] for e in epochs if e["minutes"]]
    if mins and not finished:
        eta = round(sum(mins) / len(mins) * (E - done))
    latest = jload(run / f"epoch_{done:02d}" / "report.json") if done else None
    held = jload(run / f"epoch_{done:02d}" / "heldout_report.json") if done else None
    tg = jload(run / f"epoch_{done:02d}" / "targets.json") if done else None
    start = re.match(r"\[(\d\d:\d\d:\d\d)\]", log[0]).group(1) if log and re.match(r"\[(\d\d:\d\d:\d\d)\]", log[0]) else None
    return dict(run=run.name, epochs_total=E, epochs_done=done, phase=phase, started=start, log_age_s=None if age is None else round(age),
                eta_min=eta, gpu=gpu(), gen_progress=gen[-1] if gen else None, epochs=epochs, latest=latest, held=held, targets=tg,
                log=log[-12:], crashed=crashed, finished=finished, freeze=rc.get("freeze"))


def fmt(v, d=3):
    return "–" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))


def page(s):
    e = html.escape
    status_cls = "bad" if s["crashed"] else ("ok" if s["finished"] else "run")
    g = s["gpu"]
    rows = "".join(
        f"<tr><td>{x['epoch']}</td><td>{fmt(x['steer'])}</td><td>{x['passed']}/{x['n_heads']}</td><td>{fmt(x['sel'])}</td>"
        f"<td>{fmt(x['gen'])}</td><td>{fmt(x['gen_lib'])}</td><td>{fmt(x['gen_font'])}</td>"
        f"<td>{fmt(x['loss'].get('presence'))}</td><td>{fmt(x['loss'].get('ordinal'))}</td><td>{fmt(x['loss'].get('text'), 2)}</td>"
        f"<td>{fmt(x['frozen'])}</td><td>{fmt(x['minutes'], 1)}</td></tr>" for x in s["epochs"])
    heads = ""
    if s["latest"]:
        gsc = ((s["held"] or {}).get("strata", {}).get("all", {}) or {}).get("scores", {})
        order = {"passed": 0, "improving": 1, "stalled": 2, "flagged": 3, "no_data": 4}
        items = sorted(s["latest"]["scores"].items(), key=lambda kv: (order.get(s["latest"]["states"].get(kv[0]), 9), -(kv[1].get("score") or 0)))
        heads = "".join(
            f"<tr><td>{e(h)}</td><td class='st-{s['latest']['states'].get(h)}'>{s['latest']['states'].get(h)}</td>"
            f"<td>{fmt(v.get('score'))}</td><td>{fmt(v.get('gap'))}</td><td>{fmt((gsc.get(h) or {}).get('score'))}</td>"
            f"<td>{fmt(((s['targets'] or {}).get('heads') or {}).get(h))}</td></tr>" for h, v in items)
    logs = e("\n".join(s["log"]))
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="15">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{e(s['run'])} status</title>
<style>
:root{{--bg:#fff;--fg:#1d1d1f;--mut:#6e6e73;--line:#e5e5ea;--ok:#1a7f37;--bad:#c62828;--run:#0b5cad;--card:#f6f6f8}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151517;--fg:#ececf0;--mut:#9a9aa2;--line:#2c2c31;--ok:#4cc26b;--bad:#ff6b6b;--run:#5aa6ff;--card:#1e1e22}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;margin:0;padding:16px;max-width:1200px}}
h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:15px;margin:22px 0 8px}} .mut{{color:var(--mut)}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin-top:12px}}
.card{{background:var(--card);border-radius:8px;padding:10px 12px}} .card b{{display:block;font-size:18px}}
.ok{{color:var(--ok)}} .bad{{color:var(--bad)}} .run{{color:var(--run)}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}} th,td{{text-align:left;padding:4px 8px;border-bottom:1px solid var(--line)}}
th{{color:var(--mut);font-weight:500}} .wrap{{overflow-x:auto}}
.st-passed{{color:var(--ok)}} .st-flagged{{color:var(--bad)}} .st-no_data{{color:var(--mut)}}
pre{{background:var(--card);padding:10px;border-radius:8px;overflow-x:auto;font-size:12px;white-space:pre-wrap}}
</style></head><body>
<h1>{e(s['run'])}</h1><div class="mut">refreshes every 15 s · freezing {'on' if s['freeze'] else 'off'} · started {fmt(s['started'])}</div>
<div class="cards">
<div class="card"><span class="mut">Status</span><b class="{status_cls}">{e(s['phase'])}</b></div>
<div class="card"><span class="mut">Epochs done</span><b>{s['epochs_done']} / {s['epochs_total']}</b></div>
<div class="card"><span class="mut">ETA</span><b>{'–' if s['eta_min'] is None else f"~{s['eta_min']} min"}</b></div>
<div class="card"><span class="mut">GPU</span><b>{'–' if not g else f"{g['util']}% · {g['mem']/1024:.1f}/{g['total']/1024:.0f} GB"}</b></div>
<div class="card"><span class="mut">Last log write</span><b>{fmt(s['log_age_s'])} s ago</b></div>
<div class="card"><span class="mut">Generator</span><b style="font-size:13px">{e(s['gen_progress'] or '–')}</b></div>
</div>
<h2>Per epoch <span class="mut">(mean head score; generalisation is watch-only)</span></h2>
<div class="wrap"><table><tr><th>Epoch</th><th>Steering</th><th>Passed</th><th>Selection</th><th>Generalisation</th><th>held-out libs</th>
<th>held-out fonts</th><th>presence loss</th><th>ordinal loss</th><th>text loss</th><th>Frozen ch.</th><th>Minutes</th></tr>{rows}</table></div>
<h2>Heads after the latest epoch <span class="mut">(state and score on steering; share of next batch's targeted half)</span></h2>
<div class="wrap"><table><tr><th>Head</th><th>State</th><th>Steering score</th><th>Gap to gate</th><th>Generalisation score</th><th>Target share</th></tr>{heads}</table></div>
<h2>Log</h2><pre>{logs}</pre></body></html>"""


def detail(run):
    """Text: global loss, per-head training losses and steering scores across epochs, freezing."""
    import math
    run = Path(run)
    s = collect(run)
    print(f"{s['run']}: {s['phase']}  epochs {s['epochs_done']}/{s['epochs_total']}  ETA {s['eta_min']} min")
    eps = sorted(int(d.name[6:]) for d in run.glob("epoch_*") if (d / "train_losses.json").exists())
    tls = {k: jload(run / f"epoch_{k:02d}" / "train_losses.json") for k in eps}
    reps = {k: jload(run / f"epoch_{k:02d}" / "report.json") for k in eps}
    fms = {k: jload(run / f"epoch_{k:02d}" / "freeze_map.json") for k in eps}
    print("\nGLOBAL (uncertainty-weighted: sum exp(-s_g) L_g + s_g; epoch means)")
    for k in eps:
        t = tls[k]
        g, lv = t["groups"], t["uncertainty_log_var"]
        tot = sum(math.exp(-lv[x]) * g[x] + lv[x] for x in g)
        r = reps.get(k) or {}
        print(f"  e{k:02d}  global {tot:7.4f}  presence {g['presence']:.4f} ordinal {g['ordinal']:.4f} text {g['text']:.3f} "
              f"coverage {g['coverage']:.4f}  | weights exp(-s) " + " ".join(f"{x[:4]} {math.exp(-lv[x]):.3f}" for x in g)
              + f"  | steering mean {r.get('mean_score')} passed {r.get('passed')}")
    last = eps[-1] if eps else None
    if last:
        print("\nTEXT parts (train):  " + "  ".join(f"e{k:02d} " + " ".join(f"{n.replace('slot_', 's.')}={v:.3f}" for n, v in tls[k]["text"].items()) for k in eps[-3:]))
        print("\nPER HEAD  presence loss by epoch | ordinal loss by epoch | steering score (state) latest")
        r = reps[last]
        for h, v in tls[last]["presence"].items():
            pl = " ".join(f"{tls[k]['presence'].get(h, float('nan')):.3f}" for k in eps)
            ol = " ".join(f"{tls[k]['ordinal'][h]:.3f}" for k in eps) if h in tls[last]["ordinal"] else "-"
            sc = (r["scores"].get(h) or {}).get("score")
            print(f"  {h:24s} {pl} | {ol} | {sc} ({r['states'].get(h)})")
        for h in ("length", "composition", "slots"):
            print(f"  {h:24s} score {(r['scores'].get(h) or {}).get('score')} ({r['states'].get(h)})")
        print("\nFREEZING")
        for k in eps:
            f = fms[k] or {}
            ev = f.get("events") or []
            kinds = {}
            for e in ev:
                kinds[e["event"]] = kinds.get(e["event"], 0) + 1
            print(f"  e{k:02d}  frozen {f.get('total_frozen')}/{f.get('total_channels', '-')}  per unit {f.get('counts')}  events {kinds or '-'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--detail", action="store_true", help="per-head losses, global loss, freezing across epochs (text)")
    a = ap.parse_args()
    if a.detail:
        detail(a.run)
        return
    if a.once:
        s = collect(a.run)
        print(f"{s['run']}: {s['phase']}  epochs {s['epochs_done']}/{s['epochs_total']}  ETA {s['eta_min']} min  GPU {s['gpu']}")
        for x in s["epochs"]:
            print(f"  epoch {x['epoch']:2d}: steering {fmt(x['steer'])} passed {x['passed']}/{x['n_heads']} selection {fmt(x['sel'])} "
                  f"generalisation {fmt(x['gen'])} frozen {fmt(x['frozen'])} {fmt(x['minutes'], 1)} min")
        print("\n".join(s["log"][-3:]))
        return

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = page(collect(a.run)).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    print(f"status page: http://localhost:{a.port}  (run {a.run})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
