"""Read-only sampler: CPU of the fit process tree + GPU util, 1 s cadence. No signals sent."""
import json, subprocess, sys, time, psutil
PID, CT = 137140, 1794632645.2347612
root = psutil.Process(PID); assert abs(root.create_time() - CT) < 1, 'PID reuse'
dur = int(sys.argv[1]); out = []
procs = {}
def tree():
    ps = [root] + root.children(recursive=True)
    for p in ps:
        if p.pid not in procs:procs[p.pid] = p; p.cpu_percent(None)
    return ps
tree(); time.sleep(1)
for t in range(dur):
    ps = tree(); cpu = {}
    for p in ps:
        p = procs[p.pid]
        try:cpu[p.pid] = (p.name(), p.cpu_percent(None), p.memory_info().rss/2**20)
        except psutil.Error:pass
    g = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.strip()
    out.append(dict(t=t, gpu=int(g.split()[0]) if g else None, cpu=cpu, sys=psutil.cpu_percent(None)))
    time.sleep(max(0, 1 - 0.05))
json.dump(out, open(sys.argv[2], 'w'))
# summary
big = lambda s: sorted(s['cpu'].items(), key=lambda kv: -kv[1][1])
rss = {pid: v[2] for pid, v in out[-1]['cpu'].items()}
heavy = max(out[-1]['cpu'].items(), key=lambda kv: kv[1][2])
print('n procs', len(out[-1]['cpu']), 'biggest rss MiB', round(heavy[1][2]), heavy[1][0])
for s in out:
    vals = sorted((v[1] for v in s['cpu'].values()), reverse=True)
    busy = sum(1 for v in vals[1:] if v > 50)
    print(f"t={s['t']:3d} gpu={s['gpu']:3d}% sys={s['sys']:5.1f}% top={vals[0]:6.1f} workers>50%={busy:2d} sum={sum(vals):6.0f}")
