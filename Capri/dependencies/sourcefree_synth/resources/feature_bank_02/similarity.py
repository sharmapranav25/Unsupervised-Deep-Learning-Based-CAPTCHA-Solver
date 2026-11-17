"""Pairwise code similarity (5-token shingles, Jaccard) to flag forks/derivatives."""
import glob, itertools, json, os, re

def shingles(name):
    toks = []
    for p in glob.glob(os.path.join("staging", name, "**", "*.php"), recursive=True):
        n = p.replace(os.sep, "/").lower()
        if "/vendor/" in n or "/tests/" in n:
            continue
        toks += re.findall(r"[A-Za-z_]\w+", open(p, errors="ignore").read())
    return {tuple(toks[i:i + 5]) for i in range(len(toks) - 4)}

names = sorted(os.listdir("staging"))
sh = {n: shingles(n) for n in names}
pairs = []
for a, b in itertools.combinations(names, 2):
    if sh[a] and sh[b]:
        inter = len(sh[a] & sh[b])
        cont = inter / min(len(sh[a]), len(sh[b]))  # containment of smaller in larger
        if cont > 0.25:
            pairs.append((round(cont, 2), a, b))
for p in sorted(pairs, reverse=True):
    print(*p)
json.dump(pairs, open("similarity.json", "w"), indent=1)
