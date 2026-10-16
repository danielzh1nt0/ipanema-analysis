# Kaggle (CPU, 17 Oct, F1): what the piece pickles say about team labels on the sun ground - rows per team incl. 'K'
# (neither team: referee / staff / unreadable kit) per piece, and at the by-eye "who has the ball" moments. No GPU.
import os, sys, json, glob, pickle, collections, subprocess
def sh(c): print("+", c, flush=True); r = subprocess.run(c, shell=True, capture_output=True, text=True); print((r.stdout + r.stderr)[-1500:], flush=True)
M = "{{MATCH}}"; W = "/kaggle/working"; T = "/kaggle/temp"; IA = f"{T}/ia"
sh(f"git clone -q --depth 1 https://github.com/danielzh1nt0/ipanema-analysis.git {IA}"); sys.path.insert(0, IA)
import numpy as np
from ipanema import fullmatch as FM
slugs = [x for x in "{{SOURCES_DIRS}}".split(",") if x]; caches = {}
for slug in slugs:
    d = next((x for x in glob.glob(f"/kaggle/input/{slug}") + glob.glob(f"/kaggle/input/*/{slug}") + glob.glob(f"/kaggle/input/*/*/{slug}") if os.path.isdir(x)), None)
    for c in glob.glob(f"{d}/root/cache/{M}_c*") if d else []: caches[os.path.basename(c)] = c
print(sorted(caches), flush=True)
moments = []
for fn in ("ball_key_graded.json", "poss_sample.json"):
    p = f"{IA}/reference/{M}/{fn}"
    if os.path.exists(p): moments += [(it["t"], it["owner"]) for it in json.load(open(p))["items"] if it.get("owner") in ("A", "B")]
out = {"pieces": {}, "moments": []}
for pid in sorted(caches):
    p = pickle.load(open(f"{caches[pid]}/{FM.PIECE_FILE}", "rb")); per = p["per"]; fps = p["fps"]; i = int(pid.rsplit("_c", 1)[1]); off = i * 300
    cnt = collections.Counter(); n = 0
    for k, rows in per.items():
        n += 1
        for r in rows: cnt[r[1]] += 1
    out["pieces"][pid] = {k: round(v / max(1, n), 1) for k, v in cnt.items()}; print(pid, out["pieces"][pid], flush=True)
    raw = pickle.load(open(f"{caches[pid]}/tracks_rfdetr_v1.pkl", "rb"))[0] if os.path.exists(f"{caches[pid]}/tracks_rfdetr_v1.pkl") else None
    for t, o in moments:
        if off <= t < off + 300:
            k = int(round((t - off) * fps)); rows = per.get(k, []); c = collections.Counter(r[1] for r in rows)
            rc = collections.Counter(r[1] for r in raw.get(k, [])) if raw else {}
            out["moments"].append({"t": t, "eye": o, "piece": dict(c), "raw_tracks": dict(rc)}); print(t, o, "piece", dict(c), "raw", dict(rc), flush=True)
json.dump(out, open(f"{W}/teams.json", "w"), indent=1); print("done")
