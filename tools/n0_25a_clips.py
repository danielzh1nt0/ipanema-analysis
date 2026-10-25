"""N0-25a: the app's runs (metrics.runs) on the two app clips' saved inputs, old vs new rule -> results/qa/n0-25a/clips.json
and track pictures (tracked feet grey, each new run red, old runs blue) for the by-eye check.
    PYTHONPATH=. python tools/n0_25a_clips.py <picker_inputs.pkl>... [--out dir]"""
import sys, os, json, pickle, collections, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import metrics as M

AR = {"A": True, "B": False}


def clip_runs(path, **kw):
    d = pickle.load(open(path, "rb")); per = {k: d["per"][k] for k in range(len(d["per"]))}; fps = d["fps"]
    return per, fps, M.runs(per, fps, np.full(len(per), 2), AR, **kw)


def summary(rs, minutes):
    if not rs: return {"n": 0, "per_10min": 0.0}
    return {"n": len(rs), "per_10min": round(10 * len(rs) / minutes, 1), "players": len({r["player"] for r in rs}),
            "median_m": float(np.median([r["distance_m"] for r in rs])), "median_ms": float(np.median([r["speed_ms"] for r in rs])),
            "median_s": round(float(np.median([r["t_end"] - r["t"] for r in rs])), 2), "max_ms": float(max(r["speed_ms"] for r in rs))}


def picture(per, fps, new, old, path, n=12):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    tr = collections.defaultdict(dict)
    for k in per:
        for r in per[k]: tr[r[0]][k] = np.asarray(r[2], float)
    top = [r for r in sorted(new, key=lambda r: -r["distance_m"])][:n]
    fig, ax = plt.subplots(3, 4, figsize=(16, 11)); ax = ax.ravel()
    for i, r in enumerate(top):
        a, b = int(r["t"] * fps), int(r["t_end"] * fps); pts = tr[r["player"]]
        ks = [k for k in sorted(pts) if a - 2 * fps <= k <= b + 2 * fps]; P = np.array([pts[k] for k in ks])
        ax[i].plot(P[:, 0], P[:, 1], ".", color="0.6", ms=3)
        S = np.array([pts[k] for k in ks if a <= k <= b]); ax[i].plot(S[:, 0], S[:, 1], "-", color="red", lw=2)
        ax[i].plot(*r["from_m"], "go"); ax[i].plot(*r["to_m"], "rs")
        for o in old:
            if o["player"] == r["player"] and o["t"] <= r["t_end"] and r["t"] <= o["t_end"]:
                ax[i].plot([o["from_m"][0], o["to_m"][0]], [o["from_m"][1], o["to_m"][1]], "b-", lw=3)
        ax[i].set_title(f"#{r['player']} {r['team']} t={r['t']:.1f}-{r['t_end']:.1f}s {r['distance_m']} m {r['speed_ms']} m/s", fontsize=9)
        ax[i].set_aspect("equal"); ax[i].grid(alpha=.3)
    for j in range(len(top), len(ax)): ax[j].axis("off")
    fig.suptitle("grey = tracked feet (+-2 s), red = new run, green dot start, red square end, blue = old run (if any)")
    fig.tight_layout(); fig.savefig(path, dpi=70); plt.close(fig)


if __name__ == "__main__":
    a = sys.argv[1:]; out = "results/qa/n0-25a"
    if "--out" in a: i = a.index("--out"); out = a[i + 1]; del a[i:i + 2]
    res = {}
    for p in a:
        name = os.path.basename(os.path.dirname(p)); per, fps, new = clip_runs(p, smooth=True); _, _, old = clip_runs(p, smooth=False)
        minutes = len(per) / fps / 60; res[name] = {"minutes": round(minutes, 1), "old": summary(old, minutes), "new": summary(new, minutes)}
        print(name, json.dumps(res[name]))
        picture(per, fps, new, old, os.path.join(out, f"runs_{name.split('_')[0][:12]}.png"))
    json.dump(res, open(os.path.join(out, "clips.json"), "w"), indent=1)
