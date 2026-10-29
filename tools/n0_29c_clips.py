"""N0-29c: per-player top speed on the real app clips (exact picker inputs), old (fastest row) vs new (best 1-s average
while seen), plus a picture of a few tracks' speed with both marked.
Usage: python tools/n0_29c_clips.py <picker_inputs.pkl>... --out results/qa/n0-29c"""
import sys, os, json, pickle, collections, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import analytics as AN


def tracks(path):
    d = pickle.load(open(path, "rb")); per, fps = d["per"], d["fps"]; tr = collections.defaultdict(dict)
    for k in range(len(per)):
        for r in per[k]: tr[r[0]][k] = np.asarray(r[2], float)
    out = {}
    for t, v in tr.items():
        ks = np.array(sorted(v))
        if len(ks) / fps >= 2.0: out[t] = (ks, np.array([v[k] for k in ks]))
    return out, fps


def main(paths, outdir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    res = {}
    for p in paths:
        name = os.path.basename(os.path.dirname(p)); T, fps = tracks(p); rows = []
        for t, (ks, P) in T.items():
            rows.append({"id": int(t), "seen_s": round(len(ks) / fps, 1), "old": AN.top_speed(ks, P, fps, old=True), "new": AN.top_speed(ks, P, fps, old=False)})
        o = np.array([r["old"] for r in rows]); n = np.array([r["new"] for r in rows]); big = np.array([r["seen_s"] >= 20 for r in rows])
        res[name] = {"tracks": len(rows), "old_median": round(float(np.median(o)), 2), "new_median": round(float(np.median(n)), 2),
                     "old_over_8_pct": round(100 * float(np.mean(o > 8))), "new_over_8_pct": round(100 * float(np.mean(n > 8))),
                     "new_zero_pct": round(100 * float(np.mean(n == 0))),
                     "tracks_seen_20s": int(big.sum()), "new_median_seen_20s": round(float(np.median(n[big])), 2) if big.any() else None,
                     "old_median_seen_20s": round(float(np.median(o[big])), 2) if big.any() else None, "rows": rows}
        print(name, json.dumps({k: v for k, v in res[name].items() if k != "rows"}))
        long_ = sorted(T, key=lambda t: -len(T[t][0]))[:4]
        fig, axs = plt.subplots(4, 1, figsize=(11, 10), sharey=True)
        for ax, t in zip(axs, long_):
            ks, P = T[t]; sp, ok = AN.seen_speeds(ks, P, fps); sp_old = AN.track_motion(ks, P, fps)[0]
            ax.plot(ks / fps, sp_old, ".", ms=2, color="tab:red", label="old row speed")
            ax.plot(ks / fps, sp, ".", ms=2, color="tab:blue", label="seen speed (jump-split)")
            ax.axhline(AN.top_speed(ks, P, fps, old=True), color="tab:red", ls="--", lw=1, label=f"old top {AN.top_speed(ks, P, fps, old=True)}")
            ax.axhline(AN.top_speed(ks, P, fps, old=False), color="tab:blue", ls="--", lw=1, label=f"new top {AN.top_speed(ks, P, fps, old=False)}")
            ax.set_title(f"{name} track {t} (seen {len(ks) / fps:.0f} s)", fontsize=9); ax.set_ylabel("m/s"); ax.legend(fontsize=7, loc="upper right"); ax.set_ylim(0, 10.5)
        axs[-1].set_xlabel("s"); fig.tight_layout(); fig.savefig(os.path.join(outdir, f"speed_{name}.png"), dpi=90); plt.close(fig)
    json.dump(res, open(os.path.join(outdir, "app_clips.json"), "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]; i = a.index("--out"); out = a[i + 1]; del a[i:i + 2]; os.makedirs(out, exist_ok=True); main(a, out)
