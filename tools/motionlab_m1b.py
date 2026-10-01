"""1 Oct (M1b): does the speed layer read faster where the calibration is coarse (far touchline, zoomed-out camera)?
Old = 1 s moving average; new = noise-aware Kalman smoother (ipanema/motion.py, H given). Free, local, on the exact
app inputs of SFK-BP and AIK. Graded on: speed by metres-per-pixel band (should be flat), the by-eye moments
(results/review/speedcheck_answers.json, SFK-BP only) and metres per player-minute."""
import sys, os, json, pickle, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import motion as MO
OK = {"stand": (0, 4), "walk": (2, 8), "jog": (6, 16), "run": (12, 30)}
BANDS = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.45), (0.45, 9)]


def grade(name, per, fps, H, J, mo):
    rows = {}
    for k, f in mo.items():
        for t, v in f.items():
            j = J.get((k, t))
            if v[0] is None or j is None: continue
            d = float(np.linalg.norm(j[:, 1]))                                        # metres per pixel up/down the image
            for lo, hi in BANDS:
                if lo <= d < hi: rows.setdefault((lo, hi), []).append(v[0])
    bands = {f"{lo}-{hi} m/px": {"n": len(rows.get((lo, hi), [])), "p10": round(float(np.percentile(rows[(lo, hi)], 10)), 1), "p50": round(float(np.median(rows[(lo, hi)])), 1), "p95": round(float(np.percentile(rows[(lo, hi)], 95)), 1)} for lo, hi in BANDS if len(rows.get((lo, hi), [])) > 50}
    best, span = {}, {}
    for k, f in mo.items():
        for t, v in f.items():
            best[t] = max(best.get(t, 0), v[1]); a, b = span.get(t, (k, k)); span[t] = (min(a, k), max(b, k))
    mins = sum((b - a) / fps for a, b in span.values()) / 60
    sp = [v[0] for f in mo.values() for v in f.values() if v[0] is not None]
    res = {"bands": bands, "m_per_player_min": round(sum(best.values()) / mins, 1), "kmh_p50_p95": [round(float(x), 1) for x in np.percentile(sp, [50, 95])], "hidden_pct": round(100 * (1 - len(sp) / sum(len(f) for f in mo.values())), 1)}
    if name == "SFKBP1109_s1200":
        G = json.load(open("results/review/speedcheck_answers.json"))["graded"]; M = {str(m["n"]): m for m in json.load(open("results/review/speedcheck_moments.json"))}
        ok = 0; det = []
        for n, g in sorted(G.items(), key=lambda z: int(z[0])):
            if g == "bad": continue
            m = M[n]; v = mo.get(m["frame"], {}).get(m["id"]); kmh = v[0] if v else None
            hit = kmh is None or OK[g][0] <= kmh <= OK[g][1]                       # hidden counts as fine (as in M1 round 1)
            ok += hit; det.append([int(n), g, kmh, bool(hit)])
        res["by_eye_ok"] = f"{ok}/{len(det)}"; res["by_eye"] = det
        res["by_eye_kmh_outside"] = round(sum(max(0, OK[g][0] - k, k - OK[g][1]) for _, g, k, _ in det if k is not None), 1)
    return res


if __name__ == "__main__":
    out = {}
    for name in ("SFKBP1109_s1200", "p15u-vs-aik-2026-09-21-bd09_s2520"):
        P = pickle.load(open(f"results/volume/cache/{name}/picker_inputs.pkl", "rb")); per, fps, H = P["per"], P["fps"], P["H"]
        J = MO.jacobians(per, H); out[name] = {}
        for label, kw in (("old (1 s average)", {}), ("new (Kalman, H)", {"H": H})):
            t = time.time(); mo = MO.compute(per, fps, **kw); r = grade(name, per, fps, H, J, mo); r["sec"] = round(time.time() - t, 1)
            out[name][label] = r; print(name, label, json.dumps({k: v for k, v in r.items() if k != "by_eye"}), flush=True)
    os.makedirs("results/review/m1b", exist_ok=True); json.dump(out, open("results/review/m1b/motionlab_m1b.json", "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (name, d) in zip(axs, out.items()):
        for label, c in (("old (1 s average)", "#888"), ("new (Kalman, H)", "#1f6feb")):
            b = d[label]["bands"]; ax.plot(range(len(b)), [v["p50"] for v in b.values()], "o-", color=c, label=label)
            ax.set_xticks(range(len(b))); ax.set_xticklabels([k.replace(" m/px", "") for k in b], fontsize=8)
        ax.set_title(name.split("_s")[0], fontsize=10); ax.set_xlabel("metres per pixel (near camera -> far / zoomed out)", fontsize=8); ax.grid(alpha=.3)
    axs[0].set_ylabel("median speed, km/h"); axs[0].legend(fontsize=8); fig.tight_layout(); fig.savefig("results/review/m1b/speed_by_scale.png", dpi=110)
