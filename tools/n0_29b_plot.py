"""N0-29b: per player, ours vs Metrica's events (passes tried, progressive passes, forward share, completion %),
perfect ball and flicker noise (seed 0).
    python tools/n0_29b_plot.py [results/qa/n0-29b/metrica_playerpass.json] -> results/qa/n0-29b/players.png"""
import sys, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

f = sys.argv[1] if len(sys.argv) > 1 else "results/qa/n0-29b/metrica_playerpass.json"
R = json.load(open(f))["runs"]
noises = [("False", "perfect ball"), ("flicker", "flicker noise (like our clips)")]
panels = [("passes", "passes tried", False), ("progressive_passes", "progressive passes", False),
          ("passes_forward", "forward share of his passes (%)", True), ("passes_completed", "completion % (his passes)", True)]
fig, ax = plt.subplots(2, 4, figsize=(17, 8))
for j, (nz, lab) in enumerate(noises):
    rs = [r for r in R if r["noise"] == nz and r["seed"] == 0]
    for i, (k, name, share) in enumerate(panels):
        a = ax[j, i]; allv = []
        for r in rs:
            xs, ys = [], []
            for p in r["players"].values():
                t, o = p["truth"], p["ours"]
                if share:
                    if t.get("passes", 0) < 10 or not o["passes"]: continue
                    xs.append(100 * t[k] / t["passes"]); ys.append(100 * o[k] / o["passes"])
                else:
                    xs.append(t.get(k, 0)); ys.append(o[k])
            a.scatter(xs, ys, s=14, alpha=.7, label=f"game {r['game']} half {r['half']}"); allv += xs + ys
        lo = min(allv + [0]) if not share else min(allv); hi = max(allv + [1])
        a.plot([lo, hi], [lo, hi], "k--", lw=.8); a.set_xlabel("truth (Metrica events)"); a.set_ylabel("ours")
        a.set_title(f"{name} - {lab}", fontsize=10)
        if i == 0 and j == 0: a.legend(fontsize=7)
fig.suptitle("N0-29b: per player, one id per player (what M2 gives), Metrica pro data; dashed = right; shares: players with >= 10 passes")
fig.tight_layout(); out = f.rsplit("/", 1)[0] + "/players.png"; fig.savefig(out, dpi=80); print(out)
