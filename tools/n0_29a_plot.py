"""N0-29a: per player, ours vs the same-rule truth (pressures applied, time under pressure), one panel per ball model.
    python tools/n0_29a_plot.py [results/qa/n0-29a/metrica_playerpress.json] -> results/qa/n0-29a/players.png"""
import sys, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

f = sys.argv[1] if len(sys.argv) > 1 else "results/qa/n0-29a/metrica_playerpress.json"
R = json.load(open(f))["runs"]
noises = [("False", "perfect ball"), ("streak", "streak noise"), ("flicker", "flicker noise (like our clips)")]
fig, ax = plt.subplots(2, 3, figsize=(13, 8))
for j, (nz, lab) in enumerate(noises):
    rs = [r for r in R if r["noise"] == nz and r["seed"] == 0]
    for i, (o, t, name, unit) in enumerate((("ours_press", "truth_press", "pressures applied", ""), ("ours_under_s", "truth_under_s", "time under pressure", " (s)"))):
        a = ax[i, j]
        for r in rs:
            xs = [p[t] for p in r["players"].values()]; ys = [p[o] for p in r["players"].values()]
            a.scatter(xs, ys, s=14, alpha=.7, label=f"game {r['game']} half {r['half']}")
        m = max([p[t] for r in rs for p in r["players"].values()] + [p[o] for r in rs for p in r["players"].values()] + [1])
        a.plot([0, m], [0, m], "k--", lw=.8); a.set_xlabel(f"truth (same rule, perfect inputs; carrier = ball within 2.5 m){unit}"); a.set_ylabel(f"ours{unit}")
        a.set_title(f"{name} - {lab}", fontsize=10)
        if i == 0 and j == 0: a.legend(fontsize=7)
fig.suptitle("N0-29a: per player, one id per player (what M2 gives), Metrica pro data; dashed = right")
fig.tight_layout(); out = f.rsplit("/", 1)[0] + "/players.png"; fig.savefig(out, dpi=90); print(out)
