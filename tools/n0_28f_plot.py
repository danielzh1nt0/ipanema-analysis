"""N0-28f picture: Metrica game 2, first half - each outfield player's true average spot vs what our camera view gives
(seen only) and with the 10-s hold. Both teams drawn attacking right. -> results/qa/n0-28f/avgpos_g2h1.png
    python tools/n0_28f_plot.py"""
import json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

d = json.load(open("results/qa/n0-28f/metrica_playerpos.json"))["example_game2_half1_SFKBP_view"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
for ax, team, lo, hi in ((axes[0], "home (Metrica team A)", 1100, 1200), (axes[1], "away (Metrica team B)", 1200, 1300)):
    ax.add_patch(plt.Rectangle((0, 0), 105, 68, fc="#e8f3e8", ec="#577857")); ax.axvline(35, c="#9bb59b", lw=.8); ax.axvline(70, c="#9bb59b", lw=.8); ax.axvline(52.5, c="#577857", lw=.8)
    for pid, t in d["truth"].items():
        if not lo <= int(pid) < hi: continue
        s, h = d["seen"].get(pid), d["hold10"].get(pid)
        ax.plot(*t, "o", c="black", ms=9); ax.annotate(str(int(pid) % 100), t, xytext=(5, 5), textcoords="offset points", fontsize=8)
        if s: ax.plot([t[0], s[0]], [t[1], s[1]], c="#d1495b", lw=1); ax.plot(*s, "x", c="#d1495b", ms=8, mew=2)
        if h: ax.plot([t[0], h[0]], [t[1], h[1]], c="#2e86ab", lw=1); ax.plot(*h, "+", c="#2e86ab", ms=10, mew=2)
    ax.set_xlim(-2, 107); ax.set_ylim(-2, 70); ax.set_aspect("equal"); ax.set_title(f"{team}, attacking right"); ax.set_xticks([]); ax.set_yticks([])
axes[0].plot([], [], "o", c="black", label="true average spot"); axes[0].plot([], [], "x", c="#d1495b", label="seen only (today's rule)"); axes[0].plot([], [], "+", c="#2e86ab", label="last spot held 10 s")
axes[0].legend(loc="lower left", fontsize=8)
fig.suptitle("Metrica game 2, 1st half, seen through the SFK-BP clip's camera views: average positions (keepers out)")
fig.tight_layout(); fig.savefig("results/qa/n0-28f/avgpos_g2h1.png", dpi=110)
