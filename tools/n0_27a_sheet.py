"""N0-27a: picture sheet - 12 random completed pro passes (Metrica, perfect inputs) that the rule calls 'a better pass
was open': passer (black ring), the pass played (blue arrow), the 'better' option (red dashed arrow), opponents (grey).

    PYTHONPATH=. python tools/n0_27a_sheet.py /tmp/metrica/src/data results/qa/n0-27a/sheet_metrica.png
"""
import sys, os, random, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import n0_27a_better_option as B
from n0_23a_transition import home_attacks_right


def main(d, out, g="Sample_Game_1", per=1, n=12, seed=7):
    import metricalab as ML
    gd = f"{d}/{g}"; home = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Home_Team.csv", 100); away = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Away_Team.csv", 200)
    fr = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home}
    hr = home_attacks_right(fr, per); T = [t for t in B.truth_rows(B.load_passes(f"{gd}/{g}_RawEventsData.csv"), fr, per, hr) if t["better_option"]]
    pick = random.Random(seed).sample(T, min(n, len(T))); fig, axs = plt.subplots(3, 4, figsize=(20, 10.5))
    for ax, t in zip(axs.flat, pick):
        _, ps, b = fr[t["f0"]]; pos = {pid: (x, y) for pid, x, y in ps}; A = t["team"] == "A"
        ax.add_patch(plt.Rectangle((0, 0), 105, 68, fill=False)); ax.plot([52.5, 52.5], [0, 68], "k-", lw=0.5)
        for pid, (x, y) in pos.items():
            mate = (pid < 200) == A; ax.scatter(x, y, s=40, c="tab:blue" if mate else "0.6", zorder=3)
        c = pos[t["from"]]; ax.scatter(*c, s=160, facecolors="none", edgecolors="k", lw=2, zorder=4)
        r = pos[t["to"]]; ax.annotate("", r, c, arrowprops=dict(arrowstyle="->", color="tab:blue", lw=2))
        bb = pos[t["best_to"]]; ax.annotate("", bb, c, arrowprops=dict(arrowstyle="->", color="red", lw=2, ls="--"))
        ax.set_title(f"f{t['f0']} played {t['played_value']} vs better {t['best_value']} (beats {t['best_bypassed']}) - attacks {'->' if A == hr else '<-'}", fontsize=9)
        ax.set_xlim(-2, 107); ax.set_ylim(-2, 70); ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(); fig.savefig(out, dpi=70); print(out, len(T))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
