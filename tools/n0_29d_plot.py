"""N0-29d picture: the runs' 'with the ball' label on Metrica (perfect / flicker ball), app vs spell-state labels.
    python tools/n0_29d_plot.py [results/qa/n0-29d]"""
import sys, json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, numpy as np
d = sys.argv[1] if len(sys.argv) > 1 else "results/qa/n0-29d"; S = json.load(open(f"{d}/metrica_runball.json"))["summary"]
labs = [("raw_start", "app today\n(raw state)"), ("spell_start", "NEW: spell state"), ("spell_back", "spell + 3 s back\n(lab)")]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for i, (key, title) in enumerate((("labelled_pct", "runs that get a label (%)"), ("right_of_labelled_pct", "labels right (%)"), ("share", "share of sprints in possession, off (pts)"))):
    for j, nz in enumerate(("False", "flicker")):
        v = [S[nz]["share_err_pts"][k] if key == "share" else S[nz][k][key] for k, _ in labs]
        b = ax[i].bar(np.arange(3) + (j - 0.5) * 0.38, v, 0.38, label=("perfect ball" if nz == "False" else "flicker ball (like our clips)"), color=("#4a7bd0" if j == 0 else "#e08a3c"))
        ax[i].bar_label(b, fmt="%.0f" if key != "share" else "%.1f", fontsize=9)
    if key == "share": ax[i].axhline(S["False"]["share_err_pts"]["baseline_possession_share"], color="0.4", ls="--", lw=1, label="baseline: team's possession %")
    ax[i].set_xticks(range(3)); ax[i].set_xticklabels([t for _, t in labs], fontsize=9); ax[i].set_title(title, fontsize=11); ax[i].legend(fontsize=8)
fig.suptitle("N0-29d: runs 'with the ball' label on Metrica pro data (4 halves, 1,875 runs; flicker = 2 seeds)", fontsize=12)
fig.tight_layout(); fig.savefig(f"{d}/runball.png", dpi=110); print(f"{d}/runball.png")
