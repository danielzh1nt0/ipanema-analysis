"""N0-25d picture: the app's 5-min possession curve (ours, perfect ball and flicker noise) vs the truth, game 1 both halves.
    PYTHONPATH=. python tools/n0_25d_plot.py /tmp/metrica/data -> results/qa/n0-25d/metrica_possession_curve.png"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, n0_25d_tilt_windows as T


def main(d, out="results/qa/n0-25d/metrica_possession_curve.png"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    keep = {}; T.main(d, games=("Sample_Game_1",), halves=(1, 2), noises=(False, "flicker"), seeds=(0,), keep=keep)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharey=True)
    for ax, per in zip(axes, (1, 2)):
        ow0, tw = keep[("1", per, "False", 0)]; owf, _ = keep[("1", per, "flicker", 0)]
        t = np.array([w["t"] for w in tw]) / 60
        ax.plot(t, 100 * T.rolling(tw, "possession_A", weighted=True), "k", lw=2.2, label="truth (Metrica events)")
        ax.plot(t[:len(ow0)], 100 * T.rolling(ow0, "possession_A"), color="tab:blue", label="ours, perfect ball (app's 5-min mean)")
        ax.plot(t[:len(owf)], 100 * T.rolling(owf, "possession_A"), color="tab:red", label="ours, flicker ball like our clips")
        ax.plot(t[:len(owf)], 100 * T.rolling(owf, "possession_A", weighted=True), color="tab:red", ls=":", label="flicker, weighted by control (lab)")
        ax.axhline(50, color="grey", lw=0.6); ax.set_ylabel(f"half {per}: home %"); ax.set_ylim(15, 85)
    axes[0].legend(fontsize=8, loc="upper right"); axes[1].set_xlabel("minutes into the half")
    fig.suptitle("N0-25d: 'possession over time' (5-min rolling of 15-s windows), Metrica game 1"); fig.tight_layout(); fig.savefig(out, dpi=110)


if __name__ == "__main__":
    main(sys.argv[1])
