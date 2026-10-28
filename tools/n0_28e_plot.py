"""N0-28e picture: Metrica truth pass network vs ours (perfect ball, flicker ball) for one team-half, drawn the usual way
(node = mean pass / reception spot, line width = passes, edges with >= min_n passes).

    PYTHONPATH=. python tools/n0_28e_plot.py <metrica data dir> [game 2] [half 1] [team A] [out png]
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from ipanema import metrics as MT
import n0_28e_passnet as NP

L, W = 105.0, 68.0


def draw(ax, net, title, min_n=3, ref_nodes=None):
    e, n = net
    ax.add_patch(plt.Rectangle((0, 0), L, W, fill=False, lw=1, color="0.5")); ax.plot([L / 2, L / 2], [0, W], color="0.7", lw=0.8)
    mx = max(e.values()) if e else 1
    for (i, j), c in sorted(e.items(), key=lambda kv: kv[1]):
        if c < min_n or i not in n or j not in n: continue
        ax.plot([n[i][0], n[j][0]], [n[i][1], n[j][1]], color="tab:blue", alpha=0.25 + 0.6 * c / mx, lw=0.5 + 5 * c / mx)
    for i, p in n.items():
        if ref_nodes is not None and i in ref_nodes: ax.plot([ref_nodes[i][0], p[0]], [ref_nodes[i][1], p[1]], color="tab:red", lw=0.8)
        ax.scatter([p[0]], [p[1]], s=90, color="white", edgecolor="k", zorder=3); ax.text(p[0], p[1], str(i % 100), ha="center", va="center", fontsize=7, zorder=4)
    ax.set_xlim(-3, L + 3); ax.set_ylim(W + 3, -3); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=9)


def main(d, game="2", half=1, team="A", out="results/qa/n0-28e/passnet_g2h1_A.png"):
    import metricalab as ML
    g = f"Sample_Game_{game}"; gd = f"{d}/{g}"
    home = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Home_Team.csv", 100); away = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Away_Team.csv", 200)
    fr = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home}
    T = NP.network(NP.truth_passes(f"{gd}/{g}_RawEventsData.csv", half))[team]
    nets = [("Truth (Metrica labels)", T)]
    for nz in (False, "flicker"):
        ps, _ = NP.capture(fr, half, nz, 0); O = NP.from_export(MT.pass_network(ps, NP.players_stub(ps)))[team]
        c = NP.compare(T, O); nets.append((f"Ours, {'perfect' if not nz else 'flicker'} ball: overlap {c['weighted_overlap']}, top-10 {c['top10_overlap']}/10, node off {c['node_err_median_m']} m", O))
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.6))
    for k, (ax, (t, net)) in enumerate(zip(axs, nets)): draw(ax, net, t, ref_nodes=None if k == 0 else T[1])
    fig.suptitle(f"Metrica game {game}, half {half}, team {team}: pass network (edges with >= 3 passes; red tick = node offset from truth)", fontsize=10)
    os.makedirs(os.path.dirname(out), exist_ok=True); fig.tight_layout(); fig.savefig(out, dpi=110); print(out)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], *(a[1:2] or ["2"]), *([int(a[2])] if len(a) > 2 else [1]), *(a[3:4] or ["A"]), *(a[4:5] or ["results/qa/n0-28e/passnet_g2h1_A.png"]))
