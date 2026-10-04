"""PC1 picture: who has the ball, second by second, on Metrica game 1 (first 4 min of each half): Metrica's labels vs
PathCRF from players only: exact, 35 m in view (rest filled), 0.8 m drifting error, 0.8 m error new every frame, and
view + drifting error together; pass ticks. -> results/pathcrf/pc1_timeline.png"""
import os, sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pc1lab as PC, metricalab as ML

ROOT = PC.ROOT; MIN = 4.0


def strip(ax, y, team, f0):
    col = {"home": "#d62728", "away": "#1f77b4"}
    t = (team.index.values - f0) / PC.FPS
    for tm, c in col.items():
        m = (team.values == tm)
        ax.fill_between(t, y - 0.35, y + 0.35, where=m, color=c, step="mid", linewidth=0)


def main():
    PC.ensure_repos(); model, args = PC.load_model()
    evs = ML.load_events(f"{PC.METRICA}/Sample_Game_1/Sample_Game_1_RawEventsData.csv")
    runs = {"PathCRF, all 22 exact": PC.build_tracking(1, MIN), "35 m in view": PC.build_tracking(1, MIN, view=35),
            "0.8 m drifting error": PC.build_tracking(1, MIN, 0.8, smooth_s=1.0), "0.8 m new every frame": PC.build_tracking(1, MIN, 0.8, smooth_s=0),
            "35 m view + 0.8 m drifting": PC.build_tracking(1, MIN, 0.8, view=35, smooth_s=1.0)}
    res = {k: PC.run_pathcrf(model, args, t) for k, t in runs.items()}
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    for per, ax in zip((1, 2), axes):
        out = {}
        for k, (tr, micro, ev, _) in res.items():
            st, poss = PC.stats_from_edges(tr, micro, ev, None)
            trI = tr.set_index("frame_id") if "frame_id" in tr.columns else tr
            m = (trI["period_id"] == per).values
            out[k] = (poss[m].where(trI["episode_id"].values[m] > 0), st[per]["pass_frames"])
        idx = out[next(iter(out))][0].index; f0 = idx.min()
        lab = PC.truth_team_per_frame([e for e in evs if e["period"] == per], idx, None)
        alive = runs[next(iter(runs))].set_index("frame_id").loc[idx, "episode_id"].values > 0
        R = len(out); strip(ax, R, lab.where(alive), f0)
        tp = [(e["f0"] - 1 - f0) / PC.FPS for e in evs if e["period"] == per and e["type"] == "PASS" and f0 <= e["f0"] - 1 <= idx.max()]
        ax.plot(tp, [R + 0.45] * len(tp), "k|", ms=6)
        for y, (k, (poss, pf)) in zip(range(R - 1, -1, -1), out.items()):
            strip(ax, y, poss, f0)
            pp = [(f - f0) / PC.FPS for tm in pf for f in pf[tm]]
            ax.plot(pp, [y + 0.45] * len(pp), "k|", ms=6)
        ax.set_yticks(range(R, -1, -1)); ax.set_yticklabels(["Metrica labels", *out.keys()], fontsize=8)
        ax.set_xlim(0, MIN * 60); ax.set_ylim(-0.6, R + 0.7); ax.set_title(f"Game 1, half {per}: red = home has the ball, blue = away, white = loose / dead; ticks = completed passes", fontsize=9)
    axes[-1].set_xlabel("seconds")
    plt.tight_layout(); out = os.path.join(ROOT, "results/pathcrf/pc1_timeline.png"); plt.savefig(out, dpi=110); print("->", out)


if __name__ == "__main__":
    main()
