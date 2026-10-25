"""N0-25b: the team heat map on the repo copies of the app's exports (frame files, like tools/n0_22e_app_copies.py):
players in the picture only (what the app can build today) vs the new export rule (analytics.team_heatmaps, last seen
spot held 20 s) -> results/qa/n0-25b/clips.json + clips_maps.png. Usage: python tools/n0_25b_clips.py"""
import sys, os, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "tools"); sys.path.insert(0, "."); from ipanema import analytics as AN
import n0_22e_app_copies as AC
res = {}; ms = [m for m in AC.MATCHES if os.path.exists(f"results/volume/runs/matches/{m}")]
fig, ax = plt.subplots(4, len(ms), figsize=(4 * len(ms), 9.5), squeeze=False)
th = lambda g: [round(100 * float(g[a:b].sum())) for a, b in ((0, 4), (4, 8), (8, 12))]
for c, m in enumerate(ms):
    md, st, fr = AC.load(m); L, W = md["pitch"]["length"], md["pitch"]["width"]; ar = md["attack_right"]
    ts = [f["t"] for f in fr]; fps = 1.0 / float(np.median(np.diff(ts))); per = {}; prev = None
    for f in fr:
        if prev is not None and f["t"] - prev > 2.0: per[len(per)] = []
        per[len(per)] = [[p["id"], p["team"], np.array(p["m"], float), None, None, bool(p.get("gk"))] for p in f["players"]]; prev = f["t"]
    per = [per[k] for k in range(len(per))]
    old = AN.team_heatmaps(per, fps, L, W, ar, hold_s=0); new = AN.team_heatmaps(per, fps, L, W, ar)
    res[m] = {"frames": len(fr), "fps": round(fps, 2), "attack_right": ar}
    for i, tm in enumerate("AB"):
        o, n_ = np.array(old[tm]["grid"]), np.array(new[tm]["grid"])
        res[m][tm] = {"tv_old_vs_new": round(50 * float(np.abs(o - n_).sum()), 1), "thirds_own_mid_att_old": old[tm]["thirds_pct"], "thirds_own_mid_att_new": new[tm]["thirds_pct"],
                      "frames_old": old[tm]["frames"], "frames_new": new[tm]["frames"]}
        for r, (g, lab, d) in enumerate(((o, "in picture", old[tm]), (n_, "new (held 20 s)", new[tm]))):
            a = ax[2 * i + r, c]; a.imshow(g.T, origin="lower", cmap="magma", extent=[0, L, 0, W], vmin=0, vmax=max(o.max(), n_.max()))
            for x in (L / 3, 2 * L / 3): a.axvline(x, color="w", lw=.6, ls=":")
            a.annotate("", xy=(L * (0.62 if ar[tm] else 0.38), W * 0.93), xytext=(L * (0.38 if ar[tm] else 0.62), W * 0.93), arrowprops=dict(arrowstyle="->", color="c"))
            a.set_title(f"{m[:14]} {tm}: {lab}\nown/mid/att {d['thirds_pct']}", fontsize=8); a.set_xticks([]); a.set_yticks([])
    print(m, json.dumps({k: v for k, v in res[m].items() if k in "AB"}))
fig.suptitle("Team heat maps on the app's repo copies (arrow = attack direction)", fontsize=9); fig.tight_layout(); fig.savefig("results/qa/n0-25b/clips_maps.png", dpi=105)
json.dump(res, open("results/qa/n0-25b/clips.json", "w"), indent=1)
