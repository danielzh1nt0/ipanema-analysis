"""N0-25b: picture of the team heat maps on Metrica (truth vs what our camera would give) -> results/qa/n0-25b/metrica_maps.png"""
import json, sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
src = sys.argv[1] if len(sys.argv) > 1 else "results/qa/n0-25b/metrica_grids.json"; out = sys.argv[2] if len(sys.argv) > 2 else "results/qa/n0-25b/metrica_maps.png"
G = json.load(open(src)); runs = [r for r in G if "SFKBP" in r][:2]; rules = ["truth", "frames", "tracksum", "hold20"] + (["shipped"] if "shipped" in G[runs[0]]["A"] else [])
fig, ax = plt.subplots(len(runs) * 2, len(rules), figsize=(3.2 * len(rules), 2.3 * len(runs) * 2))
for i, (r, tm) in enumerate([(r, tm) for r in runs for tm in "AB"]):
    t = np.array(G[r][tm]["truth"]); vmax = t.max() * 1.3
    for j, rule in enumerate(rules):
        g = np.array(G[r][tm][rule]); a = ax[i, j]; a.imshow(g.T, origin="lower", cmap="magma", vmin=0, vmax=vmax, extent=[0, 105, 0, 68])
        for x in (35, 70): a.axvline(x, color="w", lw=.6, ls=":")
        tv = 0.5 * np.abs(g - t).sum() * 100
        a.set_title(f"{r[:5]} {tm} {rule}" + ("" if rule == "truth" else f"  TV {tv:.0f}"), fontsize=8); a.set_xticks([]); a.set_yticks([])
fig.suptitle("Team heat maps, attacking right (Metrica pro tracking; our camera views from the SFK-BP clip)", fontsize=9); fig.tight_layout(); fig.savefig(out, dpi=110)
