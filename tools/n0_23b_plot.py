"""N0-23b: summary table + picture from results/qa/n0-23b/metrica_press*.json -> summary.json, press_vs_truth.png"""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "results/qa/n0-23b/"; w2 = lambda x: x is not None and x <= 2
rows = []
for f in ("metrica_press.json", "metrica_press_view.json", "metrica_press_from_touch_rejected.json"):
    R = json.load(open(D + f))
    for nz in dict.fromkeys(r["noise"] for r in R):
        rs = [r for r in R if r["noise"] == nz]; P = [p for r in rs for p in r["pairs"]]
        both = [(p["truth"], p["ours"]) for p in P if p["truth"] is not None and p["ours"] is not None]
        n2 = [(p["truth_near2"], p["ours_near2"]) for p in P if p["truth_near2"] is not None and p["ours_near2"] is not None]
        rows.append({"file": f, "noise": nz, "pairs": len(P), "agree_w2": sum(w2(p["truth"]) == w2(p["ours"]) for p in P),
                     "truth_yes": sum(w2(p["truth"]) for p in P), "ours_yes": sum(w2(p["ours"]) for p in P),
                     "time_err_median_s": round(float(np.median([o - t for t, o in both])), 2) if both else None,
                     "time_within_0_5s": f"{sum(abs(o - t) <= 0.5 for t, o in both)}/{len(both)}",
                     "near2_same": f"{sum(a == b for a, b in n2)}/{len(n2)}", "near2_mean_truth": round(float(np.mean([a for a, b in n2])), 2),
                     "near2_mean_ours": round(float(np.mean([b for a, b in n2])), 2),
                     "half_within2_truth": [r["truth"]["within2_pct"] for r in rs], "half_within2_ours": [r["ours"]["within2_pct"] for r in rs],
                     "half_median_truth": [r["truth"]["median_s"] for r in rs], "half_median_ours": [r["ours"]["median_s"] for r in rs]})
json.dump(rows, open(D + "summary.json", "w"), indent=1)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
lab = [("today" if "rejected" not in r["file"] else "from 1st touch") + "\n" + r["noise"].replace("False", "perfect") for r in rows]
x = np.arange(len(rows)); ax[0].bar(x, [100 * r["agree_w2"] / r["pairs"] for r in rows], color=["#3a7" if "rejected" not in r["file"] else "#bbb" for r in rows])
for i, r in enumerate(rows): ax[0].text(i, 100 * r["agree_w2"] / r["pairs"] + 1, f"{r['agree_w2']}/{r['pairs']}", ha="center", fontsize=7)
ax[0].set_xticks(x, lab, fontsize=7); ax[0].set_ylim(0, 100); ax[0].set_title("'pressed within 2 s': same answer as the truth, per lost ball (%)", fontsize=9)
k = [i for i, r in enumerate(rows) if "rejected" not in r["file"]]
ax[1].bar(np.arange(len(k)) - 0.2, [r["truth_yes"] for r in (rows[i] for i in k)], 0.4, label="truth 'yes'", color="#999")
ax[1].bar(np.arange(len(k)) + 0.2, [r["ours_yes"] for r in (rows[i] for i in k)], 0.4, label="ours 'yes'", color="#3a7")
ax[1].set_xticks(np.arange(len(k)), [lab[i] for i in k], fontsize=7); ax[1].legend(fontsize=8); ax[1].set_title("same lost balls: how many pressed within 2 s", fontsize=9)
plt.tight_layout(); plt.savefig(D + "press_vs_truth.png", dpi=110); print(json.dumps([{k_: v for k_, v in r.items() if not k_.startswith("half")} for r in rows], indent=0))
