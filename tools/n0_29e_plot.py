"""N0-29e: picture of balls lost per team-half, ours vs the truth -> results/qa/n0-29e/losses.png"""
import json, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, numpy as np

src = sys.argv[1] if len(sys.argv) > 1 else "results/qa/n0-29e/metrica_lossrec.json"
out = sys.argv[2] if len(sys.argv) > 2 else "results/qa/n0-29e/losses.png"
d = json.load(open(src)); R = d["rows"]
labels, tall, ts9, ours = [], [], [], {nz: [] for nz in R}
for h in sorted(d["truth"]):
    for t in "AB":
        labels.append(f"G{h} {t}"); tall.append(d["truth"][h][t]["loss_all"]); ts9.append(d["truth"][h][t]["loss_s9"])
        for nz in R: ours[nz].append(np.mean([r[t]["ours"]["losses"] for r in R[nz] if r["half"] == h]))
x = np.arange(len(labels)); w = 0.17
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.bar(x - 2 * w, tall, w, label="truth: every ball lost", color="#888888")
ax.bar(x - w, ts9, w, label="truth: winner kept it 3 s or passed", color="#bbbbbb")
for i, (nz, c, nm) in enumerate((("False", "#1f6fb2", "ours, perfect ball"), ("flicker", "#e07b00", "ours, flicker ball (like our clips)"), ("streak", "#7a3fa0", "ours, streak noise"))):
    if nz in ours: ax.bar(x + i * w, ours[nz], w, label=nm, color=c)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("balls lost per half"); ax.set_ylim(0, 1.3 * max(tall))
ax.set_title("Balls lost per team-half on Metrica pro data: the app's count vs the truth")
ax.legend(fontsize=8, ncol=2); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(out, dpi=110); print(out)
