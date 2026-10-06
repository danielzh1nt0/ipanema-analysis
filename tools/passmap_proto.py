"""6 Oct: prototype of the pass graphics for the app, from stats.json passes[] (from_m/to_m, team, completed, progressive).
Left: where a team's passes go (zone flows, 6 x 3 zones, arrow width = number of passes, top flows only).
Right: the team's forward passes that gained >= 10 m (completed solid, failed dashed).
    python tools/passmap_proto.py <match> [A|B] -> results/app/passmap/<match>_<team>.png"""
import sys, json, collections, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arc, Circle, FancyArrowPatch
M = sys.argv[1]; T = sys.argv[2] if len(sys.argv) > 2 else "A"
st = json.load(open(f"results/volume/runs/matches/{M}/stats.json")); L, W = 106.0, 64.0
per = st["periods"]; inwin = lambda t: any(p["t_start"] <= t <= p["t_end"] for p in per)
P = [p for p in st["passes"] if p["team"] == T and inwin(p["t"]) and p.get("from_m") and p.get("to_m")]
INK, GRASS, LINE = "#17201b", "#eef3ef", "#9fb3a6"; COL = "#17201b" if T == "A" else "#c8323c"
def pitch(ax):
    ax.add_patch(Rectangle((0, 0), L, W, fc=GRASS, ec=LINE, lw=1.2))
    ax.plot([L / 2, L / 2], [0, W], c=LINE, lw=1); ax.add_patch(Circle((L / 2, W / 2), 9.15, fill=False, ec=LINE, lw=1))
    for x0, s in ((0, 1), (L, -1)):
        ax.add_patch(Rectangle((x0 if s == 1 else x0 - 16.5, W / 2 - 20.15), 16.5, 40.3, fill=False, ec=LINE, lw=1))
        ax.add_patch(Rectangle((x0 if s == 1 else x0 - 5.5, W / 2 - 9.15), 5.5, 18.3, fill=False, ec=LINE, lw=1))
    ax.set_xlim(-3, L + 3); ax.set_ylim(W + 3, -3); ax.set_aspect("equal"); ax.axis("off")
    ax.annotate("", xy=(L * 0.62, W + 1.8), xytext=(L * 0.38, W + 1.8), arrowprops=dict(arrowstyle="->", color="#5d6b63", lw=1))
    ax.text(L / 2, W + 2.8, "attacking direction", ha="center", va="top", fontsize=8, color="#5d6b63")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2)); fig.patch.set_facecolor("white")
# zone flows
NX, NY = 6, 3; zid = lambda p: (min(NX - 1, int(p[0] / L * NX)), min(NY - 1, int(p[1] / W * NY)))
cen = lambda z: ((z[0] + 0.5) * L / NX, (z[1] + 0.5) * W / NY)
fl = collections.Counter((zid(p["from_m"]), zid(p["to_m"])) for p in P if p.get("completed"))
pitch(a1)
for i in range(1, NX): a1.plot([i * L / NX] * 2, [0, W], c="#d6ddd8", lw=0.6, ls=":")
for j in range(1, NY): a1.plot([0, L], [j * W / NY] * 2, c="#d6ddd8", lw=0.6, ls=":")
top = [(k, n) for k, n in fl.most_common() if k[0] != k[1]][:14]; mx = max([n for _, n in top] or [1])
for (z0, z1), n in top:
    (x0, y0), (x1, y1) = cen(z0), cen(z1)
    a1.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10 + 8 * n / mx, lw=1 + 6 * n / mx, color=COL, alpha=0.35 + 0.6 * n / mx, shrinkA=6, shrinkB=6, connectionstyle="arc3,rad=0.12"))
inside = collections.Counter(zid(p["from_m"]) for p in P if p.get("completed") and zid(p["from_m"]) == zid(p["to_m"]))
for z, n in inside.items():
    x, y = cen(z); a1.text(x, y, str(n), ha="center", va="center", fontsize=8, color="#5d6b63")
a1.set_title(f"Where the passes go: top {len(top)} zone-to-zone routes (completed)\nArrow width = number of passes; grey number = short passes inside a zone", fontsize=10, color=INK, loc="left")
# forward passes
pitch(a2); fw = [p for p in P if p["from_m"][0] < 2 * L / 3 <= p["to_m"][0]]
for p in fw:
    ok = bool(p.get("completed"))
    a2.add_patch(FancyArrowPatch(tuple(p["from_m"]), tuple(p["to_m"]), arrowstyle="-|>", mutation_scale=8, lw=1.3, color=COL if ok else "#9aa8a0", alpha=0.8 if ok else 0.6, ls="-" if ok else (0, (3, 2))))
n_ok = sum(bool(p.get("completed")) for p in fw)
a2.axvline(2 * L / 3, color="#d6ddd8", lw=1, ls="--"); a2.set_title(f"Passes into the final third: {len(fw)} ({n_ok} completed)\nSolid = completed, dashed grey = did not arrive", fontsize=10, color=INK, loc="left")
fig.suptitle(f"{M} · team {T} · {len(P)} passes in the playing time", fontsize=11, x=0.01, ha="left", color="#5d6b63")
plt.tight_layout(); out = f"results/app/passmap/{M}_{T}.png"; plt.savefig(out, dpi=110); print(out, len(P), len(fw), top[:3])
