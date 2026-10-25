"""N0-25c: draw the frames where the new shape timeline reads a block longer than 60 m on the repo copies (AIK, Vallentuna):
filled = in the picture, hollow = held (seconds since seen next to it). Usage: python tools/n0_25c_spikes.py <out dir>"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import n0_22e_app_copies as AC, n0_25c_app_copies as A
from ipanema import analytics as AN, metrics as ME


def main(out_dir, n=6, seed=2):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    for m in AC.MATCHES[1:]:
        md, st, fr = AC.load(m); L, W = md["pitch"]["length"], md["pitch"]["width"]; ar = md["attack_right"]
        fps = 1 / float(np.median(np.diff([f["t"] for f in fr]))); per = A.per_of(fr); sh = AN.shapes(per, L, fps)
        tl = ME.shape_timeline(sh, fps, ar, L, per=per, W=W); held = AN.held_points(per, fps, L=L, W=W)
        spk = [(tm, int(round(r["t"] * fps))) for tm in "AB" for r in tl[tm] if r["length"] is not None and r["length"] > 60]
        pick = [spk[i] for i in sorted(rng.choice(len(spk), size=min(n, len(spk)), replace=False))]
        fig, axs = plt.subplots(2, 3, figsize=(15, 6.6))
        for ax, (tm, k) in zip(axs.ravel(), pick):
            ax.add_patch(plt.Rectangle((0, 0), L, W, fill=False, color="0.5")); ax.plot([L / 2, L / 2], [0, W], color="0.7", lw=0.8)
            col = "#222222" if tm == "A" else "#d23c3c"
            sp = np.array([r[2] for r in per[k] if r[1] == tm and not r[5]]); hp = held[k][tm]
            if len(sp): ax.scatter(sp[:, 0], sp[:, 1], s=30, c=col, zorder=3)
            hs = [p for p in hp if not any(np.allclose(p, q) for q in sp)]
            for p in hs: ax.scatter([p[0]], [p[1]], s=40, facecolors="none", edgecolors=col, zorder=3)
            ax.plot([hp[:, 0].min(), hp[:, 0].max()], [W + 2, W + 2], color=col, lw=2)
            if len(sp) >= 2: ax.plot([sp[:, 0].min(), sp[:, 0].max()], [W + 4, W + 4], color=col, lw=1, ls=":")
            ax.set_xlim(-3, L + 3); ax.set_ylim(-3, W + 7); ax.set_aspect("equal"); ax.axis("off")
            ax.set_title(f"{m[:16]} team {tm} t={fr[min(k, len(fr) - 1)]['t']:.0f}s length {np.ptp(hp[:, 0]):.0f} m (seen {np.ptp(sp[:, 0]) if len(sp) else 0:.0f})", fontsize=8)
        fig.suptitle(f"{m}: seconds with block length > 60 m ({len(spk)} team-seconds); filled = seen, hollow = held < 5 s; solid bar new, dotted seen only", fontsize=9)
        fig.tight_layout(); fig.savefig(f"{out_dir}/spikes_{m[:12]}.png", dpi=85); plt.close(fig)
        print(m, len(spk))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/qa/n0-25c")


def line_clean(per, L, W, fps):
    """the pipeline's line rules (tracking.clean 1b + 9 Oct + TOUCHLINE 25 Oct) on a copy's rows - the repo copies
    (4 Oct exports) predate the last two"""
    from ipanema import tracking as TR
    life = {}
    for k in per:
        for r in per[k]: life.setdefault(r[0], []).append(r[2])
    e = lambda p: min(p[0], L - p[0], p[1], W - p[1])
    still = {t for t, pts in life.items() if len(pts) >= int(2 * fps) and all(e(p) < 2.0 for p in pts) and float(np.linalg.norm(np.max(pts, axis=0) - np.min(pts, axis=0))) < 3.0}
    still |= {t for t, pts in life.items() if len(pts) >= int(fps) and sum(1 for p in pts if e(p) < 1.0) >= 0.7 * len(pts)}
    per = {k: [r for r in v if r[0] not in still] for k, v in per.items()}
    tl = TR.touchline_rows(per, W, fps)
    return {k: [r for r in v if (k, r[0]) not in tl] for k, v in per.items()}


def share_over(out_json="results/qa/n0-25c/spikes.json"):
    import json; res = {}
    for m in AC.MATCHES:
        md, st, fr = AC.load(m); L, W = md["pitch"]["length"], md["pitch"]["width"]; ar = md["attack_right"]
        fps = 1 / float(np.median(np.diff([f["t"] for f in fr]))); per0 = A.per_of(fr); res[m] = {}
        for nm, per in (("copy", per0), ("copy + today's line rules", line_clean(per0, L, W, fps))):
            sh = AN.shapes(per, L, fps); r = {}
            for rule, tl in (("old", ME.shape_timeline(sh, fps, ar, L)), ("new", ME.shape_timeline(sh, fps, ar, L, per=per, W=W))):
                v = np.array([e["length"] for tm in "AB" for e in tl[tm] if e["length"] is not None])
                r[rule] = {"over_60m": round(float((v > 60).mean()), 3), "p95": round(float(np.percentile(v, 95)), 1), "median": round(float(np.median(v)), 1)}
            res[m][nm] = r; print(m[:14], nm, r)
    json.dump(res, open(out_json, "w"), indent=1)
