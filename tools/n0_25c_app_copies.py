"""N0-25c: the 'shape over time' timeline (metrics.shape_timeline) on the repo copies of the app's exports, old (players in
the picture only, >= 3) vs new (held spots 5 s, >= 5, like the Shape tab's medians since N0-22e), plus a picture per
match: block length / width per second, old vs new, with the team-card median. Also a Metrica picture: truth vs old vs
new over 5 minutes (one game, one clip view, seed 0).
Usage: python tools/n0_25c_app_copies.py <out dir> [/tmp/metrica/data]"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import n0_22e_app_copies as AC
from ipanema import analytics as AN, metrics as ME


def per_of(fr):
    per = {}; prev = None
    for f in fr:
        if prev is not None and f["t"] - prev > 2.0: per[len(per)] = []
        per[len(per)] = [[p["id"], p["team"], np.array(p["m"], float), None, None, bool(p.get("gk"))] for p in f["players"]]; prev = f["t"]
    return per


def summary(tl):
    out = {}
    for tm, rows in tl.items():
        d = {}
        for q in ("length", "width", "line_height"):
            v = [r[q] for r in rows if r[q] is not None]
            d[q] = {"median": round(float(np.median(v)), 1) if v else None, "covered": round(len(v) / max(1, len(rows)), 2)}
        out[tm] = d
    return out


def plot(m, tl_old, tl_new, card, out_png, title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(15, 6.5), sharex=True)
    for j, tm in enumerate(("A", "B")):
        for i, q in enumerate(("length", "width")):
            ax = axs[i, j]
            for tl, st, lab in ((tl_old, dict(color="0.6", lw=0.8), "old (players in the picture)"), (tl_new, dict(color="#d23c3c", lw=1.2), "new (held 5 s)")):
                t = [r["t"] for r in tl[tm]]; v = [np.nan if r[q] is None else r[q] for r in tl[tm]]
                ax.plot(t, v, label=lab, **st)
            if card and card.get(tm) and card[tm].get(q) is not None: ax.axhline(card[tm][q], color="k", ls="--", lw=0.8, label="card median (Shape tab)")
            if isinstance(card, dict) and "truth" in card: pass
            ax.set_title(f"{tm} block {q} (m)", fontsize=9); ax.set_ylim(0, 80)
            if i == 0 and j == 0: ax.legend(fontsize=8, loc="upper right")
    axs[1, 0].set_xlabel("s"); axs[1, 1].set_xlabel("s"); fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(out_png, dpi=80); plt.close(fig)


def metrica_plot(d, out_png, secs=(600, 900)):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import n0_22e_shape_view as SV, n0_22c_metrica as C, n0_25c_timeline as T
    runs, gaps, jit = C.our_breaks(T.CLIPS); fr = SV.load_metrica(d, "Sample_Game_1")
    v, Lv, Wv = SV.view_masks(T.CLIPS[0]); per, fps, dirA, truth = T.build(fr, v, Lv, Wv, seed=0, breaks=(runs, gaps), jit=jit)
    ks = sorted(per); hk = [k for k in ks if per[k] or truth.get(k)][: int(2700 * fps)]          # first half
    hk = [k for k in hk if k >= ks[0]]; sub = {i: per[k] for i, k in enumerate(hk)}; ar = {"A": dirA[hk[0]], "B": not dirA[hk[0]]}
    sh = AN.shapes(sub, 105.0, fps); old = ME.shape_timeline(sh, fps, ar, 105.0); new = ME.shape_timeline(sh, fps, ar, 105.0, per=sub, W=68.0)
    fig, axs = plt.subplots(2, 1, figsize=(15, 6.5), sharex=True)
    for ax, q, qi in ((axs[0], "length", 0), (axs[1], "width", 1)):
        t = [r["t"] for r in old["A"]]; sel = [i for i, x in enumerate(t) if secs[0] <= x < secs[1]]
        tv = [truth.get(hk[min(int(round(old["A"][i]["t"] * fps)), len(hk) - 1)], {}).get("A") for i in sel]
        ax.plot([t[i] for i in sel], [np.nan if x is None else x[qi] for x in tv], color="k", lw=1.5, label="truth (all 10 outfield players)")
        ax.plot([t[i] for i in sel], [np.nan if old["A"][i][q] is None else old["A"][i][q] for i in sel], color="0.6", lw=0.8, label="old: players in our camera view")
        ax.plot([t[i] for i in sel], [np.nan if new["A"][i][q] is None else new["A"][i][q] for i in sel], color="#d23c3c", lw=1.2, label="new: held 5 s")
        ax.set_title(f"Metrica game 1, home team, block {q} (m), seen through the SFK-BP clip's camera views", fontsize=9); ax.set_ylim(0, 70)
    axs[0].legend(fontsize=8, loc="upper right"); axs[1].set_xlabel("s"); fig.tight_layout(); fig.savefig(out_png, dpi=80); plt.close(fig)


def main(out_dir, metrica=None):
    os.makedirs(out_dir, exist_ok=True); res = {}
    for m in AC.MATCHES:
        if not os.path.exists(f"results/volume/runs/matches/{m}"): continue
        md, st, fr = AC.load(m); L, W = md["pitch"]["length"], md["pitch"]["width"]; ar = md["attack_right"]
        ts = [f["t"] for f in fr]; fps = 1.0 / float(np.median(np.diff(ts))) if len(ts) > 1 else 10.0
        per = per_of(fr); sh = AN.shapes(per, L, fps)
        old = ME.shape_timeline(sh, fps, ar, L); new = ME.shape_timeline(sh, fps, ar, L, per=per, W=W)
        card = AC.medians(per, fps, L, ar, AN.SHAPE_HOLD_S, W)
        res[m] = {"seconds": len(old["A"]), "old": summary(old), "new": summary(new), "card_new": card}
        print(m, json.dumps(res[m]))
        plot(m, old, new, card, f"{out_dir}/timeline_{m[:12]}.png", f"{m}: shape over time, old vs new; dashed = team card median (Shape tab, held 5 s)")
    json.dump(res, open(f"{out_dir}/app_copies.json", "w"), indent=1)
    if metrica: metrica_plot(metrica, f"{out_dir}/metrica_timeline.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/qa/n0-25c", sys.argv[2] if len(sys.argv) > 2 else None)
