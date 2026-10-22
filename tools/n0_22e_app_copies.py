"""N0-22e: team block length / width / line height on the repo copies of the app's exports (frame files, every 3rd
frame = 10 per second), old (seen players only) vs new (players just out of the picture keep their last spot for
analytics.SHAPE_HOLD_S s). Also draws a few frames: filled = in the picture, hollow = held (with seconds since seen).
Usage: python tools/n0_22e_app_copies.py <out dir>"""
import sys, os, glob, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import analytics as AN

MATCHES = ["SFKBP1109", "p15u-vs-vallentuna-2026-10-03-6cce", "p15u-vs-aik-2026-09-21-bd09"]


def load(m):
    root = f"results/volume/runs/matches/{m}"; md = json.load(open(f"{root}/match_data.json")); st = json.load(open(f"{root}/stats.json"))
    fr = []
    for f in sorted(glob.glob(f"{root}/frames_*.json")): fr += json.load(open(f))["frames"]
    if not fr: fr = md.get("frames", [])
    return md, st, fr


def medians(per, fps, L, attack_right, hold, W=None):
    held = AN.held_points(per, fps, hold, L=L, W=W) if hold else None; out = {}
    for tm in ("A", "B"):
        lens, wids, hts = [], [], []
        for k in range(len(per)):
            pts = np.array([r[2] for r in per[k] if r[1] == tm and not r[5]])
            if len(pts) >= 5:
                hts.append(pts[:, 0].min() if attack_right[tm] else L - pts[:, 0].max())
                if not hold: lens.append(np.ptp(pts[:, 0])); wids.append(np.ptp(pts[:, 1]))
            hp = held[k][tm] if held else None
            if hp is not None and len(hp) >= 5: lens.append(np.ptp(hp[:, 0])); wids.append(np.ptp(hp[:, 1]))
        out[tm] = {"length": round(float(np.median(lens)), 1) if lens else None, "width": round(float(np.median(wids)), 1) if wids else None,
                   "line": round(float(np.median(hts)), 1) if hts else None, "frames": len(lens)}
    return out


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True); res = {}
    for m in MATCHES:
        if not os.path.exists(f"results/volume/runs/matches/{m}"): continue
        md, st, fr = load(m); L, W = md["pitch"]["length"], md["pitch"]["width"]; ar = md["attack_right"]
        ts = [f["t"] for f in fr]; fps = 1.0 / float(np.median(np.diff(ts))) if len(ts) > 1 else 10.0
        per = {}; prev = None
        for i, f in enumerate(fr):
            if prev is not None and f["t"] - prev > 2.0: per[len(per)] = []          # missing frame files: a hole, not a join
            per[len(per)] = [[p["id"], p["team"], np.array(p["m"], float), None, None, bool(p.get("gk"))] for p in f["players"]]; prev = f["t"]
        res[m] = {"frames": len(fr), "fps": round(fps, 2), "old": medians(per, fps, L, ar, 0, W), "new": medians(per, fps, L, ar, AN.SHAPE_HOLD_S, W),
                  "in_app_stats_json": {t["team"]: [t.get("block_length_median_m"), t.get("block_width_median_m"), t.get("def_line_height_median_m")] for t in st.get("teams", [])}}
        held = AN.held_points(per, fps, AN.SHAPE_HOLD_S, L=L, W=W); seen_n = {"A": [], "B": []}; held_n = {"A": [], "B": []}
        for k in range(len(per)):
            for tm in ("A", "B"):
                if held[k][tm] is not None:
                    seen_n[tm].append(sum(1 for r in per[k] if r[1] == tm and not r[5])); held_n[tm].append(len(held[k][tm]))
        res[m]["per_frame_median"] = {tm: {"seen": float(np.median(seen_n[tm])) if seen_n[tm] else None, "with_held": float(np.median(held_n[tm])) if held_n[tm] else None,
                                           "with_held_p90": float(np.percentile(held_n[tm], 90)) if held_n[tm] else None} for tm in ("A", "B")}
        print(m, json.dumps(res[m]))
        draw(m, per, fps, L, W, held, out_dir)
    json.dump(res, open(f"{out_dir}/app_copies.json", "w"), indent=1)


def draw(m, per, fps, L, W, held, out_dir, n=6, seed=3):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    ok = [k for k in range(len(per)) if held[k]["A"] is not None and held[k]["B"] is not None and len(held[k]["A"]) > sum(1 for r in per[k] if r[1] == "A" and not r[5])]
    if not ok: return
    pick = sorted(rng.choice(ok, size=min(n, len(ok)), replace=False))
    fig, axs = plt.subplots(2, 3, figsize=(15, 6.6))
    last = {}
    for k in range(len(per)):
        for r in per[k]: last.setdefault(r[0], []).append(k)
    for ax, k in zip(axs.ravel(), pick):
        ax.add_patch(plt.Rectangle((0, 0), L, W, fill=False, color="0.5")); ax.plot([L / 2, L / 2], [0, W], color="0.7", lw=0.8)
        seen = {r[0]: r for r in per[k]}
        for tm, col in (("A", "#222222"), ("B", "#d23c3c")):
            sp = np.array([r[2] for r in per[k] if r[1] == tm and not r[5]]); hp = held[k][tm]
            if len(sp): ax.scatter(sp[:, 0], sp[:, 1], s=30, c=col, zorder=3)
            hs = np.array([p for p in hp if not any(np.allclose(p, q) for q in sp)]) if len(hp) else np.zeros((0, 2))
            if len(hs): ax.scatter(hs[:, 0], hs[:, 1], s=40, facecolors="none", edgecolors=col, zorder=3)
            x0, x1 = hp[:, 0].min(), hp[:, 0].max(); y = W + (2 if tm == "A" else 5)
            ax.plot([x0, x1], [y, y], color=col, lw=2); 
            if len(sp) >= 2: ax.plot([sp[:, 0].min(), sp[:, 0].max()], [y + 1, y + 1], color=col, lw=1, ls=":")
        ax.set_xlim(-3, L + 3); ax.set_ylim(-3, W + 8); ax.set_aspect("equal"); ax.set_title(f"{m[:20]} frame {k} ({k / fps:.0f} s)", fontsize=9); ax.axis("off")
    fig.suptitle("filled = in the picture, hollow = held (left the picture < 5 s ago); bars on top: block length new (solid) vs seen only (dotted)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{out_dir}/held_{m[:12]}.png", dpi=90); plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/qa/n0-22e")
