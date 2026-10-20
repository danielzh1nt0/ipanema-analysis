"""N0-20e (20 Oct, local, free): restart spots on the two app clips' exact inputs (SFK-BP + AIK picker_inputs.pkl).
possession.restarts() took the spot as the median ball from the stoppage's last second up to 1.5 s AFTER it ended - 45 of
the 75 frames are after the restart, when the ball is already on its way. Compares the old window ('after') with the
stoppage's own last second ('last1'), the kind each gives, and writes the moments for the free-runner picture sheets
(results/qa/n0-20e/moments.json; which window said what -> moments_key.json, opened only after grading).
    python tools/n0_20e_restarts.py"""
import sys, os, json, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL, possession as P, calibration as C

CLIPS = {"SFKBP1109_s1200": "results/volume/cache/SFKBP1109_s1200/picker_inputs.pkl",
         "p15u-vs-aik-2026-09-21-bd09_s2520": "results/volume/cache/p15u-vs-aik-2026-09-21-bd09_s2520/picker_inputs.pkl"}
OUT = "results/qa/n0-20e"; os.makedirs(OUT, exist_ok=True)
q = lambda *a: None


def to_px(Hk, xy):
    """metres -> picture pixels with the frame's camera (inverse of calibration.to_m)"""
    try:
        if hasattr(Hk, "to_px"): return [float(v) for v in np.asarray(Hk.to_px(np.array([xy], float))).reshape(-1)[:2]]
        p = np.asarray(Hk, float) @ np.array([xy[0], xy[1], 1.0]); return [float(p[0] / p[2]), float(p[1] / p[2])]   # to_m uses inv(H): H itself is metres -> pixels
    except Exception: return None


def main():
    report, moments, key = {}, [], {}
    for clip, path in CLIPS.items():
        I = pickle.load(open(path, "rb")); fps, H, L, W = I["fps"], I["H"], I["L"], I["W"]
        per = {k: [[r[0], r[1], np.asarray(r[2]), None if r[3] is None else np.asarray(r[3]), None, False] for r in v] for k, v in I["per"].items()}
        ball = BL.bridge(BL.pick_v2(I["cands"], H, L, W, per=per, fps=fps, log=q), fps)
        frames_, ballm = P.carriers(per, ball, H, 2.5, 5.0); state, bspeed, dstate, pinfo = P.pipeline_state(per, ball, ballm, H, fps, L, W, log=q)
        old = P.restarts(dstate, ballm, fps, L, W, spot="after"); new = P.restarts(dstate, ballm, fps, L, W, spot="last1")
        assert [r["t"] for r in old] == [r["t"] for r in new]
        rows = []
        for i, (a, b) in enumerate(zip(old, new)):
            d = None if a["x_m"] is None or b["x_m"] is None else round(float(np.hypot(a["x_m"] - b["x_m"], a["y_m"] - b["y_m"])), 1)
            rows.append({"t": a["t"], "team": a["team"], "old": [a["kind"], a["x_m"], a["y_m"]], "new": [b["kind"], b["x_m"], b["y_m"]], "moved_m": d})
            mid = f"{'sfk' if clip.startswith('SFK') else 'aik'}{i:02d}"; k_end = int(round(a["t"] * fps))
            tiles = []
            for dt in np.arange(-4.0, 3.01, 0.5):
                k = int(round(k_end + dt * fps))
                if k < 0 or k >= len(per): continue
                tiles.append({"frame": k, "t": round(k / fps, 1), "dt": round(float(dt), 1)})
            # both spots, unlabelled and in a random order (blind): which is which is in the key
            spots = [("old", a), ("new", b)]
            if sum(map(ord, mid)) % 2: spots.reverse()
            for tl in tiles:                                                   # the camera pans: each tile gets the spots in its own frame's pixels
                tl["spot_px"] = [None if r["x_m"] is None else to_px(H[tl["frame"]], (r["x_m"], r["y_m"])) for _, r in spots]
                bp = ball.get(tl["frame"]) if isinstance(ball, dict) else None
                tl["ball_px"] = [float(bp[0]), float(bp[1])] if bp is not None else None
            moments.append({"id": mid, "clip": clip, "t_end": a["t"], "tiles": tiles})
            key[mid] = {"spot1": spots[0][0], "spot2": spots[1][0], "old": a, "new": b}
        changed = sum(1 for r in rows if r["old"][0] != r["new"][0])
        report[clip] = {"restarts": len(rows), "kind_changed": changed, "moved_m_median": float(np.median([r["moved_m"] for r in rows if r["moved_m"] is not None] or [0])),
                        "kinds_old": _count(r["old"][0] for r in rows), "kinds_new": _count(r["new"][0] for r in rows), "rows": rows}
        print(f"{clip}: {len(rows)} restarts, kind changed on {changed}; old {report[clip]['kinds_old']} new {report[clip]['kinds_new']}; spot moved median {report[clip]['moved_m_median']} m")
        for r in rows: print(f"   {r['t']:7.1f} s  old {r['old']}  new {r['new']}  moved {r['moved_m']} m")
    json.dump(report, open(f"{OUT}/replay.json", "w"), indent=1)
    json.dump({"clips": {c: f"{c}/video.mp4" for c in CLIPS}, "moments": moments}, open(f"{OUT}/moments.json", "w"), indent=1)
    json.dump(key, open(f"{OUT}/moments_key.json", "w"), indent=1)


def _count(it):
    d = {}
    for x in it: d[x] = d.get(x, 0) + 1
    return d


if __name__ == "__main__": main()
