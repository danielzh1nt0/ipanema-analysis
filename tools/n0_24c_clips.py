"""N0-24c (24 Oct, local, free): final-third entries on the two app clips' exact inputs (results/volume/cache/<clip>/
picker_inputs.pkl), steps like ipanema/run.py, old rule (hold 0) vs new (IPANEMA_ENTRY_HOLD_S=3).
Writes results/qa/n0-24c/clips.json and moments.json (the entries only one rule counts, for picture sheets).
    PYTHONPATH=. python tools/n0_24c_clips.py"""
import sys, os, json, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL, possession as P, metrics as M
from ipanema.run import periods_direction

CLIPS = {"SFKBP1109_s1200": "results/volume/cache/SFKBP1109_s1200/picker_inputs.pkl",
         "p15u-vs-aik-2026-09-21-bd09_s2520": "results/volume/cache/p15u-vs-aik-2026-09-21-bd09_s2520/picker_inputs.pkl"}
OUT = "results/qa/n0-24c"; q = lambda *a: None


def line_px(Hk, right, L, W):
    """the team's 2/3 line (start of its final third), metres -> pixels through the frame's camera (H maps metres -> px)"""
    if Hk is None or hasattr(Hk, "to_m"): return None
    x = 2 * L / 3 if right else L / 3; out = []
    for y in (0.0, W):
        p = np.asarray(Hk, float) @ np.array([x, y, 1.0]); out.append([round(float(p[0] / p[2]), 1), round(float(p[1] / p[2]), 1)])
    return out


def main():
    rep, moments = {}, []
    for clip, path in CLIPS.items():
        I = pickle.load(open(path, "rb")); fps, H, L, W = I["fps"], I["H"], I["L"], I["W"]
        per = {k: [[r[0], r[1], np.asarray(r[2]), None if r[3] is None else np.asarray(r[3]), None, False] for r in v] for k, v in I["per"].items()}
        ball = BL.bridge(BL.pick_v2(I["cands"], H, L, W, per=per, fps=fps, log=q), fps)
        frames_, ballm = P.carriers(per, ball, H, 2.5, 5.0); state, bspeed, dstate, pinfo = P.pipeline_state(per, ball, ballm, H, fps, L, W, log=q)
        ar, conf = P.direction(state, ballm, log=q); rd = periods_direction(".", clip)
        if rd is not None: ar = {"A": bool(rd), "B": not rd}
        elif min(conf.values()) < 0.15: ar = P.direction_from_keepers(per, L, log=q) or P.direction_fallback(per, L, log=q)
        res = {h: M.field_tilt_and_entries(state, ballm, bspeed, fps, L, ar, *M.ENTRY_REARM, hold_s=h) for h in (0.0, 3.0)}
        rep[clip] = {"attack_right": ar, "fps": fps}
        for name, h in (("old", 0.0), ("new", 3.0)):
            rep[clip][name] = {tm: {k: res[h][tm][k] for k in ("field_tilt_pct", "entries_count", "entries_by_pass", "entries_by_carry")} for tm in "AB"}
        for tm in "AB":
            o = {e["t"] for e in res[0.0][tm]["final_third_entries"]}; n = {e["t"] for e in res[3.0][tm]["final_third_entries"]}
            rep[clip][f"only_old_{tm}"] = sorted(o - n); rep[clip][f"only_new_{tm}"] = sorted(n - o)
            for t in sorted(o ^ n):
                k = int(round(t * fps)); tiles = []
                for j in range(max(0, k - int(3 * fps)), min(len(per) - 1, k + int(1.5 * fps)) + 1, max(1, int(round(0.5 * fps)))):
                    bp = ball.get(j); tiles.append({"frame": j, "t": round(j / fps, 1), "ball_px": [float(bp[0]), float(bp[1])] if bp is not None else None,
                                                    "line_px": line_px(H.get(j), ar[tm], L, W)})
                moments.append({"id": f"{'sfk' if clip.startswith('SFK') else 'aik'}{len(moments):02d}", "clip": clip, "team": tm, "t": t, "frame": k,
                                "attacks": "right" if ar[tm] else "left", "rule": "new only" if t in n else "old only", "tiles": tiles})
        print(clip, json.dumps({k: v for k, v in rep[clip].items()}))
    os.makedirs(OUT, exist_ok=True)
    json.dump(rep, open(f"{OUT}/clips.json", "w"), indent=1)
    # sheets are graded blind: the per-clip moments files (read by the free runner) carry no rule; the key is separate
    json.dump({m["id"]: {k: m[k] for k in ("clip", "team", "t", "rule")} for m in moments}, open(f"{OUT}/moments_key.json", "w"), indent=1)
    for clip in CLIPS:
        mm = [{k: v for k, v in m.items() if k != "rule"} for m in moments if m["clip"] == clip]
        json.dump({"clip": clip, "moments": mm}, open(f"{OUT}/moments_{'sfk' if clip.startswith('SFK') else 'aik'}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
