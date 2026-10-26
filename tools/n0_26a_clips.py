"""N0-26a (26 Oct, local, free): the app clips' pressure points (metrics.pressure_points) on the exact inputs
(results/volume/cache/<clip>/picker_inputs.pkl, steps like ipanema/run.py via n0_25e_clips.clip_inputs), old rule (raw
nearest-player carrier) vs new (only while the spell state gives the carrier's team the ball). Writes
results/qa/n0-26a/clips.json (counts per team, per 5 min, thirds) and moments.json (a seeded sample of dropped and kept
points for the free-runner picture sheets, tools/n0_26a_frames.py).
    PYTHONPATH=. python tools/n0_26a_clips.py"""
import sys, os, json, random, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import possession as P, metrics as M, ball as BL
import n0_25e_clips as C

OUT = "results/qa/n0-26a"; q = lambda *a: None
N_DROP, N_KEEP = 12, 6           # per clip, for the blind sheets


def frames_of(path):
    """carriers like run.py (the same ball as n0_25e_clips) + the spell state"""
    import pickle
    I = pickle.load(open(path, "rb")); fps, H, L, W = I["fps"], I["H"], I["L"], I["W"]
    per = {k: [[r[0], r[1], np.asarray(r[2]), None if r[3] is None else np.asarray(r[3]), None, False] for r in v] for k, v in I["per"].items()}
    ball = BL.bridge(BL.pick_v2(I["cands"], H, L, W, per=per, fps=fps, log=q), fps)
    frames_, ballm = P.carriers(per, ball, H, 2.5, 5.0); state, *_ = P.pipeline_state(per, ball, ballm, H, fps, L, W, log=q)
    cstate = P.spell_state(state, fps, P.SPELL_TAKE_S, P.SPELL_JOIN_S)
    return frames_, np.asarray(state), np.asarray(cstate), fps, L, per, ball


def tiles_of(k0, carrier, tm, per, ball, fps):
    """6 tiles every 0.5 s from 1 s before to 1.5 s after (3rd tile = the point): app ball pick = red ring, yellow line =
    the marked player's feet -> the nearest opponent's feet (pixels)"""
    out = []
    for j in [k0 + int(round(d * fps)) for d in (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5)]:
        bp = ball.get(j); rows = per.get(j) or []; c = next((r for r in rows if r[0] == carrier), None); line = None
        if c is not None and c[3] is not None:
            opp = [r for r in rows if r[1] != tm and r[1] in ("A", "B") and r[3] is not None]
            if opp:
                o = min(opp, key=lambda r: float(np.linalg.norm(np.asarray(r[2]) - np.asarray(c[2]))))
                line = [[float(c[3][0]), float(c[3][1])], [float(o[3][0]), float(o[3][1])]]
        out.append({"frame": j, "t": round(j / fps, 1), "ball_px": [float(bp[0]), float(bp[1])] if bp is not None else None, "line_px": line})
    return out


def per5(pts):
    c = {"A": [0] * 1, "B": [0] * 1}
    for p in pts:
        b = int(p["t"] // 300)
        for tm in "AB":
            while len(c[tm]) <= b: c[tm].append(0)
        c[p["pressing_team"]][b] += 1
    return c


def main():
    res = {}; moments = []; rng = random.Random(26)
    for clip, path in C.CLIPS.items():
        frames_, state, cstate, fps, L, per, ball = frames_of(path)
        D = C.clip_inputs(path, clip); ar = D["ar"]
        old = M.pressure_points(None, frames_, fps); new = M.pressure_points(None, frames_, fps, state=cstate)
        key = lambda p: (p["t"], p["carrier_team"]); nk = {key(p) for p in new}
        dropped = [p for p in old if key(p) not in nk]; added = [p for p in new if key(p) not in {key(o) for o in old}]
        def thirds(pts, tm):
            s = [0, 0, 0]
            for p in pts:
                if p["pressing_team"] != tm: continue
                x = p["x_m"] if ar[tm] else L - p["x_m"]; s[min(2, max(0, int(3 * x / L)))] += 1
            n = sum(s); return [round(100 * v / n) for v in s] if n else None
        res[clip] = {"old": {tm: sum(p["pressing_team"] == tm for p in old) for tm in "AB"}, "new": {tm: sum(p["pressing_team"] == tm for p in new) for tm in "AB"},
                     "dropped": len(dropped), "added": len(added), "old_per5": per5(old), "new_per5": per5(new),
                     "thirds_old": {tm: thirds(old, tm) for tm in "AB"}, "thirds_new": {tm: thirds(new, tm) for tm in "AB"},
                     "dropped_state": {s: sum(1 for p in dropped if P.STATES[cstate[int(round(p["t"] * fps))]] == s) for s in ("A", "B", "loose", "dead")}}
        for kind, pool, n in (("dropped", dropped, N_DROP), ("kept", [p for p in new if key(p) in {key(o) for o in old}], N_KEEP)):
            for p in rng.sample(pool, min(n, len(pool))):
                k = int(round(p["t"] * fps)); f = frames_[k]
                moments.append({"clip": clip, "t": p["t"], "kind": kind, "carrier": f["carrier"], "carrier_team": p["carrier_team"], "x_m": p["x_m"], "y_m": p["y_m"],
                                "state": P.STATES[cstate[k]], "tiles": tiles_of(k, f["carrier"], p["carrier_team"], per, ball, fps)})
        print(clip, json.dumps({k: v for k, v in res[clip].items() if "per5" not in k}))
    rng.shuffle(moments)
    for i, m in enumerate(moments): m["id"] = i
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/clips.json", "w"), indent=1)
    json.dump(moments, open(f"{OUT}/moments.json", "w"), indent=1)
    return res, moments


if __name__ == "__main__":
    main()
