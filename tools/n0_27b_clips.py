"""N0-27b (27 Oct, local, free): the app clips' pass quality on the exact inputs (results/volume/cache/<clip>/picker_inputs.pkl,
steps like ipanema/run.py via n0_25e_clips.clip_inputs / n0_26a_clips.frames_of): old (a lost pass has no lane, so no
quality) vs new (IPANEMA_LOST_LANE=mate: intended receiver = the passer's team-mate nearest where the ball was won).
Writes results/qa/n0-27b/clips.json and moments.json: a seeded sample of the lost passes that now get a quality, for the
free-runner picture sheets (tools/n0_27b_frames.py). The quality (lane blocked or open) is NOT on the sheets.
    PYTHONPATH=. python tools/n0_27b_clips.py"""
import sys, os, json, random, collections, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import analytics as AN
import n0_25e_clips as C, n0_26a_clips as C26
from n0_27b_lanes import summarise

OUT = "results/qa/n0-27b"; N_PER_CLIP = 14
OFFS = (-0.5, 0.0, 0.4, 0.8, 1.2, 1.6)          # tiles around the passer's last frame on the ball (2nd tile)


def run(per, frames_, ar, fps, cstate, mode):
    old = AN.LOST_LANE; AN.LOST_LANE = mode
    try:
        ln = AN.lanes(per, frames_, ar, 1.5, 45.0); return AN.passes(per, frames_, [], ln, ar, fps, state=cstate)[0]
    finally: AN.LOST_LANE = old


def feet(per, j, tid):
    r = next((r for r in (per.get(j) or []) if r[0] == tid), None)
    return [float(r[3][0]), float(r[3][1])] if r is not None and r[3] is not None else None


def tiles_of(p, per, ball, fps):
    k = int(round(p["t"] * fps)); out = []
    for j in [k + int(round(d * fps)) for d in OFFS]:
        bp = ball.get(j)
        out.append({"frame": j, "t": round(j / fps, 2), "ball_px": [float(bp[0]), float(bp[1])] if bp is not None else None,
                    "passer_px": feet(per, j, p["from"]), "mate_px": feet(per, j, p["intended_to"]), "winner_px": feet(per, j, p["to"])})
    return out


def main():
    res = {}; moments = []; rng = random.Random(27)
    for clip, path in C.CLIPS.items():
        frames_, state, cstate, fps, L, per, ball = C26.frames_of(path)
        ar = C.clip_inputs(path, clip)["ar"]
        old = run(per, frames_, ar, fps, cstate, "old"); new = run(per, frames_, ar, fps, cstate, "mate")
        assert [(p["t"], p["from"], p["to"]) for p in old] == [(p["t"], p["from"], p["to"]) for p in new]
        same_completed = all(a == b for a, b in zip(old, new) if a["completed"])
        q = lambda ps: dict(collections.Counter(p["quality"] for p in ps))
        def lift(ps):
            v = [p for p in ps if p["lane_open"] is not None]
            return summarise({"completed": [sum(p["completed"] for p in v), sum(p["completed"] and not p["lane_open"] for p in v)],
                              "failed": [sum(not p["completed"] for p in v), sum((not p["completed"]) and not p["lane_open"] for p in v)]})
        lost_new = [p for p in new if not p["completed"] and p["quality"] in ("bad_lost", "execution_error")]
        res[clip] = {"minutes": round(len(frames_) / fps / 60, 1), "passes": len(new), "lost": sum(not p["completed"] for p in new),
                     "old_quality": q(old), "new_quality": q(new), "completed_unchanged": same_completed,
                     "better_option_old": sum(bool(p.get("better_option")) for p in old), "better_option_new": sum(bool(p.get("better_option")) for p in new),
                     "lift_new": lift(new),
                     "intended_dist_m_median": round(float(np.median([np.linalg.norm(np.asarray(p["to_m"]) - np.asarray(
                         next(r[2] for r in per[int(round(p["t"] * fps))] if r[0] == p["intended_to"]))) for p in lost_new
                         if any(r[0] == p["intended_to"] for r in per.get(int(round(p["t"] * fps)), []))])), 1) if lost_new else None}
        for p in rng.sample(lost_new, min(N_PER_CLIP, len(lost_new))):
            moments.append({"clip": clip, "t": p["t"], "passer": p["from"], "mate": p["intended_to"], "winner": p["to"], "team": p["team"],
                            "length_m": p["length_m"], "quality": p["quality"], "tiles": tiles_of(p, per, ball, fps)})
        print(clip, json.dumps(res[clip]), flush=True)
    rng.shuffle(moments)
    for i, m in enumerate(moments): m["id"] = i
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/clips.json", "w"), indent=1)
    json.dump(moments, open(f"{OUT}/moments.json", "w"), indent=1)
    return res, moments


if __name__ == "__main__":
    main()
