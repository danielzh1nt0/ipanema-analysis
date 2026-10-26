"""N0-26c (26 Oct, local, free): the app clips' per-player touches on the exact inputs
(results/volume/cache/<clip>/picker_inputs.pkl, steps like ipanema/run.py via n0_26a_clips.frames_of), old count (every
run of carrier frames, new after a 0.4 s gap) vs new (analytics.touch_counts: cleaned contacts >= 0.4 s, the spell state
as run.py's passes use it). Writes results/qa/n0-26c/clips.json and moments.json: a seeded sample of old touches the new
rule drops and of ones it keeps, for the free-runner picture sheets (tools/n0_26c_frames.py). Which rule keeps which
touch is NOT on the sheets ('kind' is opened after grading).
    PYTHONPATH=. python tools/n0_26c_clips.py"""
import sys, os, json, random, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import analytics as AN
import n0_26a_clips as C26

OUT = "results/qa/n0-26c"; N_DROP, N_KEEP = 12, 6
OFFS = (-0.5, -0.25, 0.0, 0.25, 0.5, 1.0)          # tiles around the touch's first frame (3rd tile)


def old_spells(frames_, fps):
    """today's analytics.stats rule as spells: per player, runs of carrier frames, a new one after a gap > 0.4 s"""
    on = {}
    for k, f in enumerate(frames_):
        if f["carrier"] is not None: on.setdefault(f["carrier"], []).append(k)
    out = []
    for tid, v in on.items():
        s = v[0]
        for a, b in zip(v, v[1:]):
            if b - a > int(0.4 * fps): out.append((s, a, tid)); s = b
        out.append((s, v[-1], tid))
    return sorted(out)


def counted(spell, eps, fps):
    """True when a new-rule touch (contact >= TOUCH_MIN_S by the same player) overlaps the old spell"""
    s, e, tid = spell
    return any(c[2] == tid and c[0] <= e and c[1] >= s and (c[1] - c[0] + 1) >= AN.TOUCH_MIN_S * fps for c in eps)


def tiles_of(k0, tid, per, ball, fps):
    out = []
    for j in [k0 + int(round(d * fps)) for d in OFFS]:
        bp = ball.get(j); c = next((r for r in (per.get(j) or []) if r[0] == tid), None)
        out.append({"frame": j, "t": round(j / fps, 2), "ball_px": [float(bp[0]), float(bp[1])] if bp is not None else None,
                    "feet_px": [float(c[3][0]), float(c[3][1])] if c is not None and c[3] is not None else None})
    return out


def main():
    res = {}; moments = []; rng = random.Random(2026)
    for clip, path in C26.C.CLIPS.items():
        frames_, state, cstate, fps, L, per, ball = C26.frames_of(path)
        eps = AN.touch_episodes(frames_, fps, state=cstate); new = AN.touch_counts(eps, fps)
        sp = old_spells(frames_, fps); old = {}
        for s in sp: old[s[2]] = old.get(s[2], 0) + 1
        team = {}
        for f in frames_:
            if f["carrier"] is not None: team[f["carrier"]] = f["team"]
        dropped = [s for s in sp if not counted(s, eps, fps)]; kept = [s for s in sp if counted(s, eps, fps)]
        dur = lambda S: round(float(np.median([(e - s + 1) / fps for s, e, _ in S])), 2) if S else None
        top = sorted(old, key=lambda t: -old[t])[:8]
        res[clip] = {"minutes": round(len(frames_) / fps / 60, 1),
                     "old_total": {tm: sum(v for t, v in old.items() if team.get(t) == tm) for tm in "AB"},
                     "new_total": {tm: sum(v for t, v in new.items() if team.get(t) == tm) for tm in "AB"},
                     "players_old": len(old), "players_new": len(new), "old_spells": len(sp), "dropped": len(dropped), "kept": len(kept),
                     "dropped_median_s": dur(dropped), "kept_median_s": dur(kept),
                     "top8_old_vs_new": [[int(t), old[t], new.get(t, 0)] for t in top]}
        for kind, pool, n in (("dropped", dropped, N_DROP), ("kept", kept, N_KEEP)):
            for s, e, tid in rng.sample(pool, min(n, len(pool))):
                moments.append({"clip": clip, "t": round(s / fps, 2), "kind": kind, "player": int(tid), "team": team.get(tid), "dur_s": round((e - s + 1) / fps, 2),
                                "tiles": tiles_of(s, tid, per, ball, fps)})
        print(clip, json.dumps(res[clip]), flush=True)
    rng.shuffle(moments)
    for i, m in enumerate(moments): m["id"] = i
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/clips.json", "w"), indent=1)
    json.dump(moments, open(f"{OUT}/moments.json", "w"), indent=1)
    return res, moments


if __name__ == "__main__":
    main()
