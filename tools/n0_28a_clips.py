"""N0-28a (28 Oct, local, free): the app clips' reaction labels after a loss (turnover_lost payload.reactions, players[].
counterpress_rate) on the exact inputs (results/volume/cache/<clip>/picker_inputs.pkl, steps like ipanema/run.py:
turnovers -> passes -> confirm_losses), old rule (IPANEMA_REACTION=gap) vs new ('own': the player's own run toward the
ball). Writes results/qa/n0-28a/clips.json and moments.json: a seeded sample of player-reactions whose label changes,
plus unchanged controls, for blind free-runner picture sheets (tools/n0_28a_frames.py). The labels are NOT on the sheets.
    PYTHONPATH=. python tools/n0_28a_clips.py"""
import sys, os, json, random, collections, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import possession as P, analytics as AN
import n0_25e_clips as C, n0_26a_clips as C26

OUT = "results/qa/n0-28a"; N_CHANGED, N_SAME = 12, 4      # per clip
OFFS = (0.0, 0.4, 0.8, 1.2, 1.6, 2.0)                       # tiles over the reaction window (1st = the win)


def reactions(per, frames_, D, mode):
    old = P.REACTION; P.REACTION = mode
    try:
        tvs = P.turnovers(per, frames_, D["cstate"], D["ballm"], D["fps"], D["ar"], 2.0, 5.0, min_before_s=D["pinfo"]["turnover_s"], min_after_s=D["pinfo"]["turnover_s"])
        ps, _ = AN.passes(per, frames_, tvs, {}, D["ar"], D["fps"], state=D["cstate"])
        return P.confirm_losses(tvs, D["cstate"], ps, D["fps"], P.LOSS_KEEP_S)[0]
    finally: P.REACTION = old


def labels(tvs):
    return {(t["frame"], p): c for t in tvs for c, lst in (t.get("reactions") or {}).items() for p in lst}


def tiles_of(k0, tid, per, ball, fps):
    out = []
    for j in [k0 + int(round(d * fps)) for d in OFFS]:
        bp = ball.get(j); r = next((r for r in (per.get(j) or []) if r[0] == tid), None)
        out.append({"frame": j, "t": round(j / fps, 2), "ball_px": [float(bp[0]), float(bp[1])] if bp is not None else None,
                    "player_px": [float(r[3][0]), float(r[3][1])] if r is not None and r[3] is not None else None})
    return out


def main():
    res = {}; moments = []; rng = random.Random(28)
    for clip, path in C.CLIPS.items():
        frames_, state, cstate, fps, L, per, ball = C26.frames_of(path)
        D = C.clip_inputs(path, clip)
        old = reactions(per, frames_, D, "gap"); new = reactions(per, frames_, D, "own")
        assert [t["frame"] for t in old] == [t["frame"] for t in new]
        lo, ln = labels(old), labels(new); assert set(lo) == set(ln)
        trans = collections.Counter(f"{lo[k]}->{ln[k]}" for k in lo)
        rate = lambda L_: {tm: round(100 * sum(1 for (f, p), c in L_.items() if c == "pressed" and next(t["lost_by"] for t in old if t["frame"] == f) == tm) /
                                    max(1, sum(1 for (f, p) in L_ if next(t["lost_by"] for t in old if t["frame"] == f) == tm)), 1) for tm in "AB"}
        res[clip] = {"minutes": round(len(frames_) / fps / 60, 1), "losses": len(old), "losses_with_reactions": sum(1 for t in old if t.get("reactions")),
                     "player_reactions": len(lo), "per_loss_median": float(np.median([sum(len(v) for v in t["reactions"].values()) for t in old if t.get("reactions")])) if lo else None,
                     "old": dict(collections.Counter(lo.values())), "new": dict(collections.Counter(ln.values())), "changes": dict(trans),
                     "pressed_pct_old": rate(lo), "pressed_pct_new": rate(ln)}
        ch = [k for k in lo if lo[k] != ln[k]]; same = [k for k in lo if lo[k] == ln[k]]
        team = {t["frame"]: t["lost_by"] for t in old}
        for kind, pool, n in (("changed", ch, N_CHANGED), ("same", same, N_SAME)):
            for f, tid in rng.sample(pool, min(n, len(pool))):
                moments.append({"clip": clip, "kind": kind, "frame_won": f, "player": tid, "team": team[f], "old": lo[(f, tid)], "new": ln[(f, tid)],
                                "tiles": tiles_of(f, tid, per, ball, fps)})
        print(clip, json.dumps(res[clip]), flush=True)
    rng.shuffle(moments)
    for i, m in enumerate(moments): m["id"] = f"r{i + 1:02d}"
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/clips.json", "w"), indent=1)
    json.dump([{k: v for k, v in m.items() if k not in ("kind", "old", "new")} for m in moments], open(f"{OUT}/moments.json", "w"), indent=1)   # blind: no labels
    json.dump({m["id"]: {k: m[k] for k in ("clip", "kind", "old", "new", "frame_won", "player")} for m in moments}, open(f"{OUT}/key_SEALED.json", "w"), indent=1)


if __name__ == "__main__":
    main()
