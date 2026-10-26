"""N0-26d (26 Oct, local, free): the app clips' per-player time on the ball on the exact inputs (as n0_26c_clips), old
(every carrier frame) vs new (analytics.touch_time: carrier frames inside the cleaned contacts). Also: players shown
with time on the ball but 0 touches (the table disagreeing with itself), and the share of the old seconds that sit in
spells the N0-26c blind sheets graded (results/qa/n0-26c/grades.json). -> results/qa/n0-26d/clips.json
    PYTHONPATH=. python tools/n0_26d_clips.py"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import analytics as AN
import n0_26a_clips as C26
from n0_26d_onball import onball_old

OUT = "results/qa/n0-26d"


def main():
    res = {}
    for clip, path in C26.C.CLIPS.items():
        frames_, state, cstate, fps, L, per, ball = C26.frames_of(path)
        eps = AN.touch_episodes(frames_, fps, state=cstate); tc = AN.touch_counts(eps, fps)
        old = onball_old(frames_, fps); new = AN.touch_time(frames_, eps, fps)
        team = {}
        for f in frames_:
            if f["carrier"] is not None: team[int(f["carrier"])] = f["team"]
        poss = {tm: round(float((np.asarray(cstate) == i).sum()) / fps, 1) for i, tm in enumerate("AB")}
        res[clip] = {"minutes": round(len(frames_) / fps / 60, 1), "team_possession_s": poss,
                     "old_s": {tm: round(sum(v for t, v in old.items() if team.get(t) == tm), 1) for tm in "AB"},
                     "new_s": {tm: round(sum(v for t, v in new.items() if team.get(t) == tm), 1) for tm in "AB"},
                     "players_with_time_old": sum(1 for v in old.values() if v > 0), "players_with_time_new": sum(1 for v in new.values() if v > 0),
                     "time_but_no_touch_old": sum(1 for t, v in old.items() if v > 0 and not tc.get(t)),
                     "time_but_no_touch_new": sum(1 for t, v in new.items() if v > 0 and not tc.get(t)),
                     "top5_old_vs_new_s": [[t, round(old[t], 1), round(new.get(t, 0), 1)] for t in sorted(old, key=lambda t: -old[t])[:5]]}
        print(clip, json.dumps(res[clip]), flush=True)
    os.makedirs(OUT, exist_ok=True); json.dump(res, open(f"{OUT}/clips.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    main()
