"""S2 (9 Oct, worker, local $0): set pieces from player MOTION vs from the ball's dead state, on the FULL SFK-BP first half
as the app exported it (results/volume/runs/matches/SFKBP1109/frames_*.json, fetched 8 Oct), scored against Veo's own event
list (reference/veo_events_SFKBP1109.txt) on the minutes Veo covers inside our half.

- ball rule  = the restarts in the exported match_data.json (today's app: P.restarts on the dead-ball state)
- motion rule = P.stoppages_from_motion on the exported player positions (metres); restart moment = end of the slow spell
- union      = both, a motion restart within UNION_S of a ball restart counts once

Veo minute m covers video [(m-1)*60, m*60) in the first half (validated on the 5-min clip, tools/cliplab.py). A Veo restart
is found when one of ours lands in its minute (+-5 s), one-to-one. Extras = ours in covered minutes not used by a Veo one.
    PYTHONPATH=. python tools/s2full.py            -> results/possession/s2/s2full.json (+ printed table)"""
import sys, os, json, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import possession as P

MATCH = "SFKBP1109"
VEO = "reference/veo_events_SFKBP1109.txt"
KINDS = {"Throw-in": "throw-in", "Corner": "corner", "Goal kick": "goal kick", "Free kick": "free kick"}
COVER = [(11, 31), (44, 45)]          # Veo-covered minutes fully inside our half (kick-off 555 s, half ends 3060 s)
COVER_AMBIG = [(46, 51)]              # Veo counts 46'-51' on without a kick-off: scored separately
UNION_S = 6.0
SETTINGS = [(1.0, 3.0), (1.0, 4.0), (1.4, 3.0), (1.4, 4.0)]


def load_export(match=MATCH, root="results/volume/runs/matches"):
    fr = []
    for f in sorted(glob.glob(f"{root}/{match}/frames_*.json")): fr += json.load(open(f))["frames"]
    md = json.load(open(f"{root}/{match}/match_data.json"))
    p = md["periods"][0]; fr = [f for f in fr if p["t_start"] <= f["t"] <= p["t_end"]]
    return fr, md


def per_from_frames(fr):
    """exported frames -> P-style rows [id, team, metres, px, None, gk]; frames off the calibration give no metres"""
    per = {}
    for k, f in enumerate(fr):
        per[k] = [[q["id"], q["team"], np.array(q["m"], float), None, None, q.get("gk", False)]
                  for q in f.get("players", []) if q.get("m") is not None and q.get("team") in ("A", "B")]
    return per


def veo_truth(path=VEO, cover=COVER):
    out = []
    for line in open(path):
        p = line.split()
        if not p or not p[0].isdigit(): continue
        m = int(p[0]); kind = " ".join(p[2:])
        if kind in KINDS and any(a <= m <= b for a, b in cover):
            out.append({"minute": m, "team": p[1], "kind": KINDS[kind], "t0": (m - 1) * 60.0, "t1": m * 60.0})
    return out


def in_cover(t, cover):
    return any((a - 1) * 60 - 5 <= t < b * 60 + 5 for a, b in cover)


def match(ours_t, truth, tol=5.0):
    """one-to-one: each Veo restart takes the first unused one of ours inside its minute (+-tol). -> (found, used idx)"""
    used = set(); found = []
    for v in truth:
        c = [i for i, t in enumerate(ours_t) if i not in used and v["t0"] - tol <= t < v["t1"] + tol]
        if c: used.add(c[0]); found.append((v["minute"], v["kind"], ours_t[c[0]]))
        else: found.append((v["minute"], v["kind"], None))
    return found, used


def union(ball_t, mot_t, gap=UNION_S):
    out = list(ball_t)
    for t in mot_t:
        if not any(abs(t - b) <= gap for b in ball_t): out.append(t)
    return sorted(out)


def score(ours_t, truth, cover):
    ours_c = [t for t in ours_t if in_cover(t, cover)]
    found, used = match(ours_c, truth)
    return {"veo": len(truth), "ours": len(ours_c), "found": sum(f[2] is not None for f in found),
            "extras": len(ours_c) - len(used), "missed": [(m, k) for m, k, t in found if t is None],
            "extra_t": [round(t, 1) for i, t in enumerate(ours_c) if i not in used]}


def main():
    fr, md = load_export(); ts = np.array([f["t"] for f in fr]); fps = 1.0 / float(np.median(np.diff(ts)))
    per = per_from_frames(fr)
    ball_t = [r["t"] for r in md["restarts"] if md["periods"][0]["t_start"] <= r["t"] <= md["periods"][0]["t_end"]]
    rep = {"match": MATCH, "frames": len(fr), "fps_export": round(fps, 2), "half_s": [md["periods"][0]["t_start"], md["periods"][0]["t_end"]],
           "players_per_frame_median": float(np.median([len(v) for v in per.values()])), "restarts_ball_half": len(ball_t), "settings": {}}
    T = veo_truth(cover=COVER); TA = veo_truth(cover=COVER_AMBIG)
    rep["veo_restarts_cover"] = len(T); rep["veo_restarts_ambig"] = len(TA)
    rep["ball"] = {"cover": score(ball_t, T, COVER), "ambig": score(ball_t, TA, COVER_AMBIG)}
    for thr, min_s in SETTINGS:
        st = P.stoppages_from_motion(per, fps, thr=thr, min_s=min_s)
        mot_t = [float(ts[min(b, len(ts) - 1)]) for a, b in st]
        key = f"thr{thr}_min{min_s}"
        rep["settings"][key] = {"stoppages_half": len(mot_t),
                                "motion": {"cover": score(mot_t, T, COVER), "ambig": score(mot_t, TA, COVER_AMBIG)},
                                "union": {"cover": score(union(ball_t, mot_t), T, COVER), "ambig": score(union(ball_t, mot_t), TA, COVER_AMBIG)},
                                "motion_t": [round(t, 1) for t in mot_t],
                                "motion_spans": [(round(float(ts[a]), 1), round(float(ts[min(b, len(ts) - 1)]), 1)) for a, b in st]}
    rep["ball_t"] = [round(t, 1) for t in ball_t]
    os.makedirs("results/possession/s2", exist_ok=True)
    json.dump(rep, open("results/possession/s2/s2full.json", "w"), indent=1)
    print(f"Veo restarts in covered minutes {len(T)} (+{len(TA)} in 46'-51')")
    print("rule | ours | found | extras || 46-51: ours found extras")
    def row(name, s): print(f"{name:22s} | {s['cover']['ours']} | {s['cover']['found']}/{s['cover']['veo']} | {s['cover']['extras']} || {s['ambig']['ours']} {s['ambig']['found']}/{s['ambig']['veo']} {s['ambig']['extras']}")
    row("ball (app)", rep["ball"])
    for k, v in rep["settings"].items():
        row("motion " + k, v["motion"]); row("union  " + k, v["union"])
    return rep


if __name__ == "__main__":
    main()
