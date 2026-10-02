"""S2 (3 Oct): set pieces from player MOTION (P.stoppages_from_motion: everybody slows for a few seconds) vs from the ball's
dead state (today's P.restarts), on Metrica's two pro games where every set piece is labelled. A stoppage's END is our
restart moment; a labelled set piece counts as found when one of ours ends within +-tol. Sweep thr / min_s.
    PYTHONPATH=. python tools/s2lab.py <metrica data dir>   -> results/possession/s2lab_<date>.json"""
import sys, os, json, time, itertools, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "tools")
from metricalab import load_tracking, load_events, truth, match_frames, L, W, FPS
from ipanema import possession as P
D = sys.argv[1]; out = {}
for g in ("Sample_Game_1", "Sample_Game_2"):
    d = f"{D}/{g}"; home = load_tracking(f"{d}/{g}_RawTrackingData_Home_Team.csv", 100); away = load_tracking(f"{d}/{g}_RawTrackingData_Away_Team.csv", 200)
    frames = {k: (home[k][0], home[k][1] + away[k][1]) for k in home}; byper = {}
    for k, (p, _) in frames.items(): byper.setdefault(p, []).append(k)
    T = truth(load_events(f"{d}/{g}_RawEventsData.csv"), byper)
    for per in sorted(T):
        ks = sorted(byper[per]); k0 = ks[0]
        perd = {k - k0: [[pid, "A" if pid < 200 else "B", np.array([x, y]), None, None, False] for pid, x, y in frames[k][1]] for k in ks}
        for k in range(len(ks)): perd.setdefault(k, [])
        truth_f = T[per]["set_piece_frames"]
        for thr, min_s in itertools.product((1.0, 1.4, 1.8, 2.2), (2.0, 3.0, 4.0, 6.0)):
            st = P.stoppages_from_motion(perd, FPS, thr=thr, min_s=min_s)
            ends = [k0 + b for a, b in st]; hit = match_frames(truth_f, ends, tol=8 * FPS)
            out[f"{g}|p{per}|thr{thr}|min{min_s}"] = {"truth": len(truth_f), "ours": len(ends), "found": hit}
    print(g, "done", flush=True)
# summary per setting over all 4 halves
summ = {}
for key, v in out.items():
    s = key.split("|", 2)[2]; a = summ.setdefault(s, {"truth": 0, "ours": 0, "found": 0})
    for k2 in a: a[k2] += v[k2]
rows = sorted(summ.items(), key=lambda kv: -(kv[1]["found"] / max(1, kv[1]["truth"]) - 0.5 * max(0, kv[1]["ours"] - kv[1]["found"]) / max(1, kv[1]["truth"])))
print("setting | labelled set pieces | ours | found (recall) | extras")
for s, a in rows: print(f"{s:14s} | {a['truth']} | {a['ours']} | {a['found']} ({100 * a['found'] // max(1, a['truth'])}%) | {a['ours'] - a['found']}")
json.dump({"per_half": out, "summary": summ}, open(f"results/possession/s2lab_{time.strftime('%Y-%m-%d')}.json", "w"), indent=1)
