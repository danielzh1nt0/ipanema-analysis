"""28 Sep: stoppages from player movement, scored on Metrica's hand-labelled stoppages (clean and with our tracking noise)."""
import sys, os, json, random, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metricalab as ML
from ipanema import possession as P
FPS = ML.FPS

def truth_stops(ev, per):
    """a stoppage runs from the end of the event before a restart to the restart (kick-offs excluded: they follow goals/half starts)"""
    E = sorted([e for e in ev if e["period"] == per], key=lambda e: e["f0"]); out = []
    for i, e in enumerate(E):
        if (e["type"] == "SET PIECE" and e["sub"] != "KICK OFF") or (e["type"] == "PASS" and e["sub"] == "GOAL KICK"):
            prev = E[i - 1] if i else None; a = max(prev["f1"] or prev["f0"], prev["f0"]) if prev else e["f0"] - 50
            out.append((a, e["f0"]))
    return out

def players(frames, per, noisy, seed=0):
    rng = random.Random(seed); ks = sorted(k for k in frames if frames[k][0] == per); perd = {}; off = {}
    for i, k in enumerate(ks):
        rows = [[pid, "A" if pid < 200 else "B", np.array([x, y]), None, None, False] for pid, x, y, in frames[k][1]]
        if noisy:
            rows = [r for r in rows if rng.random() < 0.65]
            for r in rows:                                                     # slowly drifting error, like a real tracker
                o = off.get(r[0], np.zeros(2)); o = 0.95 * o + np.array([rng.gauss(0, 0.25), rng.gauss(0, 0.25)]); off[r[0]] = o; r[2] = r[2] + o
        perd[i] = rows
    return perd, ks[0]

def score(found, truth, k0, n):
    tr = [(a - k0, b - k0) for a, b in truth]; hit = sum(any(fa <= b + 2 * FPS and fb >= a - 2 * FPS for fa, fb in found) for a, b in tr)
    extra = sum(not any(fa <= b + 2 * FPS and fb >= a - 2 * FPS for a, b in tr) for fa, fb in found)
    dt = np.zeros(n, bool); df = np.zeros(n, bool)
    for a, b in tr: dt[max(0, a):max(0, b)] = True
    for a, b in found: df[a:b + 1] = True
    return {"found": f"{hit}/{len(tr)}", "extra": extra, "true_dead_pct": round(100 * dt.mean()), "our_dead_pct": round(100 * df.mean()), "frame_agreement_pct": round(100 * (dt == df).mean())}

if __name__ == "__main__":
    M = sys.argv[1]; cfgs = [(1.2, 4.0), (1.4, 4.0), (1.6, 4.0), (1.4, 3.0), (1.6, 3.0), (1.8, 3.0)]; rep = []
    for g in ("Sample_Game_1", "Sample_Game_2"):
        d = f"{M}/{g}"; home = ML.load_tracking(f"{d}/{g}_RawTrackingData_Home_Team.csv", 100); away = ML.load_tracking(f"{d}/{g}_RawTrackingData_Away_Team.csv", 200)
        frames = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home}; ev = ML.load_events(f"{d}/{g}_RawEventsData.csv")
        for per in (1, 2):
            T = truth_stops(ev, per)
            for noisy in (False, True):
                perd, k0 = players(frames, per, noisy)
                for thr, ms in cfgs:
                    f = P.stoppages_from_motion(perd, FPS, thr, ms); r = dict(game=g[-1], period=per, noisy=noisy, thr=thr, min_s=ms, **score(f, T, k0, len(perd)))
                    rep.append(r); print(json.dumps(r), flush=True)
    os.makedirs("results/metrica", exist_ok=True); json.dump(rep, open("results/metrica/stoppages.json", "w"), indent=1)
