"""S1b (28 Sep): balls lost on Metrica's free pro games. Our turnovers vs their hand-labelled BALL LOST events, matched in
time (a loss counts as found if ours is within TOL s and the same team), for different 'real possession' limits.
Clean positions and realistic noise ('streak' = errors in runs, like our Veo tracking). Free, CPU.
    python tools/turnoverlab.py <metrica data dir with Sample_Game_1, Sample_Game_2>"""
import sys, os, json, random, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import possession as P, analytics as AN
import tools.metricalab as ML
FPS = ML.FPS; TOL = 2.0

def prepare(frames, per, noisy, seed=0, speed_win_s=None):
    """the expensive part of metricalab.ours (positions -> carriers -> possession state), done once per setting"""
    rng = random.Random(seed); ks = sorted(k for k in frames if frames[k][0] == per); k0 = ks[0]
    perd, ballm, H, run = {}, {}, {}, [0, "ok", None]
    for i, k in enumerate(ks):
        _, ps, b = frames[k]
        rows = [[pid, "A" if pid < 200 else "B", np.array([x, y]), None, None, False] for pid, x, y in ps]
        if noisy:
            rows = [r for r in rows if rng.random() < 0.65]
            for r in rows: r[2] = r[2] + np.array([rng.gauss(0, 0.8), rng.gauss(0, 0.8)])
        perd[i] = rows; H[i] = np.eye(3)
        if noisy:
            if run[0] <= 0:
                u = rng.random(); run[1] = "miss" if u < 0.25 else "wrong" if u < 0.35 else "ok"
                run[0] = int(rng.expovariate(1 / ({"miss": 0.6, "wrong": 0.4, "ok": 1.2}[run[1]] * FPS))) + 1
                run[2] = np.array([rng.uniform(-25, 25), rng.uniform(-15, 15)])
            run[0] -= 1
        if b is not None:
            bm = np.array(b)
            if noisy:
                if run[1] == "miss": continue
                bm = bm + (run[2] if run[1] == "wrong" else np.array([rng.gauss(0, 0.7), rng.gauss(0, 0.7)]))
            ballm[i] = bm
    frames_, bm = P.carriers(perd, dict(ballm), H)
    state, _ = P.viterbi(perd, P.clean_ball(bm, ML.L, ML.W), FPS, ML.L, ML.W, speed_win_s=speed_win_s)
    ar = P.direction_from_keepers(perd, ML.L, log=lambda *a: None) or {"A": True, "B": False}
    return dict(k0=k0, perd=perd, frames_=frames_, state=state, bm=bm, ar=ar)

def match(truth, ours, tol=TOL * FPS):
    """truth/ours: [(frame, team)] -> found (truth matched by an unused same-team 'ours' within tol)"""
    used = set(); hit = 0
    for f, t in sorted(truth):
        c = [(abs(g - f), i) for i, (g, u) in enumerate(ours) if u == t and i not in used and abs(g - f) <= tol]
        if c: used.add(min(c)[1]); hit += 1
    return hit

if __name__ == "__main__":
    root = sys.argv[1]; settings = [(3.0, 3.0), (2.0, 2.0), (1.5, 1.5), (1.0, 1.0), (1.0, 2.0), (0.5, 1.0), (0.5, 0.5)]
    res = []
    for g in ("Sample_Game_1", "Sample_Game_2"):
        d = f"{root}/{g}"
        home = ML.load_tracking(f"{d}/{g}_RawTrackingData_Home_Team.csv", 100); away = ML.load_tracking(f"{d}/{g}_RawTrackingData_Away_Team.csv", 200)
        frames = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home}
        ev = ML.load_events(f"{d}/{g}_RawEventsData.csv")
        for per in (1, 2):
            tl = [(e["f0"], "A" if e["team"] == "Home" else "B") for e in ev if e["period"] == per and e["type"] == "BALL LOST"]
            tp = {t: sum(1 for e in ev if e["period"] == per and e["type"] == "PASS" and e["team"] == t) for t in ("Home", "Away")}
            for noisy in (False, "streak"):
                S = prepare(frames, per, noisy)
                for bef, aft in settings:
                    tvs = P.turnovers(S["perd"], S["frames_"], S["state"], S["bm"], FPS, S["ar"], min_before_s=bef, min_after_s=aft)
                    ol = [(S["k0"] + t["frame_lost"], t["lost_by"]) for t in tvs]
                    ps, _ = AN.passes(S["perd"], S["frames_"], tvs, {}, S["ar"], FPS, min_touch_s=0.6, floor_s=0.08, sandwich=True)
                    op = {"Home": sum(p["completed"] for p in ps if p["team"] == "A"), "Away": sum(p["completed"] for p in ps if p["team"] == "B")}
                    hit = match(tl, ol)
                    r = {"game": g, "half": per, "noise": noisy or "clean", "before_s": bef, "after_s": aft, "truth_lost": len(tl), "ours_lost": len(ol),
                         "found": hit, "recall": round(hit / max(1, len(tl)), 2), "precision": round(hit / max(1, len(ol)), 2),
                         "passes_truth": tp, "passes_ours": op, "passes_err_pct": round(100 * (sum(op.values()) - sum(tp.values())) / sum(tp.values()))}
                    res.append(r); print(json.dumps(r), flush=True)
    os.makedirs("results/metrica", exist_ok=True); json.dump(res, open("results/metrica/turnoverlab.json", "w"), indent=1)
    print("\nsummary (all 4 halves):")
    for noisy in ("clean", "streak"):
        for bef, aft in settings:
            R = [r for r in res if r["noise"] == noisy and r["before_s"] == bef and r["after_s"] == aft]
            T, O, H = sum(r["truth_lost"] for r in R), sum(r["ours_lost"] for r in R), sum(r["found"] for r in R)
            pe = [abs(r["passes_err_pct"]) for r in R]
            print(f"{noisy:6s} before {bef:>3} s after {aft:>3} s | lost: truth {T} ours {O} found {H} (recall {H/T:.0%}, precision {H/max(1,O):.0%}) | passes off {min(pe)}-{max(pe)}%")
