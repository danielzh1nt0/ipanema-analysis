"""Stats answer key without anyone clicking (28 Sep): Metrica Sports' free sample games have every player and the ball
at 25 fps plus hand-labelled events. We feed their positions through OUR stats code (possession, set pieces, turnovers,
passes, dead time) and compare with their labels. This measures the stats LOGIC with perfect inputs; with --noisy the
ball and players are degraded to what our Veo tracking actually delivers, which predicts what a coach would see.

    python tools/metricalab.py <metrica data dir> [--noisy]
"""
import sys, os, csv, json, random, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import possession as P, analytics as AN

L, W, FPS = 105.0, 68.0, 25.0

def load_tracking(path, team):
    rows = list(csv.reader(open(path))); hdr = rows[2]; out = {}
    cols = [(i, hdr[i]) for i in range(3, len(hdr) - 2, 2) if hdr[i].startswith("Player")]
    for r in rows[3:]:
        per, fr = int(r[0]), int(r[1]); ps = []
        for i, name in cols:
            if r[i] in ("NaN", "") or r[i + 1] in ("NaN", ""): continue
            ps.append((team + int(name[6:]), float(r[i]) * L, float(r[i + 1]) * W))
        b = None if r[-2] in ("NaN", "") else (float(r[-2]) * L, float(r[-1]) * W)
        out[fr] = (per, ps, b)
    return out

def load_events(path):
    ev = []
    for r in csv.DictReader(open(path)):
        ev.append({"team": r["Team"], "type": r["Type"], "sub": r["Subtype"], "period": int(r["Period"]), "f0": int(r["Start Frame"]), "f1": int(r["End Frame"] or 0)})
    return ev

def truth(ev, frames):
    """per period: set pieces (incl. goal kicks), passes (completed), balls lost, possession share, dead share"""
    out = {}
    for per in sorted({e["period"] for e in ev}):
        E = [e for e in ev if e["period"] == per]; f0, f1 = min(frames[per]), max(frames[per])
        sp = [e for e in E if e["type"] == "SET PIECE" or (e["type"] == "PASS" and e["sub"] == "GOAL KICK")]
        own = {}; last = None; dead = np.zeros(f1 - f0 + 1, bool); out_at = None
        for e in sorted(E, key=lambda e: e["f0"]):
            if e["type"] == "BALL OUT": out_at = e["f0"]
            if out_at is not None and (e["type"] == "SET PIECE" or e["sub"] == "GOAL KICK"): dead[out_at - f0:e["f0"] - f0] = True; out_at = None
        seq = sorted(E, key=lambda e: e["f0"]); share = {"Home": 0, "Away": 0}
        for a, b in zip(seq, seq[1:]):
            if a["type"] in ("PASS", "CARRY", "RECOVERY", "SET PIECE", "SHOT") and a["team"] in share:
                share[a["team"]] += max(0, min(b["f0"], f1) - a["f0"] - int(dead[max(0, a["f0"] - f0):max(0, b["f0"] - f0)].sum()))
        tot = max(1, sum(share.values()))
        out[per] = {"set_pieces": len(sp), "set_piece_frames": [e["f0"] for e in sp],
                    "passes": {t: sum(1 for e in E if e["type"] == "PASS" and e["team"] == t) for t in ("Home", "Away")},
                    "balls_lost": {t: sum(1 for e in E if e["type"] == "BALL LOST" and e["team"] == t) for t in ("Home", "Away")},
                    "possession_pct": {t: round(100 * v / tot) for t, v in share.items()}, "dead_pct": round(100 * dead.mean())}
    return out

def ours(frames, per, noisy=False, seed=0, min_touch_s=0.6, floor_s=0.08, sandwich=True):
    rng = random.Random(seed); ks = sorted(k for k in frames if frames[k][0] == per); k0 = ks[0]
    perd = {}; ballm = {}; H = {}; run = [0, "ok", None]
    for i, k in enumerate(ks):
        _, ps, b = frames[k]
        rows = [[pid, "A" if pid < 200 else "B", np.array([x, y]), None, None, False] for pid, x, y in ps]
        if noisy:                                                              # what our Veo follow-cam delivers (measured 27-28 Sep)
            rows = [r for r in rows if rng.random() < 0.65]                     # ~7 of 11 per team visible
            for r in rows: r[2] = r[2] + np.array([rng.gauss(0, 0.8), rng.gauss(0, 0.8)])
        perd[i] = rows; H[i] = np.eye(3)
        if noisy == "streak":                                                  # errors come in runs, like real tracking
            if run[0] <= 0:
                u = rng.random(); run[1] = "miss" if u < 0.25 else "wrong" if u < 0.35 else "ok"
                run[0] = int(rng.expovariate(1 / ({"miss": 0.6, "wrong": 0.4, "ok": 1.2}[run[1]] * FPS))) + 1
                run[2] = np.array([rng.uniform(-25, 25), rng.uniform(-15, 15)])
            run[0] -= 1
        if b is not None:
            bm = np.array(b)
            if noisy == "streak":
                if run[1] == "miss": continue
                bm = bm + (run[2] if run[1] == "wrong" else np.array([rng.gauss(0, 0.7), rng.gauss(0, 0.7)]))
            elif noisy:
                u = rng.random()
                if u < 0.25: continue                                           # ball not found
                if u < 0.35: bm = bm + np.array([rng.uniform(-25, 25), rng.uniform(-15, 15)])   # wrong pick
                else: bm = bm + np.array([rng.gauss(0, 0.7), rng.gauss(0, 0.7)])
            ballm[i] = bm
    ball_px = {k: v for k, v in ballm.items()}                                 # H = identity: "pixels" are metres
    frames_, bm = P.carriers(perd, ball_px, H)
    state, bspeed = P.viterbi(perd, P.clean_ball(bm, L, W), FPS, L, W)
    ar = P.direction_from_keepers(perd, L, log=lambda *a: None) or {"A": True, "B": False}
    rst = P.restarts(state, bm, FPS, L, W)
    tvs = P.turnovers(perd, frames_, state, bm, FPS, ar)
    ps, _ = AN.passes(perd, frames_, tvs, {}, ar, FPS, min_touch_s=min_touch_s, floor_s=floor_s, sandwich=sandwich)
    st = np.asarray(state)
    live = np.isin(st, [0, 1]).sum()
    return {"set_pieces": len(rst), "set_piece_frames": [k0 + int(r["t"] * FPS) for r in rst],
            "passes": {"Home": sum(p["completed"] for p in ps if p["team"] == "A"), "Away": sum(p["completed"] for p in ps if p["team"] == "B")},
            "balls_lost": {"Home": sum(t["lost_by"] == "A" for t in tvs), "Away": sum(t["lost_by"] == "B" for t in tvs)},
            "possession_pct": {"Home": round(100 * (st == 0).sum() / max(1, live)), "Away": round(100 * (st == 1).sum() / max(1, live))},
            "dead_pct": round(100 * (st == 3).mean())}

def match_frames(a, b, tol=5 * FPS):
    used = set(); hit = 0
    for f in a:
        c = [i for i, g in enumerate(b) if i not in used and abs(g - f) <= tol]
        if c: used.add(c[0]); hit += 1
    return hit

if __name__ == "__main__":
    d = sys.argv[1]; noisy = "--noisy" in sys.argv; g = os.path.basename(d.rstrip("/"))
    home = load_tracking(f"{d}/{g}_RawTrackingData_Home_Team.csv", 100); away = load_tracking(f"{d}/{g}_RawTrackingData_Away_Team.csv", 200)
    frames = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home}
    byper = {}
    for k, (p, _, _) in frames.items(): byper.setdefault(p, []).append(k)
    T = truth(load_events(f"{d}/{g}_RawEventsData.csv"), byper); rep = {}
    for per in sorted(T):
        O = ours(frames, per, noisy)
        hit = match_frames(T[per]["set_piece_frames"], O["set_piece_frames"])
        rep[per] = {"truth": {k: v for k, v in T[per].items() if k != "set_piece_frames"}, "ours": {k: v for k, v in O.items() if k != "set_piece_frames"},
                    "set_pieces_found": f"{hit}/{T[per]['set_pieces']}", "set_pieces_extra": O["set_pieces"] - hit}
        print(g, "period", per, "noisy" if noisy else "clean", json.dumps(rep[per]), flush=True)
    os.makedirs("results/metrica", exist_ok=True)
    json.dump(rep, open(f"results/metrica/{g}_{'noisy' if noisy else 'clean'}.json", "w"), indent=1)
