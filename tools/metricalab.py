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
FLICK_P, FLICK_R, FLICK_S = 0.02, 10.0, (0.3, 0.6)   # S6 "flicker" noise: chance per frame that the ball dot hops to the nearest opponent (within 10 m of the ball) for 0.3-0.6 s. Metrica: 17 raw team switches per min, raw state agrees with the clean one on 78% of held frames (our clips: 82/99 who-has-the-ball, 35-41 switches). Heavy test (tools/s6lab.py): 0.1, 10, (0.3, 1.0) = 28 per min, 48%

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

def truth_sequences(ev, per):
    """E5: chains of on-ball events (PASS, CARRY, SHOT, SET PIECE, RECOVERY, BALL LOST) by one team; a chain ends when
    the other team acts or the ball goes out -> [{team A/B, start, end}] in Metrica frame numbers"""
    E = sorted([e for e in ev if e["period"] == per], key=lambda e: (e["f0"], e["f1"])); out = []; cur = None
    for e in E:
        if e["type"] == "BALL OUT":
            if cur: out.append(cur); cur = None
            continue
        if e["type"] not in ("PASS", "CARRY", "SHOT", "SET PIECE", "RECOVERY", "BALL LOST") or e["team"] not in ("Home", "Away"): continue
        tm = "A" if e["team"] == "Home" else "B"
        if cur and cur["team"] == tm: cur["end"] = max(cur["end"], e["f1"] or e["f0"]); continue
        if cur: out.append(cur)
        cur = {"team": tm, "start": e["f0"], "end": max(e["f0"], e["f1"] or e["f0"])}
    if cur: out.append(cur)
    return out

def ours(frames, per, noisy=False, seed=0, min_touch_s=0.6, floor_s=0.08, sandwich=True, mode=None, spell_take=None, spell_join=None, loss_keep_s=None):
    """S9 (8 Oct): loss_keep_s = P.confirm_losses on the balls lost (None = the pipeline default, 0 = off).
    S6 (2 Oct): sequences and balls lost read P.spell_state like ipanema/run.py since S8 (spell_take=0 = raw state,
    None = the pipeline default). E5 (29 Sep): the possession state comes from P.pipeline_state, as in ipanema/run.py (default possession_simple,
    'feet' = metres because H = identity; mode='viterbi' or IPANEMA_POSSESSION=viterbi = the old model, which the
    nightly scored until 29 Sep). Set pieces and dead time come from the old model's dead runs, as in the pipeline."""
    rng = random.Random(seed); ks = sorted(k for k in frames if frames[k][0] == per); k0 = ks[0]
    perd = {}; ballm = {}; H = {}; run = [0, "ok", None]
    for i, k in enumerate(ks):
        _, ps, b = frames[k]
        rows = [[pid, "A" if pid < 200 else "B", np.array([x, y]), None, None, False] for pid, x, y in ps]
        if noisy and noisy != "flicker":                                       # what our Veo follow-cam delivers (measured 27-28 Sep)
            rows = [r for r in rows if rng.random() < 0.65]                     # ~7 of 11 per team visible
            for r in rows: r[2] = r[2] + np.array([rng.gauss(0, 0.8), rng.gauss(0, 0.8)])
        for r in rows: r[3] = r[2]                                             # 'pixel' feet = metres (H = identity)
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
            elif noisy == "flicker":                                             # S6: the ball dot hops to a nearby other player for
                if run[0] <= 0 and rng.random() < FLICK_P:                       # 0.3-1.0 s (our clips: state flips 27-33 per minute)
                    run[0] = int(rng.uniform(FLICK_S[0], FLICK_S[1]) * FPS) + 1
                if run[0] > 0:                                                   # hop to the nearest player of the OTHER team than
                    run[0] -= 1; d = sorted((float(np.hypot(x - b[0], y - b[1])), pid, x, y) for pid, x, y in ps)   # the one on the ball
                    o = next((q for q in d[1:] if (q[1] < 200) != (d[0][1] < 200)), None) if d else None
                    if o is not None and o[0] < FLICK_R: bm = np.array([o[2], o[3]])
                bm = bm + np.array([rng.gauss(0, 0.7), rng.gauss(0, 0.7)])
            elif noisy:
                u = rng.random()
                if u < 0.25: continue                                           # ball not found
                if u < 0.35: bm = bm + np.array([rng.uniform(-25, 25), rng.uniform(-15, 15)])   # wrong pick
                else: bm = bm + np.array([rng.gauss(0, 0.7), rng.gauss(0, 0.7)])
            ballm[i] = bm
    ball_px = {k: v for k, v in ballm.items()}                                 # H = identity: "pixels" are metres
    frames_, bm = P.carriers(perd, ball_px, H)
    state, bspeed, dstate, info = P.pipeline_state(perd, {k: tuple(v) for k, v in ball_px.items()}, bm, H, FPS, L, W, mode=mode, log=lambda *a: None)
    ar = P.direction_from_keepers(perd, L, log=lambda *a: None) or {"A": True, "B": False}
    rst = P.restarts(dstate, bm, FPS, L, W)
    cstate = P.spell_state(state, FPS, P.SPELL_TAKE_S if spell_take is None else spell_take, P.SPELL_JOIN_S if spell_join is None else spell_join)
    tvs = P.turnovers(perd, frames_, cstate, bm, FPS, ar, min_before_s=info["turnover_s"], min_after_s=info["turnover_s"])
    seqs = P.sequences(cstate, bm, bspeed, FPS, L, ar, **info["seq"])
    ps, _ = AN.passes(perd, frames_, tvs, {}, ar, FPS, min_touch_s=min_touch_s, floor_s=floor_s, sandwich=sandwich)
    tvs, _drop = P.confirm_losses(tvs, cstate, ps, FPS, P.LOSS_KEEP_S if loss_keep_s is None else loss_keep_s)
    st = np.asarray(state)
    live = np.isin(st, [0, 1]).sum()
    return {"set_pieces": len(rst), "set_piece_frames": [k0 + int(r["t"] * FPS) for r in rst],
            "passes": {"Home": sum(p["completed"] for p in ps if p["team"] == "A"), "Away": sum(p["completed"] for p in ps if p["team"] == "B")},
            "balls_lost": {"Home": sum(t["lost_by"] == "A" for t in tvs), "Away": sum(t["lost_by"] == "B" for t in tvs)}, "loss_frames": [(k0 + t["frame_lost"], t["lost_by"]) for t in tvs],
            "possession_pct": {"Home": round(100 * (st == 0).sum() / max(1, live)), "Away": round(100 * (st == 1).sum() / max(1, live))},
            "dead_pct": round(100 * (np.asarray(dstate) == 3).mean()), "mode": info["mode"],
            "sequences": len(seqs), "sequence_starts": [(k0 + s["start"], s["team"]) for s in seqs],
            "switches_per_min_raw": switches_per_min(state, FPS), "switches_per_min": switches_per_min(cstate, FPS)}

def switches_per_min(st, fps):
    """S6: team changes per minute, loose/dead frames skipped (A, loose, B = one switch); our clips: 35-41 raw, 4-5 spell"""
    t = [int(x) for x in np.asarray(st) if x < 2]
    return round(sum(1 for a, b in zip(t, t[1:]) if a != b) / max(1e-6, len(st) / fps / 60), 1)

def seq_found(truth, ours_starts, tol=2 * FPS):
    """E5: true sequences whose start has one of ours (same team) within +-2 s -> (found, ours that are real)"""
    pairs = sorted((abs(o - t["start"]), i, j) for i, t in enumerate(truth) for j, (o, tm) in enumerate(ours_starts) if tm == t["team"] and abs(o - t["start"]) <= tol)
    ut, uo = set(), set()
    for _, i, j in pairs:
        if i in ut or j in uo: continue
        ut.add(i); uo.add(j)
    return len(ut), len(uo)

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
