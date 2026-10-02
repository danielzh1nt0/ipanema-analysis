"""S1 (29 Sep): pass counting on Metrica's free pro games. Our completed passes vs their hand-labelled PASS events,
matched in time (ours within +-1 s of the release, same team). Misses and extras are sorted into kinds (short, long,
one-touch, header, set piece, during a loose ball ...) so we can see which rule is wrong. Clean positions and 'streak'
noise (errors in runs, like our Veo tracking). Also compares two possession states: the old viterbi model and
possession_simple (nearest feet to the ball; on Metrica H = identity, metres used as 'pixels', near_m = 1.5).
Free, CPU. The slow part (positions -> carriers -> states) is cached in the scratch dir given by PASSLAB_CACHE.
    python tools/metrica_passlab.py <metrica data dir with Sample_Game_1, Sample_Game_2> [base|grid|fine|defl|states]"""
import sys, os, json, pickle, random, collections, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import possession as P, analytics as AN
import tools.metricalab as ML
FPS = ML.FPS; TOL = 1.0
CACHE = os.environ.get("PASSLAB_CACHE", "/tmp/passlab_cache")

def prepare(frames, per, noisy, seed=0):
    """same noise model as tools/turnoverlab.py (so numbers line up), plus the possession_simple state"""
    rng = random.Random(seed); ks = sorted(k for k in frames if frames[k][0] == per); k0 = ks[0]
    perd, ballm, H, run = {}, {}, {}, [0, "ok", None]
    for i, k in enumerate(ks):
        _, ps, b = frames[k]
        rows = [[pid, "A" if pid < 200 else "B", np.array([x, y]), None, None, False] for pid, x, y in ps]
        if noisy:
            rows = [r for r in rows if rng.random() < 0.65]
            for r in rows: r[2] = r[2] + np.array([rng.gauss(0, 0.8), rng.gauss(0, 0.8)])
        for r in rows: r[3] = r[2]                                             # 'pixel' feet = metres (H = identity)
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
    sv, _ = P.viterbi(perd, P.clean_ball(bm, ML.L, ML.W), FPS, ML.L, ML.W)
    ss = P.possession_simple(perd, {k: tuple(v) for k, v in ballm.items()}, H, len(perd), near_m=1.5, smooth=6)
    ar = P.direction_from_keepers(perd, ML.L, log=lambda *a: None) or {"A": True, "B": False}
    return dict(k0=k0, perd=perd, frames_=frames_, bm=bm, ballm=ballm, state_viterbi=np.asarray(sv), state_simple=np.asarray(ss), ar=ar)

def load_game(root, g):
    d = f"{root}/{g}"
    home = ML.load_tracking(f"{d}/{g}_RawTrackingData_Home_Team.csv", 100); away = ML.load_tracking(f"{d}/{g}_RawTrackingData_Away_Team.csv", 200)
    frames = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home}
    return frames, ML.load_events(f"{d}/{g}_RawEventsData.csv")

def cached(root, g, per, noisy):
    os.makedirs(CACHE, exist_ok=True); fn = f"{CACHE}/{g}_{per}_{noisy or 'clean'}.pkl"
    if os.path.exists(fn): return pickle.load(open(fn, "rb"))
    frames, _ = load_game(root, g); S = prepare(frames, per, noisy); pickle.dump(S, open(fn, "wb")); return S

def truth_passes(ev, per):
    """PASS events of one half with kinds. One-touch = passer received the ball < 0.5 s before; set piece = a SET PIECE
    event at the same frame (or a goal kick); loose = a CHALLENGE/RECOVERY/BALL LOST within 1 s before."""
    E = sorted([e for e in ev if e["period"] == per], key=lambda e: e["f0"]); out = []
    for i, e in enumerate(E):
        if e["type"] != "PASS": continue
        prev = [x for x in E[max(0, i - 6):i] if x["f0"] >= e["f0"] - 1.0 * FPS]
        rec = next((x for x in reversed(E[:i]) if x["type"] == "PASS"), None)
        dur = (e["f1"] - e["f0"]) / FPS
        kinds = []
        if e["sub"] == "GOAL KICK" or any(x["type"] == "SET PIECE" and abs(x["f0"] - e["f0"]) <= 2 for x in E[max(0, i - 3):i + 1]): kinds.append("set piece")
        if "HEAD" in e["sub"]: kinds.append("header")
        if rec is not None and rec["team"] == e["team"] and 0 <= e["f0"] - rec["f1"] < 0.5 * FPS: kinds.append("one-touch")
        if any(x["type"] in ("CHALLENGE", "RECOVERY", "BALL LOST") for x in prev): kinds.append("after duel/loose")
        kinds.append("short (<0.6 s)" if dur < 0.6 else "long (>2 s)" if dur > 2.0 else "medium")
        out.append({"f": e["f0"], "f1": e["f1"], "team": "A" if e["team"] == "Home" else "B", "kinds": kinds, "dur": dur})
    return out

def match(tp, op, tol=TOL * FPS):
    """greedy nearest match, same team -> (matched truth idx set, matched ours idx set)"""
    pairs = sorted((abs(o["f"] - t["f"]), i, j) for i, t in enumerate(tp) for j, o in enumerate(op) if o["team"] == t["team"] and abs(o["f"] - t["f"]) <= tol)
    ut, uo = set(), set()
    for _, i, j in pairs:
        if i in ut or j in uo: continue
        ut.add(i); uo.add(j)
    return ut, uo

def extra_kind(o, tp, ev, per, state):
    E = [e for e in ev if e["period"] == per]
    near = [t for t in tp if abs(t["f"] - o["f"]) <= 3 * FPS and t["team"] == o["team"]]
    if any(e["type"] == "BALL OUT" and 0 <= o["f"] - e["f0"] <= 15 * FPS for e in E) and not near: return "dead ball"
    if any(t["f"] < o["f"] < t["f1"] for t in tp if t["team"] == o["team"]): return "during a pass (ball passed a teammate)"
    if near: return "near a real pass (split/double)"
    if any(e["type"] in ("CHALLENGE", "BALL LOST", "RECOVERY") and abs(e["f0"] - o["f"]) <= 2 * FPS for e in E): return "duel/loose ball"
    return "other (carry/dribble)"

def our_passes(S, variant):
    """variant = passes() keyword args, plus state='viterbi'|'simple' (filter carriers by that state) and
    carriers_from='simple'|'viterbi' (carrier = nearest player of the team the state names, AN.carriers_from_state)"""
    v = dict(variant); src = v.pop("state", None); cf = v.pop("carriers_from", None); cm = v.pop("carrier_m", 2.5); fr = S["frames_"]
    take = v.pop("take_s", None); join = v.pop("join_s", 3.0)
    if src is not None: v["state"] = S[f"state_{src}"]
    if take is not None:                                                      # S8 (2 Oct): the de-flickered state as the filter
        from tools.statelab import clean_state
        v["state"] = clean_state(S["state_simple"], FPS, take, join)
    if v.get("pass_by_mps"): v["ballm"] = S["bm"]
    if cf is not None: fr = AN.carriers_from_state(S["perd"], S["ballm"], S[f"state_{cf}"], max_m=cm)
    ps, _ = AN.passes(S["perd"], fr, [], {}, S["ar"], FPS, **v)
    return [{"f": S["k0"] + int(round(p["t"] * FPS)), "team": p["team"], "len": p["length_m"]} for p in ps if p["completed"]]

OLD = {"team_sandwich_s": 0.0, "min_pass_m": 0.0}                           # the rule before S1 (29 Sep)
VARIANTS = {"now": dict(OLD), "new": {}}
GRID = {f"ts{a}_ms{b}_len{c}": {"team_sandwich_s": a, "mate_sandwich_s": b, "min_pass_m": c}
        for a in (0.0, 0.3, 0.6) for b in (0.0, 0.2, 0.3, 0.4) for c in (0.0, 3.0, 5.0)}
FINE = {f"ts{a}_ms{b}_len{c}_mt{m}": {"team_sandwich_s": a, "mate_sandwich_s": b, "min_pass_m": c, "min_touch_s": m}
        for a in (0.2, 0.3, 0.4) for b in (0.0, 0.1, 0.2) for c in (4.0, 5.0, 6.0) for m in (0.4, 0.6, 0.8)}
DEFL = {f"ts{a}_ms{b}_len{c}_pb{d}": {"team_sandwich_s": a, "mate_sandwich_s": b, "min_pass_m": c, "pass_by_mps": d}
        for a in (0.4, 0.6) for b in (0.0, 0.3, 0.6) for c in (0.0, 3.0, 5.0) for d in (4.0, 6.0, 8.0)}
NEW = {"team_sandwich_s": 0.3, "min_pass_m": 5.0}                           # the S1 rule (now the default)
STATES = {"now": dict(OLD), "new": dict(NEW)}
CLEAN = {"new": dict(NEW), "filter_simple+new": dict(NEW, state="simple")}
for _t in (0.5, 1.0, 1.5, 2.0): CLEAN[f"clean{_t}+new"] = dict(NEW, take_s=_t)
for _nm, _kw in {"filter_viterbi": {"state": "viterbi"}, "filter_simple": {"state": "simple"}, "from_viterbi": {"carriers_from": "viterbi"},
                 "from_simple": {"carriers_from": "simple"}}.items():
    STATES[_nm] = dict(_kw, **OLD); STATES[_nm + "+new"] = dict(_kw, **NEW)

def summary(res):
    by = collections.defaultdict(list)
    for r in res: by[r["variant"]].append(r)
    out = []
    for v, rs in by.items():
        m, t, o = (sum(r[k] for r in rs) for k in ("matched", "truth", "ours"))
        out.append({"variant": v, "max_abs_err_pct": max(abs(r["err_pct"]) for r in rs), "errs": [r["err_pct"] for r in rs],
                    "recall": round(m / t, 3), "precision": round(m / o, 3)})
    return sorted(out, key=lambda x: x["max_abs_err_pct"])

def events(root, g):
    return ML.load_events(f"{root}/{g}/{g}_RawEventsData.csv")

def run(root, variants, noises=(False, "streak"), detail=True):
    res = []
    for g in ("Sample_Game_1", "Sample_Game_2"):
        ev = events(root, g)
        for per in (1, 2):
            tp = truth_passes(ev, per)
            for noisy in noises:
                S = cached(root, g, per, noisy)
                for name, var in variants.items():
                    op = our_passes(S, var); ut, uo = match(tp, op)
                    miss = collections.Counter(k for i, t in enumerate(tp) if i not in ut for k in t["kinds"])
                    allk = collections.Counter(k for t in tp for k in t["kinds"])
                    extra = collections.Counter(extra_kind(o, tp, ev, per, None) for j, o in enumerate(op) if j not in uo)
                    r = {"game": g[-1], "half": per, "noise": noisy or "clean", "variant": name, "truth": len(tp), "ours": len(op), "matched": len(ut),
                         "err_pct": round(100 * (len(op) - len(tp)) / len(tp), 1), "missed": len(tp) - len(ut), "extra": len(op) - len(uo),
                         "missed_by_kind": {k: f"{miss[k]}/{allk[k]}" for k in allk}, "extra_by_kind": dict(extra)}
                    res.append(r); print(json.dumps(r), flush=True)
    return res

if __name__ == "__main__":
    root = sys.argv[1]; mode = sys.argv[2] if len(sys.argv) > 2 else "base"
    V = {"base": VARIANTS, "grid": GRID, "fine": FINE, "defl": DEFL, "states": STATES, "clean": CLEAN}[mode]
    res = run(root, V); summ = summary(res)
    for x in summ: print(json.dumps(x))
    os.makedirs("results/metrica", exist_ok=True); json.dump({"runs": res, "summary": summ}, open(f"results/metrica/passlab_{mode}.json", "w"), indent=1)
