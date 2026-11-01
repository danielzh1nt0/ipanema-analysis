"""N0-01c (1 Nov): 'passes' from a player to himself. N0-01b's side note: 4-8% of the passes in the app copies have
from == to (SFK-BP 8/167, Vallentuna 37/487). A pass to yourself is not a pass in any event feed (it is a dribble /
knock-on); with our tracks it can also be a real pass whose receiver's track slid onto the passer.

Metrica's 2 pro games, 4 halves, real ids: the pipeline's passes (tools/n0_24d_ppda.ours_run, perfect ball + 'streak' /
'flicker' ball noise, 2 seeds) are matched to Metrica's pass attempts (PASS + BALL LOST that is a pass, N0-24a's rule)
of the same team whose Start Frame is within +-1 s of our release (greedy nearest, like metrica_passlab.match).
Per pass kind (self = from == to, other): how many are real (matched), and for matched self passes whether Metrica's
passer is ours. Also: what happens to the team pass count / completion % if self passes are dropped.

    PYTHONPATH=. python tools/n0_01c_selfpass.py <metrica sample-data/data dir> [--out results/qa/n0-01c/metrica_selfpass.json]
    PYTHONPATH=. python tools/n0_01c_selfpass.py --dry
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FPS = 25.0
NOT_A_PASS = {"THEFT", "FORCED", "HEAD-FORCED", "END HALF", "OFFSIDE", "REFEREE HIT"}


def pid_of(name, team):
    if not name or not name.startswith("Player"): return None
    return (100 if team == "Home" else 200) + int(name[6:])


def truth_attempts(ev, per):
    out = []
    for r in ev:
        if int(r["Period"]) != per or r["Team"] not in ("Home", "Away"): continue
        if not (r["Type"] == "PASS" or (r["Type"] == "BALL LOST" and r["Subtype"] not in NOT_A_PASS)): continue
        out.append({"f": int(r["Start Frame"]), "team": "A" if r["Team"] == "Home" else "B", "from": pid_of(r["From"], r["Team"]),
                    "to": pid_of(r["To"], r["Team"]), "completed": r["Type"] == "PASS"})
    return out


def is_self(p): return p.get("to") is not None and int(p["from"]) == int(p["to"])


def classify(ours, truth, f0, tol_s=1.0):
    """-> list of dicts per our pass: self, matched (bool), same_passer (matched and Metrica's From == ours)"""
    tol = tol_s * FPS
    pairs = sorted((abs(f0 + o["t"] * FPS - t["f"]), i, j) for i, t in enumerate(truth) for j, o in enumerate(ours)
                   if o["team"] == t["team"] and abs(f0 + o["t"] * FPS - t["f"]) <= tol)
    ut, m = set(), {}
    for _, i, j in pairs:
        if i in ut or j in m: continue
        ut.add(i); m[j] = i
    res = []
    for j, o in enumerate(ours):
        t = truth[m[j]] if j in m else None
        res.append({"self": is_self(o), "matched": t is not None, "same_passer": bool(t and t["from"] == int(o["from"])),
                    "completed": bool(o["completed"]), "length_m": o["length_m"]})
    return res, len(truth)


def tally(rows):
    """rows: classify() outputs of many runs -> {self/other: n, real, real_pct}; drop effect on found / real share"""
    out = {}
    for kind in ("self", "other"):
        R = [r for r in rows if r["self"] == (kind == "self")]
        out[kind] = {"n": len(R), "real": sum(r["matched"] for r in R), "real_pct": round(100 * sum(r["matched"] for r in R) / max(1, len(R)), 1),
                     "same_passer": sum(r["same_passer"] for r in R), "completed_pct": round(100 * sum(r["completed"] for r in R) / max(1, len(R)), 1)}
    out["self_share_pct"] = round(100 * out["self"]["n"] / max(1, out["self"]["n"] + out["other"]["n"]), 1)
    return out


def dry():
    """stand-in: 3 truth passes, ours has 2 matched others + 1 matched self (right passer) + 1 unmatched self"""
    truth = [{"f": 100, "team": "A", "from": 101, "to": 102, "completed": True}, {"f": 200, "team": "A", "from": 102, "to": 103, "completed": True},
             {"f": 300, "team": "B", "from": 201, "to": 202, "completed": True}]
    ours = [{"t": 4.0, "team": "A", "from": 101, "to": 102, "completed": True, "length_m": 10},
            {"t": 8.0, "team": "A", "from": 102, "to": 102, "completed": True, "length_m": 6},
            {"t": 12.0, "team": "B", "from": 201, "to": 202, "completed": True, "length_m": 9},
            {"t": 20.0, "team": "B", "from": 202, "to": 202, "completed": True, "length_m": 7}]
    rows, n = classify(ours, truth, 0)
    return tally(rows)


def main(d, out):
    import n0_24d_ppda as PP
    from ipanema import analytics as AN
    AN.PASS_SELF = True                                                        # measure the old rule (self passes kept)
    runs, allrows = [], {"clean": [], "noisy": []}
    for g in ("Sample_Game_1", "Sample_Game_2"):
        fr, ev = PP.load_game(d, g)
        for per in (1, 2):
            T = truth_attempts(ev, per); f0 = min(k for k in fr if fr[k][0] == per)
            for noisy in (False, "streak", "flicker"):
                for seed in ((0, 1) if noisy else (0,)):
                    st, got = PP.ours_run(fr, per, noisy, seed)
                    rows, nt = classify(got["ps"], T, f0)
                    allrows["noisy" if noisy else "clean"] += rows
                    t = tally(rows)
                    found_all = sum(r["matched"] for r in rows); found_drop = sum(r["matched"] for r in rows if not r["self"])
                    n_all = len(rows); n_drop = sum(not r["self"] for r in rows)
                    # completion % per team as the app shows it, with and without self passes
                    comp = {}
                    for tm in ("A", "B"):
                        P_ = [p for p in got["ps"] if p["team"] == tm]; Q = [p for p in P_ if not is_self(p)]
                        Tt = [x for x in T if x["team"] == tm]
                        comp[tm] = {"truth": round(100 * sum(x["completed"] for x in Tt) / max(1, len(Tt)), 1),
                                    "ours": round(100 * sum(p["completed"] for p in P_) / max(1, len(P_)), 1),
                                    "ours_no_self": round(100 * sum(p["completed"] for p in Q) / max(1, len(Q)), 1)}
                    r = {"game": g[-1], "half": per, "noise": str(noisy), "seed": seed, "truth_attempts": nt, "ours": n_all, "ours_no_self": n_drop,
                         "found_pct": round(100 * found_all / nt, 1), "found_pct_no_self": round(100 * found_drop / nt, 1),
                         "real_pct": round(100 * found_all / max(1, n_all), 1), "real_pct_no_self": round(100 * found_drop / max(1, n_drop), 1),
                         "tally": t, "completion": comp}
                    runs.append(r); print(json.dumps({k: r[k] for k in ("game", "half", "noise", "seed", "truth_attempts", "ours", "ours_no_self", "found_pct", "found_pct_no_self", "real_pct", "real_pct_no_self")}), json.dumps(t["self"]), flush=True)
    summ = {k: tally(v) for k, v in allrows.items()}
    print(json.dumps(summ, indent=1))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"summary": summ, "runs": runs}, open(out, "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--dry" in a: print(json.dumps(dry(), indent=1)); sys.exit()
    main(a[0], a[a.index("--out") + 1] if "--out" in a else "results/qa/n0-01c/metrica_selfpass.json")
