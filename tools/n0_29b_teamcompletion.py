"""N0-29b (side finding): the app's TEAM pass completion % (teams[].pass_completion_pct) with the passes as run.py makes
them today (state = spell state, IPANEMA_PASS_STATE=spell) vs other pass states, on Metrica's 2 pro games.
N0-24a measured passes WITHOUT a state (its lab did not pass one), so its '-1 pt clean / -16 pts flicker' is not what
the app shows. Variants: spell (app), raw (pipeline possession state), none (N0-24a), none + hand-backs dropped
(n0_24a_completion.drop_handbacks, 2 s). Truth = n0_29b_playerpass.truth_counts (PASS / attempts), all players.

    PYTHONPATH=. python tools/n0_29b_teamcompletion.py <metrica sample-data/data dir> [--out results/qa/n0-29b/metrica_teamcompletion.json]
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import analytics as AN
from n0_29b_playerpass import truth_counts


def team_pct(ps, tm):
    r = [p for p in ps if p["team"] == tm]
    return round(100 * sum(p["completed"] for p in r) / len(r), 1) if r else None


def truth_pct(T, tm):
    ids = [i for i in T if (i < 200) == (tm == "A")]; a = sum(T[i]["passes"] for i in ids)
    return round(100 * sum(T[i]["passes_completed"] for i in ids) / a, 1) if a else None


def variants(got, fps=25.0):
    from n0_24a_completion import drop_handbacks
    run = lambda st: AN.passes(got["per"], got["frames_"], got["tvs"], {}, got["ar"], fps, state=st)[0]
    none = run(None)
    return {"spell (app)": run(got["cstate"]), "raw state": run(got["state"]), "none (N0-24a)": none, "none + hand-backs 2 s": drop_handbacks(none, 2.0)}


def main(d, out):
    import n0_24d_ppda as PP
    rows = []
    for g in ("Sample_Game_1", "Sample_Game_2"):
        fr, ev = PP.load_game(d, g)
        for per in (1, 2):
            T = truth_counts(ev, per, PP.truth_direction(fr, per))
            for noisy in (False, "streak", "flicker"):
                for seed in ((0, 1) if noisy else (0,)):
                    _, got = PP.ours_run(fr, per, noisy, seed); V = variants(got)
                    for tm in ("A", "B"):
                        t = truth_pct(T, tm)
                        row = {"game": g[-1], "half": per, "noise": str(noisy), "seed": seed, "team": tm, "truth": t,
                               "err": {k: round(team_pct(v, tm) - t, 1) for k, v in V.items()}, "n": {k: sum(p["team"] == tm for p in v) for k, v in V.items()}}
                        rows.append(row); print(json.dumps(row), flush=True)
    S = {}
    for nz in ("False", "streak", "flicker"):
        R = [r for r in rows if r["noise"] == nz]
        S[nz] = {k: {"mean": round(float(np.mean([r["err"][k] for r in R])), 1), "min": min(r["err"][k] for r in R), "max": max(r["err"][k] for r in R),
                     "mean_abs": round(float(np.mean([abs(r["err"][k]) for r in R])), 1)} for k in R[0]["err"]}
    print(json.dumps(S, indent=1)); json.dump({"summary": S, "rows": rows}, open(out, "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[a.index("--out") + 1] if "--out" in a else "results/qa/n0-29b/metrica_teamcompletion.json")
