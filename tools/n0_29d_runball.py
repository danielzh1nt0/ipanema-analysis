"""N0-29d: the 'with the ball' label on the app's runs (stats.metrics.runs[].with_ball) - never checked before.

What the app writes (metrics.runs -> metrics._run): with_ball = did the runner's TEAM have the ball at the run's first
frame, read from the RAW possession state (ipanema/run.py passes `state`); loose / dead at that frame -> None. It splits
sprints into 'in possession' (runs to get free / carries) and 'out of possession' (chasing, pressing, recovery runs).

Truth (Metrica's 2 pro games, 4 halves): the team of the truth possession chain (metricalab.truth_sequences: chains of
on-ball events) that covers the run's first frame; a frame between chains (contested / dead) has no truth and is not
scored. Runs = the app's own runs (metrics.runs) on Metrica's positions as the pipeline sees them, so only the LABEL is
graded, not whether the run is real (N0-25a did that).

Labels compared (state = the pipeline's raw state, spell = P.spell_state like run.py's counted stats):
  raw_start   - the app today: raw state at the first frame
  spell_start - spell state at the first frame
  spell_back  - spell state at the first frame, else the last team that had it within LOOK_S before (like metrics.shots)
  spell_major - the team holding the spell state most of the run's frames, else spell_back
Scores per noise model: labelled (not None) share, right among labelled+known, right of all known (None = wrong), and
per team-half 'share of sprints made in possession' ours vs truth (points), vs a baseline giving every run the team's
possession share.

    PYTHONPATH=. python tools/n0_29d_runball.py <metrica sample-data/data dir> [--out results/qa/n0-29d/metrica_runball.json]
    PYTHONPATH=. python tools/n0_29d_runball.py --dry
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import metrics as M
from ipanema.possession import STATES

FPS = 25.0
LOOK_S = 3.0
LABELS = ("raw_start", "spell_start", "spell_back", "spell_major")


def _team(s): return STATES[int(s)] if int(s) < 2 else None


def label(kind, state, cstate, a, b, team, fps=FPS):
    """-> True (team had the ball) / False (the other team) / None (nobody)"""
    if kind == "raw_start": t = _team(state[a])
    elif kind == "spell_start": t = _team(cstate[a])
    elif kind == "spell_back":
        t = next((_team(cstate[j]) for j in range(a, max(-1, a - int(LOOK_S * fps)), -1) if int(cstate[j]) < 2), None)
    elif kind == "spell_major":
        seg = [int(x) for x in cstate[a:b + 1] if int(x) < 2]
        na, nb = seg.count(0), seg.count(1)
        t = "A" if na > nb else "B" if nb > na else label("spell_back", state, cstate, a, b, team, fps)
        return t if isinstance(t, bool) or t is None else t == team
    else: raise ValueError(kind)
    return None if t is None else t == team


def truth_team_at(seqs, f):
    """team of the truth chain covering Metrica frame f, else None"""
    for s in seqs:
        if s["start"] <= f <= s["end"]: return s["team"]
    return None


def score_runs(runs_, state, cstate, seqs, k0, fps=FPS):
    """-> list of {team, truth (bool/None), label: {kind: bool/None}, dist}"""
    out = []
    for r in runs_:
        a, b = int(round(r["t"] * fps)), int(round(r["t_end"] * fps)); tt = truth_team_at(seqs, k0 + a)
        out.append({"team": r["team"], "truth": None if tt is None else tt == r["team"], "dist": r["distance_m"],
                    "app": r["with_ball"], "label": {k: label(k, state, cstate, a, b, r["team"], fps) for k in LABELS}})
    return out


def summarise(rows):
    """rows = scored runs of one noise model (several halves) -> per label scores"""
    known = [r for r in rows if r["truth"] is not None]; S = {"runs": len(rows), "truth_known_pct": round(100 * len(known) / max(1, len(rows)))}
    base = np.mean([r["truth"] for r in known]) if known else 0
    S["always_out_of_possession_right_pct"] = round(100 * (1 - base)) if known else None
    for k in LABELS:
        lab = [r for r in known if r["label"][k] is not None]
        S[k] = {"labelled_pct": round(100 * sum(r["label"][k] is not None for r in rows) / max(1, len(rows))),
                "right_of_labelled_pct": round(100 * np.mean([r["label"][k] == r["truth"] for r in lab])) if lab else None,
                "right_of_known_pct": round(100 * sum(r["label"][k] == r["truth"] for r in known) / max(1, len(known)))}
    return S


def share_err(halves):
    """per team-half: share of runs labelled 'in possession' (of labelled) vs truth share (of known); baseline = the team's
    truth possession share of the half. halves = [(scored runs, {team: truth possession share 0-1})]"""
    err = {k: [] for k in LABELS}; berr = []
    for rows, pshare in halves:
        for tm in ("A", "B"):
            R = [r for r in rows if r["team"] == tm and r["truth"] is not None]
            if len(R) < 10: continue
            t = 100 * np.mean([r["truth"] for r in R]); berr.append(abs(100 * pshare[tm] - t))
            for k in LABELS:
                lab = [r["label"][k] for r in R if r["label"][k] is not None]
                if lab: err[k].append(abs(100 * np.mean(lab) - t))
    return {"baseline_possession_share": round(float(np.mean(berr)), 1) if berr else None, **{k: round(float(np.mean(v)), 1) if v else None for k, v in err.items()}}


def truth_poss_share(seqs):
    d = {"A": 0, "B": 0}
    for s in seqs: d[s["team"]] += s["end"] - s["start"] + 1
    tot = max(1, d["A"] + d["B"]); return {t: d[t] / tot for t in d}


def one_half(fr, ev_rows, per, noisy, seed):
    import metricalab as ML, n0_24d_ppda as PP
    _, got = PP.ours_run(fr, per, noisy, seed)
    k0 = min(k for k in fr if fr[k][0] == per)
    seqs = ML.truth_sequences(PP.ML_events(ev_rows), per)
    rs = M.runs(got["per"], FPS, got["state"], got["ar"])
    return score_runs(rs, np.asarray(got["state"]), np.asarray(got["cstate"]), seqs, k0), truth_poss_share(seqs)


def main(d, out):
    import n0_24d_ppda as PP
    by = {}; halves = {}
    for g in ("Sample_Game_1", "Sample_Game_2"):
        fr, ev = PP.load_game(d, g)
        for per in (1, 2):
            for noisy in (False, "streak", "flicker"):
                for seed in ((0, 1) if noisy else (0,)):
                    rows, ps = one_half(fr, ev, per, noisy, seed); nz = str(noisy)
                    assert all(r["app"] == r["label"]["raw_start"] for r in rows), "lab label != app label"
                    by.setdefault(nz, []).extend(rows); halves.setdefault(nz, []).append((rows, ps))
                    print(g, per, nz, seed, len(rows), flush=True)
    S = {nz: {**summarise(R), "share_err_pts": share_err(halves[nz])} for nz, R in by.items()}
    print(json.dumps(S, indent=1))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"summary": S, "runs": {nz: [{k: v for k, v in r.items()} for r in R] for nz, R in by.items()}}, open(out, "w"))
    return S


def dry():
    """stand-in: two players, A sprints while A has the ball, B sprints while B has it"""
    fps = FPS; n = int(20 * fps); per = {}
    for k in range(n):
        xa = 10 + (7 * k / fps if k < 5 * fps else 35); xb = 60 + (7 * (k - 10 * fps) / fps if 10 * fps <= k < 15 * fps else (0 if k < 10 * fps else 35))
        w = 0.05 * np.sin(k * 1.3)                                              # a wobble: exact straight lines are dropped as gap fills
        per[k] = [[1, "A", np.array([xa, 30.0 + w]), None, None, False], [2, "B", np.array([xb, 40.0 + w]), None, None, False]]
    state = np.array([0 if k < 8 * fps else 2 if k < 10 * fps else 1 for k in range(n)]); cstate = state.copy()
    seqs = [{"team": "A", "start": 0, "end": int(8 * fps)}, {"team": "B", "start": int(10 * fps), "end": n}]
    rs = M.runs(per, fps, state, {"A": True, "B": False})
    rows = score_runs(rs, state, cstate, seqs, 0)
    return rows, summarise(rows)


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--dry" in a:
        rows, S = dry(); print(json.dumps(S, indent=1)); sys.exit(0)
    main(a[0], a[a.index("--out") + 1] if "--out" in a else "results/qa/n0-29d/metrica_runball.json")
