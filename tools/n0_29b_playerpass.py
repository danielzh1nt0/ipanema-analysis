"""N0-29b: are the app's PER-PLAYER passing figures right? Never checked on pro data before (N0-24a checked the TEAM
completion %, N0-23c the team forward share / progressive passes, N0-28e the pass network).

What the app exports per player (stats.json players[], analytics.stats; hidden in the app until M2 gives one id per
player - today they belong to track pieces):
  passes, passes_completed, passes_forward / _sideways / _backward (gain > 3 m / < -3 m / between, all passes incl.
  ones to an opponent), progressive_passes (completed, gain >= 10 m).

Metrica's 2 pro games, 4 halves, real ids (what M2 will give). Truth from Metrica's events, by the 'From' player:
  attempts = PASS + BALL LOST that is a pass (N0-24a's rule: not THEFT / FORCED / HEAD-FORCED / END HALF / OFFSIDE /
             REFEREE HIT); completed = PASS; kind from the ball's start -> end x (attack direction from kick-off spots);
  progressive = completed and gain >= 10 m.
Baselines (does the per-player figure say more than something trivial?):
  counts -> 'who is near the play': seconds within 10 m of the true ball (rank only);
  shares -> every player gets his TEAM's true share (forward share, completion %): if ours is no closer per player than
            that, the per-player split adds nothing.
Ours = the pipeline (metricalab.ours: perfect ball, 'streak', 'flicker' x 2 seeds) + analytics.stats like run.py.

Scores per team-half (players with >= 10 min): ratio = sum ours / sum truth; err = median per-player |ours - truth| /
truth (truth >= 5); rank = Spearman; top3 = truth's 3 busiest in ours' top 3. Shares: median |ours - truth| in points
over players with >= 10 truth attempts (forward share over all attempts; completion % = completed / attempts).

    PYTHONPATH=. python tools/n0_29b_playerpass.py <metrica sample-data/data dir> [--out results/qa/n0-29b/metrica_playerpass.json]
    PYTHONPATH=. python tools/n0_29b_playerpass.py --dry
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from n0_29a_playerpress import team_scores

L, W, FPS = 105.0, 68.0, 25.0
NOT_A_PASS = {"THEFT", "FORCED", "HEAD-FORCED", "END HALF", "OFFSIDE", "REFEREE HIT"}
MIN_PLAY_MIN = 10.0
MIN_ATT = 10
COUNTS = ("passes", "passes_completed", "passes_forward", "passes_sideways", "passes_backward", "progressive_passes")


def kind(g, fwd=3.0): return "forward" if g > fwd else "backward" if g < -fwd else "sideways"


def _f(s): return None if s in ("", "NaN", None) else float(s)


def is_attempt(r):
    return r["Type"] == "PASS" or (r["Type"] == "BALL LOST" and r["Subtype"] not in NOT_A_PASS)


def truth_counts(ev, per, attack_right):
    """Metrica event rows -> {pid: {passes, passes_completed, passes_forward, ..., progressive_passes}} (same fields as
    the app). attack_right: {'A': bool, 'B': bool}; Home = A (ids 1xx), Away = B (2xx)."""
    from n0_24d_ppda import pid_of
    T = {}
    for r in ev:
        if int(r["Period"]) != per or r["Team"] not in ("Home", "Away") or not is_attempt(r): continue
        p = pid_of(r["From"], r["Team"])
        if p is None: continue
        t = T.setdefault(p, {k: 0 for k in COUNTS}); tm = "A" if r["Team"] == "Home" else "B"; ok = r["Type"] == "PASS"
        t["passes"] += 1; t["passes_completed"] += int(ok)
        x0, x1 = _f(r["Start X"]), _f(r["End X"])
        if x0 is None or x1 is None: continue
        g = (x1 - x0) * L * (1 if attack_right[tm] else -1)
        t["passes_" + kind(g)] += 1; t["progressive_passes"] += int(ok and g >= 10.0)
    return T


def near_ball(fr, per):
    """{pid: (seconds within 10 m of the true ball, seconds on the pitch)}"""
    N = {}
    for k in (k for k in fr if fr[k][0] == per):
        _, ps, b = fr[k]
        for pid, x, y in ps:
            n = N.setdefault(pid, [0.0, 0.0]); n[1] += 1 / FPS
            if b is not None and np.hypot(x - b[0], y - b[1]) <= 10: n[0] += 1 / FPS
    return N


def share_err(o, t, ids, key, base):
    """median |ours share - truth share| (pts) of key / passes over players with >= MIN_ATT truth attempts; base = the
    team's true share given to everyone"""
    e, eb = [], []
    for i in ids:
        tt = t.get(i, {}); oo = o.get(i, {})
        if tt.get("passes", 0) < MIN_ATT or not oo.get("passes"): continue
        ts = tt[key] / tt["passes"]; e.append(abs(oo[key] / oo["passes"] - ts)); eb.append(abs(base - ts))
    if not e: return {"err_pts": None, "base_err_pts": None, "n": 0}
    return {"err_pts": round(100 * float(np.median(e)), 1), "base_err_pts": round(100 * float(np.median(eb)), 1), "n": len(e)}


def score_half(st_players, T, N):
    """-> {team: {<count>: team_scores, near_vs_<count>, fwd_share, completion}}"""
    O = {p["id"]: p for p in st_players}; out = {}
    for tm, lo in (("A", lambda i: i < 200), ("B", lambda i: i >= 200)):
        ids = [i for i, n in N.items() if lo(i) and n[1] >= MIN_PLAY_MIN * 60]
        o = {i: {k: O.get(i, {}).get(k, 0) for k in COUNTS} for i in ids}; t = {i: T.get(i, {k: 0 for k in COUNTS}) for i in ids}
        r = {}
        for k in COUNTS:
            r[k] = team_scores({i: o[i][k] for i in ids}, {i: t[i][k] for i in ids}, "o", "t", 5, ids)
        nb = {i: N[i][0] for i in ids}
        r["near_vs_passes"] = team_scores(nb, {i: t[i]["passes"] for i in ids}, "o", "t", 5, ids)
        r["near_vs_progressive"] = team_scores(nb, {i: t[i]["progressive_passes"] for i in ids}, "o", "t", 5, ids)
        ta = sum(t[i]["passes"] for i in ids) or 1
        r["fwd_share"] = share_err(o, t, ids, "passes_forward", sum(t[i]["passes_forward"] for i in ids) / ta)
        r["back_share"] = share_err(o, t, ids, "passes_backward", sum(t[i]["passes_backward"] for i in ids) / ta)
        r["completion"] = share_err(o, t, ids, "passes_completed", sum(t[i]["passes_completed"] for i in ids) / ta)
        r["truth_totals"] = {k: sum(t[i][k] for i in ids) for k in COUNTS}; r["ours_totals"] = {k: sum(o[i][k] for i in ids) for k in COUNTS}
        out[tm] = r
    return out


def summarise(rows):
    out = {}
    for nz in sorted({r["noise"] for r in rows}):
        rs = [t for r in rows if r["noise"] == nz for t in r["teams"].values()]; out[nz] = {"team_halves": len(rs)}
        for k in COUNTS + ("near_vs_passes", "near_vs_progressive"):
            d = {}
            for q in ("ratio", "err", "rank", "top3"):
                v = [t[k][q] for t in rs if t[k][q] is not None]; d[q] = round(float(np.mean(v)), 2) if v else None
            d["rank_min"] = min((t[k]["rank"] for t in rs if t[k]["rank"] is not None), default=None); out[nz][k] = d
        for k in ("fwd_share", "back_share", "completion"):
            out[nz][k] = {q: round(float(np.mean([t[k][q] for t in rs if t[k][q] is not None])), 1) if any(t[k][q] is not None for t in rs) else None for q in ("err_pts", "base_err_pts")}
            out[nz][k]["beats_base"] = f"{sum(1 for t in rs if t[k]['err_pts'] is not None and t[k]['err_pts'] < t[k]['base_err_pts'])}/{sum(1 for t in rs if t[k]['err_pts'] is not None)}"
    return out


def ours_players(fr, per, noisy, seed):
    from n0_24d_ppda import ours_run
    st, got = ours_run(fr, per, noisy, seed)
    return st["players"]


def dry():
    """stand-ins: Home player 1 passes forward a lot, player 2 sideways; ours reads player 1 right and player 2 short"""
    ev = []
    def row(tp, sub, who, x0, x1, team="Home"):
        ev.append({"Team": team, "Type": tp, "Subtype": sub, "Period": "1", "From": who, "Start X": str(x0), "End X": str(x1)})
    for _ in range(12): row("PASS", "", "Player1", 0.4, 0.6)                   # +21 m: forward, progressive
    for _ in range(3): row("BALL LOST", "INTERCEPTION", "Player1", 0.4, 0.6)   # failed forward
    row("BALL LOST", "THEFT", "Player1", 0.4, 0.4)                              # dispossessed: not a pass
    for _ in range(10): row("PASS", "", "Player2", 0.5, 0.51)                  # sideways
    T = truth_counts(ev, 1, {"A": True, "B": False})
    N = {101: [300.0, 2700.0], 102: [200.0, 2700.0], 103: [50.0, 2700.0]}
    st = [{"id": 101, "passes": 14, "passes_completed": 11, "passes_forward": 13, "passes_sideways": 1, "passes_backward": 0, "progressive_passes": 10},
          {"id": 102, "passes": 6, "passes_completed": 6, "passes_forward": 0, "passes_sideways": 6, "passes_backward": 0, "progressive_passes": 0}]
    return {"truth": {str(k): v for k, v in T.items()}, "scores": score_half(st, T, N)["A"]}


def main(d, out):
    import n0_24d_ppda as PP
    rows = []
    for g in ("Sample_Game_1", "Sample_Game_2"):
        fr, ev = PP.load_game(d, g)
        for per in (1, 2):
            T = truth_counts(ev, per, PP.truth_direction(fr, per)); N = near_ball(fr, per)
            for noisy in (False, "streak", "flicker"):
                for seed in ((0, 1) if noisy else (0,)):
                    pl = ours_players(fr, per, noisy, seed); sc = score_half(pl, T, N); O = {p["id"]: p for p in pl}
                    rows.append({"game": g[-1], "half": per, "noise": str(noisy), "seed": seed, "teams": sc,
                                 "players": {str(i): {"truth": T.get(i, {}), "ours": {k: O.get(i, {}).get(k, 0) for k in COUNTS}, "near_s": round(N[i][0], 1)}
                                             for i in N if N[i][1] >= MIN_PLAY_MIN * 60}})
                    print(g, per, noisy, seed, json.dumps({tm: {k: s[k] for k in ("passes", "progressive_passes", "fwd_share", "completion")} for tm, s in sc.items()}), flush=True)
    summ = summarise(rows); print(json.dumps(summ, indent=1))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"summary": summ, "runs": rows}, open(out, "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--dry" in a: print(json.dumps(dry(), indent=1)); sys.exit()
    main(a[0], a[a.index("--out") + 1] if "--out" in a else "results/qa/n0-29b/metrica_playerpass.json")
