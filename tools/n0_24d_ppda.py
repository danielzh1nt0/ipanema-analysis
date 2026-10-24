"""N0-24d: are the app's pressures and PPDA right? Never checked on pro data before.

The app (analytics.stats, as ipanema/run.py calls it):
  players[].pressures_applied = per opponent possession sequence (P.sequences on the spell state), each player of the
      other team who comes within press_r = 2 m of the carrier counts once (S7, 2 Oct).
  teams[].pressures_applied = the team's sum.
  teams[].ppda_opp_passes_per_def_action = the other team's passes (whole pitch) / the team's pressures.

Truth (Metrica's labelled events + tracking, 2 games x 2 halves):
  same-rule pressures: possession sequences = metricalab.truth_sequences (chains of on-ball events); carrier = the
      'From' player of the latest on-ball event, or the 'To' player after a pass has arrived, nobody while a pass is in
      flight; a defender within 2 m of him (tracking) counts once per sequence. This is OUR definition with perfect
      inputs, so it measures what our ball / possession noise does to the number.
  standard PPDA (the usual pro-data stat): the other team's passes tried (PASS + BALL LOST but THEFT) in its own 60% of the pitch / the team's defensive
      actions there (tackles + ground challenges + interceptions + steals as RECOVERY + fouls = the other team's FAULT
      RECEIVED), from Metrica's events. Lower = presses more. Measures whether our number tells a coach the same story.
Ours = the pipeline (metricalab.ours' noise, plus passes on the spell state like run.py) on Metrica's positions:
perfect ball, 'streak' and 'flicker' ball noise (2 seeds).

    PYTHONPATH=. python tools/n0_24d_ppda.py <metrica sample-data/data dir> [--out results/qa/n0-24d/metrica_ppda.json]
        [--zone]   lab: also our PPDA with the standard zone (opponent passes in its own 60%, pressures there)
        [--refresh-truth]   only recompute the truth fields of the saved --out file
"""
import sys, os, json, csv, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import possession as P, analytics as AN

L, W, FPS = 105.0, 68.0, 25.0
ON_BALL = ("PASS", "CARRY", "RECOVERY", "SHOT", "SET PIECE", "BALL LOST")
PRESS_R = 2.0


def pid_of(name, team):
    """Metrica 'Player10' of Home -> 110, of Away -> 210 (metricalab.load_tracking's ids)"""
    if not name or not name.startswith("Player"): return None
    return (100 if team == "Home" else 200) + int(name[6:])


def truth_direction(fr, per):
    k = min(k for k in fr if fr[k][0] == per); ps = fr[k][1]
    ma = np.mean([x for pid, x, y in ps if pid < 200]); mb = np.mean([x for pid, x, y in ps if pid >= 200])
    return {"A": bool(ma < mb), "B": bool(mb < ma)}


def own_x(x, attacks_right):
    """x in the team's own direction: 0 = its own goal line, L = the goal it attacks"""
    return x if attacks_right else L - x


def truth_carrier(ev_rows, per, k0, k1):
    """-> {frame: player id} from the events (see the module doc)"""
    E = sorted([r for r in ev_rows if int(r["Period"]) == per and r["Type"] in ON_BALL and r["Team"] in ("Home", "Away")],
               key=lambda r: int(r["Start Frame"]))
    car = {}
    for a, b in zip(E, E[1:] + [None]):
        f0 = int(a["Start Frame"]); f1 = max(f0, int(a["End Frame"] or 0)); nxt = int(b["Start Frame"]) if b is not None else k1 + 1
        frm = pid_of(a["From"], a["Team"])
        if a["Type"] == "PASS":
            if frm is not None: car[f0] = frm
            to = pid_of(a["To"], a["Team"])
            if to is not None:
                for k in range(f1, max(f1, nxt)): car[k] = to                    # received: his until the next event
        elif frm is not None:
            for k in range(f0, max(f1 + 1, nxt) if a["Type"] in ("CARRY", "RECOVERY") else f1 + 1): car[k] = frm
    return {k: v for k, v in car.items() if k0 <= k <= k1}


def truth_pressures(ev, fr, per, car):
    """same rule as the app with perfect inputs -> ({team: pressures applied}, {team: pressures in its attacking 60%})"""
    import metricalab as ML
    seqs = ML.truth_sequences(ML_events(ev), per); ar = truth_direction(fr, per)
    out = {"A": 0, "B": 0}; zone = {"A": 0, "B": 0}
    for sq in seqs:
        opp = "B" if sq["team"] == "A" else "A"; seen = set()
        for k in range(sq["start"], sq["end"] + 1):
            c = car.get(k); f = fr.get(k)
            if c is None or f is None or (c < 200) != (sq["team"] == "A"): continue
            cp = next(((x, y) for pid, x, y in f[1] if pid == c), None)
            if cp is None: continue
            for pid, x, y in f[1]:
                if (pid >= 200) == (opp == "B") and pid not in seen and np.hypot(x - cp[0], y - cp[1]) <= PRESS_R:
                    seen.add(pid); out[opp] += 1
                    if own_x(cp[0], ar[opp]) >= 0.4 * L: zone[opp] += 1
    return out, zone


_EV = {}
def ML_events(ev_rows):
    key = id(ev_rows)
    if key not in _EV or _EV[key][0] is not ev_rows:
        _EV[key] = ev_rows, [{"team": r["Team"], "type": r["Type"], "sub": r["Subtype"], "period": int(r["Period"]), "f0": int(r["Start Frame"]), "f1": int(r["End Frame"] or 0)} for r in ev_rows]
    return _EV[key][1]


DEF_CHALLENGE = ("TACKLE", "GROUND")


def is_pass_try(r):
    """a pass tried: Metrica's PASS rows are completed passes; a failed one is a BALL LOST (all but THEFT = dispossessed)"""
    return r["Type"] == "PASS" or (r["Type"] == "BALL LOST" and r["Subtype"] != "THEFT")


def refresh_truth(rows, d):
    """recompute the event-only truth fields of saved rows (no pipeline re-run)"""
    std, tps = {}, {}
    for g in sorted({r["game"] for r in rows}):
        fr, ev = load_game(d, f"Sample_Game_{g}")
        for per in sorted({r["half"] for r in rows if r["game"] == g}):
            ks = sorted(k for k in fr if fr[k][0] == per); ar = truth_direction(fr, per)
            std[(g, per)] = truth_standard(ev, fr, per, ar); tps[(g, per)] = truth_pressures(ev, fr, per, truth_carrier(ev, per, ks[0], ks[-1]))
    for r in rows:
        s_, (tp, tz) = std[(r["game"], r["half"])], tps[(r["game"], r["half"])]
        for tm, t in r["teams"].items():
            t.update({"truth_pressures": tp[tm], "truth_pressures_zone": tz[tm], "truth_ppda_same_rule": round(s_[tm]["opp_passes_all"] / tp[tm], 1) if tp[tm] else None,
                      "truth_ppda_standard": s_[tm]["ppda"], "truth_ppda_strict": s_[tm]["ppda_strict"], "truth_opp_passes": s_[tm]["opp_passes_all"],
                      "truth_opp_passes_own60": s_[tm]["opp_passes_own60"], "truth_def_actions": s_[tm]["def_actions"]})
    return rows


def truth_standard(ev_rows, fr, per, ar):
    """standard PPDA per team: other team's passes in its own 60% / the team's defensive actions in that zone"""
    E = [r for r in ev_rows if int(r["Period"]) == per and r["Team"] in ("Home", "Away")]
    def x_of(r):
        if r["Start X"] in ("NaN", ""):
            b = fr.get(int(r["Start Frame"]), (None, None, None))[2]; return b[0] if b is not None else None
        return float(r["Start X"]) * L
    res = {}
    for tm, team, oteam in (("A", "Home", "Away"), ("B", "Away", "Home")):
        opp = "B" if tm == "A" else "A"
        tried = [r for r in E if r["Team"] == oteam and is_pass_try(r)]
        opp_passes = sum(1 for r in tried if x_of(r) is not None and own_x(x_of(r), ar[opp]) < 0.6 * L)
        opp_passes_all = len(tried)
        da = 0; strict = 0
        for r in E:
            x = x_of(r)
            if x is None: continue
            if r["Team"] == team and ((r["Type"] == "CHALLENGE" and r["Subtype"].startswith(DEF_CHALLENGE)) or
                                      (r["Type"] == "RECOVERY" and r["Subtype"] in ("INTERCEPTION", "THEFT"))):
                pass
            elif r["Team"] == oteam and r["Type"] == "FAULT RECEIVED":
                pass
            else:
                continue
            if own_x(x, ar[tm]) >= 0.4 * L:
                da += 1
                strict += not (r["Type"] == "RECOVERY" and r["Subtype"] == "INTERCEPTION")   # Metrica tags most loose-ball pickups 'interception'
        res[tm] = {"opp_passes_own60": opp_passes, "opp_passes_all": opp_passes_all, "def_actions": da, "ppda": round(opp_passes / da, 1) if da else None,
                   "def_actions_strict": strict, "ppda_strict": round(opp_passes / strict, 1) if strict else None}
    return res


def ours_run(fr, per, noisy, seed):
    """the pipeline via metricalab.ours (its noise models), passes on the spell state like ipanema/run.py, then
    analytics.stats like run.py -> (team rows, players, captured state)"""
    import metricalab as ML
    got = {}
    o_ps, o_car, o_sp, o_seq, o_pass, o_cl = P.pipeline_state, P.carriers, P.spell_state, P.sequences, AN.passes, P.confirm_losses
    def ps_(*a, **k): r = o_ps(*a, **k); got["state"] = r[0]; return r
    def car_(*a, **k): r = o_car(*a, **k); got["frames_"] = r[0]; got["ballm"] = r[1]; got["per"] = a[0]; return r
    def sp_(*a, **k): r = o_sp(*a, **k); got["cstate"] = r; return r
    def seq_(*a, **k): r = o_seq(*a, **k); got["seqs"] = r; return r
    def pass_(*a, **k): r = o_pass(*a, **{**k, "state": got["cstate"]}); got["ps"] = r[0]; got["tracks"] = r[1]; return r   # run.py: state = spell state
    def cl_(*a, **k): r = o_cl(*a, **k); got["tvs"] = r[0]; return r
    P.pipeline_state, P.carriers, P.spell_state, P.sequences, AN.passes, P.confirm_losses = ps_, car_, sp_, seq_, pass_, cl_
    try: o = ML.ours(fr, per, noisy, seed=seed)
    finally: P.pipeline_state, P.carriers, P.spell_state, P.sequences, AN.passes, P.confirm_losses = o_ps, o_car, o_sp, o_seq, o_pass, o_cl
    ar = o["attack_right"]
    st = AN.stats(got["per"], got["frames_"], got["tvs"], got["ps"], got["tracks"], got["state"], FPS, L, W, ar, PRESS_R, 5.0, sequences_=got["seqs"])
    got["ar"] = ar
    return st, got


def ours_zone(st, got):
    """lab: our PPDA with the standard zone - opponent passes starting in its own 60% / pressures on a carrier in it"""
    per, frames_, tracks, ar = got["per"], got["frames_"], got["tracks"], got["ar"]; n = len(per); res = {}
    for tm in ("A", "B"):
        opp = "B" if tm == "A" else "A"
        op = [p for p in got["ps"] if p["team"] == opp]
        opz = sum(1 for p in op if own_x(float(p["from_m"][0]), ar[opp]) < 0.6 * L)
        pr = 0
        for sq in got["seqs"]:
            if sq["team"] != opp: continue
            seen = set()
            for k in range(sq["start"], min(n, sq["end"] + 1)):
                f = frames_[k]; c = f["carrier"]
                if c is None or f["team"] != opp or c not in tracks or k not in tracks[c]: continue
                cp = tracks[c][k]
                if own_x(float(cp[0]), ar[tm]) < 0.4 * L: continue
                for r in per[k]:
                    if r[1] == tm and r[0] not in seen and np.linalg.norm(r[2] - cp) <= PRESS_R: seen.add(r[0]); pr += 1
        res[tm] = {"opp_passes_own60": opz, "pressures_zone": pr, "ppda_zone": round(opz / pr, 1) if pr else None}
    return res


def load_game(d, g):
    import metricalab as ML
    gd = f"{d}/{g}"
    home = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Home_Team.csv", 100); away = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Away_Team.csv", 200)
    fr = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home}
    ev = list(csv.DictReader(open(f"{gd}/{g}_RawEventsData.csv")))
    return fr, ev


def main(d, games=("Sample_Game_1", "Sample_Game_2"), halves=(1, 2), noises=(False, "streak", "flicker"), seeds=(0, 1), zone=False):
    rows = []
    for g in games:
        fr, ev = load_game(d, g)
        for per in halves:
            ks = sorted(k for k in fr if fr[k][0] == per); k0, k1 = ks[0], ks[-1]; ar_t = truth_direction(fr, per)
            car = truth_carrier(ev, per, k0, k1)
            tp, tpz = truth_pressures(ev, fr, per, car); std = truth_standard(ev, fr, per, ar_t)
            mins = len(ks) / FPS / 60
            for noisy in noises:
                for seed in (seeds if noisy else (0,)):
                    st, got = ours_run(fr, per, noisy, seed)
                    zz = ours_zone(st, got) if zone else None
                    row = {"game": g[-1], "half": per, "noise": str(noisy), "seed": seed, "minutes": round(mins, 1), "direction_same": got["ar"] == ar_t, "teams": {}}
                    for t in st["teams"]:
                        tm = t["team"]; opp = "B" if tm == "A" else "A"
                        row["teams"][tm] = {"ours_pressures": t["pressures_applied"], "ours_ppda": t["ppda_opp_passes_per_def_action"],
                                            "ours_opp_passes": next(x["passes"] for x in st["teams"] if x["team"] == opp), "ours_sequences_opp": next(x["sequences"] for x in st["teams"] if x["team"] == opp),
                                            "truth_pressures": tp[tm], "truth_pressures_zone": tpz[tm],
                                            "truth_ppda_same_rule": round(std[tm]["opp_passes_all"] / tp[tm], 1) if tp[tm] else None,
                                            "truth_ppda_standard": std[tm]["ppda"], "truth_opp_passes": std[tm]["opp_passes_all"],
                                            "truth_opp_passes_own60": std[tm]["opp_passes_own60"], "truth_def_actions": std[tm]["def_actions"],
                                            "truth_ppda_strict": std[tm]["ppda_strict"]}
                        if zz: row["teams"][tm].update({"ours_" + k: v for k, v in zz[tm].items()})
                    rows.append(row); print(json.dumps(row), flush=True)
    return rows


if __name__ == "__main__":
    a = sys.argv[1:]; out = a[a.index("--out") + 1] if "--out" in a else "results/qa/n0-24d/metrica_ppda.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if "--refresh-truth" in a: rows = refresh_truth(json.load(open(out)), a[0])
    else: rows = main(a[0], zone="--zone" in a)
    json.dump(rows, open(out, "w"), indent=1)
