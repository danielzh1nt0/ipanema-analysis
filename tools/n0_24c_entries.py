"""N0-24c: are the app's final-third entries right? Never checked on pro data before (N0-20b only by eye).

  metrics.field[tm].final_third_entries / entries_count / entries_by_pass / entries_by_carry
      (metrics.field_tilt_and_entries): the ball, under the team's control in the RAW possession state, goes from below
      the 2/3 line into the final third (the previous controlled frame <= 3 frames earlier); N0-20b re-arm: the next one
      counts only after the ball is 3 m back behind the line or 1 s out of control. 'pass' if ball speed > 9 m/s.

Truth (Metrica's labelled events + clean tracked ball):
  owner per frame = team of the last on-ball event (PASS / CARRY / RECOVERY / SHOT / SET PIECE / BALL LOST), dead time
  (BALL OUT -> next on-ball event) left out - so a pass in flight belongs to the passer's team.
  entry = the owned ball crosses the 2/3 line into the team's final third; the same re-arm as the app (3 m back or owner
  change) so line jitter is not counted twice. 'pass' if the crossing frame lies inside one of the team's PASS events.
Ours = the pipeline (tools/metricalab.ours) on Metrica's positions, perfect ball and our 'streak' / 'flicker' ball
noise (2 seeds each). Matching: each truth entry -> nearest unused one of ours, same team, within 2 s.

    PYTHONPATH=. python tools/n0_24c_entries.py <metrica sample-data/data dir> [--out results/qa/n0-24c/metrica_entries.json]
        [--hold 0,1,3]   lab: control carries over loose frames up to X s (0 = today's 3-frame rule)
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipanema import possession as P, metrics as M
from n0_24b_hightov import truth_direction, ON_BALL

L, W, FPS = 105.0, 68.0, 25.0
REARM_M = 3.0


def depth(x, attacks_right): return x if attacks_right else L - x


def truth_owner(ev_rows, fr, per):
    """-> {metrica frame: 'A'/'B'} for live frames, and per team the list of PASS (f0, f1) spans"""
    E = sorted([r for r in ev_rows if int(r["Period"]) == per], key=lambda r: int(r["Start Frame"]))
    ks = sorted(k for k in fr if fr[k][0] == per); f1 = ks[-1]
    own = {}; cur = None; dead = False; passes = {"A": [], "B": []}
    for a, b in zip(E, E[1:] + [None]):
        ty = a["Type"]
        if ty == "BALL OUT": dead = True; cur = None
        elif ty in ON_BALL and a["Team"] in ("Home", "Away"):
            dead = False; cur = "A" if a["Team"] == "Home" else "B"
            if ty == "PASS": passes[cur].append((int(a["Start Frame"]), max(int(a["Start Frame"]), int(a["End Frame"] or 0))))
        end = int(b["Start Frame"]) if b is not None else f1 + 1
        if cur and not dead:
            for k in range(int(a["Start Frame"]), min(end, f1 + 1)): own[k] = cur
    return own, passes


def truth_entries(own, passes, fr, ar):
    out = []
    for tm in ("A", "B"):
        prev = None; armed = True; last_k = None
        for k in sorted(own):
            if last_k is not None and k - last_k > 1: prev = None; armed = True          # dead time between
            last_k = k
            b = fr.get(k, (0, 0, None))[2]
            if own[k] != tm: prev = None; armed = True; continue                        # other team's ball: chain over
            if b is None: continue
            d = depth(b[0], ar[tm])
            if not armed and d <= 2 * L / 3 - REARM_M: armed = True
            if prev is not None and prev < 2 * L / 3 <= d and armed:
                how = "pass" if any(f0 <= k <= f1 for f0, f1 in passes[tm]) else "carry"
                out.append({"team": tm, "f": k, "how": how, "y_m": round(b[1], 1)}); armed = False
            prev = d
    return out


def entries_hold(state, ballm, bspeed, fps, attack_right, hold_s, rearm=M.ENTRY_REARM):
    """lab: like metrics.field_tilt_and_entries, but the team keeps control over loose frames for up to hold_s
    (a pass in flight is loose in our state, so a pass into the final third is today seen only if the receiver's
    first controlled frame comes <= 3 frames after the passer's last)"""
    out = {}
    for tm in ("A", "B"):
        sgn = attack_right[tm]; entries = []; prev = None; last = None; armed = True
        for k in sorted(ballm):
            st = P.STATES[state[k]]
            if st != tm:
                if st in ("A", "B"): prev = None        # the other team has it: no crossing across it; the re-arm stays
                continue                                # time-based as in the app (a 0.3-0.6 s flicker hop must not re-arm)
            dd = depth(ballm[k][0], sgn)
            if last is not None and k - last > max(3, rearm[1] * fps, hold_s * fps): armed = True
            if not armed and dd <= 2 * L / 3 - rearm[0]: armed = True
            if prev is not None and prev[0] < 2 * L / 3 <= dd and k - prev[1] <= max(3, hold_s * fps) and armed:
                how = "pass" if (k - prev[1] > 3 or bspeed.get(k, 0) > 9) else "carry"
                entries.append({"t": round(k / fps, 2), "how": how, "y_m": round(float(ballm[k][1]), 1)}); armed = False
            prev = (dd, k); last = k
        out[tm] = {"final_third_entries": entries, "entries_count": len(entries)}
    return out


def ours_run(fr, per, noisy, seed, holds):
    import metricalab as ML
    got = {}; o_ps, o_car = P.pipeline_state, P.carriers
    def ps_(*a, **k): r = o_ps(*a, **k); got["state"] = r[0]; got["bspeed"] = r[1]; return r
    def car_(*a, **k): r = o_car(*a, **k); got["ballm"] = r[1]; return r
    P.pipeline_state, P.carriers = ps_, car_
    try: o = ML.ours(fr, per, noisy, seed=seed)
    finally: P.pipeline_state, P.carriers = o_ps, o_car
    ar = o["attack_right"]
    res = {"app": M.field_tilt_and_entries(got["state"], got["ballm"], got["bspeed"], FPS, L, ar, *M.ENTRY_REARM)}
    for h in holds:
        if h: res[f"hold{h:g}"] = entries_hold(got["state"], got["ballm"], got["bspeed"], FPS, ar, h)
    return res, ar


def match(te, ours, k0, tol_s=2.0):
    """te: truth entries; ours: {tm: [{t, how}]} -> found, real, how agreement"""
    flat = [(tm, int(round(e["t"] * FPS)) + k0, e["how"]) for tm in ours for e in ours[tm]]
    used = set(); found = 0; how_same = 0; pairs = []
    for t in sorted(te, key=lambda t: t["f"]):
        best = None
        for j, (tm, f, how) in enumerate(flat):
            if j in used or tm != t["team"]: continue
            dt = abs(f - t["f"])
            if dt <= tol_s * FPS and (best is None or dt < best[0]): best = (dt, j)
        if best is not None:
            used.add(best[1]); found += 1; how_same += flat[best[1]][2] == t["how"]; pairs.append((t["how"], flat[best[1]][2]))
    return {"truth": len(te), "ours": len(flat), "found": found, "ours_real": len(used), "how_same": how_same,
            "truth_pass": sum(t["how"] == "pass" for t in te), "found_pass": sum(1 for a, b in pairs if a == "pass"),
            "found_carry": sum(1 for a, b in pairs if a == "carry")}


def main(d, games=("Sample_Game_1", "Sample_Game_2"), halves=(1, 2), noises=(False, "streak", "flicker"), seeds=(0, 1), holds=(0,)):
    import csv, metricalab as ML
    rows = []
    for g in games:
        gd = f"{d}/{g}"; home = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Home_Team.csv", 100); away = ML.load_tracking(f"{gd}/{g}_RawTrackingData_Away_Team.csv", 200)
        fr = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home}
        ev = list(csv.DictReader(open(f"{gd}/{g}_RawEventsData.csv")))
        for per in halves:
            k0 = min(k for k in fr if fr[k][0] == per); ar_t = truth_direction(fr, per)
            own, passes = truth_owner(ev, fr, per); te = truth_entries(own, passes, fr, ar_t)
            for noisy in noises:
                for seed in (seeds if noisy else (0,)):
                    res, ar = ours_run(fr, per, noisy, seed, holds)
                    for rule, field in res.items():
                        ours = {tm: field[tm]["final_third_entries"] for tm in ("A", "B")}
                        row = {"game": g[-1], "half": per, "noise": str(noisy), "seed": seed, "rule": rule, "direction_same": ar == ar_t,
                               "teams": {tm: {"truth": sum(t["team"] == tm for t in te), "ours": len(ours[tm]),
                                              "truth_pass": sum(t["team"] == tm and t["how"] == "pass" for t in te),
                                              "ours_pass": sum(e["how"] == "pass" for e in ours[tm])} for tm in ("A", "B")},
                               "matched": match(te, ours, k0)}
                        rows.append(row); print(json.dumps(row), flush=True)
    return rows


if __name__ == "__main__":
    a = sys.argv[1:]; out = a[a.index("--out") + 1] if "--out" in a else "results/qa/n0-24c/metrica_entries.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    holds = tuple(float(h) for h in a[a.index("--hold") + 1].split(",")) if "--hold" in a else (0,)
    rows = main(a[0], holds=holds); json.dump(rows, open(out, "w"), indent=1)
